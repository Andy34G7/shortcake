import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

from config import ModelConfig
from mamba import MambaBlock


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (numerically stable in FP16/BF16)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        out = x_f * torch.rsqrt(variance + self.eps)
        return out.to(x.dtype) * self.weight


class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(8 / 3 * config.d_model)
        # Ensure hidden_dim is a multiple of 64
        hidden_dim = 64 * ((hidden_dim + 63) // 64)

        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for Attention layers."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,  # (B, H, L, D_head)
        k: torch.Tensor,  # (B, H, L, D_head)
        pos_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        l = q.shape[2]
        positions = torch.arange(pos_offset, pos_offset + l, device=q.device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)  # (L, D_head // 2)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]

        def _rotate(x: torch.Tensor) -> torch.Tensor:
            x_f = x.float()
            x1, x2 = x_f.chunk(2, dim=-1)
            out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return out.to(x.dtype)

        return _rotate(q), _rotate(k)


class AttentionBlock(nn.Module):
    """Multi-Head Self-Attention with QK-Norm, RoPE, KV-Cache, and PyTorch SDPA."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = RotaryEmbedding(self.head_dim)
        self.norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)
        self.mlp_norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        is_causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, l, d = x.shape
        h = self.norm(x)

        # QKV Projections
        qkv = self.qkv_proj(h)  # (B, L, 3 * D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B, H, L, D_head)
        q = q.view(b, l, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.n_head, self.head_dim).transpose(1, 2)

        # QK-Norm for attention logit stability
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE with cached position offset
        past_k = cache.get("k") if cache is not None else None
        past_v = cache.get("v") if cache is not None else None
        pos_offset = cache.get("pos", 0) if cache is not None else 0
        q, k = self.rope(q, k, pos_offset=pos_offset)

        if cache is not None:
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            # Keep KV cache bounded to max_seq_len
            if k.shape[2] > self.config.max_seq_len:
                k = k[:, :, -self.config.max_seq_len :, :]
                v = v[:, :, -self.config.max_seq_len :, :]
            cache["k"] = k
            cache["v"] = v
            cache["pos"] = pos_offset + l

        # Scaled Dot Product Attention
        dropout_p = self.dropout if self.training else 0.0
        if past_k is None:
            if is_causal:
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=dropout_p
                )
            elif attention_mask is not None:
                # Bidirectional attention with key-padding mask: (B, 1, 1, L) bool
                pad_mask = attention_mask.to(dtype=torch.bool)[:, None, None, :]
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=pad_mask, is_causal=False, dropout_p=dropout_p
                )
            else:
                attn_out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=False, dropout_p=dropout_p
                )
        elif l == 1:
            # Single-token decode attends to all cached past keys + current key
            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=dropout_p
            )
        else:
            # Multi-token continuation with existing cache
            total_kv = k.shape[2]
            past_len = total_kv - l
            q_idx = torch.arange(l, device=x.device).unsqueeze(1) + past_len
            k_idx = torch.arange(total_kv, device=x.device).unsqueeze(0)
            attn_mask = k_idx <= q_idx
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(b, l, d)
        attn_out = self.out_proj(attn_out)

        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MambaBlockWrapper(nn.Module):
    """Mamba SSM Layer wrapped with RMSNorm, MLP, and residual connections."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mamba = MambaBlock(config)
        self.mlp = SwiGLUMLP(config)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        bidirectional: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        bwd_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm(x)
        fwd = self.mamba(h, cache=cache)
        if bidirectional and cache is None:
            if attention_mask is not None:
                # Reverse only valid (non-pad) prefix per row so right-padding never pollutes backward SSM state
                lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
                h_rev = torch.zeros_like(h)
                for i in range(h.shape[0]):
                    n = int(lengths[i].item())
                    if n > 0:
                        h_rev[i, :n] = h[i, :n].flip(0)
                bwd_rev = self.mamba(h_rev, cache=None)
                bwd = torch.zeros_like(bwd_rev)
                for i in range(h.shape[0]):
                    n = int(lengths[i].item())
                    if n > 0:
                        bwd[i, :n] = bwd_rev[i, :n].flip(0)
            else:
                bwd = self.mamba(h.flip(1), cache=None).flip(1)
            gate = bwd_gate.to(dtype=bwd.dtype) if bwd_gate is not None else 1.0
            x = x + fwd + gate * bwd
        else:
            x = x + fwd
        x = x + self.mlp(self.mlp_norm(x))
        return x


class JEPAPredictor(nn.Module):
    """Predictor network for JEPA latent block representation learning."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        self.proj_in = nn.Linear(config.d_model, config.jepa_predictor_d_model)
        self.predictor_layer = nn.TransformerEncoderLayer(
            d_model=config.jepa_predictor_d_model,
            nhead=4,
            dim_feedforward=config.jepa_predictor_d_model * 2,
            batch_first=True,
            norm_first=True,
        )
        self.proj_out = nn.Linear(config.jepa_predictor_d_model, config.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replaces masked positions with a learned mask_token before predicting their latents."""
        mask_tok = self.mask_token.expand(x.shape[0], x.shape[1], -1).to(x.dtype)
        masked_x = torch.where(mask.unsqueeze(-1), mask_tok, x)
        h = self.proj_in(masked_x)
        h = self.predictor_layer(h)
        return self.proj_out(h)


class Shortcake(nn.Module):
    """Interleaved 3 Mamba layers per 1 Attention layer (~20M parameters)."""


    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        layers = []
        for i in range(config.n_layer):
            if (i + 1) % (config.mamba_ratio + 1) == 0:
                layers.append(AttentionBlock(config))
            else:
                layers.append(MambaBlockWrapper(config))
        self.layers = nn.ModuleList(layers)

        self.norm_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying between input embedding and output lm_head
        if config.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        # JEPA (experiment)
        if config.jepa_enabled:
            self.jepa_predictor = JEPAPredictor(config)

        # Initialize weights
        self.apply(self._init_weights)

        # Apply GPT-2 / Mamba residual projection scaling (1 / sqrt(2 * n_layer))
        # and restore Mamba dt_proj weight & bias initialization overwritten by _init_weights
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("mlp.w2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=residual_std)
        for m in self.modules():
            if isinstance(m, MambaBlock):
                m._reset_dt_parameters()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Returns total non-embedding parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if self.config.tie_weights:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def init_cache(self) -> List[Dict[str, Any]]:
        """Initialize an empty per-layer inference cache (SSM states + Attention KV cache)."""
        return [{} for _ in range(len(self.layers))]

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        compute_jepa: bool = False,
        cache: Optional[List[Dict[str, Any]]] = None,
        return_logits: bool = True,
        bidirectional: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        bwd_gates: Optional[nn.ParameterList] = None,
    ) -> Dict[str, Any]:
        """
        input_ids: (B, L)
        targets: (B, L) optional ground truth token targets for Causal LM loss
        cache: Optional list of per-layer dicts returned by init_cache() for fast O(1) decode
        return_logits: Set False to skip lm_head projection (e.g., when using SPLADE head)
        bidirectional: When True (cache must be None), runs bidirectional Attention + BiMamba SSM
        """
        b, l = input_ids.shape
        x = self.tok_embeddings(input_ids)
        x = self.drop(x)

        # Pass through interleaved backbone
        mamba_idx = 0
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            if isinstance(layer, AttentionBlock):
                x = layer(x, cache=layer_cache, is_causal=not bidirectional, attention_mask=attention_mask)
            else:
                gate = bwd_gates[mamba_idx] if (bwd_gates is not None and mamba_idx < len(bwd_gates)) else None
                mamba_idx += 1
                x = layer(x, cache=layer_cache, bidirectional=bidirectional, attention_mask=attention_mask, bwd_gate=gate)

        h = self.norm_f(x)  # Hidden representation (B, L, D)

        # Output LM logits (skip when only hidden_states are needed)
        logits = self.lm_head(h) if (return_logits or targets is not None) else None

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

        output = {
            "logits": logits,
            "hidden_states": h,
            "loss": loss,
        }

        # Optional JEPA Latent Loss calculation (masked positions replaced with mask_token)
        if compute_jepa and hasattr(self, "jepa_predictor"):
            mask_ratio = self.config.jepa_mask_ratio
            mask = torch.rand(b, l, device=input_ids.device) < mask_ratio
            if mask.sum() > 0:
                pred_h = self.jepa_predictor(h, mask)
                jepa_loss = F.mse_loss(pred_h[mask], h[mask].detach())
                output["jepa_loss"] = jepa_loss
                if loss is not None:
                    output["total_loss"] = loss + 0.5 * jepa_loss

        return output
