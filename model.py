import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from config import ModelConfig
from mamba import MambaBlock


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


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
    """Rotary position embeddings (RoPE, Su et al. 2021) for q/k of shape (B, H, L, head_dim).

    No learned parameters. The cos/sin tables are non-persistent buffers, so they follow
    .to(device) but never appear in state_dict. They are kept in float32 and rebuilt lazily
    if a longer sequence (or a different device/dtype after e.g. model.half()) shows up.
    Uses the "rotate half" pairing (dim i with dim i + head_dim/2).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"RoPE requires an even head_dim, got {head_dim}"
        self.head_dim = head_dim
        self.base = base
        self._build_cache(max_seq_len, torch.device("cpu"))

    def _build_cache(self, seq_len: int, device: torch.device):
        # Elementwise ops only (no matmul/outer), so autocast cannot downcast the table.
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = t[:, None] * inv_freq[None, :]  # (L, head_dim / 2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        x1, x2 = xf.chunk(2, dim=-1)
        out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.to(x.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        l = q.shape[-2]
        cached = self.cos_cached
        if l > cached.shape[0] or cached.device != q.device or cached.dtype != torch.float32:
            self._build_cache(max(l, cached.shape[0]), q.device)
        cos = self.cos_cached[:l]
        sin = self.sin_cached[:l]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)


class AttentionBlock(nn.Module):
    """Multi-Head Self-Attention with Causal Mask and PyTorch SDPA."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)
        self.mlp_norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)
        self.dropout = config.dropout

        # Old pickled configs predate `rope`; ModelConfig.__setstate__ restores it as False for
        # them, and the getattr default covers config objects that are not ModelConfig at all.
        self.rotary = (
            RotaryEmbedding(self.head_dim, config.max_seq_len) if getattr(config, "rope", False) else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        h = self.norm(x)

        # QKV Projections
        qkv = self.qkv_proj(h) # (B, L, 3 * D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B, H, L, D_head)
        q = q.view(b, l, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.n_head, self.head_dim).transpose(1, 2)

        if self.rotary is not None:
            q, k = self.rotary(q, k)

        # Scaled Dot Product Attention with Causal Mask
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        ) # (B, H, L, D_head)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mamba(self.norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class JEPAPredictor(nn.Module):
    """Predictor network for the JEPA objective: context-pass hidden states -> clean hidden states.

    A single bidirectional TransformerEncoderLayer behind a bottleneck. It has no positional
    encoding of its own on purpose: it predicts slot-for-slot (output i is the prediction for
    position i), and its inputs are backbone states that already carry position (causal Mamba
    recurrence/convolution plus RoPE attention), so an extra positional signal is not needed.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.proj_in = nn.Linear(config.d_model, config.jepa_predictor_d_model)
        self.predictor_layer = nn.TransformerEncoderLayer(
            d_model=config.jepa_predictor_d_model,
            nhead=4,
            dim_feedforward=config.jepa_predictor_d_model * 2,
            batch_first=True,
            norm_first=True,
        )
        self.proj_out = nn.Linear(config.jepa_predictor_d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        h = self.predictor_layer(h)
        return self.proj_out(h)


def jepa_span_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float,
    span_len: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """Sample a (B, L) bool mask of contiguous spans covering round(mask_ratio * L) positions per row.

    Spans have length `span_len` except one shorter span holding the remainder (its place
    among the spans is random). Spans never overlap, so every row masks exactly the target
    count; adjacent spans may touch and form a longer run. At least one position always stays
    visible. Returns None when nothing would be masked (very short sequences).
    """
    n_mask = min(int(round(mask_ratio * seq_len)), seq_len - 1)
    if n_mask <= 0:
        return None
    span_len = max(1, min(int(span_len), n_mask))
    n_spans = math.ceil(n_mask / span_len)
    n_visible = seq_len - n_mask

    lens = torch.full((n_spans,), span_len, dtype=torch.long, device=device)
    lens[-1] = n_mask - (n_spans - 1) * span_len
    # Shuffle span lengths per row so the short remainder span is not always the last one.
    order = torch.rand(batch_size, n_spans, device=device, generator=generator).argsort(dim=1)
    lens = lens[order]  # (B, n_spans)

    # Distribute the visible positions into the n_spans + 1 gaps around the spans:
    # gaps[:, i] = number of visible positions before span i (non-decreasing).
    gaps = torch.randint(0, n_visible + 1, (batch_size, n_spans), device=device, generator=generator)
    gaps = gaps.sort(dim=1).values
    starts = gaps + (lens.cumsum(dim=1) - lens)  # (B, n_spans)

    offsets = torch.arange(span_len, device=device)
    pos = starts.unsqueeze(-1) + offsets  # (B, n_spans, span_len)
    pos = torch.where(offsets < lens.unsqueeze(-1), pos, torch.full_like(pos, seq_len))  # pad -> dummy col
    mask = torch.zeros(batch_size, seq_len + 1, dtype=torch.bool, device=device)
    mask.scatter_(1, pos.view(batch_size, -1), True)
    return mask[:, :seq_len]


class Shortcake(nn.Module):
    """Interleaved 3 Mamba layers per 1 Attention layer (~20M parameters).

    With config.jepa_enabled and forward(..., compute_jepa=True) an auxiliary JEPA loss is
    added. It costs roughly one extra backbone forward + backward per training step (a second,
    masked "context" pass over the same input); inference (compute_jepa=False) is unaffected.
    """

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

        # JEPA (experiment): predictor + learned embedding that replaces hidden token embeddings
        if config.jepa_enabled:
            self.jepa_predictor = JEPAPredictor(config)
            self.jepa_mask_embedding = nn.Parameter(torch.zeros(config.d_model))

        # Keep MambaBlock's own dt_proj init (weight ~ U(+-dt_rank**-0.5), bias = inverse-softplus
        # of dt in [0.001, 0.1]); the generic Linear init below would otherwise overwrite it.
        dt_proj_init = {
            id(m): (m.dt_proj.weight.detach().clone(), m.dt_proj.bias.detach().clone())
            for m in self.modules()
            if isinstance(m, MambaBlock)
        }

        # Initialize weights
        self.apply(self._init_weights)

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, MambaBlock):
                    weight, bias = dt_proj_init[id(m)]
                    m.dt_proj.weight.copy_(weight)
                    m.dt_proj.bias.copy_(bias)

            # GPT-2 style scaled init for the projections that write into the residual stream.
            # Selected by module type (not by parameter name) so the JEPA predictor's
            # nn.MultiheadAttention.out_proj is left alone.
            residual_std = 0.02 / math.sqrt(2 * config.n_layer)
            for m in self.modules():
                if isinstance(m, (AttentionBlock, MambaBlock)):
                    nn.init.normal_(m.out_proj.weight, mean=0.0, std=residual_std)
                elif isinstance(m, SwiGLUMLP):
                    nn.init.normal_(m.w2.weight, mean=0.0, std=residual_std)

            if config.jepa_enabled:
                nn.init.normal_(self.jepa_mask_embedding, mean=0.0, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints from the earlier JEPA variant have a predictor but no mask embedding.
        # Keep the fresh init for it so strict loading (generate.py, chat.py) still works.
        key = prefix + "jepa_mask_embedding"
        if hasattr(self, "jepa_mask_embedding") and key not in state_dict:
            state_dict[key] = self.jepa_mask_embedding.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_num_params(self) -> int:
        """Returns total non-embedding parameter count."""
        n_params = sum(p.numel() for p in self.parameters())
        if self.config.tie_weights:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def _embed(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Token embeddings (B, L, D); positions where `mask` is True get the JEPA mask embedding."""
        x = self.tok_embeddings(input_ids)
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.jepa_mask_embedding.to(x.dtype), x)
        return self.drop(x)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Interleaved Mamba/Attention layers + final norm: (B, L, D) embeddings -> hidden states."""
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)

    def _jepa_loss(
        self, input_ids: torch.Tensor, target_h: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Joint-embedding predictive loss.

        Target: the clean-pass hidden states, detached (stop-gradient). There is deliberately
        no EMA target network, to keep parameters and memory down; the LM loss on the same
        hidden states is what keeps them from collapsing to a trivial constant.
        Context: a second backbone pass where contiguous spans of positions have their token
        embeddings replaced by `jepa_mask_embedding`.
        Predictor: maps context states to target states; MSE only on the masked positions.
        """
        b, l = input_ids.shape
        mask = jepa_span_mask(
            b,
            l,
            getattr(self.config, "jepa_mask_ratio", 0.15),
            getattr(self.config, "jepa_span_len", 8),
            device=input_ids.device,
        )
        if mask is None:
            return None, None

        target = target_h.detach()
        context_h = self._backbone(self._embed(input_ids, mask))
        pred = self.jepa_predictor(context_h)
        jepa_loss = F.mse_loss(pred[mask].float(), target[mask].float())
        return jepa_loss, mask

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        compute_jepa: bool = False,
    ) -> Dict[str, Any]:
        """
        input_ids: (B, L)
        targets: (B, L) optional ground truth token targets for Causal LM loss
        compute_jepa: also compute the JEPA loss (only if the model was built with jepa_enabled).
            Runs a second backbone pass; logits/hidden_states/loss always come from the clean pass.
        """
        # Clean pass: LM logits and loss (and the JEPA target) come only from here.
        h = self._backbone(self._embed(input_ids))  # Hidden representation (B, L, D)

        # Output LM logits
        logits = self.lm_head(h) # (B, L, Vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))

        output = {
            "logits": logits,
            "hidden_states": h,
            "loss": loss,
        }

        # Optional JEPA latent loss (masked context pass -> predict clean hidden states)
        if compute_jepa and hasattr(self, "jepa_predictor"):
            jepa_loss, mask = self._jepa_loss(input_ids, h)
            if jepa_loss is not None:
                output["jepa_loss"] = jepa_loss
                output["jepa_mask"] = mask
                if loss is not None:
                    output["total_loss"] = loss + getattr(self.config, "jepa_loss_weight", 0.5) * jepa_loss

        return output
