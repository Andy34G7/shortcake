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
    """Predictor network for JEPA latent block representation learning."""

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

        # Re-apply Mamba dt_proj bias initialization (prevent zeroing by _init_weights)
        for m in self.modules():
            if isinstance(m, MambaBlock):
                m._reset_dt_bias()

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

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        compute_jepa: bool = False,
    ) -> Dict[str, Any]:
        """
        input_ids: (B, L)
        targets: (B, L) optional ground truth token targets for Causal LM loss
        """
        b, l = input_ids.shape
        x = self.tok_embeddings(input_ids)
        x = self.drop(x)

        # Pass through interleaved backbone
        for layer in self.layers:
            x = layer(x)

        h = self.norm_f(x) # Hidden representation (B, L, D)

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

        # Optional JEPA Latent Loss calculation
        if compute_jepa and hasattr(self, "jepa_predictor"):
            # Simple JEPA loss: predict hidden states of random token positions
            mask_ratio = self.config.jepa_mask_ratio
            mask = torch.rand(b, l, device=input_ids.device) < mask_ratio
            if mask.sum() > 0:
                pred_h = self.jepa_predictor(h)
                jepa_loss = F.mse_loss(pred_h[mask], h[mask].detach())
                output["jepa_loss"] = jepa_loss
                if loss is not None:
                    output["total_loss"] = loss + 0.5 * jepa_loss

        return output
