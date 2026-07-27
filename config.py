import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for Shortcake"""


    vocab_size: int = 16384
    d_model: int = 384
    n_layer: int = 8
    n_head: int = 6
    mamba_ratio: int = 3  # 3 Mamba layers per 1 Attention layer
    max_seq_len: int = 2048
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4  # Causal 1D convolution kernel width
    expand_factor: int = 2  # Block expansion multiplier
    dt_rank: Optional[int] = None  # Rank of dt projection (defaults to d_model / 16)
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_weights: bool = True
    
    # JEPA latent predictor settings
    jepa_enabled: bool = False
    jepa_mask_ratio: float = 0.15
    jepa_predictor_d_model: int = 192

    def __post_init__(self):
        if self.dt_rank is None:
            self.dt_rank = math.ceil(self.d_model / 16)
        assert self.d_model % self.n_head == 0, f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"

    @property
    def d_inner(self) -> int:
        return self.expand_factor * self.d_model

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head
