import math
from dataclasses import dataclass
from typing import Optional, Dict, Any


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny": {"d_model": 192, "n_layer": 4, "n_head": 3},        # ~5.2M params
    "small": {"d_model": 384, "n_layer": 8, "n_head": 6},       # ~16.4M params (default)
    "base": {"d_model": 512, "n_layer": 12, "n_head": 8},       # ~48.5M params
    "medium": {"d_model": 768, "n_layer": 12, "n_head": 12},    # ~108.2M params
    "large": {"d_model": 1024, "n_layer": 24, "n_head": 16},    # ~345.8M params
}


@dataclass
class ModelConfig:
    """Configuration for Shortcake (Hybrid Mamba State Space Model + Self Attention)"""

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

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "ModelConfig":
        """Construct a ModelConfig from a preset name ('tiny'/'5m', 'small'/'16m', 'base'/'50m', 'medium'/'110m', 'large'/'350m')."""
        clean_name = str(name).lower().replace("shortcake-", "").strip()
        if clean_name.endswith("m") and clean_name[:-1].isdigit():
            clean_name = clean_name[:-1]

        aliases = {
            "5": "tiny",
            "16": "small",
            "20": "small",
            "50": "base",
            "110": "medium",
            "350": "large",
        }
        preset_key = aliases.get(clean_name, clean_name)
        if preset_key not in MODEL_PRESETS:
            valid_keys = list(MODEL_PRESETS.keys()) + [f"{k}m" for k in aliases.keys()]
            raise ValueError(f"Unknown model preset '{name}'. Choose from: {valid_keys}")

        config_dict = MODEL_PRESETS[preset_key].copy()
        config_dict.update(kwargs)
        return cls(**config_dict)
