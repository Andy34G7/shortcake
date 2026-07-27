"""
Mamba State Space Model (SSM) Block.

Based on the original Mamba architecture:
"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
by Albert Gu and Tri Dao (2023).
Official Repository: https://github.com/state-spaces/mamba
Licensed under the Apache License, Version 2.0.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


from config import ModelConfig

# Try importing official compiled CUDA kernels if installed on environment
HAS_MAMBA_CUDA = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    selective_scan_fn = None


def selective_scan_pytorch(
    u: torch.Tensor,        # (B, D, L)
    delta: torch.Tensor,    # (B, D, L)
    A: torch.Tensor,        # (D, N)
    B: torch.Tensor,        # (B, N, L)
    C: torch.Tensor,        # (B, N, L)
    D: Optional[torch.Tensor] = None, # (D,)
) -> torch.Tensor:
    """Pure PyTorch native selective scan algorithm for Mamba SSM.
    
    u: (B, D, L)
    delta: (B, D, L)
    A: (D, N)
    B: (B, N, L)
    C: (B, N, L)
    D: (D,)
    Returns: (B, D, L)
    """
    b, d, l = u.shape
    n = A.shape[1]

    # Discretize A and B:
    # delta: (B, D, L) -> (B, D, L, 1)
    # A: (D, N) -> (1, D, 1, N)
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)) # (B, D, L, N)
    
    # delta: (B, D, L, 1)
    # B: (B, 1, L, N)
    # u: (B, D, L, 1)
    deltaB_u = delta.unsqueeze(-1) * B.permute(0, 2, 1).unsqueeze(1) * u.unsqueeze(-1) # (B, D, L, N)

    # Recurrent scan loop across sequence length L
    x = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
    ys = []
    
    C_perm = C.permute(0, 2, 1) # (B, L, N)

    for i in range(l):
        x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :] # (B, D, N)
        # y_i = sum_n (x_i * C_i)
        y_i = torch.einsum("bdn,bn->bd", x, C_perm[:, i, :]) # (B, D)
        ys.append(y_i)

    y = torch.stack(ys, dim=-1) # (B, D, L)

    if D is not None:
        y = y + u * D.unsqueeze(-1)

    return y


class MambaBlock(nn.Module):
    """Mamba State Space Model (SSM) Layer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.dt_rank = config.dt_rank

        # In projection for x and gate z
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Causal 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )

        # Projections for dt, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D real initialization for A
        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32), repeats=self.d_inner
        ).reshape(self.d_inner, self.d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Out projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Parameter initialization
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt_proj bias such that softplus(dt_proj.bias) ~ dt_init
        dt_scale = 1.0
        dt_min = 0.001
        dt_max = 0.1
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        Returns: (B, L, D)
        """
        b, l, d = x.shape

        # 1. Project input to inner dimension and split into x and gate z
        xz = self.in_proj(x) # (B, L, 2 * d_inner)
        x_inner, z = xz.chunk(2, dim=-1) # (B, L, d_inner)

        # 2. 1D Causal Convolution
        x_conv = x_inner.transpose(1, 2) # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :l] # Causal pad truncation
        x_conv = F.silu(x_conv) # (B, d_inner, L)

        # 3. Project to dt, B, C parameters
        x_flat = x_conv.transpose(1, 2) # (B, L, d_inner)
        x_db = self.x_proj(x_flat) # (B, L, dt_rank + 2 * d_state)
        dt, B_ssm, C_ssm = torch.split(
            x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = self.dt_proj(dt) # (B, L, d_inner)
        dt = F.softplus(dt)

        # Transpose for selective scan: (B, d_inner, L)
        u = x_conv
        delta = dt.transpose(1, 2)
        B_ssm = B_ssm.transpose(1, 2) # (B, d_state, L)
        C_ssm = C_ssm.transpose(1, 2) # (B, d_state, L)
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)

        # 4. Run Selective Scan
        if HAS_MAMBA_CUDA and not self.training:
            # Use fast CUDA kernel if present
            y = selective_scan_fn(
                u, delta, A, B_ssm, C_ssm, self.D.float(), z=None, delta_bias=None, delta_softplus=False
            )
        else:
            # Native PyTorch scan
            y = selective_scan_pytorch(u, delta, A, B_ssm, C_ssm, self.D.float())

        # 5. Multiply by gate z (SiLU(z)) and project out
        y = y.transpose(1, 2) # (B, L, d_inner)
        y = y * F.silu(z)
        out = self.out_proj(y) # (B, L, d_model)

        return out
