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


SCAN_CHUNK_SIZE = 16  # timesteps per chunk in the pure-PyTorch scan (see selective_scan_pytorch)


def _scan_chunk_terms(dt, w, B_t, A, dA_out, dBu_out):
    """Fill dA = exp(dt * A) and dBu = (dt * u) * B for one chunk into preallocated buffers.

    dt, w (= dt * u): (T, B, D); B_t: (T, B, N); A: (D, N); outputs: (T, B, D, N).
    """
    torch.mul(dt.unsqueeze(-1), A, out=dA_out).exp_()
    torch.mul(w.unsqueeze(-1), B_t.unsqueeze(2), out=dBu_out)


def _scan_chunk_states(dA, dBu, H):
    """Run h_t = dA_t * h_{t-1} + dBu_t over one chunk.

    H: (T + 1, B, D, N) buffer; H[0] must hold the incoming state, H[1:] receives h_1..h_T.
    """
    h = H[0]
    for t in range(dA.shape[0]):
        h = torch.addcmul(dBu[t], dA[t], h, out=H[t + 1])


def _scan_forward(u, delta, A, B, C, chunk, save_starts):
    """Chunked forward scan, time-major. Returns y (B, D, L) and the chunk-start states (or None)."""
    b, d, l = u.shape
    n = A.shape[1]
    dt = delta.permute(2, 0, 1).contiguous()  # (L, B, D)
    w = dt * u.permute(2, 0, 1)                 # (L, B, D)  dt * u
    B_t = B.permute(2, 0, 1).contiguous()       # (L, B, N)
    C_t = C.permute(2, 0, 1).contiguous()       # (L, B, N)

    T = max(1, min(chunk, l))
    n_chunks = (l + chunk - 1) // chunk
    dA = u.new_empty(T, b, d, n)
    dBu = torch.empty_like(dA)
    H = u.new_empty(T + 1, b, d, n)
    H[0].zero_()
    starts = u.new_empty(n_chunks, b, d, n) if save_starts else None
    y = u.new_empty(l, b, d, 1)

    for ci, s in enumerate(range(0, l, chunk)):
        e = min(l, s + chunk)
        tl = e - s
        if starts is not None:
            starts[ci].copy_(H[0])
        _scan_chunk_terms(dt[s:e], w[s:e], B_t[s:e], A, dA[:tl], dBu[:tl])
        _scan_chunk_states(dA[:tl], dBu[:tl], H[: tl + 1])
        # y_t = <h_t, C_t> over the state dimension
        torch.matmul(H[1 : tl + 1], C_t[s:e].unsqueeze(-1), out=y[s:e])
        H[0].copy_(H[tl])

    return y.squeeze(-1).permute(1, 2, 0), starts


class _SelectiveScanFn(torch.autograd.Function):
    """Selective scan with a hand-written backward (reverse-time adjoint recurrence).

    Only the inputs plus one (B, D, N) state per chunk are saved for backward; per-chunk
    states are recomputed in backward, so memory is O(B*D*N*(chunk + L/chunk)) rather than
    the O(B*L*D*N) that autograd through a Python loop would store.
    Inputs arrive as float32/float64 (the wrapper casts) and autocast is disabled inside.
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, chunk):
        with torch.autocast(device_type=u.device.type, enabled=False):
            y, starts = _scan_forward(u, delta, A, B, C, chunk, save_starts=True)
        ctx.chunk = chunk
        ctx.save_for_backward(u, delta, A, B, C, starts)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        u, delta, A, B, C, starts = ctx.saved_tensors
        chunk = ctx.chunk
        b, d, l = u.shape
        n = A.shape[1]
        with torch.autocast(device_type=u.device.type, enabled=False):
            dy = dy.to(u.dtype)
            u_t = u.permute(2, 0, 1)                   # (L, B, D)
            dt = delta.permute(2, 0, 1).contiguous()   # (L, B, D)
            w = dt * u_t                               # (L, B, D)
            B_t = B.permute(2, 0, 1).contiguous()      # (L, B, N)
            C_t = C.permute(2, 0, 1).contiguous()      # (L, B, N)
            dy_t = dy.permute(2, 0, 1).contiguous()    # (L, B, D)

            T = max(1, min(chunk, l))
            dA = u.new_empty(T, b, d, n)
            dBu = torch.empty_like(dA)                 # reused for dy*C, then for E below
            H = u.new_empty(T + 1, b, d, n)            # H[0] = h_{s-1}, H[1:] = h_s..h_{e-1}
            G = u.new_empty(T + 1, b, d, n)            # G[t] = dL/dh_t; G[T] = carry from next chunk
            G[T].zero_()
            dA_next = u.new_zeros(b, d, n)             # dA of the first step of the following chunk

            dw = u.new_empty(l, b, d, 1)               # dL/d(dt*u)
            d_dt_from_A = u.new_empty(l, b, d)
            dB = u.new_empty(l, b, 1, n)
            dC = u.new_empty(l, b, 1, n)
            dA_param = torch.zeros_like(A)

            chunk_starts = list(range(0, l, chunk))
            for ci in reversed(range(len(chunk_starts))):
                s = chunk_starts[ci]
                e = min(l, s + chunk)
                tl = e - s
                # Recompute this chunk's states from its saved starting state.
                _scan_chunk_terms(dt[s:e], w[s:e], B_t[s:e], A, dA[:tl], dBu[:tl])
                H[0].copy_(starts[ci])
                _scan_chunk_states(dA[:tl], dBu[:tl], H[: tl + 1])

                # Adjoint recurrence: g_t = dy_t * C_t + dA_{t+1} * g_{t+1}
                if tl < T:
                    G[tl].copy_(G[T])
                q = torch.mul(dy_t[s:e].unsqueeze(-1), C_t[s:e].unsqueeze(2), out=dBu[:tl])
                g = G[tl]
                for t in reversed(range(tl)):
                    g = torch.addcmul(q[t], dA_next, g, out=G[t])
                    dA_next = dA[t]
                dA_next = dA_next.clone()  # dA buffer is overwritten by the next chunk
                G[T].copy_(G[0])
                Gc = G[:tl]

                torch.matmul(dy_t[s:e].unsqueeze(-2), H[1 : tl + 1], out=dC[s:e])  # sum_d dy * h
                torch.matmul(Gc, B_t[s:e].unsqueeze(-1), out=dw[s:e])            # sum_n g * B
                torch.matmul(w[s:e].unsqueeze(-2), Gc, out=dB[s:e])              # sum_d g * dt*u
                # E = dL/d(dt*A) = g_t * h_{t-1} * dA_t
                E = torch.mul(Gc, dA[:tl], out=q).mul_(H[:tl])
                torch.sum(torch.mul(E, A, out=dA[:tl]), -1, out=d_dt_from_A[s:e])
                dA_param.add_(E.mul_(dt[s:e].unsqueeze(-1)).sum((0, 1)))

            dw = dw.squeeze(-1)
            du = dw * dt
            ddelta = d_dt_from_A + dw * u_t
        return (
            du.permute(1, 2, 0),
            ddelta.permute(1, 2, 0),
            dA_param,
            dB.squeeze(2).permute(1, 2, 0),
            dC.squeeze(2).permute(1, 2, 0),
            None,
        )


def selective_scan_pytorch(
    u: torch.Tensor,        # (B, D, L)
    delta: torch.Tensor,    # (B, D, L)
    A: torch.Tensor,        # (D, N)
    B: torch.Tensor,        # (B, N, L)
    C: torch.Tensor,        # (B, N, L)
    D: Optional[torch.Tensor] = None, # (D,)
    chunk_size: int = SCAN_CHUNK_SIZE,
) -> torch.Tensor:
    """Pure-PyTorch selective scan (fallback when the mamba_ssm CUDA kernel is unavailable).

    Computes h_t = exp(delta_t * A) * h_{t-1} + delta_t * u_t * B_t and
    y_t = <h_t, C_t> (+ D * u_t), exactly (no approximation), sequentially over time.

    The sequence is processed in chunks of `chunk_size` steps: exp(delta*A) and delta*u*B are
    computed for a whole chunk at once into reused buffers, leaving only one fused
    multiply-add per timestep in the Python loop. Gradients come from a hand-written
    reverse-time recurrence (`_SelectiveScanFn`) that recomputes states chunk by chunk, so the
    (B, L, D, N) state history is never materialised.

    The recurrence runs in float32 (float64 if the inputs are float64) with autocast disabled. The SSM output is cast back to
    u.dtype before the D skip term is added, matching the dtype behaviour of the original
    loop implementation. Double backward is not supported.
    """
    with torch.autocast(device_type=u.device.type, enabled=False):
        # At least float32 (float64 inputs stay float64), like the state in the original loop.
        dtype = torch.promote_types(torch.promote_types(u.dtype, A.dtype), torch.float32)
        args = tuple(t.to(dtype) for t in (u, delta, A, B, C))
        if torch.is_grad_enabled() and any(t.requires_grad for t in (u, delta, A, B, C)):
            y = _SelectiveScanFn.apply(*args, chunk_size)
        else:
            y, _ = _scan_forward(*args, chunk_size, save_starts=False)
    y = y.to(u.dtype)

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
        self._reset_dt_bias()

    def _reset_dt_bias(self):
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
        delta = dt.transpose(1, 2).to(u.dtype)
        B_ssm = B_ssm.transpose(1, 2).to(u.dtype)
        C_ssm = C_ssm.transpose(1, 2).to(u.dtype)
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)

        # 4. Run Selective Scan
        if HAS_MAMBA_CUDA:
            # Use fast CUDA kernel if present
            y = selective_scan_fn(
                u, delta, A, B_ssm, C_ssm, self.D.float(), z=None, delta_bias=None, delta_softplus=False
            )
        else:
            # Native PyTorch scan. Its custom backward already recomputes states chunk by chunk,
            # so no gradient checkpointing is needed to keep memory low.
            y = selective_scan_pytorch(u, delta, A, B_ssm, C_ssm, self.D.float())

        # 5. Multiply by gate z (SiLU(z)) and project out
        y = y.transpose(1, 2) # (B, L, d_inner)
        y = y * F.silu(z)
        out = self.out_proj(y) # (B, L, d_model)

        return out
