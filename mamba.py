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
from typing import Optional, Tuple, Dict, Union


from config import ModelConfig

# Try importing official compiled CUDA kernels if installed on environment
HAS_MAMBA_CUDA = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    selective_scan_fn = None

HAS_CAUSAL_CONV1D = False
try:
    from causal_conv1d import causal_conv1d_fn
    HAS_CAUSAL_CONV1D = True
except ImportError:
    causal_conv1d_fn = None

HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None


if HAS_TRITON:
    @triton.jit
    def _ssm_fwd_kernel(
        u_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, y_ptr, h_ckpts_ptr, last_state_ptr,
        B_batch, L, D_dim, N_state: tl.constexpr,
        CHUNK_L: tl.constexpr, BLOCK_D: tl.constexpr,
        SAVE_CKPTS: tl.constexpr, SAVE_LAST: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_dim
        offs_n = tl.arange(0, N_state)

        a_val = tl.load(A_ptr + offs_d[:, None] * N_state + offs_n[None, :], mask=mask_d[:, None], other=0.0).to(tl.float32)
        d_skip = tl.load(D_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

        h = tl.zeros((BLOCK_D, N_state), dtype=tl.float32)
        num_chunks = (L + CHUNK_L - 1) // CHUNK_L

        u_base = pid_b * L * D_dim + offs_d
        bc_base = pid_b * L * N_state + offs_n

        for t in range(L):
            if SAVE_CKPTS:
                if t % CHUNK_L == 0:
                    c_idx = t // CHUNK_L
                    ckpt_off = ((pid_b * num_chunks + c_idx) * D_dim + offs_d[:, None]) * N_state + offs_n[None, :]
                    tl.store(h_ckpts_ptr + ckpt_off, h, mask=mask_d[:, None])

            u_t = tl.load(u_ptr + u_base + t * D_dim, mask=mask_d, other=0.0).to(tl.float32)
            dt_t = tl.load(dt_ptr + u_base + t * D_dim, mask=mask_d, other=0.0).to(tl.float32)
            B_t = tl.load(B_ptr + bc_base + t * N_state).to(tl.float32)
            C_t = tl.load(C_ptr + bc_base + t * N_state).to(tl.float32)

            dA = tl.exp(dt_t[:, None] * a_val)
            h = dA * h + (dt_t * u_t)[:, None] * B_t[None, :]
            y_t = tl.sum(h * C_t[None, :], axis=1) + u_t * d_skip
            tl.store(y_ptr + u_base + t * D_dim, y_t, mask=mask_d)

        if SAVE_LAST:
            last_off = (pid_b * D_dim + offs_d[:, None]) * N_state + offs_n[None, :]
            tl.store(last_state_ptr + last_off, h, mask=mask_d[:, None])

    @triton.jit
    def _ssm_bwd_kernel(
        u_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, dy_ptr, h_ckpts_ptr,
        du_ptr, ddt_ptr, dA_ptr, dB_ptr, dC_ptr, dD_ptr,
        B_batch, L, D_dim, N_state: tl.constexpr,
        CHUNK_L: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_dim
        offs_n = tl.arange(0, N_state)

        a_val = tl.load(A_ptr + offs_d[:, None] * N_state + offs_n[None, :], mask=mask_d[:, None], other=0.0).to(tl.float32)
        d_skip = tl.load(D_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

        num_chunks = (L + CHUNK_L - 1) // CHUNK_L
        u_base = pid_b * L * D_dim + offs_d
        bc_base = pid_b * L * N_state + offs_n

        dh = tl.zeros((BLOCK_D, N_state), dtype=tl.float32)
        dA_acc = tl.zeros((BLOCK_D, N_state), dtype=tl.float32)
        dD_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for c_rev in range(num_chunks):
            c_idx = num_chunks - 1 - c_rev
            t_start = c_idx * CHUNK_L
            t_end = tl.minimum(t_start + CHUNK_L, L)
            c_len = t_end - t_start

            ckpt_off = ((pid_b * num_chunks + c_idx) * D_dim + offs_d[:, None]) * N_state + offs_n[None, :]
            h_init = tl.load(h_ckpts_ptr + ckpt_off, mask=mask_d[:, None], other=0.0).to(tl.float32)

            for k in range(CHUNK_L):
                step_in_chunk = c_len - 1 - k
                if step_in_chunk >= 0:
                    t = t_start + step_in_chunk
                    h_prev = h_init
                    for j in range(CHUNK_L):
                        if j < step_in_chunk:
                            tj = t_start + j
                            uj = tl.load(u_ptr + u_base + tj * D_dim, mask=mask_d, other=0.0).to(tl.float32)
                            dtj = tl.load(dt_ptr + u_base + tj * D_dim, mask=mask_d, other=0.0).to(tl.float32)
                            Bj = tl.load(B_ptr + bc_base + tj * N_state).to(tl.float32)
                            h_prev = tl.exp(dtj[:, None] * a_val) * h_prev + (dtj * uj)[:, None] * Bj[None, :]

                    u_t = tl.load(u_ptr + u_base + t * D_dim, mask=mask_d, other=0.0).to(tl.float32)
                    dt_t = tl.load(dt_ptr + u_base + t * D_dim, mask=mask_d, other=0.0).to(tl.float32)
                    B_t = tl.load(B_ptr + bc_base + t * N_state).to(tl.float32)
                    C_t = tl.load(C_ptr + bc_base + t * N_state).to(tl.float32)
                    dy_t = tl.load(dy_ptr + u_base + t * D_dim, mask=mask_d, other=0.0).to(tl.float32)

                    dA_t = tl.exp(dt_t[:, None] * a_val)
                    h_t = dA_t * h_prev + (dt_t * u_t)[:, None] * B_t[None, :]

                    dh = dh + dy_t[:, None] * C_t[None, :]
                    dC_part = tl.sum(tl.where(mask_d[:, None], dy_t[:, None] * h_t, 0.0), axis=0)
                    tl.atomic_add(dC_ptr + bc_base + t * N_state, dC_part)
                    dD_acc += tl.where(mask_d, dy_t * u_t, 0.0)

                    d_dt_A = dh * h_prev * dA_t
                    dA_acc += d_dt_A * dt_t[:, None]
                    d_dt_from_A = tl.sum(d_dt_A * a_val, axis=1)

                    d_dtu = tl.sum(dh * B_t[None, :], axis=1)
                    ddt_t = d_dt_from_A + d_dtu * u_t
                    du_t = dy_t * d_skip + d_dtu * dt_t

                    dB_part = tl.sum(tl.where(mask_d[:, None], dh * (dt_t * u_t)[:, None], 0.0), axis=0)
                    tl.atomic_add(dB_ptr + bc_base + t * N_state, dB_part)

                    tl.store(du_ptr + u_base + t * D_dim, du_t, mask=mask_d)
                    tl.store(ddt_ptr + u_base + t * D_dim, ddt_t, mask=mask_d)

                    dh = dh * dA_t

        tl.atomic_add(dA_ptr + offs_d[:, None] * N_state + offs_n[None, :], dA_acc, mask=mask_d[:, None])
        tl.atomic_add(dD_ptr + offs_d, dD_acc, mask=mask_d)


    class _TritonSelectiveScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, delta, A, B_ssm, C_ssm, D_skip, return_last_state=False):
            u_bld = u.transpose(1, 2).contiguous()
            dt_bld = delta.transpose(1, 2).contiguous()
            A_dn = A.float().contiguous()
            B_bln = B_ssm.transpose(1, 2).contiguous()
            C_bln = C_ssm.transpose(1, 2).contiguous()
            D_d = D_skip.float().contiguous()

            B_batch, L, D_dim = u_bld.shape
            N_state = A_dn.shape[1]
            CHUNK_L = 16
            BLOCK_D = 32
            num_chunks = (L + CHUNK_L - 1) // CHUNK_L

            y_bld = torch.empty_like(u_bld)
            save_ckpts = any(ctx.needs_input_grad[:5])
            h_ckpts = (
                torch.empty((B_batch, num_chunks, D_dim, N_state), device=u.device, dtype=torch.float32)
                if save_ckpts
                else torch.empty(1, device=u.device, dtype=torch.float32)
            )
            last_state = (
                torch.empty((B_batch, D_dim, N_state), device=u.device, dtype=torch.float32)
                if return_last_state
                else torch.empty(1, device=u.device, dtype=torch.float32)
            )

            grid = (B_batch, triton.cdiv(D_dim, BLOCK_D))
            _ssm_fwd_kernel[grid](
                u_bld, dt_bld, A_dn, B_bln, C_bln, D_d, y_bld, h_ckpts, last_state,
                B_batch, L, D_dim, N_state=N_state, CHUNK_L=CHUNK_L, BLOCK_D=BLOCK_D,
                SAVE_CKPTS=save_ckpts, SAVE_LAST=return_last_state,
            )
            ctx.save_for_backward(u_bld, dt_bld, A_dn, B_bln, C_bln, D_d, h_ckpts)
            ctx.CHUNK_L = CHUNK_L
            ctx.BLOCK_D = BLOCK_D
            y = y_bld.transpose(1, 2)
            if return_last_state:
                return y, last_state
            return y

        @staticmethod
        def backward(ctx, *grad_outputs):
            dy = grad_outputs[0]
            u_bld, dt_bld, A_dn, B_bln, C_bln, D_d, h_ckpts = ctx.saved_tensors
            dy_bld = dy.transpose(1, 2).contiguous()
            B_batch, L, D_dim = u_bld.shape
            N_state = A_dn.shape[1]

            du_bld = torch.empty_like(u_bld)
            ddt_bld = torch.empty_like(dt_bld)
            dA = torch.zeros_like(A_dn)
            dB_bln = torch.zeros((B_batch, L, N_state), device=u_bld.device, dtype=torch.float32)
            dC_bln = torch.zeros((B_batch, L, N_state), device=u_bld.device, dtype=torch.float32)
            dD = torch.zeros_like(D_d)

            grid = (B_batch, triton.cdiv(D_dim, ctx.BLOCK_D))
            _ssm_bwd_kernel[grid](
                u_bld, dt_bld, A_dn, B_bln, C_bln, D_d, dy_bld, h_ckpts,
                du_bld, ddt_bld, dA, dB_bln, dC_bln, dD,
                B_batch, L, D_dim, N_state=N_state, CHUNK_L=ctx.CHUNK_L, BLOCK_D=ctx.BLOCK_D,
            )
            return (
                du_bld.transpose(1, 2),
                ddt_bld.transpose(1, 2),
                dA,
                dB_bln.to(B_bln.dtype).transpose(1, 2),
                dC_bln.to(C_bln.dtype).transpose(1, 2),
                dD,
                None,
            )


def selective_scan_triton(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    return_last_state: bool = False,
):
    """Fused Triton CUDA selective scan with chunked state checkpointing."""
    return _TritonSelectiveScanFn.apply(u, delta, A, B, C, D, return_last_state)


def selective_scan_pytorch(
    u: torch.Tensor,                  # (B, D, L)
    delta: torch.Tensor,              # (B, D, L)
    A: torch.Tensor,                  # (D, N)
    B: torch.Tensor,                  # (B, N, L)
    C: torch.Tensor,                  # (B, N, L)
    D: Optional[torch.Tensor] = None, # (D,)
    chunk_size: int = 128,
    return_last_state: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Chunked parallel associative selective scan in float32 for numerical stability and speed.

    Uses Hillis-Steele parallel prefix scan within each chunk (O(log C) steps instead of O(L)),
    keeping recurrent state accumulation strictly in float32 to prevent FP16 overflow/underflow.
    """
    orig_dtype = u.dtype
    b, d, l = u.shape
    n = A.shape[1]

    # Always compute SSM discretization and recurrence in float32 for numerical stability
    u_f = u.float()
    delta_f = delta.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()

    h = torch.zeros((b, d, n), device=u.device, dtype=torch.float32)
    y_chunks = []

    for start in range(0, l, chunk_size):
        end = min(start + chunk_size, l)
        c = end - start

        u_c = u_f[:, :, start:end]         # (B, D, c)
        dt_c = delta_f[:, :, start:end]    # (B, D, c)
        B_c = B_f[:, :, start:end]         # (B, N, c)
        C_c = C_f[:, :, start:end]         # (B, N, c)

        # Discretize within chunk: (B, D, c, N)
        # a_t = exp(dt_t * A), b_t = dt_t * u_t * B_t
        aa = torch.exp(dt_c.unsqueeze(-1) * A_f.view(1, d, 1, n))
        bb = (dt_c * u_c).unsqueeze(-1) * B_c.transpose(1, 2).unsqueeze(1)

        # Hillis-Steele parallel prefix scan over chunk length c: O(log2(c)) steps
        stride = 1
        while stride < c:
            aa_prev = aa[:, :, :-stride, :]
            bb_prev = bb[:, :, :-stride, :]
            aa_curr = aa[:, :, stride:, :]
            bb_curr = bb[:, :, stride:, :]

            bb = torch.cat([bb[:, :, :stride, :], aa_curr * bb_prev + bb_curr], dim=2)
            aa = torch.cat([aa[:, :, :stride, :], aa_curr * aa_prev], dim=2)
            stride <<= 1

        # Incorporate carried state h from previous chunk
        x_c = aa * h.unsqueeze(2) + bb     # (B, D, c, N)
        h = x_c[:, :, -1, :]               # (B, D, N)

        # Project state with C: (B, D, c)
        y_c = torch.sum(x_c * C_c.transpose(1, 2).unsqueeze(1), dim=-1)
        y_chunks.append(y_c)

    ys = torch.cat(y_chunks, dim=2)        # (B, D, L)

    if D is not None:
        ys = ys + u_f * D.float().unsqueeze(-1)

    ys = ys.to(orig_dtype)
    if return_last_state:
        return ys, h
    return ys




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

        # S4D real initialization for A: each channel gets distinct decay rates [1, 2, ..., d_state]
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Out projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Initialize dt_proj weight and bias
        self._reset_dt_parameters()

    def _reset_dt_parameters(self):
        """Initialize dt_proj.weight and dt_proj.bias for calibrated SSM step sizes."""
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        self._reset_dt_bias()

    def _reset_dt_bias(self):
        dt_min = 0.001
        dt_max = 0.1
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def _step(
        self,
        x_inner: torch.Tensor,  # (B, d_inner)
        z: torch.Tensor,        # (B, d_inner)
        cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """O(1) single-token recurrent decode step using cached conv_state and ssm_state."""
        conv_state = torch.roll(cache["conv_state"], shifts=-1, dims=-1)
        conv_state[:, :, -1] = x_inner
        cache["conv_state"] = conv_state

        # Causal depthwise conv1d + SiLU
        x_conv = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = F.silu(x_conv)  # (B, d_inner)

        # Project dt, B, C
        x_db = self.x_proj(x_conv)  # (B, dt_rank + 2 * d_state)
        dt, B_ssm, C_ssm = torch.split(
            x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # (B, d_inner)

        # Single-step SSM recurrence in float32
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        dt_f = dt.float().unsqueeze(-1)     # (B, d_inner, 1)
        dA = torch.exp(dt_f * A.unsqueeze(0))
        dB_u = (dt_f * x_conv.float().unsqueeze(-1)) * B_ssm.float().unsqueeze(1)

        ssm_state = dA * cache["ssm_state"] + dB_u  # (B, d_inner, d_state)
        cache["ssm_state"] = ssm_state

        y = torch.sum(ssm_state * C_ssm.float().unsqueeze(1), dim=-1) + x_conv.float() * self.D.float()
        y = y.to(x_inner.dtype) * F.silu(z)
        return self.out_proj(y).unsqueeze(1)  # (B, 1, d_model)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, D)
        cache: Optional per-layer dict holding 'conv_state' and 'ssm_state' for O(1) decoding.
        Returns: (B, L, D)
        """
        b, l, d = x.shape

        # 1. Project input to inner dimension and split into x and gate z
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # (B, L, d_inner)

        # Fast O(1) single-token decode path when cache is populated
        if cache is not None and cache.get("conv_state") is not None and l == 1:
            return self._step(x_inner[:, 0, :], z[:, 0, :], cache)

        # 2. 1D Causal Convolution + SiLU (use fused CUDA kernel when available)
        x_conv_in = x_inner.transpose(1, 2)  # (B, d_inner, L)
        if cache is not None:
            # Save last d_conv inputs (left-padded with zeros if prompt is shorter than d_conv)
            if l >= self.d_conv:
                cache["conv_state"] = x_conv_in[:, :, -self.d_conv :].contiguous()
            else:
                pad = torch.zeros(
                    (b, self.d_inner, self.d_conv - l),
                    device=x.device,
                    dtype=x_conv_in.dtype,
                )
                cache["conv_state"] = torch.cat([pad, x_conv_in], dim=-1).contiguous()

        if HAS_CAUSAL_CONV1D and x_conv_in.is_cuda and self.d_conv in (2, 3, 4):
            x_conv = causal_conv1d_fn(
                x_conv_in.contiguous(),
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation="silu",
            )
        else:
            x_conv = self.conv1d(x_conv_in)[:, :, :l]  # Causal pad truncation
            x_conv = F.silu(x_conv)  # (B, d_inner, L)

        # 3. Project to dt, B, C parameters
        x_flat = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_db = self.x_proj(x_flat)  # (B, L, dt_rank + 2 * d_state)
        dt, B_ssm, C_ssm = torch.split(
            x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        u = x_conv
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        need_state = cache is not None

        # 4. Run Selective Scan + Gate z
        if HAS_MAMBA_CUDA and u.is_cuda:
            # Fused CUDA kernel: combines dt_bias + softplus + selective scan + SiLU(z) gating
            dt = F.linear(dt, self.dt_proj.weight)  # (B, L, d_inner), bias fused in kernel
            delta = dt.transpose(1, 2).contiguous().to(u.dtype)
            B_ssm = B_ssm.transpose(1, 2).contiguous().to(u.dtype)
            C_ssm = C_ssm.transpose(1, 2).contiguous().to(u.dtype)
            z_t = z.transpose(1, 2).contiguous().to(u.dtype)
            scan_out = selective_scan_fn(
                u.contiguous(),
                delta,
                A,
                B_ssm,
                C_ssm,
                self.D.float(),
                z=z_t,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=need_state,
            )
            if need_state:
                y, last_state = scan_out
                cache["ssm_state"] = last_state.float()
            else:
                y = scan_out
            y = y.transpose(1, 2)  # (B, L, d_inner)
        elif HAS_TRITON and u.is_cuda and (self.d_state & (self.d_state - 1) == 0):
            # Fused Triton CUDA kernel (fast SRAM register recurrence + low VRAM)
            dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
            delta = dt.transpose(1, 2).to(u.dtype)
            B_ssm = B_ssm.transpose(1, 2).to(u.dtype)
            C_ssm = C_ssm.transpose(1, 2).to(u.dtype)
            scan_out = selective_scan_triton(
                u, delta, A, B_ssm, C_ssm, self.D.float(), return_last_state=need_state
            )
            if need_state:
                y, last_state = scan_out
                cache["ssm_state"] = last_state.float()
            else:
                y = scan_out
            y = y.transpose(1, 2)  # (B, L, d_inner)
            y = y * F.silu(z)
        else:
            dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
            delta = dt.transpose(1, 2).to(u.dtype)
            B_ssm = B_ssm.transpose(1, 2).to(u.dtype)
            C_ssm = C_ssm.transpose(1, 2).to(u.dtype)

            # Chunked parallel associative scan (checkpointed during training to save VRAM)
            if self.training and not need_state:
                y = torch.utils.checkpoint.checkpoint(
                    selective_scan_pytorch, u, delta, A, B_ssm, C_ssm, self.D.float(), use_reentrant=False
                )
            else:
                scan_out = selective_scan_pytorch(
                    u, delta, A, B_ssm, C_ssm, self.D.float(), return_last_state=need_state
                )
                if need_state:
                    y, last_state = scan_out
                    cache["ssm_state"] = last_state.float()
                else:
                    y = scan_out

            # Multiply by gate z (SiLU(z))
            y = y.transpose(1, 2)  # (B, L, d_inner)
            y = y * F.silu(z)

        # 5. Output projection
        out = self.out_proj(y)  # (B, L, d_model)
        return out
