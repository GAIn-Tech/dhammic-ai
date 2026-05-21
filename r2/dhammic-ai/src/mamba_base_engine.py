"""
Mamba-3 Base Engine v3 — Fully fused Triton kernels + cuBLAS hybrid.

All elementwise/reduction ops are fused into custom Triton kernels;
matmuls go through cuBLAS for tensor-core throughput. Production-only:
no eager fallbacks.

See ``src/kernels/`` for the kernel surface. The two-SSD decomposition
(gamma-path + beta-path) preserves the eager cache-sharing pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor

from kernels import (
    FusedRMSNormModule,
    FusedInProjSplitModule,
    fused_rope,
    fused_trapezoidal,
    fused_ssd_scan,
)


# --- Legacy-compat alias kept around for any external imports ---
# Production path uses FusedRMSNormModule everywhere internally.
RMSNorm = FusedRMSNormModule


# --- Mamba-3 Block (Fused) ---

class Mamba3Block(nn.Module):
    """
    Fast Mamba-3 using fully-fused Triton kernels + cuBLAS hybrid.

    Pipeline (fully fused):
        fused_inproj_split  : single GEMM → 7 zero-copy views
        fused_rmsnorm       : B/C RMSNorm
        fused_rope          : per-head data-dependent rotary
        fused_trapezoidal   : softplus + sigmoid + exp + (1-λ)·dt·α + λ·dt
        fused_ssd_scan      : chunked-parallel SSD (Triton + cuBLAS bmm)
        fused_rmsnorm       : final gate-norm
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 n_heads: int = 4, chunk_size: int = 256,
                 device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.n_heads = n_heads
        self.chunk_size = chunk_size
        self.d_head = self.d_inner // n_heads

        assert self.d_inner % n_heads == 0
        assert d_state % 2 == 0, "d_state must be even for RoPE"

        # Input projection: -> (z, x_ssm, B, C, dt, lambda, theta)
        self.theta_dim = d_state // 2
        self.split_sizes = [
            self.d_inner,     # z (gate)
            self.d_inner,     # x_ssm
            d_state,          # B
            d_state,          # C
            n_heads,          # dt_raw
            n_heads,          # lam_raw
            self.theta_dim,   # theta
        ]
        # Fused in-proj + split: single GEMM + zero-copy views.
        # Default to bf16 for production (param dtype upcast handled at
        # model-level .to(dtype) if needed).
        in_proj_dtype = dtype if dtype is not None else torch.bfloat16
        self.in_proj = FusedInProjSplitModule(
            d_model, self.split_sizes,
            dtype=in_proj_dtype, device=device,
        )

        # SSM parameters
        self.A_log = nn.Parameter(torch.empty(n_heads))
        self.D = nn.Parameter(torch.empty(n_heads))
        self.dt_bias = nn.Parameter(torch.empty(n_heads))

        # QK-Normalization on B, C (fused RMSNorm)
        self.B_norm = FusedRMSNormModule(d_state)
        self.C_norm = FusedRMSNormModule(d_state)

        # Learnable BC bias (per-head)
        self.B_bias = nn.Parameter(torch.ones(n_heads, d_state))
        self.C_bias = nn.Parameter(torch.ones(n_heads, d_state))

        # Output
        self.out_proj = nn.Linear(
            self.d_inner, d_model, bias=False,
            device=device, dtype=in_proj_dtype,
        )
        self.norm = FusedRMSNormModule(self.d_inner)

        self._init_ssm_params()

    def _init_ssm_params(self):
        nn.init.uniform_(self.A_log, -4, -1)
        nn.init.ones_(self.D)
        nn.init.uniform_(self.dt_bias, 0.001, 0.1)

    def forward(self, x: Tensor) -> Tensor:
        B_dim, T, D = x.shape

        # The fused kernels expect a uniform dtype across the forward.
        # Mixing autocast bf16 with native fp32 ops (e.g. torch.cumsum)
        # silently promotes intermediates and the backward pass then
        # trips dtype checks. Disable autocast for the entire block and
        # cast the input to the in_proj's parameter dtype explicitly.
        x = x.to(self.in_proj.weight.dtype)
        with torch.amp.autocast("cuda", enabled=False):
            return self._forward_impl(x)

    def _forward_impl(self, x: Tensor) -> Tensor:
        B_dim, T, D = x.shape

        # K2: fused in_proj + split → 7 zero-copy views
        z, x_ssm, B_proj, C_proj, dt_raw, lam_raw, theta = self.in_proj(x)

        # K4: fused trapezoidal (softplus + sigmoid + exp + combo)
        # Returns (dt, dA, beta, gamma) directly; α and λ are absorbed.
        dt, dA, beta, gamma = fused_trapezoidal(
            dt_raw, lam_raw, self.dt_bias, self.A_log,
        )

        # K1: fused RMSNorm on B, C
        B_proj = self.B_norm(B_proj)
        C_proj = self.C_norm(C_proj)

        # Data-dependent RoPE — per-head angles (B, T, H, D/2).
        raw_angles = dt.unsqueeze(-1) * theta.unsqueeze(2)
        cum_angles = -torch.cumsum(raw_angles, dim=1)

        B_h = B_proj.unsqueeze(2) + self.B_bias
        C_h = C_proj.unsqueeze(2) + self.C_bias
        # fused_rope accepts per-head angles (B, T, H, D/2) directly.
        # The kernel returns same dtype as input. B_bias/C_bias are
        # fp32 parameters (initialized via torch.ones default), so
        # B_h/C_h here are fp32. Cast back to x_ssm's dtype so
        # fused_ssd_scan sees a uniform dtype across (x, A, B, C) —
        # otherwise its einsum trips a dtype check.
        target_dtype = x_ssm.dtype
        B_h = fused_rope(B_h, cum_angles).to(target_dtype)
        C_h = fused_rope(C_h, cum_angles).to(target_dtype)

        x_heads = x_ssm.view(B_dim, T, self.n_heads, self.d_head)

        # Pad to chunk_size for SSD
        pad = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad > 0:
            x_heads_p = F.pad(x_heads, (0, 0, 0, 0, 0, pad))
            dA_p = F.pad(dA, (0, 0, 0, pad))
            B_h_p = F.pad(B_h, (0, 0, 0, 0, 0, pad))
            C_h_p = F.pad(C_h, (0, 0, 0, 0, 0, pad))
            gamma_p = F.pad(gamma, (0, 0, 0, pad))
            beta_p = F.pad(beta, (0, 0, 0, pad))
        else:
            x_heads_p, dA_p, B_h_p, C_h_p = x_heads, dA, B_h, C_h
            gamma_p, beta_p = gamma, beta

        # K5: fused two-SSD decomposition with cache reuse.
        # All inputs already share target_dtype because _forward_impl
        # runs with autocast disabled and we cast the input above.
        y_gamma, _, ssd_cache = fused_ssd_scan(
            x_heads_p * gamma_p.unsqueeze(-1),
            dA_p, B_h_p, C_h_p, self.chunk_size,
        )

        B_prev = F.pad(B_h_p[:, :-1], (0, 0, 0, 0, 1, 0))
        x_prev = F.pad(x_heads_p[:, :-1], (0, 0, 0, 0, 1, 0))
        y_beta, _, _ = fused_ssd_scan(
            x_prev * beta_p.unsqueeze(-1),
            dA_p, B_prev, C_h_p, self.chunk_size,
            cache=ssd_cache,
        )

        y = (y_gamma + y_beta)[:, :T]
        # self.D is fp32; cast to x's dtype so the dtype stays uniform
        # through the residual+gate path (otherwise the final F.linear in
        # out_proj sees fp32 input vs bf16 weight and fails).
        y = y + x_heads * self.D.to(x.dtype).view(1, 1, -1, 1)

        # Gate + final RMSNorm + project
        y = y.reshape(B_dim, T, -1)
        y = self.norm(y) * F.silu(z)
        return self.out_proj(y)


# --- Legacy compatibility wrapper ---
# Keep old API for anything that imported Mamba3Config / Mamba3BaseEngine
from dataclasses import dataclass

@dataclass
class Mamba3Config:
    d_model: int = 768
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 4
    chunk_size: int = 256
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None
    # Legacy fields (unused by fast path)
    dt_rank: str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True
    layer_idx: Optional[int] = None
    rms_norm_eps: float = 1e-5
    residual_in_fp32: bool = True
    fused_add_norm: bool = True


class Mamba3BaseEngine(nn.Module):
    """Compatibility wrapper: delegates to fast Mamba3Block."""

    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        self._amp_disabled = False  # SSD is bf16-safe
        self.block = Mamba3Block(
            d_model=config.d_model,
            d_state=config.d_state,
            expand=config.expand,
            n_heads=config.n_heads,
            chunk_size=config.chunk_size,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self, hidden_states: torch.Tensor, inference_params=None):
        return self.block(hidden_states)


def create_mamba3_base_engine(
    d_model: int = 768, d_state: int = 16, d_conv: int = 4, expand: int = 2,
    n_heads: int = 4, chunk_size: int = 256,
    device=None, dtype=None,
) -> Mamba3BaseEngine:
    config = Mamba3Config(
        d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
        n_heads=n_heads, chunk_size=chunk_size, device=device, dtype=dtype,
    )
    return Mamba3BaseEngine(config)


if __name__ == "__main__":
    model = create_mamba3_base_engine(d_model=64, d_state=8, n_heads=4)
    x = torch.randn(2, 256, 64)  # Must be divisible by chunk_size
    output = model(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print("Mamba-3 SSD Engine test successful!")
