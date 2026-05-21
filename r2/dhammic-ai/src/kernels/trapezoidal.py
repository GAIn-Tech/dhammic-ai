"""
K4: Fused Trapezoidal Discretization Coefficients — Triton kernel for RTX 3060 (SM86).

Fuses (from mamba_base_engine.Mamba3Block.forward, lines ~206-226):

    dt    = softplus(dt_raw + dt_bias)          # (B, T, H)
    lam   = sigmoid(lam_raw)                    # (B, T, H)
    A     = -exp(A_log)                         # (H,)
    dA    = dt * A                              # (B, T, H)
    alpha = exp(dA)                             # intermediate
    beta  = (1 - lam) * dt * alpha              # (B, T, H)
    gamma = lam * dt                            # (B, T, H)

Outputs (consumed downstream by RoPE / ssd / pad+ssd): dt, dA, beta, gamma.
alpha is purely intermediate.

Inputs:
    dt_raw   : (B, T, H)  bf16/fp32
    lam_raw  : (B, T, H)  bf16/fp32
    dt_bias  : (H,)       bf16/fp32   (broadcast over B,T)
    A_log    : (H,)       bf16/fp32   (broadcast over B,T)

Layout: vectorize a single program over (B*T, H) — but each tile in our kernel
covers one (b,t) row across all heads with BLOCK_H >= H. RTX 3060 prefers small
warp counts; H is typically 8-16 so one warp does the whole row trivially.

bf16 in, bf16 out. fp32 compute internally.

Backward derivation:
    Let s = dt_raw + dt_bias, so dt = softplus(s), ds/d(dt_raw) = ds/d(dt_bias) = 1.
    d(softplus)/ds = sigmoid(s).
    d(sigmoid)/d(lam_raw) = lam*(1-lam).
    A = -exp(A_log)  =>  dA/d(A_log) = -exp(A_log) = A.

    Path contributions to dt:
        g_dt_total = g_dt
                    + g_dA   * A                                       (dA = dt*A)
                    + g_beta * (1-lam) * alpha * (1 + dt*A)            (beta = (1-lam)*dt*exp(dt*A))
                    + g_gamma* lam                                     (gamma = lam*dt)

    Path contributions to lam:
        g_lam = -g_beta * dt * alpha + g_gamma * dt

    Path contributions to A (per-head, reduced over B,T):
        g_A_perBT = g_dA * dt + g_beta * (1-lam) * dt^2 * alpha

    Then:
        g_dt_raw  = g_dt_total * sigmoid(s)
        g_dt_bias = sum_{B,T} g_dt_raw                (reduce to (H,))
        g_lam_raw = g_lam * lam * (1-lam)
        g_A_log   = (sum_{B,T} g_A_perBT) * A         (since dA/dA_log = A)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


# --------------------------------------------------------------------------- #
# Autotune configurations
# --------------------------------------------------------------------------- #
# Typical n_heads in dhammic-ai: 8 or 16. BLOCK_H is rounded up to next pow2.
def _fwd_configs():
    return [
        triton.Config({"BLOCK_H": 8}, num_warps=1),
        triton.Config({"BLOCK_H": 16}, num_warps=1),
        triton.Config({"BLOCK_H": 32}, num_warps=2),
        triton.Config({"BLOCK_H": 64}, num_warps=2),
        triton.Config({"BLOCK_H": 128}, num_warps=4),
    ]


def _bwd_configs():
    return [
        triton.Config({"BLOCK_H": 8}, num_warps=1),
        triton.Config({"BLOCK_H": 16}, num_warps=1),
        triton.Config({"BLOCK_H": 32}, num_warps=2),
        triton.Config({"BLOCK_H": 64}, num_warps=2),
        triton.Config({"BLOCK_H": 128}, num_warps=4),
    ]


# --------------------------------------------------------------------------- #
# Forward kernel
# --------------------------------------------------------------------------- #
@triton.autotune(configs=_fwd_configs(), key=["H"])
@triton.jit
def _trapz_fwd_kernel(
    DT_RAW_ptr, LAM_RAW_ptr, DT_BIAS_ptr, A_LOG_ptr,
    DT_ptr, DA_ptr, BETA_ptr, GAMMA_ptr,
    stride_row,           # stride between (b,t) rows in elements (== H for contiguous)
    M, H,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    base = row * stride_row
    dt_raw = tl.load(DT_RAW_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    lam_raw = tl.load(LAM_RAW_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(DT_BIAS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    A_log = tl.load(A_LOG_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Forward
    s = dt_raw + dt_bias
    # Numerically stable softplus: log(1 + exp(s)) = max(s, 0) + log1p(exp(-|s|))
    abs_s = tl.abs(s)
    dt = tl.maximum(s, 0.0) + tl.log(1.0 + tl.exp(-abs_s))

    # Sigmoid via softplus identity to keep it stable: sigmoid(x) = 1/(1+exp(-x))
    lam = 1.0 / (1.0 + tl.exp(-lam_raw))

    A = -tl.exp(A_log)
    dA = dt * A
    alpha = tl.exp(dA)
    beta = (1.0 - lam) * dt * alpha
    gamma = lam * dt

    tl.store(DT_ptr    + base + cols, dt.to(DT_ptr.dtype.element_ty),    mask=mask)
    tl.store(DA_ptr    + base + cols, dA.to(DA_ptr.dtype.element_ty),    mask=mask)
    tl.store(BETA_ptr  + base + cols, beta.to(BETA_ptr.dtype.element_ty),  mask=mask)
    tl.store(GAMMA_ptr + base + cols, gamma.to(GAMMA_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# Backward kernel: per-(b,t) row.
#
# Writes:
#   DX_DT_RAW (B,T,H)        — gradient w.r.t. dt_raw
#   DX_LAM_RAW (B,T,H)       — gradient w.r.t. lam_raw
#   DW_BIAS_partial (M,H) fp32 — per-row contribution to dt_bias grad
#   DW_ALOG_partial (M,H) fp32 — per-row contribution to A_log grad
#
# The two partial buffers are reduced in PyTorch with .sum(0).
# --------------------------------------------------------------------------- #
@triton.autotune(configs=_bwd_configs(), key=["H"])
@triton.jit
def _trapz_bwd_kernel(
    DT_RAW_ptr, LAM_RAW_ptr, DT_BIAS_ptr, A_LOG_ptr,
    G_DT_ptr, G_DA_ptr, G_BETA_ptr, G_GAMMA_ptr,
    DX_DT_RAW_ptr, DX_LAM_RAW_ptr,
    DW_BIAS_PART_ptr, DW_ALOG_PART_ptr,
    stride_row, stride_part_row,
    M, H,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_H)
    mask = cols < H

    base = row * stride_row
    pbase = row * stride_part_row

    dt_raw = tl.load(DT_RAW_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    lam_raw = tl.load(LAM_RAW_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(DT_BIAS_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    A_log = tl.load(A_LOG_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    g_dt    = tl.load(G_DT_ptr    + base + cols, mask=mask, other=0.0).to(tl.float32)
    g_dA    = tl.load(G_DA_ptr    + base + cols, mask=mask, other=0.0).to(tl.float32)
    g_beta  = tl.load(G_BETA_ptr  + base + cols, mask=mask, other=0.0).to(tl.float32)
    g_gamma = tl.load(G_GAMMA_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)

    # Recompute forward quantities (cheap; avoids saving alpha/dt/lam in fp32)
    s = dt_raw + dt_bias
    abs_s = tl.abs(s)
    dt = tl.maximum(s, 0.0) + tl.log(1.0 + tl.exp(-abs_s))
    # d(softplus)/ds = sigmoid(s)
    sig_s = 1.0 / (1.0 + tl.exp(-s))

    lam = 1.0 / (1.0 + tl.exp(-lam_raw))
    one_minus_lam = 1.0 - lam

    A = -tl.exp(A_log)
    dA_val = dt * A
    alpha = tl.exp(dA_val)

    # ---- gradients ----
    # g_dt_total: sum of all paths into dt
    g_dt_total = (
        g_dt
        + g_dA * A
        + g_beta * one_minus_lam * alpha * (1.0 + dA_val)
        + g_gamma * lam
    )

    # g_lam: paths into lam
    g_lam = -g_beta * dt * alpha + g_gamma * dt

    # g_A per-row (per-head): paths into A
    g_A_perrow = g_dA * dt + g_beta * one_minus_lam * dt * dt * alpha

    # Chain through activations
    g_dt_raw = g_dt_total * sig_s          # ds/d(dt_raw) = 1
    g_dt_bias_row = g_dt_raw               # ds/d(dt_bias) = 1; reduced over rows in host
    g_lam_raw = g_lam * lam * one_minus_lam
    g_A_log_row = g_A_perrow * A           # dA/dA_log = -exp(A_log) = A

    tl.store(DX_DT_RAW_ptr  + base + cols, g_dt_raw.to(DX_DT_RAW_ptr.dtype.element_ty),  mask=mask)
    tl.store(DX_LAM_RAW_ptr + base + cols, g_lam_raw.to(DX_LAM_RAW_ptr.dtype.element_ty), mask=mask)

    # Partial buffers kept in fp32 for accurate sum-reduction host-side.
    tl.store(DW_BIAS_PART_ptr + pbase + cols, g_dt_bias_row, mask=mask)
    tl.store(DW_ALOG_PART_ptr + pbase + cols, g_A_log_row,   mask=mask)


# --------------------------------------------------------------------------- #
# autograd.Function
# --------------------------------------------------------------------------- #
class FusedTrapezoidal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dt_raw: Tensor, lam_raw: Tensor, dt_bias: Tensor, A_log: Tensor):
        assert dt_raw.is_cuda and lam_raw.is_cuda and dt_bias.is_cuda and A_log.is_cuda, (
            "FusedTrapezoidal requires CUDA tensors"
        )
        assert dt_raw.shape == lam_raw.shape, (
            f"dt_raw {dt_raw.shape} vs lam_raw {lam_raw.shape}"
        )
        assert dt_raw.dim() >= 1, "expected at least (..., H) tensor"
        H = dt_raw.shape[-1]
        assert dt_bias.dim() == 1 and dt_bias.shape[0] == H, (
            f"dt_bias shape {dt_bias.shape} mismatches H={H}"
        )
        assert A_log.dim() == 1 and A_log.shape[0] == H, (
            f"A_log shape {A_log.shape} mismatches H={H}"
        )

        out_dtype = dt_raw.dtype
        dt_raw_c = dt_raw.contiguous()
        lam_raw_c = lam_raw.contiguous()
        # Cast params to the working dtype so loads & stores share the same dtype path
        dt_bias_c = dt_bias.contiguous().to(out_dtype)
        A_log_c = A_log.contiguous().to(out_dtype)

        orig_shape = dt_raw_c.shape
        dt_flat = dt_raw_c.view(-1, H)
        lam_flat = lam_raw_c.view(-1, H)
        M = dt_flat.shape[0]

        dt    = torch.empty_like(dt_flat)
        dA    = torch.empty_like(dt_flat)
        beta  = torch.empty_like(dt_flat)
        gamma = torch.empty_like(dt_flat)

        grid = (M,)
        _trapz_fwd_kernel[grid](
            dt_flat, lam_flat, dt_bias_c, A_log_c,
            dt, dA, beta, gamma,
            dt_flat.stride(0),
            M, H,
        )

        ctx.save_for_backward(dt_flat, lam_flat, dt_bias_c, A_log_c)
        ctx.orig_shape = orig_shape
        ctx.input_dtype = out_dtype
        ctx.bias_dtype = dt_bias.dtype
        ctx.alog_dtype = A_log.dtype
        return (
            dt.view(orig_shape),
            dA.view(orig_shape),
            beta.view(orig_shape),
            gamma.view(orig_shape),
        )

    @staticmethod
    def backward(ctx, g_dt: Tensor, g_dA: Tensor, g_beta: Tensor, g_gamma: Tensor):
        dt_flat, lam_flat, dt_bias_c, A_log_c = ctx.saved_tensors
        H = dt_flat.shape[-1]
        M = dt_flat.shape[0]
        in_dtype = ctx.input_dtype

        g_dt_c    = g_dt.contiguous().view(-1, H).to(in_dtype)
        g_dA_c    = g_dA.contiguous().view(-1, H).to(in_dtype)
        g_beta_c  = g_beta.contiguous().view(-1, H).to(in_dtype)
        g_gamma_c = g_gamma.contiguous().view(-1, H).to(in_dtype)

        dx_dt_raw  = torch.empty_like(dt_flat)
        dx_lam_raw = torch.empty_like(dt_flat)
        dw_bias_part = torch.empty((M, H), dtype=torch.float32, device=dt_flat.device)
        dw_alog_part = torch.empty((M, H), dtype=torch.float32, device=dt_flat.device)

        grid = (M,)
        _trapz_bwd_kernel[grid](
            dt_flat, lam_flat, dt_bias_c, A_log_c,
            g_dt_c, g_dA_c, g_beta_c, g_gamma_c,
            dx_dt_raw, dx_lam_raw,
            dw_bias_part, dw_alog_part,
            dt_flat.stride(0),
            dw_bias_part.stride(0),
            M, H,
        )

        d_dt_bias = dw_bias_part.sum(dim=0).to(ctx.bias_dtype)
        d_A_log = dw_alog_part.sum(dim=0).to(ctx.alog_dtype)

        return (
            dx_dt_raw.view(ctx.orig_shape),
            dx_lam_raw.view(ctx.orig_shape),
            d_dt_bias,
            d_A_log,
        )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fused_trapezoidal(
    dt_raw: Tensor, lam_raw: Tensor, dt_bias: Tensor, A_log: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute trapezoidal-discretization SSM coefficients.

    Args:
        dt_raw:  (..., H) tensor (typically (B, T, H))
        lam_raw: (..., H) tensor, same shape as dt_raw
        dt_bias: (H,)
        A_log:   (H,)

    Returns:
        (dt, dA, beta, gamma) — each shape == dt_raw.shape, dtype == dt_raw.dtype.
    """
    return FusedTrapezoidal.apply(dt_raw, lam_raw, dt_bias, A_log)
