"""
K1: Fused RMSNorm — Triton kernel for RTX 3060 (SM86).

Fuses: x.pow(2).mean(-1) + rsqrt + eps + weight multiply.
Forward/backward implemented as torch.autograd.Function.

Production-only: bf16 in / bf16 out, fp32 accumulation in-kernel.
Also supports fp32 in / fp32 out for tests and parity checks.

Shapes:
- (B, T, D) or (B, T, H, D) — reduce over last dim D.
- Internally we flatten everything to (M, D) where M = prod(other dims).

Backward derivation (per row, D-dim vector x, weight w, eps e):
    ms = mean(x^2)
    r  = rsqrt(ms + e)
    y  = x * r * w

    dy/dx_i:
        dy_j/dx_i = w_j * [ delta_ij * r  +  x_j * d(r)/dx_i ]
        d(ms)/dx_i = 2 x_i / D
        d(r)/dx_i  = -0.5 * r^3 * d(ms)/dx_i = -(x_i / D) * r^3
    => dy_j/dx_i = w_j * delta_ij * r  -  w_j * x_j * x_i * r^3 / D

    dL/dx_i = sum_j dL/dy_j * dy_j/dx_i
            = w_i * r * dL/dy_i  -  (x_i * r^3 / D) * sum_j (w_j * x_j * dL/dy_j)

    Let s = sum_j (w_j * x_j * dL/dy_j) = sum_j ( (w_j*x_j) * grad_y_j )
        = sum_j ( y_j/r * grad_y_j )

    dL/dx_i = r * (w_i * grad_y_i  -  (x_i / D) * r^2 * s)

    dL/dw_j  is accumulated across all rows:
        dL/dw_j = sum_row (x_j * r * grad_y_j) = sum_row (y_j/w_j * grad_y_j)
        (We compute it directly as x * r * grad_y to avoid divide-by-w.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch import Tensor


# --------------------------------------------------------------------------- #
# BLOCK_SIZE selection
# --------------------------------------------------------------------------- #
# RTX 3060 (SM86): 28 SMs, 1024 max threads/block. Each program handles one
# row, with BLOCK_SIZE >= D so the whole row is in registers — no tile loop
# needed. BLOCK_SIZE is the next power of two >= D, capped at 16384 (8 KB row
# at fp16 is plenty for our small D values: 32..1024 typical).
def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pick_num_warps(block_size: int) -> int:
    # Cheap heuristic that matches typical Triton tuning for elementwise / reduction.
    if block_size <= 128:
        return 2
    if block_size <= 512:
        return 4
    if block_size <= 2048:
        return 8
    return 16


# --------------------------------------------------------------------------- #
# Forward kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr, W_ptr, Y_ptr, RSTD_ptr,
    stride_x_row,  # stride between rows in X (and Y) in elements
    M, D,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    # Load row in fp32 accumulator (even if input is bf16)
    x_ptrs = X_ptr + row * stride_x_row + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # mean(x^2) using fp32 accumulator
    ms = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(ms + eps)

    # Save rstd for backward
    tl.store(RSTD_ptr + row, rstd)

    # Weight is broadcast across rows. Load as fp32 then re-cast on store.
    w_ptrs = W_ptr + cols
    w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    y = x * rstd * w
    y_ptrs = Y_ptr + row * stride_x_row + cols
    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# Backward kernel — dx and partial dweight per row
# --------------------------------------------------------------------------- #
@triton.jit
def _rmsnorm_bwd_kernel(
    X_ptr, W_ptr, DY_ptr, RSTD_ptr,
    DX_ptr, DW_partial_ptr,
    stride_x_row,
    stride_dw_partial_row,
    M, D,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    x_ptrs = X_ptr + row * stride_x_row + cols
    dy_ptrs = DY_ptr + row * stride_x_row + cols
    dx_ptrs = DX_ptr + row * stride_x_row + cols
    dw_ptrs = DW_partial_ptr + row * stride_dw_partial_row + cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row)

    # dL/dx_i = r * (w_i * dy_i - (x_i / D) * r^2 * s),  s = sum_j (w_j * x_j * dy_j)
    wx_dy = w * x * dy
    s = tl.sum(wx_dy, axis=0)

    dx = rstd * (w * dy - (x / D) * (rstd * rstd) * s)
    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=mask)

    # Partial dW for this row: x * rstd * dy (fp32 — reduced over rows in PyTorch)
    dw_row = x * rstd * dy
    tl.store(dw_ptrs, dw_row, mask=mask)


# --------------------------------------------------------------------------- #
# autograd.Function
# --------------------------------------------------------------------------- #
class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        assert x.is_cuda and weight.is_cuda, "FusedRMSNorm requires CUDA tensors"
        assert weight.dim() == 1, f"weight must be 1-D, got shape {tuple(weight.shape)}"
        D = x.shape[-1]
        assert weight.shape[0] == D, (
            f"weight shape {weight.shape} mismatches x last dim {D}"
        )

        x_contig = x.contiguous()
        orig_shape = x_contig.shape
        x_flat = x_contig.view(-1, D)
        M = x_flat.shape[0]

        y_flat = torch.empty_like(x_flat)
        # rstd stored in fp32 for accurate backward
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        block_size = _next_pow2(D)
        num_warps = _pick_num_warps(block_size)
        grid = (M,)
        _rmsnorm_fwd_kernel[grid](
            x_flat, weight, y_flat, rstd,
            x_flat.stride(0),
            M, D,
            eps,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        ctx.save_for_backward(x_flat, weight, rstd)
        ctx.orig_shape = orig_shape
        ctx.eps = eps
        ctx.input_dtype = x.dtype
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        return y_flat.view(orig_shape)

    @staticmethod
    def backward(ctx, dy: Tensor):
        x_flat, weight, rstd = ctx.saved_tensors
        D = x_flat.shape[-1]
        M = x_flat.shape[0]

        dy_contig = dy.contiguous().view(-1, D)
        dx_flat = torch.empty_like(x_flat)
        # Per-row partial dW in fp32; reduced in PyTorch after the kernel.
        dw_partial = torch.empty((M, D), dtype=torch.float32, device=x_flat.device)

        grid = (M,)
        _rmsnorm_bwd_kernel[grid](
            x_flat, weight, dy_contig, rstd,
            dx_flat, dw_partial,
            x_flat.stride(0),
            dw_partial.stride(0),
            M, D,
            BLOCK_SIZE=ctx.block_size,
            num_warps=ctx.num_warps,
        )

        dw = dw_partial.sum(dim=0).to(weight.dtype)
        return dx_flat.view(ctx.orig_shape), dw, None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fused_rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Fused RMSNorm: x * rsqrt(mean(x^2) + eps) * weight, reduced over last dim.

    Accepts shapes (B, T, D) and (B, T, H, D) (or any leading dims + D).
    Weight must be 1-D of size D. Eager output dtype == x.dtype.
    Internally accumulates the mean(x^2) in fp32 for stability.
    """
    return FusedRMSNorm.apply(x, weight, eps)


class FusedRMSNormModule(nn.Module):
    """Drop-in replacement for nn.RMSNorm with fused Triton kernel."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
        self.d = d

    def forward(self, x: Tensor) -> Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"d={self.d}, eps={self.eps}"
