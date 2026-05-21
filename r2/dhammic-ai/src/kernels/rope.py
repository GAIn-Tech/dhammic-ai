"""
Fused Data-Dependent Rotary Position Embedding (RoPE) — Triton kernel.

Reference eager implementation (mamba_base_engine.apply_rope):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos_a = cos(angles); sin_a = sin(angles)
    y_even = cos_a * x1 - sin_a * x2
    y_odd  = sin_a * x1 + cos_a * x2
    return stack([y_even, y_odd], dim=-1).flatten(-2)

Shapes:
    x:      (B, T, H, D)                       bf16 (or fp32) — D must be even
    angles: (B, T, 1, D/2) OR (B, T, H, D/2)   bf16 (or fp32)
    out:    (B, T, H, D)                       same dtype as x

The kernel reads x as pairs along the last axis, computes cos/sin inline
(no cos/sin tensor materialization), writes the rotated output. Backward
computes dx and dangles analytically. When angles is broadcast over H
(aH == 1), the dangles kernel reduces across the H axis; when angles is
per-head (aH == H), each head writes its own dangles slice with no
reduction.

Production-only path: bf16 in/out, fp32 internal compute.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _rope_fwd_kernel(
    x_ptr, angles_ptr, out_ptr,
    B, T, H, D_half,
    sx_b, sx_t, sx_h, sx_d,
    sa_b, sa_t, sa_h, sa_d,
    so_b, so_t, so_h, so_d,
    BLOCK_D: tl.constexpr,
):
    # One program per (b, t, h)
    pid = tl.program_id(0)
    bt = pid // H
    h = pid % H
    b = bt // T
    t = bt % T

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D_half

    # Base pointer for this (b, t, h) row in x and out, viewed as pairs.
    # Element x[..., 0::2] is at index 2*d, x[..., 1::2] at index 2*d+1.
    x_base = x_ptr + b * sx_b + t * sx_t + h * sx_h
    o_base = out_ptr + b * so_b + t * so_t + h * so_h
    # sa_h is the stride of angles along the H axis. When angles is
    # broadcast (aH == 1) the contiguous view yields sa_h == 0, so the
    # stride-multiply naturally broadcasts; when per-head, sa_h is the
    # real H stride (D/2 elements).
    a_base = angles_ptr + b * sa_b + t * sa_t + h * sa_h

    x1 = tl.load(x_base + (2 * d_offs) * sx_d, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_base + (2 * d_offs + 1) * sx_d, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_base + d_offs * sa_d, mask=mask, other=0.0).to(tl.float32)

    c = tl.cos(a)
    s = tl.sin(a)

    y_even = c * x1 - s * x2
    y_odd = s * x1 + c * x2

    tl.store(o_base + (2 * d_offs) * so_d, y_even, mask=mask)
    tl.store(o_base + (2 * d_offs + 1) * so_d, y_odd, mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel for dx: works per (b, t, h) row, no reduction needed.
# ---------------------------------------------------------------------------

@triton.jit
def _rope_bwd_dx_kernel(
    dy_ptr, angles_ptr, dx_ptr,
    B, T, H, D_half,
    sy_b, sy_t, sy_h, sy_d,
    sa_b, sa_t, sa_h, sa_d,
    sdx_b, sdx_t, sdx_h, sdx_d,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    bt = pid // H
    h = pid % H
    b = bt // T
    t = bt % T

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D_half

    dy_base = dy_ptr + b * sy_b + t * sy_t + h * sy_h
    dx_base = dx_ptr + b * sdx_b + t * sdx_t + h * sdx_h
    # Same broadcasting trick as fwd: sa_h == 0 when angles is broadcast.
    a_base = angles_ptr + b * sa_b + t * sa_t + h * sa_h

    dy_e = tl.load(dy_base + (2 * d_offs) * sy_d, mask=mask, other=0.0).to(tl.float32)
    dy_o = tl.load(dy_base + (2 * d_offs + 1) * sy_d, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_base + d_offs * sa_d, mask=mask, other=0.0).to(tl.float32)

    c = tl.cos(a)
    s = tl.sin(a)

    dx1 = c * dy_e + s * dy_o
    dx2 = -s * dy_e + c * dy_o

    tl.store(dx_base + (2 * d_offs) * sdx_d, dx1, mask=mask)
    tl.store(dx_base + (2 * d_offs + 1) * sdx_d, dx2, mask=mask)


# ---------------------------------------------------------------------------
# Backward kernel for dangles: branches on whether angles is broadcast over H.
#
# - BROADCAST_H=True (aH == 1): one program per (b, t), reduces across H.
# - BROADCAST_H=False (aH == H): one program per (b, t, h), no reduction.
# ---------------------------------------------------------------------------

@triton.jit
def _rope_bwd_dangles_kernel(
    x_ptr, dy_ptr, angles_ptr, dangles_ptr,
    B, T, H, D_half,
    sx_b, sx_t, sx_h, sx_d,
    sy_b, sy_t, sy_h, sy_d,
    sa_b, sa_t, sa_h, sa_d,
    sda_b, sda_t, sda_h, sda_d,
    BLOCK_D: tl.constexpr,
    BROADCAST_H: tl.constexpr,
):
    pid = tl.program_id(0)
    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D_half

    if BROADCAST_H:
        # pid encodes (b, t); reduce across H inside the kernel.
        b = pid // T
        t = pid % T

        a_base = angles_ptr + b * sa_b + t * sa_t  # aH == 1 → no h term
        a = tl.load(a_base + d_offs * sa_d, mask=mask, other=0.0).to(tl.float32)
        c = tl.cos(a)
        s = tl.sin(a)

        # dL/dangles[d] = sum_h [ dy_e*(-s*x1 - c*x2) + dy_o*(c*x1 - s*x2) ]
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for h in range(0, H):
            x_base = x_ptr + b * sx_b + t * sx_t + h * sx_h
            dy_base = dy_ptr + b * sy_b + t * sy_t + h * sy_h

            x1 = tl.load(x_base + (2 * d_offs) * sx_d, mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(x_base + (2 * d_offs + 1) * sx_d, mask=mask, other=0.0).to(tl.float32)
            dy_e = tl.load(dy_base + (2 * d_offs) * sy_d, mask=mask, other=0.0).to(tl.float32)
            dy_o = tl.load(dy_base + (2 * d_offs + 1) * sy_d, mask=mask, other=0.0).to(tl.float32)

            acc += dy_e * (-s * x1 - c * x2) + dy_o * (c * x1 - s * x2)

        da_base = dangles_ptr + b * sda_b + t * sda_t  # broadcast: no h
        tl.store(da_base + d_offs * sda_d, acc, mask=mask)
    else:
        # Per-head angles: pid encodes (b, t, h); each head writes its own row.
        bt = pid // H
        h = pid % H
        b = bt // T
        t = bt % T

        a_base = angles_ptr + b * sa_b + t * sa_t + h * sa_h
        a = tl.load(a_base + d_offs * sa_d, mask=mask, other=0.0).to(tl.float32)
        c = tl.cos(a)
        s = tl.sin(a)

        x_base = x_ptr + b * sx_b + t * sx_t + h * sx_h
        dy_base = dy_ptr + b * sy_b + t * sy_t + h * sy_h

        x1 = tl.load(x_base + (2 * d_offs) * sx_d, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_base + (2 * d_offs + 1) * sx_d, mask=mask, other=0.0).to(tl.float32)
        dy_e = tl.load(dy_base + (2 * d_offs) * sy_d, mask=mask, other=0.0).to(tl.float32)
        dy_o = tl.load(dy_base + (2 * d_offs + 1) * sy_d, mask=mask, other=0.0).to(tl.float32)

        val = dy_e * (-s * x1 - c * x2) + dy_o * (c * x1 - s * x2)

        da_base = dangles_ptr + b * sda_b + t * sda_t + h * sda_h
        tl.store(da_base + d_offs * sda_d, val, mask=mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _check_inputs(x: Tensor, angles: Tensor) -> tuple[int, int, int, int, bool]:
    assert x.is_cuda and angles.is_cuda, "fused_rope requires CUDA tensors"
    assert x.dim() == 4 and angles.dim() == 4, \
        f"expected x.dim()==4, angles.dim()==4, got {x.dim()}, {angles.dim()}"
    B, T, H, D = x.shape
    assert D % 2 == 0, f"x last dim must be even, got {D}"
    D_half = D // 2
    aB, aT, aH, aDh = angles.shape
    assert aB == B and aT == T and aDh == D_half, \
        f"angles shape {tuple(angles.shape)} mismatch (B,T,{{1|H}},D/2)=" \
        f"({B},{T},{{1|{H}}},{D_half})"
    assert aH == 1 or aH == H, \
        f"angles H dim must be 1 (broadcast) or {H} (per-head), got {aH}"
    broadcast_h = (aH == 1)
    return B, T, H, D_half, broadcast_h


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

class FusedRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, angles: Tensor) -> Tensor:
        B, T, H, D_half, broadcast_h = _check_inputs(x, angles)
        D = 2 * D_half

        x_c = x.contiguous()
        angles_c = angles.contiguous()
        out = torch.empty_like(x_c)

        BLOCK_D = _next_power_of_2(max(D_half, 1))
        grid = (B * T * H,)
        # Strides: when aH == 1, angles_c.stride(2) is 0 in PyTorch's
        # contiguous layout? Actually no — contiguous() yields stride
        # (T*1*D_half, 1*D_half, D_half, 1). The H stride is D_half even
        # when aH==1, because the singleton dim is materialized. We must
        # pass 0 explicitly in the broadcast case so the kernel broadcasts.
        sa_h = 0 if broadcast_h else angles_c.stride(2)
        _rope_fwd_kernel[grid](
            x_c, angles_c, out,
            B, T, H, D_half,
            x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
            angles_c.stride(0), angles_c.stride(1), sa_h, angles_c.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x_c, angles_c)
        ctx.shape = (B, T, H, D_half)
        ctx.broadcast_h = broadcast_h
        return out

    @staticmethod
    def backward(ctx, dy: Tensor):
        x_c, angles_c = ctx.saved_tensors
        B, T, H, D_half = ctx.shape
        broadcast_h = ctx.broadcast_h

        dy_c = dy.contiguous()
        dx = torch.empty_like(x_c)
        # dangles accumulator must be fp32 for numerical stability across H.
        aH_out = 1 if broadcast_h else H
        dangles_f32 = torch.empty(
            (B, T, aH_out, D_half), device=x_c.device, dtype=torch.float32
        )

        BLOCK_D = _next_power_of_2(max(D_half, 1))
        sa_h = 0 if broadcast_h else angles_c.stride(2)

        _rope_bwd_dx_kernel[(B * T * H,)](
            dy_c, angles_c, dx,
            B, T, H, D_half,
            dy_c.stride(0), dy_c.stride(1), dy_c.stride(2), dy_c.stride(3),
            angles_c.stride(0), angles_c.stride(1), sa_h, angles_c.stride(3),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            BLOCK_D=BLOCK_D,
        )

        # Grid for dangles depends on broadcast mode.
        if broadcast_h:
            da_grid = (B * T,)
        else:
            da_grid = (B * T * H,)

        # dangles output strides: when broadcast, sda_h is unused — pass 0.
        sda_h_out = 0 if broadcast_h else dangles_f32.stride(2)

        _rope_bwd_dangles_kernel[da_grid](
            x_c, dy_c, angles_c, dangles_f32,
            B, T, H, D_half,
            x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
            dy_c.stride(0), dy_c.stride(1), dy_c.stride(2), dy_c.stride(3),
            angles_c.stride(0), angles_c.stride(1), sa_h, angles_c.stride(3),
            dangles_f32.stride(0), dangles_f32.stride(1), sda_h_out, dangles_f32.stride(3),
            BLOCK_D=BLOCK_D,
            BROADCAST_H=broadcast_h,
        )

        dangles = dangles_f32.to(angles_c.dtype)
        return dx, dangles


def fused_rope(x: Tensor, angles: Tensor) -> Tensor:
    """Fused data-dependent RoPE. See module docstring for the math.

    Args:
        x: (B, T, H, D), bf16 or fp32, last dim even.
        angles: (B, T, 1, D/2) for broadcast or (B, T, H, D/2) for per-head,
            bf16 or fp32.

    Returns:
        Tensor of same shape/dtype as x.
    """
    return FusedRoPE.apply(x, angles)
