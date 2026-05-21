"""
K6: Fused SwiGLU activation — Triton kernel for RTX 3060 (SM86).

Fuses: chunk(2, dim=-1) + silu(gate) * val into a single elementwise pass.
The two enclosing GEMMs (mlp_up, mlp_down) stay in cuBLAS — only the
activation+mul is fused.

Production-only: bf16 in / bf16 out by default, fp32 accumulation in-kernel.
Also supports fp32 in / fp32 out (used for parity tests).

Shapes:
  up tensor: (..., 2*mlp_dim)
  output:    (..., mlp_dim)
Internally flattens leading dims into one (M,) batch axis.

Backward derivation
-------------------
Let u = up tensor of shape (..., 2*mlp_dim). Split as u = [g | v] along last dim.
Forward: y = silu(g) * v, where silu(g) = g * sigmoid(g).

Let s = sigmoid(g), so silu(g) = g * s.
d(silu)/dg = s + g * s * (1 - s) = silu(g) + s * (1 - silu(g))

dL/dy is shape (..., mlp_dim).
  dL/dv_i = dL/dy_i * silu(g_i)
  dL/dg_i = dL/dy_i * v_i * (silu(g_i) + s_i * (1 - silu(g_i)))

We return d(up) of shape (..., 2*mlp_dim) = concat([dg, dv], dim=-1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch import Tensor


# --------------------------------------------------------------------------- #
# Autotune configurations
# --------------------------------------------------------------------------- #
# Each program handles BLOCK_M rows × BLOCK_N cols of the OUTPUT (mlp_dim).
# RTX 3060 has 28 SMs; we want enough grid programs to saturate them.
def _fwd_configs():
    return [
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 64}, num_warps=2),
    ]


# --------------------------------------------------------------------------- #
# Forward kernel
# --------------------------------------------------------------------------- #
@triton.autotune(configs=_fwd_configs(), key=["N"])
@triton.jit
def _swiglu_act_fwd_kernel(
    UP_ptr, Y_ptr,
    stride_up_m, stride_y_m,
    M, N,  # N = mlp_dim (output cols)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # gate = up[:, :N]  (first half), val = up[:, N:2N]  (second half)
    g_ptrs = UP_ptr + rows[:, None] * stride_up_m + cols[None, :]
    v_ptrs = UP_ptr + rows[:, None] * stride_up_m + (cols[None, :] + N)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)

    s = tl.sigmoid(g)
    silu_g = g * s
    y = silu_g * v

    y_ptrs = Y_ptr + rows[:, None] * stride_y_m + cols[None, :]
    tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# Backward kernel
# --------------------------------------------------------------------------- #
@triton.autotune(configs=_fwd_configs(), key=["N"])
@triton.jit
def _swiglu_act_bwd_kernel(
    UP_ptr, DY_ptr, DUP_ptr,
    stride_up_m, stride_dy_m, stride_dup_m,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    g_ptrs = UP_ptr + rows[:, None] * stride_up_m + cols[None, :]
    v_ptrs = UP_ptr + rows[:, None] * stride_up_m + (cols[None, :] + N)
    dy_ptrs = DY_ptr + rows[:, None] * stride_dy_m + cols[None, :]

    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    s = tl.sigmoid(g)
    silu_g = g * s
    # d(silu)/dg = silu(g) + s * (1 - silu(g))
    dsilu_dg = silu_g + s * (1.0 - silu_g)

    dv = dy * silu_g
    dg = dy * v * dsilu_dg

    dg_ptrs = DUP_ptr + rows[:, None] * stride_dup_m + cols[None, :]
    dv_ptrs = DUP_ptr + rows[:, None] * stride_dup_m + (cols[None, :] + N)
    tl.store(dg_ptrs, dg.to(DUP_ptr.dtype.element_ty), mask=mask)
    tl.store(dv_ptrs, dv.to(DUP_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# autograd.Function
# --------------------------------------------------------------------------- #
class FusedSwiGLUAct(torch.autograd.Function):
    """Fused SwiGLU activation: takes (..., 2*mlp_dim), returns (..., mlp_dim).

    Splits the trailing dim in half: gate, val = up.chunk(2, dim=-1).
    Computes silu(gate) * val in a single Triton pass.
    """

    @staticmethod
    def forward(ctx, up: Tensor) -> Tensor:
        assert up.is_cuda, "FusedSwiGLUAct requires CUDA tensors"
        last = up.shape[-1]
        assert last % 2 == 0, (
            f"last dim of up must be even (= 2*mlp_dim), got {last}"
        )
        N = last // 2  # mlp_dim

        up_contig = up.contiguous()
        orig_shape = up_contig.shape
        up_flat = up_contig.view(-1, last)
        M = up_flat.shape[0]

        out_shape = orig_shape[:-1] + (N,)
        y_flat = torch.empty((M, N), dtype=up.dtype, device=up.device)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        _swiglu_act_fwd_kernel[grid](
            up_flat, y_flat,
            up_flat.stride(0), y_flat.stride(0),
            M, N,
        )

        ctx.save_for_backward(up_flat)
        ctx.orig_shape = orig_shape
        ctx.out_shape = out_shape
        ctx.N = N
        return y_flat.view(out_shape)

    @staticmethod
    def backward(ctx, dy: Tensor):
        (up_flat,) = ctx.saved_tensors
        N = ctx.N
        last = 2 * N
        M = up_flat.shape[0]

        dy_contig = dy.contiguous().view(-1, N)
        dup_flat = torch.empty((M, last), dtype=up_flat.dtype, device=up_flat.device)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        _swiglu_act_bwd_kernel[grid](
            up_flat, dy_contig, dup_flat,
            up_flat.stride(0), dy_contig.stride(0), dup_flat.stride(0),
            M, N,
        )

        return dup_flat.view(ctx.orig_shape)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fused_swiglu_act(up: Tensor) -> Tensor:
    """Fused SwiGLU activation.

    up: (..., 2*mlp_dim) — output of mlp_up linear.
    returns: (..., mlp_dim) — silu(gate) * val, where (gate, val) = chunk(2, -1).
    """
    return FusedSwiGLUAct.apply(up)


class FusedSwiGLUMLP(nn.Module):
    """Drop-in fused SwiGLU MLP module.

    Equivalent eager path:
        up = mlp_up(x)                     # (..., 2*mlp_dim)
        gate, val = up.chunk(2, dim=-1)
        h = F.silu(gate) * val             # (..., mlp_dim)
        out = mlp_down(h)                  # (..., d_model)

    Fused path: cuBLAS GEMM → Triton fused_swiglu_act → cuBLAS GEMM.
    Weight names match the eager module so state_dict transfer is direct.
    """

    def __init__(
        self,
        d_model: int,
        mlp_dim: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.mlp_dim = mlp_dim
        factory = {"dtype": dtype, "device": device}
        self.mlp_up = nn.Linear(d_model, mlp_dim * 2, bias=bias, **factory)
        self.mlp_down = nn.Linear(mlp_dim, d_model, bias=bias, **factory)

    def forward(self, x: Tensor) -> Tensor:
        up = self.mlp_up(x)
        h = fused_swiglu_act(up)
        return self.mlp_down(h)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, mlp_dim={self.mlp_dim}"
