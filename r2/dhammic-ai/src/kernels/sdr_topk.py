"""
K7: Fused SDR Top-K with Gumbel STE — Triton kernel for RTX 3060 (SM86).

Fuses (training):
    noisy = (logits + gumbel) / temperature
    topk_vals, topk_idx = noisy.topk(k_active, dim=-1)
    threshold = topk_vals[..., -1:]
    soft_mask = sigmoid((noisy - threshold) * 5.0)
    hard_mask = zeros.scatter_(-1, topk_idx, 1.0)
    sdr = hard_mask - soft_mask.detach() + soft_mask    # STE

Fuses (eval):
    _, topk_idx = logits.topk(k_active, dim=-1)
    sdr = zeros.scatter_(-1, topk_idx, 1.0)

Numerically the STE output equals `hard_mask` in forward, but the gradient
flows through `soft_mask` only. We therefore store hard_mask as output (bf16)
and compute the analytic backward through the sigmoid:

    d(sdr)/d(logits_i) = sigmoid'(z_i) * 5 / temperature,
        with z_i = (noisy_i - threshold) * 5.

`threshold` is treated as constant w.r.t. logits (same as eager: it's
extracted via topk values then index-detached).

Top-k is computed via torch.topk (cuBLAS / Thrust path) — Triton just runs
the elementwise + scatter + STE construction. This is the production choice
documented in the plan: K7 = topk-from-torch + Triton scatter/STE kernel.

Shapes: arbitrary leading dims (B, T, D) or (B, T, H, D). Flattened to
(M, D) internally, where D == sdr_dim.

dtype:
- logits: bf16 or fp32
- output (sdr): same dtype as logits (binary values, bf16 is fine)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch import Tensor


# --------------------------------------------------------------------------- #
# Forward kernel — combined eval + train paths
# --------------------------------------------------------------------------- #
def _num_warps_for(block_size: int) -> int:
    if block_size <= 256:
        return 4
    if block_size <= 1024:
        return 8
    return 16


@triton.jit
def _sdr_train_fwd_kernel(
    LOGITS_ptr, GUMBEL_ptr, IDX_ptr, THR_ptr,
    OUT_ptr, Z_ptr,
    stride_row,          # stride of LOGITS / OUT (== D)
    stride_idx_row,      # stride of IDX (== K)
    M, D, K,
    temperature,
    BLOCK_SIZE: tl.constexpr,
    K_BLOCK: tl.constexpr,
):
    """Training forward.

    For each row m:
      - Load threshold THR[m].
      - Compute noisy_i = (logits_i + gumbel_i) / temperature.
      - Compute z_i = (noisy_i - threshold) * 5.0   (stored for backward).
      - Build OUT row by broadcasting the K topk indices vs all D column
        positions: hard_mask[i] = 1 if i ∈ idxs else 0.  This is gather-style
        with O(D*K) compares per row, all in registers, and writes the row
        with a single tl.store — no scatter ordering issue.
    """
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    logits_ptrs = LOGITS_ptr + row * stride_row + cols
    gumbel_ptrs = GUMBEL_ptr + row * stride_row + cols
    out_ptrs = OUT_ptr + row * stride_row + cols
    z_ptrs = Z_ptr + row * stride_row + cols

    logits = tl.load(logits_ptrs, mask=mask, other=0.0).to(tl.float32)
    gumbel = tl.load(gumbel_ptrs, mask=mask, other=0.0).to(tl.float32)
    threshold = tl.load(THR_ptr + row).to(tl.float32)

    inv_T = 1.0 / temperature
    noisy = (logits + gumbel) * inv_T
    z = (noisy - threshold) * 5.0

    # Save z for backward (fp32).
    tl.store(z_ptrs, z, mask=mask)

    # Load all K topk indices for this row.
    ks = tl.arange(0, K_BLOCK)
    k_mask = ks < K
    idx_ptrs = IDX_ptr + row * stride_idx_row + ks
    # Set out-of-range slots to a sentinel that will never match a valid col
    # (D is past valid range and the col mask kills it anyway).
    idxs = tl.load(idx_ptrs, mask=k_mask, other=-1)  # (K_BLOCK,) int32

    # Broadcast compare: (BLOCK_SIZE, K_BLOCK) → reduce-any along K.
    cols_2d = cols[:, None]                 # (BLOCK, 1)
    idxs_2d = idxs[None, :]                 # (1, K_BLOCK)
    matches = (cols_2d == idxs_2d).to(tl.int32)
    # tl.sum along K gives 1 if the column matches any selected index, else 0.
    hard = tl.sum(matches, axis=1)
    hard_f = hard.to(OUT_ptr.dtype.element_ty)
    tl.store(out_ptrs, hard_f, mask=mask)


@triton.jit
def _sdr_eval_fwd_kernel(
    IDX_ptr, OUT_ptr,
    stride_row,
    stride_idx_row,
    M, D, K,
    BLOCK_SIZE: tl.constexpr,
    K_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D
    out_ptrs = OUT_ptr + row * stride_row + cols

    ks = tl.arange(0, K_BLOCK)
    k_mask = ks < K
    idx_ptrs = IDX_ptr + row * stride_idx_row + ks
    idxs = tl.load(idx_ptrs, mask=k_mask, other=-1)

    cols_2d = cols[:, None]
    idxs_2d = idxs[None, :]
    matches = (cols_2d == idxs_2d).to(tl.int32)
    hard = tl.sum(matches, axis=1)
    hard_f = hard.to(OUT_ptr.dtype.element_ty)
    tl.store(out_ptrs, hard_f, mask=mask)


# --------------------------------------------------------------------------- #
# Backward kernel — analytic gradient through sigmoid(z) only.
# d(sdr)/d(logits_i) = sigmoid'(z_i) * 5 / temperature, since the STE
# expression equals hard_mask + (soft - soft.detach()) in value, and the
# only differentiable part is `soft`. `hard` and `soft.detach()` contribute
# zero gradient.
# --------------------------------------------------------------------------- #
@triton.jit
def _sdr_bwd_kernel(
    Z_ptr, DOUT_ptr, DLOGITS_ptr,
    stride_row,
    M, D,
    scale,                  # = 5.0 / temperature
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    z = tl.load(Z_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(DOUT_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # sigmoid(z) and its derivative sigmoid'(z) = s*(1-s).
    s = tl.sigmoid(z)
    sprime = s * (1.0 - s)

    dlogits = dout * sprime * scale
    out_ptrs = DLOGITS_ptr + row * stride_row + cols
    tl.store(out_ptrs, dlogits.to(DLOGITS_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# K_BLOCK selector — must be a power-of-two constexpr >= K.
# K_active ≤ 64 by design (plan). We support up to 256 for safety.
# --------------------------------------------------------------------------- #
def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _k_block_for(k: int) -> int:
    # Triton constexpr must be >= 1; pick the smallest power of 2 >= k.
    return max(1, _next_pow2(k))


# --------------------------------------------------------------------------- #
# autograd.Function
# --------------------------------------------------------------------------- #
class FusedSDRTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: Tensor, k_active: int, temperature: float,
                training: bool) -> Tensor:
        assert logits.is_cuda, "FusedSDRTopK requires CUDA tensors"
        assert temperature > 0, f"temperature must be > 0, got {temperature}"
        D = logits.shape[-1]
        assert 1 <= k_active <= D, f"k_active {k_active} must be in [1, D={D}]"

        logits_contig = logits.contiguous()
        orig_shape = logits_contig.shape
        x = logits_contig.view(-1, D)
        M = x.shape[0]

        K_BLOCK = _k_block_for(k_active)
        BLOCK_SIZE = _next_pow2(D)
        num_warps = _num_warps_for(BLOCK_SIZE)
        out = torch.empty_like(x)

        if training:
            # Sample Gumbel noise. Float32 sampling for stability, then we work
            # in fp32 inside the kernel.
            u = torch.rand_like(x, dtype=torch.float32).clamp_(min=1e-8)
            gumbel = -torch.log(-torch.log(u))

            # Compute noisy and use torch.topk for k-th value + indices.
            # We need temperature-scaled noisy for both threshold and z.
            noisy = (x.to(torch.float32) + gumbel) / float(temperature)
            topk_vals, topk_idx = torch.topk(noisy, k_active, dim=-1)
            threshold = topk_vals[:, -1].contiguous()  # (M,) fp32

            z = torch.empty_like(x, dtype=torch.float32)
            grid = (M,)
            _sdr_train_fwd_kernel[grid](
                x, gumbel, topk_idx.to(torch.int32), threshold,
                out, z,
                x.stride(0),
                topk_idx.stride(0),
                M, D, k_active,
                float(temperature),
                BLOCK_SIZE=BLOCK_SIZE,
                K_BLOCK=K_BLOCK,
                num_warps=num_warps,
            )

            ctx.save_for_backward(z)
            ctx.scale = 5.0 / float(temperature)
            ctx.needs_grad = True
        else:
            _, topk_idx = torch.topk(x, k_active, dim=-1)
            grid = (M,)
            _sdr_eval_fwd_kernel[grid](
                topk_idx.to(torch.int32), out,
                x.stride(0),
                topk_idx.stride(0),
                M, D, k_active,
                BLOCK_SIZE=BLOCK_SIZE,
                K_BLOCK=K_BLOCK,
                num_warps=num_warps,
            )
            ctx.needs_grad = False

        ctx.orig_shape = orig_shape
        return out.view(orig_shape)

    @staticmethod
    def backward(ctx, dout: Tensor):
        if not ctx.needs_grad:
            return None, None, None, None

        (z,) = ctx.saved_tensors
        D = z.shape[-1]
        M = z.shape[0]

        dout_flat = dout.contiguous().view(M, D)
        dlogits = torch.empty_like(z)  # fp32 grad to keep precision

        BLOCK_SIZE = _next_pow2(D)
        num_warps = _num_warps_for(BLOCK_SIZE)
        grid = (M,)
        _sdr_bwd_kernel[grid](
            z, dout_flat, dlogits,
            z.stride(0),
            M, D,
            float(ctx.scale),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        # Match logits' input dtype: original was either fp32 or bf16.
        # ctx.orig_shape holds the original shape; saved tensors lost dtype info,
        # so we return fp32-cast-to-dout-dtype.
        dl = dlogits.view(ctx.orig_shape).to(dout.dtype)
        return dl, None, None, None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fused_sdr_topk(logits: Tensor, k_active: int, temperature: float,
                   training: bool) -> Tensor:
    """Fused SDR top-k with Gumbel STE (training) / pure top-k scatter (eval).

    Args:
        logits: (..., D). Last dim is the SDR feature axis.
        k_active: number of active positions.
        temperature: Gumbel temperature (training only).
        training: True -> Gumbel STE path; False -> deterministic topk.

    Returns:
        sdr: same shape and dtype as logits. Each row is sparse with
             exactly k_active values equal to 1.0; all others 0.0.
             In training mode, gradient flows back through the sigmoid STE.
    """
    return FusedSDRTopK.apply(logits, k_active, temperature, training)


class FusedSDRTopKModule(nn.Module):
    """nn.Module wrapper. Holds a temperature buffer; forward returns the
    sparse-distributed-representation mask. Caller is responsible for
    applying any subsequent linear projection (e.g. `from_sdr`).
    """

    def __init__(self, k_active: int, temperature: float = 0.5):
        super().__init__()
        self.k_active = k_active
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, logits: Tensor) -> Tensor:
        T = self.temperature.clamp(min=0.1).item()
        return fused_sdr_topk(logits, self.k_active, T, self.training)

    def extra_repr(self) -> str:
        return f"k_active={self.k_active}, temperature={float(self.temperature):.3f}"
