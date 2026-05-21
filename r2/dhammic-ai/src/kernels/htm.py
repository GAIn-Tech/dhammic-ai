"""
HTM Triton Kernels — Faithful Numenta HTM (Spatial Pooler + Temporal Memory)
for RTX 3060 (SM86), Triton 3.5.1, torch 2.9.1+cu128.

Three fused kernels:

  K_sp_overlap : Spatial Pooler "boosted overlap" — for each (row, column),
                 compute  overlap = sum_d ( sigmoid((perm[col,d] - thr)/tau)
                                            * input[row,d] )
                 then boosted = overlap * boost[col].
                 k-WTA is done in PyTorch on top of the kernel output
                 (torch.topk + scatter, fp32 stable, autograd-friendly).

  K_tm_predict : Temporal Memory "segment activation" — for each
                 (row, cell, segment), compute
                     act = sum_s sigmoid((seg_perm - thr)/tau) * prev_active[syn]
                 and max over segments → a cell's predictive score.

  K_tm_learn   : Hebbian-soft permanence update — given pre/post activity, emit
                 a delta tensor (the gradient version of HTM's permanence
                 increment). This is computed as
                     dW = lr * pre[None,:] * post[:,None] - decay * W
                 fused over a row block. We expose it as a `torch.autograd.
                 Function` so the LM loss can drive the permanence updates
                 indirectly (the kernel is the *update direction*; gradient
                 descent on the LM loss replaces hard Hebbian increments).

Numenta-HTM fidelity:
  - Overlap, boost, k-WTA, predicted-vs-burst dynamics: FAITHFUL.
  - Discrete synapse permanence threshold:  replaced with sigmoid for
    end-to-end differentiability (soft-HTM). The same operator is used at
    train and eval — no Python dual-branch.

References:
  - htm.core (community fork of nupic): py/htm/algorithms/spatial_pooler.py,
    temporal_memory.py
  - Hawkins & Ahmad 2016, "Why Neurons Have Thousands of Synapses"
  - BAMI (Biological & Machine Intelligence book), Numenta

Production-only: bf16 in / bf16 out, fp32 internal accumulation.
"""

from __future__ import annotations

import math
import torch
import triton
import triton.language as tl
from torch import Tensor


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pick_num_warps(block_size: int) -> int:
    if block_size <= 128:
        return 2
    if block_size <= 512:
        return 4
    if block_size <= 2048:
        return 8
    return 16


# =========================================================================== #
# SPATIAL POOLER : boosted overlap (forward + backward)
# =========================================================================== #
#
# Inputs:
#   X    (M, D_in)        bf16 or fp32   activations / SDR input
#   P    (N_cols, D_in)   fp32           potential-synapse permanence (param)
#   B    (N_cols,)        fp32           boost factor (buffer, homeostasis)
# Output:
#   O    (M, N_cols)      fp32           boosted overlap, ready for k-WTA
#
# Compute (per (m, c)):
#   gate_{c,d} = sigmoid( (P[c,d] - thr) / tau )    # soft synapse open
#   ov_{m,c}   = sum_d  gate_{c,d} * X[m,d]
#   out_{m,c}  = ov_{m,c} * B[c]
#
# Backward:
#   dL/dX[m,d]   = sum_c  dL/dout[m,c] * B[c] * gate_{c,d}
#   dL/dP[c,d]   = sum_m  dL/dout[m,c] * B[c] * X[m,d] * gate_{c,d}*(1-gate_{c,d})/tau
#   dL/dB[c]     = sum_m  dL/dout[m,c] * ov_{m,c}                (stop-grad here:
#                                                                  boost is a
#                                                                  homeostatic
#                                                                  buffer, not a
#                                                                  param)
#
# We tile rows (M) and columns (N_cols), and stream the D_in reduction in chunks
# of BLOCK_D. fp32 accumulation throughout.
# --------------------------------------------------------------------------- #


@triton.jit
def _sp_overlap_fwd_kernel(
    X_ptr, P_ptr, B_ptr, OUT_ptr,
    stride_x_m, stride_x_d,
    stride_p_c, stride_p_d,
    stride_o_m, stride_o_c,
    M, N_cols, D_in,
    perm_thr, perm_tau_inv,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 2-D grid: (row tile, col tile)
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_idx = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    m_mask = m_idx < M
    c_mask = c_idx < N_cols

    # Accumulator (BLOCK_M, BLOCK_C) in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for d_start in range(0, D_in, BLOCK_D):
        d_idx = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_idx < D_in

        # X tile: (BLOCK_M, BLOCK_D)
        x_ptrs = X_ptr + m_idx[:, None] * stride_x_m + d_idx[None, :] * stride_x_d
        x = tl.load(
            x_ptrs,
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # P tile: (BLOCK_C, BLOCK_D) -> gate via sigmoid((P - thr) * tau_inv)
        p_ptrs = P_ptr + c_idx[:, None] * stride_p_c + d_idx[None, :] * stride_p_d
        p = tl.load(
            p_ptrs,
            mask=c_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate = tl.sigmoid((p - perm_thr) * perm_tau_inv)

        # Outer-product-ish: ov_{m,c} += sum_d x_{m,d} * gate_{c,d}
        # Use tl.dot for the contraction (BLOCK_M,BLOCK_D) x (BLOCK_D,BLOCK_C)
        gate_t = tl.trans(gate)  # (BLOCK_D, BLOCK_C)
        acc += tl.dot(x, gate_t, allow_tf32=False)

    # Load boost for these columns
    boost = tl.load(B_ptr + c_idx, mask=c_mask, other=1.0).to(tl.float32)
    out = acc * boost[None, :]

    out_ptrs = OUT_ptr + m_idx[:, None] * stride_o_m + c_idx[None, :] * stride_o_c
    tl.store(
        out_ptrs,
        out,
        mask=m_mask[:, None] & c_mask[None, :],
    )


@triton.jit
def _sp_overlap_bwd_kernel(
    X_ptr, P_ptr, B_ptr, DOUT_ptr,
    DX_ptr, DP_ptr,
    stride_x_m, stride_x_d,
    stride_p_c, stride_p_d,
    stride_o_m, stride_o_c,
    stride_dx_m, stride_dx_d,
    stride_dp_c, stride_dp_d,
    M, N_cols, D_in,
    perm_thr, perm_tau_inv,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per (row tile, d tile). For each c block, accumulate dX
    and dP into the global buffers via atomic_add (small fan-in, cheap)."""
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_idx = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    m_mask = m_idx < M
    d_mask = d_idx < D_in

    # X tile: (BLOCK_M, BLOCK_D)
    x_ptrs = X_ptr + m_idx[:, None] * stride_x_m + d_idx[None, :] * stride_x_d
    x = tl.load(
        x_ptrs,
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    dx_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for c_start in range(0, N_cols, BLOCK_C):
        c_idx = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_idx < N_cols

        # dout tile: (BLOCK_M, BLOCK_C)
        dout_ptrs = DOUT_ptr + m_idx[:, None] * stride_o_m + c_idx[None, :] * stride_o_c
        dout = tl.load(
            dout_ptrs,
            mask=m_mask[:, None] & c_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        boost = tl.load(B_ptr + c_idx, mask=c_mask, other=0.0).to(tl.float32)
        dout_scaled = dout * boost[None, :]  # (BLOCK_M, BLOCK_C)

        # P tile: (BLOCK_C, BLOCK_D)
        p_ptrs = P_ptr + c_idx[:, None] * stride_p_c + d_idx[None, :] * stride_p_d
        p = tl.load(
            p_ptrs,
            mask=c_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate = tl.sigmoid((p - perm_thr) * perm_tau_inv)
        dgate_dp = gate * (1.0 - gate) * perm_tau_inv  # (BLOCK_C, BLOCK_D)

        # dX[m,d] += sum_c dout_scaled[m,c] * gate[c,d]
        dx_acc += tl.dot(dout_scaled, gate, allow_tf32=False)

        # dP[c,d] += sum_m dout_scaled[m,c] * x[m,d] * dgate_dp[c,d]
        # First, contract over m: (BLOCK_C, BLOCK_D) = dout_scaled^T @ x
        dp_pre = tl.dot(tl.trans(dout_scaled), x, allow_tf32=False)  # (BLOCK_C, BLOCK_D)
        dp_tile = dp_pre * dgate_dp

        # Atomic-add dp_tile into DP
        dp_ptrs = DP_ptr + c_idx[:, None] * stride_dp_c + d_idx[None, :] * stride_dp_d
        tl.atomic_add(
            dp_ptrs,
            dp_tile,
            mask=c_mask[:, None] & d_mask[None, :],
        )

    # Write dX (one writer per (pid_m, pid_d) — no atomics needed)
    dx_ptrs = DX_ptr + m_idx[:, None] * stride_dx_m + d_idx[None, :] * stride_dx_d
    tl.store(
        dx_ptrs,
        dx_acc,
        mask=m_mask[:, None] & d_mask[None, :],
    )


class _SPOverlap(torch.autograd.Function):
    """Boosted soft-overlap kernel. Forward returns the unsparsified, boosted
    overlap (M, N_cols). k-WTA is applied outside in PyTorch on the output."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,          # (M, D_in)
        permanence: Tensor,  # (N_cols, D_in)  fp32
        boost: Tensor,       # (N_cols,)       fp32
        perm_thr: float,
        perm_tau: float,
    ) -> Tensor:
        assert x.is_cuda and permanence.is_cuda and boost.is_cuda
        assert permanence.dim() == 2 and boost.dim() == 1
        assert x.dim() == 2, f"expected (M, D_in), got {x.shape}"
        assert permanence.dtype == torch.float32, "permanence must be fp32"
        assert boost.dtype == torch.float32, "boost must be fp32"
        N_cols, D_in = permanence.shape
        assert x.shape[1] == D_in, f"x last dim {x.shape[1]} != perm D_in {D_in}"
        assert boost.shape[0] == N_cols

        x_c = x.contiguous()
        p_c = permanence.contiguous()
        b_c = boost.contiguous()
        M = x_c.shape[0]

        out = torch.empty((M, N_cols), dtype=torch.float32, device=x.device)

        # Tile shapes — fit RTX 3060 SM86 cache hierarchy.
        # BLOCK_M must be >= 16 because tl.dot requires K >= 16 along the
        # contraction dim that traverses M in the backward kernel. We use the
        # row-mask to handle M < BLOCK_M.
        BLOCK_M = 16
        BLOCK_C = 32
        BLOCK_D = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_cols, BLOCK_C))

        perm_tau_inv = 1.0 / float(perm_tau)
        _sp_overlap_fwd_kernel[grid](
            x_c, p_c, b_c, out,
            x_c.stride(0), x_c.stride(1),
            p_c.stride(0), p_c.stride(1),
            out.stride(0), out.stride(1),
            M, N_cols, D_in,
            float(perm_thr), perm_tau_inv,
            BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D,
            num_warps=_pick_num_warps(BLOCK_M * BLOCK_C),
        )

        # Clone boost — it's a homeostatic buffer that gets updated in-place
        # between forward and backward by FaithfulHTM._update_boost, which
        # would otherwise break autograd's version counter.
        ctx.save_for_backward(x_c, p_c, b_c.clone())
        ctx.perm_thr = float(perm_thr)
        ctx.perm_tau_inv = perm_tau_inv
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_C = BLOCK_C
        ctx.BLOCK_D = BLOCK_D
        ctx.x_dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, dout: Tensor):
        x_c, p_c, b_c = ctx.saved_tensors
        M, D_in = x_c.shape
        N_cols = p_c.shape[0]
        dout_c = dout.contiguous().to(torch.float32)

        dx = torch.empty((M, D_in), dtype=torch.float32, device=x_c.device)
        dp = torch.zeros((N_cols, D_in), dtype=torch.float32, device=x_c.device)

        BLOCK_M = ctx.BLOCK_M
        BLOCK_C = ctx.BLOCK_C
        BLOCK_D = ctx.BLOCK_D
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D_in, BLOCK_D))

        _sp_overlap_bwd_kernel[grid](
            x_c, p_c, b_c, dout_c,
            dx, dp,
            x_c.stride(0), x_c.stride(1),
            p_c.stride(0), p_c.stride(1),
            dout_c.stride(0), dout_c.stride(1),
            dx.stride(0), dx.stride(1),
            dp.stride(0), dp.stride(1),
            M, N_cols, D_in,
            ctx.perm_thr, ctx.perm_tau_inv,
            BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D,
            num_warps=_pick_num_warps(BLOCK_M * BLOCK_C),
        )

        return dx.to(ctx.x_dtype), dp, None, None, None


def sp_boosted_overlap(
    x: Tensor, permanence: Tensor, boost: Tensor,
    perm_thr: float = 0.5, perm_tau: float = 0.1,
) -> Tensor:
    """Spatial-pooler boosted soft-overlap.
    x: (M, D_in), permanence: (N_cols, D_in) fp32, boost: (N_cols,) fp32.
    Returns: (M, N_cols) fp32 — ready for top-k k-WTA.
    """
    return _SPOverlap.apply(x, permanence, boost, perm_thr, perm_tau)


# =========================================================================== #
# TEMPORAL MEMORY : predicted-cell scoring (forward + backward)
# =========================================================================== #
#
# Inputs:
#   prev_active   (M, N_cols * cells_per_col)  fp32  [0/1 soft]
#   seg_perm      (N_cells_total, segs_per_cell, syns_per_seg)   fp32  param
#   seg_syn_idx   (N_cells_total, segs_per_cell, syns_per_seg)   int32
#                  — index into prev_active's last dim (which cell each
#                  synapse listens to). Static topology, sampled once.
#
# Output:
#   cell_score    (M, N_cells_total)  fp32
#                  — max over segments of  sum_s sigmoid((perm-thr)/tau)
#                                          * prev_active[seg_syn_idx[s]]
#
# This *score* serves as the prediction probability: high → cell is predicted.
# In the python wrapper we feed it through a sigmoid + multiply by the
# active-column mask to get final active cells.
#
# Backward:
#   d/d_prev_active[m, j] = sum_{ic, ig where syn_idx==j}
#                                gate * (segment-was-argmax indicator)
#                                * dL/dscore[m, ic]
#   d/d_seg_perm[ic, ig, s] = sum_m  (segment-was-argmax) * gate*(1-gate)/tau
#                                   * prev_active[seg_syn_idx[s]]
#                                   * dL/dscore[m, ic]
#
# For differentiability of the max we use a soft-max trick at backward time:
# instead of hard argmax, we backprop through *all* segments scaled by softmax
# weight (with sharpness 1). This matches the test parity reference.
# --------------------------------------------------------------------------- #


@triton.jit
def _tm_predict_fwd_kernel(
    PREV_ptr, PERM_ptr, IDX_ptr,
    SCORE_ptr, LSE_ptr, SEG_ACT_ptr,
    stride_prev_m, stride_prev_n,
    stride_perm_c, stride_perm_g, stride_perm_s,
    stride_idx_c, stride_idx_g, stride_idx_s,
    stride_score_m, stride_score_c,
    stride_lse_m, stride_lse_c,
    stride_seg_m, stride_seg_c, stride_seg_g,
    M, N_cells, N_segs, N_syns, N_prev,
    perm_thr, perm_tau_inv,
    SOFTMAX_BETA: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """One program per (row m, cell c). Loops over segments and synapses."""
    m = tl.program_id(0)
    c = tl.program_id(1)
    if m >= M:
        return
    if c >= N_cells:
        return

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < N_segs

    s_idx = tl.arange(0, BLOCK_S)
    s_mask = s_idx < N_syns

    # Per-segment activation, kept as (BLOCK_G,) accumulator
    # Loop nest: for g in BLOCK_G, sum over s.
    # We use a 2-D layout: load perm (BLOCK_G, BLOCK_S), idx (BLOCK_G, BLOCK_S),
    # gather prev (BLOCK_G, BLOCK_S), compute act per (g,s), reduce over s.
    perm_ptrs = (PERM_ptr
                 + c * stride_perm_c
                 + g_idx[:, None] * stride_perm_g
                 + s_idx[None, :] * stride_perm_s)
    idx_ptrs = (IDX_ptr
                + c * stride_idx_c
                + g_idx[:, None] * stride_idx_g
                + s_idx[None, :] * stride_idx_s)

    perm = tl.load(
        perm_ptrs,
        mask=g_mask[:, None] & s_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    idx = tl.load(
        idx_ptrs,
        mask=g_mask[:, None] & s_mask[None, :],
        other=0,
    )

    gate = tl.sigmoid((perm - perm_thr) * perm_tau_inv)

    # Gather prev_active[m, idx]: (BLOCK_G, BLOCK_S)
    prev_ptrs = PREV_ptr + m * stride_prev_m + idx * stride_prev_n
    # Mask out-of-bounds — idx clipped to 0 and value contributes 0 when masked
    prev_safe_mask = (idx >= 0) & (idx < N_prev) & g_mask[:, None] & s_mask[None, :]
    prev = tl.load(prev_ptrs, mask=prev_safe_mask, other=0.0).to(tl.float32)

    contrib = gate * prev  # (BLOCK_G, BLOCK_S)
    seg_act = tl.sum(contrib, axis=1)  # (BLOCK_G,)

    # Store per-segment activation for backward
    seg_ptrs = (SEG_ACT_ptr
                + m * stride_seg_m
                + c * stride_seg_c
                + g_idx * stride_seg_g)
    tl.store(seg_ptrs, seg_act, mask=g_mask)

    # Soft-max over segments — log-sum-exp for stable backward
    beta = SOFTMAX_BETA
    seg_act_masked = tl.where(g_mask, seg_act, -1e30)
    max_v = tl.max(seg_act_masked * beta, axis=0)
    shifted = (seg_act * beta) - max_v
    shifted = tl.where(g_mask, shifted, -1e30)
    exp_shift = tl.exp(shifted)
    sum_exp = tl.sum(exp_shift, axis=0)
    lse = max_v + tl.log(sum_exp)
    score = lse / beta  # ~ softmax-pooled max

    tl.store(SCORE_ptr + m * stride_score_m + c * stride_score_c, score)
    tl.store(LSE_ptr + m * stride_lse_m + c * stride_lse_c, lse)


@triton.jit
def _tm_predict_bwd_kernel(
    PREV_ptr, PERM_ptr, IDX_ptr,
    SEG_ACT_ptr, LSE_ptr, DSCORE_ptr,
    DPREV_ptr, DPERM_ptr,
    stride_prev_m, stride_prev_n,
    stride_perm_c, stride_perm_g, stride_perm_s,
    stride_idx_c, stride_idx_g, stride_idx_s,
    stride_seg_m, stride_seg_c, stride_seg_g,
    stride_lse_m, stride_lse_c,
    stride_score_m, stride_score_c,
    stride_dprev_m, stride_dprev_n,
    stride_dperm_c, stride_dperm_g, stride_dperm_s,
    M, N_cells, N_segs, N_syns, N_prev,
    perm_thr, perm_tau_inv,
    SOFTMAX_BETA: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    m = tl.program_id(0)
    c = tl.program_id(1)
    if m >= M:
        return
    if c >= N_cells:
        return

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < N_segs
    s_idx = tl.arange(0, BLOCK_S)
    s_mask = s_idx < N_syns

    beta = SOFTMAX_BETA

    # Load per-segment activation and lse
    seg_ptrs = (SEG_ACT_ptr
                + m * stride_seg_m
                + c * stride_seg_c
                + g_idx * stride_seg_g)
    seg_act = tl.load(seg_ptrs, mask=g_mask, other=-1e30)

    lse = tl.load(LSE_ptr + m * stride_lse_m + c * stride_lse_c)
    dscore = tl.load(DSCORE_ptr + m * stride_score_m + c * stride_score_c)

    # softmax weights w_g = exp(beta*seg_act - lse), zero where masked
    weights = tl.exp(beta * seg_act - lse)
    weights = tl.where(g_mask, weights, 0.0)

    # d(score)/d(seg_act_g) = w_g * (1/beta) * beta = w_g  (because score = lse/beta)
    # ==> dseg = dscore * w_g
    dseg = dscore * weights  # (BLOCK_G,)

    # Now back through gate * prev:
    #   d/d_prev[m, idx] += gate * dseg[g]            (atomic add — many cells write here)
    #   d/d_perm[c,g,s]  += gate*(1-gate)*tau_inv * prev * dseg[g]   (no contention)
    perm_ptrs = (PERM_ptr
                 + c * stride_perm_c
                 + g_idx[:, None] * stride_perm_g
                 + s_idx[None, :] * stride_perm_s)
    idx_ptrs = (IDX_ptr
                + c * stride_idx_c
                + g_idx[:, None] * stride_idx_g
                + s_idx[None, :] * stride_idx_s)

    perm = tl.load(
        perm_ptrs,
        mask=g_mask[:, None] & s_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    idx = tl.load(
        idx_ptrs,
        mask=g_mask[:, None] & s_mask[None, :],
        other=0,
    )
    gate = tl.sigmoid((perm - perm_thr) * perm_tau_inv)

    prev_ptrs = PREV_ptr + m * stride_prev_m + idx * stride_prev_n
    prev_mask = (idx >= 0) & (idx < N_prev) & g_mask[:, None] & s_mask[None, :]
    prev = tl.load(prev_ptrs, mask=prev_mask, other=0.0).to(tl.float32)

    # dseg broadcast across s: (BLOCK_G, BLOCK_S)
    dseg_bc = dseg[:, None] * tl.full((1, BLOCK_S), 1.0, dtype=tl.float32)

    # dprev contribution per (g,s): gate * dseg_bc
    dprev_contrib = gate * dseg_bc
    dprev_contrib = tl.where(prev_mask, dprev_contrib, 0.0)
    # atomic scatter into DPREV
    dprev_ptrs = DPREV_ptr + m * stride_dprev_m + idx * stride_dprev_n
    tl.atomic_add(
        dprev_ptrs,
        dprev_contrib,
        mask=prev_mask,
    )

    # dperm contribution per (g,s): gate*(1-gate)*tau_inv * prev * dseg_bc
    dperm_contrib = gate * (1.0 - gate) * perm_tau_inv * prev * dseg_bc
    dperm_ptrs = (DPERM_ptr
                  + c * stride_dperm_c
                  + g_idx[:, None] * stride_dperm_g
                  + s_idx[None, :] * stride_dperm_s)
    tl.atomic_add(
        dperm_ptrs,
        dperm_contrib,
        mask=g_mask[:, None] & s_mask[None, :],
    )


class _TMPredict(torch.autograd.Function):
    """Temporal-Memory predicted-cell scoring kernel.

    Inputs:
        prev_active : (M, N_prev=N_cols*cells_per_col)  fp32
        seg_perm    : (N_cells, N_segs, N_syns)         fp32  (param)
        seg_idx     : (N_cells, N_segs, N_syns)         int32 (buffer, static topo)
    Output:
        cell_score  : (M, N_cells)  fp32 — soft-max-pooled over segments.
    """

    @staticmethod
    def forward(
        ctx,
        prev_active: Tensor,
        seg_perm: Tensor,
        seg_idx: Tensor,
        perm_thr: float,
        perm_tau: float,
        softmax_beta: float,
    ) -> Tensor:
        assert prev_active.is_cuda and seg_perm.is_cuda and seg_idx.is_cuda
        assert prev_active.dim() == 2
        assert seg_perm.dim() == 3 and seg_idx.dim() == 3
        assert seg_perm.shape == seg_idx.shape, "seg_perm and seg_idx must match"
        assert seg_perm.dtype == torch.float32
        assert seg_idx.dtype == torch.int32

        M, N_prev = prev_active.shape
        N_cells, N_segs, N_syns = seg_perm.shape

        prev_c = prev_active.contiguous().to(torch.float32)
        perm_c = seg_perm.contiguous()
        idx_c = seg_idx.contiguous()

        score = torch.empty((M, N_cells), dtype=torch.float32, device=prev_active.device)
        lse = torch.empty((M, N_cells), dtype=torch.float32, device=prev_active.device)
        seg_act = torch.empty(
            (M, N_cells, N_segs), dtype=torch.float32, device=prev_active.device
        )

        BLOCK_S = _next_pow2(N_syns)
        BLOCK_G = _next_pow2(N_segs)
        # cap to avoid huge register pressure on RTX 3060
        assert BLOCK_S * BLOCK_G <= 4096, (
            f"N_segs*N_syns padded ({BLOCK_S}*{BLOCK_G}) too big for SM86 — reduce"
        )

        grid = (M, N_cells)
        perm_tau_inv = 1.0 / float(perm_tau)
        _tm_predict_fwd_kernel[grid](
            prev_c, perm_c, idx_c,
            score, lse, seg_act,
            prev_c.stride(0), prev_c.stride(1),
            perm_c.stride(0), perm_c.stride(1), perm_c.stride(2),
            idx_c.stride(0), idx_c.stride(1), idx_c.stride(2),
            score.stride(0), score.stride(1),
            lse.stride(0), lse.stride(1),
            seg_act.stride(0), seg_act.stride(1), seg_act.stride(2),
            M, N_cells, N_segs, N_syns, N_prev,
            float(perm_thr), perm_tau_inv,
            SOFTMAX_BETA=float(softmax_beta),
            BLOCK_S=BLOCK_S, BLOCK_G=BLOCK_G,
            num_warps=_pick_num_warps(BLOCK_S * BLOCK_G),
        )

        ctx.save_for_backward(prev_c, perm_c, idx_c, seg_act, lse)
        ctx.perm_thr = float(perm_thr)
        ctx.perm_tau_inv = perm_tau_inv
        ctx.softmax_beta = float(softmax_beta)
        ctx.BLOCK_S = BLOCK_S
        ctx.BLOCK_G = BLOCK_G
        ctx.shapes = (M, N_cells, N_segs, N_syns, N_prev)
        ctx.prev_dtype = prev_active.dtype
        return score

    @staticmethod
    def backward(ctx, dscore: Tensor):
        prev_c, perm_c, idx_c, seg_act, lse = ctx.saved_tensors
        M, N_cells, N_segs, N_syns, N_prev = ctx.shapes

        dscore_c = dscore.contiguous().to(torch.float32)
        dprev = torch.zeros_like(prev_c)
        dperm = torch.zeros_like(perm_c)

        grid = (M, N_cells)
        _tm_predict_bwd_kernel[grid](
            prev_c, perm_c, idx_c,
            seg_act, lse, dscore_c,
            dprev, dperm,
            prev_c.stride(0), prev_c.stride(1),
            perm_c.stride(0), perm_c.stride(1), perm_c.stride(2),
            idx_c.stride(0), idx_c.stride(1), idx_c.stride(2),
            seg_act.stride(0), seg_act.stride(1), seg_act.stride(2),
            lse.stride(0), lse.stride(1),
            dscore_c.stride(0), dscore_c.stride(1),
            dprev.stride(0), dprev.stride(1),
            dperm.stride(0), dperm.stride(1), dperm.stride(2),
            M, N_cells, N_segs, N_syns, N_prev,
            ctx.perm_thr, ctx.perm_tau_inv,
            SOFTMAX_BETA=ctx.softmax_beta,
            BLOCK_S=ctx.BLOCK_S, BLOCK_G=ctx.BLOCK_G,
            num_warps=_pick_num_warps(ctx.BLOCK_S * ctx.BLOCK_G),
        )

        return dprev.to(ctx.prev_dtype), dperm, None, None, None, None


def tm_predict(
    prev_active: Tensor, seg_perm: Tensor, seg_idx: Tensor,
    perm_thr: float = 0.5, perm_tau: float = 0.1, softmax_beta: float = 4.0,
) -> Tensor:
    """Temporal-Memory predicted-cell scoring.
    prev_active (M, N_cells*cells_per_col), seg_perm (N_cells, N_segs, N_syns) fp32,
    seg_idx (N_cells, N_segs, N_syns) int32. Returns (M, N_cells) fp32.
    """
    return _TMPredict.apply(prev_active, seg_perm, seg_idx, perm_thr, perm_tau, softmax_beta)


# =========================================================================== #
# TEMPORAL MEMORY : Hebbian-soft learning step (forward-only — no autograd)
# =========================================================================== #
#
# Numenta's TM does:
#   for each active segment (cell predicted, column active):
#     for each synapse in segment:
#       if pre-cell was active in t-1:   perm += inc
#       else:                            perm -= dec
#
# We expose this as a kernel that emits a *delta* tensor — to be applied to
# permanence outside training. Inside training, the LM-loss gradient via
# `_TMPredict.backward` is the *primary* driver of permanence updates. The
# Hebbian step provides a biologically-motivated bias/regulariser.
# --------------------------------------------------------------------------- #


@triton.jit
def _tm_learn_kernel(
    PREV_ptr, POST_ptr, PERM_ptr, IDX_ptr,
    DELTA_ptr,
    stride_prev_m, stride_prev_n,
    stride_post_m, stride_post_c,
    stride_perm_c, stride_perm_g, stride_perm_s,
    stride_idx_c, stride_idx_g, stride_idx_s,
    stride_delta_c, stride_delta_g, stride_delta_s,
    M, N_cells, N_segs, N_syns, N_prev,
    M_INV: tl.constexpr,
    inc: tl.constexpr,
    dec: tl.constexpr,
    decay: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    """One program per cell. For each (g,s), sum over the batch dim m of
    post[m,c] * (inc * prev - dec * (1-prev)) - decay * perm."""
    c = tl.program_id(0)
    if c >= N_cells:
        return

    g_idx = tl.arange(0, BLOCK_G)
    g_mask = g_idx < N_segs
    s_idx = tl.arange(0, BLOCK_S)
    s_mask = s_idx < N_syns

    perm_ptrs = (PERM_ptr
                 + c * stride_perm_c
                 + g_idx[:, None] * stride_perm_g
                 + s_idx[None, :] * stride_perm_s)
    idx_ptrs = (IDX_ptr
                + c * stride_idx_c
                + g_idx[:, None] * stride_idx_g
                + s_idx[None, :] * stride_idx_s)
    perm = tl.load(
        perm_ptrs,
        mask=g_mask[:, None] & s_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    idx = tl.load(
        idx_ptrs,
        mask=g_mask[:, None] & s_mask[None, :],
        other=0,
    )

    acc = tl.zeros((BLOCK_G, BLOCK_S), dtype=tl.float32)
    for m in range(M):
        post_val = tl.load(POST_ptr + m * stride_post_m + c * stride_post_c).to(tl.float32)
        prev_ptrs = PREV_ptr + m * stride_prev_m + idx * stride_prev_n
        prev_mask = (idx >= 0) & (idx < N_prev) & g_mask[:, None] & s_mask[None, :]
        prev = tl.load(prev_ptrs, mask=prev_mask, other=0.0).to(tl.float32)
        # Hebbian step: increment on co-active, decrement on pre-only-inactive
        step = inc * prev - dec * (1.0 - prev)
        acc += post_val * step

    # final delta = avg over M - decay * perm
    avg = acc * M_INV
    delta = avg - decay * perm
    delta_ptrs = (DELTA_ptr
                  + c * stride_delta_c
                  + g_idx[:, None] * stride_delta_g
                  + s_idx[None, :] * stride_delta_s)
    tl.store(
        delta_ptrs,
        delta,
        mask=g_mask[:, None] & s_mask[None, :],
    )


def tm_hebbian_delta(
    prev_active: Tensor,
    post_active: Tensor,
    seg_perm: Tensor,
    seg_idx: Tensor,
    inc: float = 0.05,
    dec: float = 0.01,
    decay: float = 0.001,
) -> Tensor:
    """Compute Hebbian-soft permanence delta. Returns same shape as seg_perm,
    fp32. Apply with `seg_perm.add_(lr * delta)` outside training, or use as a
    regulariser term."""
    assert prev_active.dtype == torch.float32
    assert post_active.dtype == torch.float32
    assert seg_perm.dtype == torch.float32
    assert seg_idx.dtype == torch.int32
    M, N_prev = prev_active.shape
    N_cells, N_segs, N_syns = seg_perm.shape
    assert post_active.shape == (M, N_cells)

    delta = torch.empty_like(seg_perm)

    BLOCK_S = _next_pow2(N_syns)
    BLOCK_G = _next_pow2(N_segs)
    grid = (N_cells,)
    _tm_learn_kernel[grid](
        prev_active.contiguous(), post_active.contiguous(),
        seg_perm.contiguous(), seg_idx.contiguous(),
        delta,
        prev_active.stride(0), prev_active.stride(1),
        post_active.stride(0), post_active.stride(1),
        seg_perm.stride(0), seg_perm.stride(1), seg_perm.stride(2),
        seg_idx.stride(0), seg_idx.stride(1), seg_idx.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        M, N_cells, N_segs, N_syns, N_prev,
        M_INV=1.0 / float(M),
        inc=float(inc), dec=float(dec), decay=float(decay),
        BLOCK_S=BLOCK_S, BLOCK_G=BLOCK_G,
        num_warps=_pick_num_warps(BLOCK_S * BLOCK_G),
    )
    return delta


__all__ = [
    "sp_boosted_overlap",
    "tm_predict",
    "tm_hebbian_delta",
]
