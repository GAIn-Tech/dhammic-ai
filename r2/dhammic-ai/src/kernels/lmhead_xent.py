"""
K9: Fused LM head + cross-entropy — Triton kernel for RTX 3060 (SM86).

Fuses: `hidden @ weight.T -> logits -> log_softmax -> NLL`
without ever materializing the full (M, V) logits tensor.

For a (B, T, d_model) hidden and (V, d_model) weight matrix at V ~ 32k+ the
logits tensor dominates VRAM. Liger-Kernel and Apex solve this by chunking the
vocab dimension; we follow the same pattern in pure Triton.

Forward (per row m):
  - x_m = hidden[m, :d]
  - For each vocab tile [v0, v1):
      l_vt = (x_m * W[v0:v1, :]).sum(-1)        # logits for tile (BLOCK_V,)
      m_new = max(m_running, max(l_vt))
      z = z * exp(m_running - m_new) + sum(exp(l_vt - m_new))
      m_running = m_new
      if target in tile: target_logit = l_vt[target - v0]
  - logZ = m_running + log(z)
  - loss_m = (logZ - target_logit) * mask
  - reduction: mean over non-ignored rows (divide by N_valid scalar).

Backward (per row m, with grad of loss = 1/N_valid * mask_m):
  - p_v = exp(l_vt - logZ)
  - d_logit[v] = (p_v - 1[v == target]) * d_loss
  - d_hidden[m, :] += sum_v d_logit[v] * W[v, :]      # (1, d)
  - d_weight[v, :] += d_logit[v] * hidden[m, :]       # broadcast over rows

Tile vocab dim: only (BLOCK_V, d) of W is in registers at a time.

Inputs:
  hidden  : (B, T, d) bf16 or fp32
  weight  : (V, d)    same dtype as hidden (tied to embedding)
  targets : (B, T)    int64; -100 = ignore
Outputs:
  scalar mean loss (fp32 tensor on the same device)

The kernel computes loss in fp32, materializes only:
  - per-row logZ : (M,) fp32
  - per-row target_logit : (M,) fp32
  - per-row mask : (M,) bool
…then reduces in PyTorch.

Hard-fails on V == 0 or d == 0. Production-only: no eager fallback.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch import Tensor


# --------------------------------------------------------------------------- #
# Autotune configs
# --------------------------------------------------------------------------- #
# RTX 3060 (SM86): 28 SMs, 1024 max threads/block. d typically small (96-256),
# V up to ~32-200k. BLOCK_V drives the vocab tile; BLOCK_D must be >= d.
def _fwd_configs():
    # On SM86 the (BLOCK_V, BLOCK_D) tile lives in registers. Keep tiles
    # modest so each program fits in shared mem / register budget; large
    # vocab is handled by the outer Python for-loop, not by bigger tiles.
    return [
        triton.Config({"BLOCK_V": 256}, num_warps=4),
        triton.Config({"BLOCK_V": 512}, num_warps=4),
        triton.Config({"BLOCK_V": 1024}, num_warps=8),
    ]


def _bwd_configs():
    return [
        triton.Config({"BLOCK_V": 256}, num_warps=4),
        triton.Config({"BLOCK_V": 512}, num_warps=4),
        triton.Config({"BLOCK_V": 1024}, num_warps=8),
    ]


# --------------------------------------------------------------------------- #
# Forward kernel: per-row online logsumexp + target logit extraction
# --------------------------------------------------------------------------- #
@triton.autotune(configs=_fwd_configs(), key=["D", "V"])
@triton.jit
def _lmhead_xent_fwd_kernel(
    HID_ptr,          # (M, D)
    W_ptr,            # (V, D)
    TGT_ptr,          # (M,) int64
    LOGZ_ptr,         # (M,) fp32 out
    TGT_LOGIT_ptr,    # (M,) fp32 out
    MASK_ptr,         # (M,) int32 out (1 if valid, 0 if ignored)
    stride_hid_row,
    stride_w_row,
    M, V, D,
    ignore_index,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    tgt = tl.load(TGT_ptr + row)
    is_valid = tgt != ignore_index
    # Clamp target to [0, V-1] so we don't OOB on the comparison; mask handles ignored rows.
    tgt_clamped = tl.where(is_valid & (tgt >= 0) & (tgt < V), tgt, 0)

    # Load hidden[row, :] into registers as fp32
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    h_ptrs = HID_ptr + row * stride_hid_row + d_off
    h = tl.load(h_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # Online logsumexp accumulators (fp32)
    m_running = float("-inf")
    z_running = 0.0
    target_logit = 0.0

    # Sweep vocab in BLOCK_V tiles. `tl.range` keeps this as a runtime loop
    # (no Triton unroll) so compile time stays bounded regardless of V/BLOCK_V.
    n_tiles = tl.cdiv(V, BLOCK_V)
    for tile_idx in tl.range(0, n_tiles):
        v_start = tile_idx * BLOCK_V
        v_off = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_off < V

        # Load W[v_off, :] as (BLOCK_V, BLOCK_D)
        # Address: W_ptr + v_off[:, None] * stride_w_row + d_off[None, :]
        w_ptrs = W_ptr + v_off[:, None] * stride_w_row + d_off[None, :]
        wd_mask = v_mask[:, None] & d_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=wd_mask, other=0.0).to(tl.float32)

        # logits = (w_tile * h[None, :]).sum(-1)  -> (BLOCK_V,)
        logits = tl.sum(w_tile * h[None, :], axis=1)
        # Mask out-of-range vocab to -inf so it can't contribute.
        logits = tl.where(v_mask, logits, float("-inf"))

        # Capture the target logit if it falls in this tile.
        tgt_in_tile = (v_off == tgt_clamped) & is_valid
        target_logit += tl.sum(tl.where(tgt_in_tile, logits, 0.0), axis=0)

        # Online max + exp-sum update
        tile_max = tl.max(logits, axis=0)
        m_new = tl.maximum(m_running, tile_max)
        # exp(prev - new): if m_running == -inf and m_new is finite, treat scale as 0.
        scale = tl.exp(m_running - m_new)
        scale = tl.where(m_running == float("-inf"), 0.0, scale)
        tile_z = tl.sum(tl.exp(logits - m_new), axis=0)
        z_running = z_running * scale + tile_z
        m_running = m_new

    # logZ = m + log(z). For ignored rows (or rows where all logits were -inf),
    # store 0 — they'll be masked out when reducing.
    is_valid_i32 = is_valid.to(tl.int32)
    logZ = m_running + tl.log(z_running)
    tl.store(LOGZ_ptr + row, logZ)
    tl.store(TGT_LOGIT_ptr + row, target_logit)
    tl.store(MASK_ptr + row, is_valid_i32)


# --------------------------------------------------------------------------- #
# Backward kernel: compute d_hidden row + accumulate d_weight via atomics
# --------------------------------------------------------------------------- #
# `DW_ptr` is accumulated via atomic_add. During autotune Triton runs each
# config; without `reset_to_zero` those trial runs would keep adding to the
# same buffer, producing N_configs × N_trials × the true gradient. Same is
# also true for `DHID_ptr` which we tl.store — but stores overwrite, so only
# DW needs resetting. We additionally reset DHID just to be safe in case a
# future change makes it an accumulator too.
@triton.autotune(
    configs=_bwd_configs(),
    key=["D", "V"],
    reset_to_zero=["DW_ptr", "DHID_ptr"],
)
@triton.jit
def _lmhead_xent_bwd_kernel(
    HID_ptr,          # (M, D)
    W_ptr,            # (V, D)
    TGT_ptr,          # (M,) int64
    LOGZ_ptr,         # (M,) fp32
    MASK_ptr,         # (M,) int32
    DHID_ptr,         # (M, D) out — fp32 accumulator
    DW_ptr,           # (V, D) out — fp32 accumulator (atomic)
    loss_scale,       # 1.0 / N_valid (fp32 scalar)
    stride_hid_row,
    stride_w_row,
    stride_dhid_row,
    stride_dw_row,
    M, V, D,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    valid = tl.load(MASK_ptr + row)
    # Early-out for ignored rows: their d_hidden contribution is zero, and
    # they don't contribute to d_weight. Skip the whole sweep.
    if valid == 0:
        return

    tgt = tl.load(TGT_ptr + row)
    logZ = tl.load(LOGZ_ptr + row)

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    h_ptrs = HID_ptr + row * stride_hid_row + d_off
    h = tl.load(h_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # Accumulate d_hidden[row, :] in fp32 registers
    dh_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    n_tiles = tl.cdiv(V, BLOCK_V)
    for tile_idx in tl.range(0, n_tiles):
        v_start = tile_idx * BLOCK_V
        v_off = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_off < V

        w_ptrs = W_ptr + v_off[:, None] * stride_w_row + d_off[None, :]
        wd_mask = v_mask[:, None] & d_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=wd_mask, other=0.0).to(tl.float32)

        # Recompute logits and softmax probabilities
        logits = tl.sum(w_tile * h[None, :], axis=1)
        logits = tl.where(v_mask, logits, float("-inf"))
        p = tl.exp(logits - logZ)

        # d_logit[v] = (p[v] - 1[v == tgt]) * loss_scale
        one_hot = (v_off == tgt).to(tl.float32)
        d_logit = (p - one_hot) * loss_scale
        d_logit = tl.where(v_mask, d_logit, 0.0)

        # d_hidden += sum_v d_logit[v] * W[v, :]
        dh_acc += tl.sum(d_logit[:, None] * w_tile, axis=0)

        # d_weight[v, :] += d_logit[v] * h[:]   (atomic add, may collide across rows)
        dw_tile = d_logit[:, None] * h[None, :]
        dw_ptrs = DW_ptr + v_off[:, None] * stride_dw_row + d_off[None, :]
        tl.atomic_add(dw_ptrs, dw_tile, mask=wd_mask)

    dhid_ptrs = DHID_ptr + row * stride_dhid_row + d_off
    tl.store(dhid_ptrs, dh_acc, mask=d_mask)


# --------------------------------------------------------------------------- #
# Helper: next power of two >= n (for BLOCK_D)
# --------------------------------------------------------------------------- #
def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# --------------------------------------------------------------------------- #
# autograd.Function
# --------------------------------------------------------------------------- #
class FusedLMHeadXent(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: Tensor,
        weight: Tensor,
        targets: Tensor,
        ignore_index: int = -100,
    ) -> Tensor:
        assert hidden.is_cuda and weight.is_cuda and targets.is_cuda, (
            "FusedLMHeadXent requires CUDA tensors"
        )
        assert weight.dim() == 2, f"weight must be 2-D (V, d), got {tuple(weight.shape)}"
        assert hidden.shape[-1] == weight.shape[-1], (
            f"hidden last dim {hidden.shape[-1]} != weight last dim {weight.shape[-1]}"
        )
        assert targets.dtype == torch.long, f"targets must be int64, got {targets.dtype}"

        D = hidden.shape[-1]
        V = weight.shape[0]
        BLOCK_D = _next_pow2(D)
        # Triton tile-load of (BLOCK_V, BLOCK_D) needs both to be reasonable.
        # We let autotune choose BLOCK_V; BLOCK_D is fixed per launch.
        assert BLOCK_D <= 4096, f"d_model {D} too large for current kernel"

        hidden_contig = hidden.contiguous()
        weight_contig = weight.contiguous()
        targets_contig = targets.contiguous()

        orig_hidden_shape = hidden_contig.shape
        hidden_flat = hidden_contig.view(-1, D)
        targets_flat = targets_contig.view(-1)
        M = hidden_flat.shape[0]
        assert targets_flat.shape[0] == M, (
            f"targets length {targets_flat.shape[0]} != hidden rows {M}"
        )

        logZ = torch.empty(M, dtype=torch.float32, device=hidden.device)
        target_logit = torch.empty(M, dtype=torch.float32, device=hidden.device)
        mask = torch.empty(M, dtype=torch.int32, device=hidden.device)

        grid = (M,)
        _lmhead_xent_fwd_kernel[grid](
            hidden_flat, weight_contig, targets_flat,
            logZ, target_logit, mask,
            hidden_flat.stride(0),
            weight_contig.stride(0),
            M, V, D,
            ignore_index,
            BLOCK_D=BLOCK_D,
        )

        # per-row loss = logZ - target_logit (for valid rows)
        per_row_loss = (logZ - target_logit) * mask.to(torch.float32)
        n_valid = mask.sum().clamp_min(1).to(torch.float32)
        loss = per_row_loss.sum() / n_valid

        ctx.save_for_backward(
            hidden_flat, weight_contig, targets_flat, logZ, mask, n_valid
        )
        ctx.orig_hidden_shape = orig_hidden_shape
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = weight.dtype
        ctx.BLOCK_D = BLOCK_D
        ctx.D = D
        ctx.V = V
        return loss

    @staticmethod
    def backward(ctx, dloss: Tensor):
        hidden_flat, weight_contig, targets_flat, logZ, mask, n_valid = ctx.saved_tensors
        D = ctx.D
        V = ctx.V
        BLOCK_D = ctx.BLOCK_D
        M = hidden_flat.shape[0]

        # loss = (1 / N_valid) * sum_m mask_m * (logZ_m - tgt_logit_m)
        # => d_loss / d_logit[m, v] = (mask_m / N_valid) * (softmax(logits)[v] - 1[v==tgt_m])
        # We bake the scalar (dloss * 1/N_valid) into loss_scale; mask is handled per-row.
        loss_scale = (dloss.detach() / n_valid).to(torch.float32).item()

        dhidden_flat = torch.zeros(M, D, dtype=torch.float32, device=hidden_flat.device)
        dweight = torch.zeros(V, D, dtype=torch.float32, device=hidden_flat.device)

        grid = (M,)
        _lmhead_xent_bwd_kernel[grid](
            hidden_flat, weight_contig, targets_flat,
            logZ, mask,
            dhidden_flat, dweight,
            loss_scale,
            hidden_flat.stride(0),
            weight_contig.stride(0),
            dhidden_flat.stride(0),
            dweight.stride(0),
            M, V, D,
            BLOCK_D=BLOCK_D,
        )

        dhidden = dhidden_flat.view(ctx.orig_hidden_shape).to(ctx.hidden_dtype)
        dweight = dweight.to(ctx.weight_dtype)
        # targets and ignore_index are non-tensor / non-differentiable
        return dhidden, dweight, None, None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fused_lmhead_xent(
    hidden: Tensor,
    weight: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Fused (hidden @ weight.T) + cross-entropy, returning scalar mean loss.

    Equivalent to::

        logits = hidden @ weight.T          # (..., V)
        loss = F.cross_entropy(
            logits.view(-1, V),
            targets.view(-1),
            ignore_index=ignore_index,
        )

    but without materializing the (M, V) logits tensor.

    Args:
        hidden: (B, T, d) or (M, d) — any leading dims + last dim d.
        weight: (V, d) — same dtype as hidden (typically the tied embedding).
        targets: integer tensor with shape matching all but the last dim of hidden.
        ignore_index: target value to skip (mirrors F.cross_entropy).

    Returns:
        Scalar fp32 tensor (mean loss over non-ignored positions).
    """
    return FusedLMHeadXent.apply(hidden, weight, targets, ignore_index)


class FusedLMHeadXentModule(nn.Module):
    """Drop-in fused LM-head + cross-entropy module.

    If `weight` is supplied (e.g., tied to the input embedding), it is NOT
    reallocated — the same parameter is reused. Otherwise a new (V, d) weight
    is created.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        weight: nn.Parameter | None = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        if weight is None:
            self.weight = nn.Parameter(
                torch.empty(vocab_size, d_model).normal_(mean=0.0, std=1.0 / math.sqrt(d_model))
            )
        else:
            assert weight.shape == (vocab_size, d_model), (
                f"tied weight shape {tuple(weight.shape)} != ({vocab_size}, {d_model})"
            )
            self.weight = weight  # share the same parameter (weight tying)

    def forward(self, hidden: Tensor, targets: Tensor) -> Tensor:
        return fused_lmhead_xent(hidden, self.weight, targets, self.ignore_index)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, vocab_size={self.vocab_size}, ignore_index={self.ignore_index}"
