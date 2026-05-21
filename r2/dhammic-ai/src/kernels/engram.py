"""Faithful DeepSeek Engram — fused multi-head N-gram embedding gather (Triton).

This kernel implements the *core* hot path of the Engram module from DeepSeek-AI's
"Conditional Memory via Scalable Lookup" (arXiv:2601.07372, Jan 2026,
github.com/deepseek-ai/Engram):

    For each token position t, for each n-gram size n in {2..N_MAX}, for each
    of H hash heads, look up an embedding row from a per-(ngram_size, head)
    embedding table. Concatenate the H heads to form an `n_embed_per_ngram`-
    dimensional vector per (ngram_size, t). Sum over ngram_sizes (or
    concatenate — we sum to keep d_value fixed, matching multi-head additivity).

Public Triton entry point:
    fused_engram_lookup(
        hash_ids,        # (B, T, N_GROUPS) int64, each in [0, head_vocab_size[g])
        head_offsets,    # (N_GROUPS,)      int64, additive offset into the combined
                         #                  embedding table per (ngram,head) group
        head_strides,    # (N_GROUPS,)      int64, indicates which output slice
                         #                  (d_head-sized contiguous block) this
                         #                  group writes into
        embed_weight,    # (TOTAL_ROWS, D_HEAD) bf16/fp16/fp32
        D_OUT            # int, n_embed_per_ngram (= H * d_head)
    ) -> retrieved (B, T, D_OUT)

The Python wrapper `FaithfulEngram` (see src/engram_layer.py) handles the
n-gram hash computation (cheap, vectorized in PyTorch) and provides the
gating/value projection wrapper.

Production-only: bf16 in / bf16 out, fp32 internal accumulation.
Backward writes gradients into `embed_weight` via `tl.atomic_add`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


# --------------------------------------------------------------------------- #
# Forward kernel:
#   grid = (M = B*T, n_d_tiles)
#   Each program covers one token row and a BLOCK_D-sized slice of the
#   concatenated output. For each group g (ngram x head), if the slice
#   overlaps the destination band [g*d_head, (g+1)*d_head), this program
#   loads the embedding row for that group's hash_id and writes it.
# --------------------------------------------------------------------------- #
def _fwd_configs():
    return [
        triton.Config({"BLOCK_D": 32}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ]


@triton.autotune(configs=_fwd_configs(), key=["D_HEAD", "N_GROUPS"])
@triton.jit
def _engram_lookup_fwd_kernel(
    HASH_ptr,          # (M, N_GROUPS) int64
    OFFS_ptr,          # (N_GROUPS,)   int64
    W_ptr,             # (TOTAL_ROWS, D_HEAD) bf16/fp32
    OUT_ptr,           # (M, D_OUT) bf16/fp32 (D_OUT = N_GROUPS_PER_BAND * D_HEAD)
    M, D_HEAD, D_OUT, N_GROUPS,
    n_groups_per_band,  # number of groups whose outputs are SUMMED into the same band
    n_bands,            # D_OUT / D_HEAD
    stride_hash_m,
    stride_w_row,
    stride_out_m,
    BLOCK_D: tl.constexpr,
):
    """Forward: out[m, b*D_HEAD + d] = sum_{g in band b} W[hash[m,g] + offs[g], d].

    Group-to-band mapping is implicit: group g writes into band (g % n_bands).
    That is, head index = g % n_bands, and groups within the same band
    (different n-gram sizes for the same head) are SUMMED. This implements
    the "sum over n-gram sizes, concatenate over heads" pattern faithful to
    the Engram paper.
    """
    pid_m = tl.program_id(0)
    pid_d_band = tl.program_id(1)

    if pid_m >= M:
        return

    d_offs = tl.arange(0, BLOCK_D)
    # Each band is exactly D_HEAD wide; the kernel covers band `pid_d_band`.
    d_mask = d_offs < D_HEAD

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Iterate over groups in this band: groups g where g % n_bands == pid_d_band.
    # For efficient indexing we iterate g = pid_d_band + n_bands * i, i = 0..n_groups_per_band-1.
    for i in range(0, n_groups_per_band):
        g = pid_d_band + i * n_bands
        # Safety: g must be < N_GROUPS. (Should be — caller ensures shape.)
        hash_id = tl.load(HASH_ptr + pid_m * stride_hash_m + g).to(tl.int64)
        offs = tl.load(OFFS_ptr + g).to(tl.int64)
        row = hash_id + offs
        emb_ptrs = W_ptr + row * stride_w_row + d_offs
        emb = tl.load(emb_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        acc += emb

    out_d = pid_d_band * D_HEAD + d_offs
    out_ptrs = OUT_ptr + pid_m * stride_out_m + out_d
    tl.store(out_ptrs, acc.to(OUT_ptr.dtype.element_ty), mask=d_mask)


# --------------------------------------------------------------------------- #
# Backward kernel:
#   d_W[hash[m,g] + offs[g], d] += dY[m, (g % n_bands)*D_HEAD + d]
#   atomically accumulated into the embedding bank.
# --------------------------------------------------------------------------- #
def _bwd_configs():
    return [
        triton.Config({"BLOCK_D": 32}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
    ]


@triton.autotune(
    configs=_bwd_configs(),
    key=["D_HEAD", "N_GROUPS"],
    reset_to_zero=["DW_ptr"],  # zero gradient bank between autotune trials
)
@triton.jit
def _engram_lookup_bwd_kernel(
    DY_ptr,            # (M, D_OUT) fp32 (we pre-cast caller-side)
    HASH_ptr,          # (M, N_GROUPS) int64
    OFFS_ptr,          # (N_GROUPS,)   int64
    DW_ptr,            # (TOTAL_ROWS, D_HEAD) fp32
    M, D_HEAD, D_OUT, N_GROUPS,
    n_bands,
    stride_dy_m,
    stride_hash_m,
    stride_dw_row,
    BLOCK_D: tl.constexpr,
):
    """Scatter dY back into embedding gradient bank via atomic_add."""
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)  # group id

    if pid_m >= M or pid_g >= N_GROUPS:
        return

    band = pid_g % n_bands
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D_HEAD

    # Load dY slice for this group's band.
    dy_d = band * D_HEAD + d_offs
    dy_ptrs = DY_ptr + pid_m * stride_dy_m + dy_d
    dy = tl.load(dy_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    hash_id = tl.load(HASH_ptr + pid_m * stride_hash_m + pid_g).to(tl.int64)
    offs = tl.load(OFFS_ptr + pid_g).to(tl.int64)
    row = hash_id + offs

    dw_ptrs = DW_ptr + row * stride_dw_row + d_offs
    tl.atomic_add(dw_ptrs, dy, mask=d_mask)


# --------------------------------------------------------------------------- #
# autograd.Function
# --------------------------------------------------------------------------- #
class _FusedEngramLookup(torch.autograd.Function):
    """Fused N-gram multi-head embedding gather.

    Inputs
    ------
    hash_ids      : (B, T, N_GROUPS) int64
                    Per-token, per-group hash id. Group g is uniquely identified
                    by (ngram_size, head); ordering is up to the caller, but
                    by convention: group g writes into band (g % n_bands).
                    Groups in the same band are SUMMED (e.g. summing across
                    ngram-sizes for a given head).
    offsets       : (N_GROUPS,) int64
                    Additive row offset per group into the shared embedding
                    table. Caller computes by laying out per-group sub-tables
                    contiguously: offsets[g] = sum(head_vocab_sizes[:g]).
    embed_weight  : (TOTAL_ROWS, D_HEAD) bf16/fp32 — the combined embedding
                    table. TOTAL_ROWS = sum(head_vocab_sizes).
    d_head        : int. D_OUT = n_bands * d_head, where n_bands is the number
                    of heads (= H).
    n_bands       : int. Number of output bands (heads).

    Output
    ------
    retrieved : (B, T, D_OUT) — same dtype as embed_weight.
    """

    @staticmethod
    def forward(ctx, hash_ids: Tensor, offsets: Tensor,
                embed_weight: Tensor, d_head: int, n_bands: int) -> Tensor:
        assert hash_ids.is_cuda and offsets.is_cuda and embed_weight.is_cuda
        assert hash_ids.dtype == torch.int64, f"hash_ids must be int64, got {hash_ids.dtype}"
        assert offsets.dtype == torch.int64, f"offsets must be int64, got {offsets.dtype}"
        assert embed_weight.shape[-1] == d_head, (
            f"embed_weight last dim {embed_weight.shape[-1]} != d_head {d_head}"
        )

        B, T, N_GROUPS = hash_ids.shape
        assert offsets.shape == (N_GROUPS,), (
            f"offsets shape {offsets.shape} != ({N_GROUPS},)"
        )
        assert N_GROUPS % n_bands == 0, (
            f"N_GROUPS ({N_GROUPS}) must be divisible by n_bands ({n_bands})"
        )
        n_groups_per_band = N_GROUPS // n_bands

        D_OUT = n_bands * d_head
        M = B * T

        hash_flat = hash_ids.contiguous().view(M, N_GROUPS)
        offs_c = offsets.contiguous()
        W_c = embed_weight.contiguous()

        out_dtype = embed_weight.dtype
        out = torch.empty((M, D_OUT), dtype=out_dtype, device=W_c.device)

        grid = (M, n_bands)
        _engram_lookup_fwd_kernel[grid](
            hash_flat, offs_c, W_c, out,
            M, d_head, D_OUT, N_GROUPS,
            n_groups_per_band, n_bands,
            hash_flat.stride(0),
            W_c.stride(0),
            out.stride(0),
        )

        ctx.save_for_backward(hash_flat, offs_c, W_c)
        ctx.shapes = (B, T, N_GROUPS, d_head, D_OUT, n_bands)
        return out.view(B, T, D_OUT)

    @staticmethod
    def backward(ctx, dY: Tensor):
        hash_flat, offs_c, W_c = ctx.saved_tensors
        B, T, N_GROUPS, d_head, D_OUT, n_bands = ctx.shapes
        M = B * T

        # Scatter into a fp32 grad bank, then cast to W dtype.
        dW = torch.zeros_like(W_c, dtype=torch.float32)
        dY_f = dY.contiguous().view(M, D_OUT).to(torch.float32)

        grid = (M, N_GROUPS)
        _engram_lookup_bwd_kernel[grid](
            dY_f, hash_flat, offs_c, dW,
            M, d_head, D_OUT, N_GROUPS,
            n_bands,
            dY_f.stride(0),
            hash_flat.stride(0),
            dW.stride(0),
        )

        dW = dW.to(W_c.dtype)
        return None, None, dW, None, None


# --------------------------------------------------------------------------- #
# Functional entry point
# --------------------------------------------------------------------------- #
def fused_engram_lookup(
    hash_ids: Tensor,
    offsets: Tensor,
    embed_weight: Tensor,
    d_head: int,
    n_bands: int,
) -> Tensor:
    """See :class:`_FusedEngramLookup` for argument semantics."""
    return _FusedEngramLookup.apply(hash_ids, offsets, embed_weight, d_head, n_bands)
