"""K8: Fused Engram Gather — Triton kernel for RTX 3060 (SM86).

Fuses the HTM Engram retrieval hot path:
    col_logits = col_proj(x)
    topk_scores, topk_cols = col_logits.topk(k_active, dim=-1)
    col_attn = softmax(topk_scores, dim=-1)
    ctx = ctx_proj(x)
    cell_weights = gumbel_softmax(ctx, tau=1, hard=True)  # train
                   one_hot(ctx.argmax(-1), cells_per_col).float()  # eval
    all_cell_ids = topk_cols * cells_per_col + arange(cells_per_col)
    all_cell_embs = cell_embed(all_cell_ids)           # (B,T,k,C,D)
    cell_embs = (all_cell_embs * cell_weights[:,:,None,:,None]).sum(3)  # (B,T,k,D)
    retrieved = einsum("btk,btkd->btd", col_attn, cell_embs)

The fused Triton kernel handles the *embedding gather + weighted sum* part:
    given col_attn (B,T,k), cell_weights (B,T,C), topk_cols (B,T,k), and
    cell_embed weight W_E (N_cells, D), it produces
        retrieved[b,t,d] = sum_{k,c} col_attn[b,t,k] * cell_weights[b,t,c]
                           * W_E[topk_cols[b,t,k]*C + c, d]
    in one pass with fp32 accumulation.

The col_proj / ctx_proj / topk / softmax / gumbel_softmax stay in PyTorch:
those are cheap matmuls + small softmax over k or C. The expensive
embedding gather is what we fuse — and its backward (scatter_add) gets
PyTorch's efficient index_add_ implementation, with d_col_attn /
d_cell_weights computed in dedicated Triton kernels.

Production-only: bf16 in / bf16 out, fp32 internal accumulation.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


# --------------------------------------------------------------------------- #
# Forward kernel:
#   one program per (row = b*T) covering a tile of D columns.
#   loops over k_active * cells_per_col gather sites.
# --------------------------------------------------------------------------- #
def _fwd_configs():
    return [
        triton.Config({"BLOCK_D": 32}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=8),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ]


@triton.autotune(configs=_fwd_configs(), key=["D"])
@triton.jit
def _engram_fwd_kernel(
    COL_ATTN_ptr,        # (M, K) fp32
    CELL_W_ptr,          # (M, C) fp32
    TOPK_COLS_ptr,       # (M, K) int64
    W_E_ptr,             # (N_cells, D) bf16/fp32
    OUT_ptr,             # (M, D) bf16/fp32 (same dtype as W_E for store cast)
    M, D, K, C,
    cells_per_col,
    stride_col_attn_m,
    stride_cell_w_m,
    stride_topk_m,
    stride_we_n,
    stride_out_m,
    BLOCK_D: tl.constexpr,
    K_CONST: tl.constexpr,
    C_CONST: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    if pid_m >= M:
        return

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Loop k
    for k in tl.static_range(0, K_CONST):
        col_id = tl.load(TOPK_COLS_ptr + pid_m * stride_topk_m + k).to(tl.int64)
        a_k = tl.load(COL_ATTN_ptr + pid_m * stride_col_attn_m + k).to(tl.float32)
        base = col_id * cells_per_col
        # Loop c
        for c in tl.static_range(0, C_CONST):
            w_c = tl.load(CELL_W_ptr + pid_m * stride_cell_w_m + c).to(tl.float32)
            cell_id = base + c
            row_ptr = W_E_ptr + cell_id * stride_we_n + d_offs
            emb = tl.load(row_ptr, mask=d_mask, other=0.0).to(tl.float32)
            acc += (a_k * w_c) * emb

    out_ptr = OUT_ptr + pid_m * stride_out_m + d_offs
    tl.store(out_ptr, acc.to(OUT_ptr.dtype.element_ty), mask=d_mask)


# --------------------------------------------------------------------------- #
# Backward kernels.
# Given dL/dretrieved (B,T,D) = dY (M,D), we need:
#   d_col_attn[m, k]  = sum_d dY[m,d] * sum_c cell_w[m,c] * W_E[topk[m,k]*C+c, d]
#   d_cell_w[m, c]    = sum_d dY[m,d] * sum_k col_attn[m,k] * W_E[topk[m,k]*C+c, d]
#   d_W_E[cell_id, d] += dY[m,d] * col_attn[m,k] * cell_w[m,c]
#     where cell_id = topk[m,k]*C + c  (scatter — done via PyTorch index_add_)
# --------------------------------------------------------------------------- #
def _bwd_kc_configs():
    return [
        triton.Config({"BLOCK_D": 32}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 64}, num_warps=4),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
        triton.Config({"BLOCK_D": 256}, num_warps=8),
    ]


@triton.autotune(configs=_bwd_kc_configs(), key=["D"])
@triton.jit
def _engram_bwd_kc_kernel(
    DY_ptr,              # (M, D) fp32
    COL_ATTN_ptr,        # (M, K) fp32
    CELL_W_ptr,          # (M, C) fp32
    TOPK_COLS_ptr,       # (M, K) int64
    W_E_ptr,             # (N_cells, D) bf16/fp32
    D_COL_ATTN_ptr,      # (M, K) fp32
    D_CELL_W_ptr,        # (M, C) fp32
    M, D, K, C,
    cells_per_col,
    stride_dy_m,
    stride_col_attn_m,
    stride_cell_w_m,
    stride_topk_m,
    stride_we_n,
    stride_d_col_attn_m,
    stride_d_cell_w_m,
    BLOCK_D: tl.constexpr,
    K_CONST: tl.constexpr,
    C_CONST: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    d_offs = tl.arange(0, BLOCK_D)
    n_tiles = (D + BLOCK_D - 1) // BLOCK_D

    # Accumulators for this row
    d_col_attn = tl.zeros((K_CONST,), dtype=tl.float32)
    d_cell_w = tl.zeros((C_CONST,), dtype=tl.float32)

    # Preload col_attn and cell_w into registers (small).
    col_attn_vec = tl.zeros((K_CONST,), dtype=tl.float32)
    for k in tl.static_range(0, K_CONST):
        col_attn_vec_k = tl.load(COL_ATTN_ptr + pid_m * stride_col_attn_m + k).to(tl.float32)
        # Manually write via broadcasted increment trick (Triton has no scatter to register vec)
        col_attn_vec += tl.where(tl.arange(0, K_CONST) == k, col_attn_vec_k, 0.0)

    cell_w_vec = tl.zeros((C_CONST,), dtype=tl.float32)
    for c in tl.static_range(0, C_CONST):
        cell_w_vec_c = tl.load(CELL_W_ptr + pid_m * stride_cell_w_m + c).to(tl.float32)
        cell_w_vec += tl.where(tl.arange(0, C_CONST) == c, cell_w_vec_c, 0.0)

    for tile in range(0, n_tiles):
        d_idx = tile * BLOCK_D + d_offs
        d_mask = d_idx < D
        dy = tl.load(DY_ptr + pid_m * stride_dy_m + d_idx,
                     mask=d_mask, other=0.0).to(tl.float32)

        # For each (k, c), load embedding tile and accumulate partials.
        for k in tl.static_range(0, K_CONST):
            col_id = tl.load(TOPK_COLS_ptr + pid_m * stride_topk_m + k).to(tl.int64)
            base = col_id * cells_per_col
            a_k = tl.load(COL_ATTN_ptr + pid_m * stride_col_attn_m + k).to(tl.float32)
            # cumulants over c
            for c in tl.static_range(0, C_CONST):
                w_c = tl.load(CELL_W_ptr + pid_m * stride_cell_w_m + c).to(tl.float32)
                emb_ptr = W_E_ptr + (base + c) * stride_we_n + d_idx
                emb = tl.load(emb_ptr, mask=d_mask, other=0.0).to(tl.float32)
                dot_dy_emb = tl.sum(dy * emb, axis=0)
                # d_col_attn[k] += w_c * sum_d(dy * emb)
                d_col_attn += tl.where(
                    tl.arange(0, K_CONST) == k, w_c * dot_dy_emb, 0.0
                )
                # d_cell_w[c] += a_k * sum_d(dy * emb)
                d_cell_w += tl.where(
                    tl.arange(0, C_CONST) == c, a_k * dot_dy_emb, 0.0
                )

    # Write outputs
    k_idx = tl.arange(0, K_CONST)
    tl.store(D_COL_ATTN_ptr + pid_m * stride_d_col_attn_m + k_idx,
             d_col_attn, mask=k_idx < K)
    c_idx = tl.arange(0, C_CONST)
    tl.store(D_CELL_W_ptr + pid_m * stride_d_cell_w_m + c_idx,
             d_cell_w, mask=c_idx < C)


# --------------------------------------------------------------------------- #
# autograd.Function
# --------------------------------------------------------------------------- #
class _FusedEngramKernel(torch.autograd.Function):
    """Pure Triton-fused gather+weighted-sum portion.

    Inputs:
        col_attn:  (B, T, K)   fp32 or bf16 (will be promoted to fp32 inside)
        cell_w:    (B, T, C)   fp32 or bf16
        topk_cols: (B, T, K)   int64 (column ids, < N_columns)
        W_E:       (N_cells, D)  cell_embed weight
        cells_per_col: int (C = cells_per_col, dim of cell_w == cells_per_col)

    Output: retrieved (B, T, D), same dtype as W_E.
    """

    @staticmethod
    def forward(ctx, col_attn: Tensor, cell_w: Tensor, topk_cols: Tensor,
                W_E: Tensor, cells_per_col: int) -> Tensor:
        assert col_attn.is_cuda and cell_w.is_cuda and topk_cols.is_cuda and W_E.is_cuda
        assert topk_cols.dtype == torch.int64

        B, T, K = col_attn.shape
        C = cell_w.shape[-1]
        D = W_E.shape[-1]
        assert cell_w.shape[:2] == (B, T), (
            f"cell_w shape {cell_w.shape} mismatches (B,T)=({B},{T})"
        )
        assert topk_cols.shape == (B, T, K), (
            f"topk_cols shape {topk_cols.shape} mismatches (B,T,K)=({B},{T},{K})"
        )
        assert C == cells_per_col, f"cell_w last dim {C} != cells_per_col {cells_per_col}"

        col_attn_f = col_attn.contiguous().view(-1, K).to(torch.float32)
        cell_w_f = cell_w.contiguous().view(-1, C).to(torch.float32)
        topk_cols_f = topk_cols.contiguous().view(-1, K)
        W_E_c = W_E.contiguous()
        M = col_attn_f.shape[0]

        out_dtype = W_E.dtype
        out = torch.empty((M, D), dtype=out_dtype, device=W_E.device)

        # Round K and C up to next power of two for tl.static_range over constexpr.
        K_CONST = _next_pow2(K)
        C_CONST = _next_pow2(C)
        # We rely on K_CONST == K and C_CONST == C for correctness inside the kernel
        # (it iterates K_CONST/C_CONST times). Enforce exact pow-of-two:
        assert K_CONST == K, (
            f"k_active must be power of two for this kernel (got {K})"
        )
        assert C_CONST == C, (
            f"cells_per_col must be power of two (got {C})"
        )

        grid = lambda meta: (M, triton.cdiv(D, meta["BLOCK_D"]))
        _engram_fwd_kernel[grid](
            col_attn_f, cell_w_f, topk_cols_f, W_E_c, out,
            M, D, K, C, cells_per_col,
            col_attn_f.stride(0),
            cell_w_f.stride(0),
            topk_cols_f.stride(0),
            W_E_c.stride(0),
            out.stride(0),
            K_CONST=K_CONST, C_CONST=C_CONST,
        )

        ctx.save_for_backward(col_attn_f, cell_w_f, topk_cols_f, W_E_c)
        ctx.shapes = (B, T, K, C, D)
        ctx.cells_per_col = cells_per_col
        ctx.K_CONST = K_CONST
        ctx.C_CONST = C_CONST
        ctx.out_dtype = out_dtype
        ctx.col_attn_in_dtype = col_attn.dtype
        ctx.cell_w_in_dtype = cell_w.dtype
        return out.view(B, T, D)

    @staticmethod
    def backward(ctx, dY: Tensor):
        col_attn_f, cell_w_f, topk_cols_f, W_E_c = ctx.saved_tensors
        B, T, K, C, D = ctx.shapes
        cells_per_col = ctx.cells_per_col
        K_CONST = ctx.K_CONST
        C_CONST = ctx.C_CONST

        M = col_attn_f.shape[0]
        dY_f = dY.contiguous().view(M, D).to(torch.float32)

        d_col_attn = torch.zeros((M, K), dtype=torch.float32, device=dY.device)
        d_cell_w = torch.zeros((M, C), dtype=torch.float32, device=dY.device)

        grid = (M,)
        _engram_bwd_kc_kernel[grid](
            dY_f, col_attn_f, cell_w_f, topk_cols_f, W_E_c,
            d_col_attn, d_cell_w,
            M, D, K, C, cells_per_col,
            dY_f.stride(0),
            col_attn_f.stride(0),
            cell_w_f.stride(0),
            topk_cols_f.stride(0),
            W_E_c.stride(0),
            d_col_attn.stride(0),
            d_cell_w.stride(0),
            K_CONST=K_CONST, C_CONST=C_CONST,
        )

        # d_W_E via PyTorch scatter_add: for each (m, k, c)
        #   cell_id = topk[m,k]*C + c
        #   d_W_E[cell_id, :] += col_attn[m,k] * cell_w[m,c] * dY[m,:]
        N_cells = W_E_c.shape[0]
        # Build scaled dY broadcasted over (M, K, C, D) — but that's huge.
        # Use a loop over (k, c) which is small (K*C is ~160) to keep memory bounded.
        d_W_E = torch.zeros_like(W_E_c, dtype=torch.float32)
        topk_view = topk_cols_f  # (M, K)
        for k in range(K):
            col_ids_k = topk_view[:, k]  # (M,)
            a_k = col_attn_f[:, k]       # (M,)
            base_k = col_ids_k * cells_per_col  # (M,)
            for c in range(C):
                w_c = cell_w_f[:, c]     # (M,)
                cell_ids = base_k + c    # (M,)
                scale = (a_k * w_c).unsqueeze(-1)  # (M, 1)
                contrib = scale * dY_f             # (M, D)
                d_W_E.index_add_(0, cell_ids, contrib)

        d_W_E = d_W_E.to(W_E_c.dtype)

        # Restore shapes / dtypes for upstream grads
        d_col_attn = d_col_attn.view(B, T, K).to(ctx.col_attn_in_dtype)
        d_cell_w = d_cell_w.view(B, T, C).to(ctx.cell_w_in_dtype)

        # topk_cols / cells_per_col are non-differentiable
        return d_col_attn, d_cell_w, None, d_W_E, None


# --------------------------------------------------------------------------- #
# High-level fused entry point (training-aware)
# --------------------------------------------------------------------------- #
def fused_engram_gather(
    col_logits: Tensor,
    ctx_tensor: Tensor,
    cell_embed_weight: Tensor,
    k_active: int,
    cells_per_col: int,
    training: bool,
) -> Tensor:
    """Fused engram retrieval: returns retrieved (B, T, D).

    Args:
        col_logits:        (B, T, n_columns) — output of col_proj(x)
        ctx_tensor:        (B, T, cells_per_col) — output of ctx_proj(x)
        cell_embed_weight: (n_columns*cells_per_col, D) — cell embedding table
        k_active:          int, number of active columns per token
        cells_per_col:     int
        training:          if True use Gumbel-Softmax(hard=True), else one_hot(argmax)

    Output dtype matches cell_embed_weight dtype (bf16 typical).
    """
    assert col_logits.is_cuda and ctx_tensor.is_cuda and cell_embed_weight.is_cuda
    B, T, N_cols = col_logits.shape
    assert ctx_tensor.shape[:2] == (B, T)
    C = ctx_tensor.shape[-1]
    assert C == cells_per_col
    D = cell_embed_weight.shape[-1]
    assert cell_embed_weight.shape[0] == N_cols * cells_per_col

    # 1) topk + softmax over columns
    topk_scores, topk_cols = col_logits.topk(k_active, dim=-1)  # (B,T,k)
    col_attn = F.softmax(topk_scores, dim=-1)                   # (B,T,k)

    # 2) cell weights from ctx
    if training:
        # gumbel_softmax(hard=True) returns one-hot of argmax (after gumbel noise)
        # with straight-through gradient. Equivalent to eval-mode one_hot when
        # gradients aren't backed through cell_weights downstream, but its
        # forward output is still a STE one-hot — perfect.
        cell_weights = F.gumbel_softmax(ctx_tensor, tau=1.0, hard=True, dim=-1)
    else:
        cell_weights = F.one_hot(ctx_tensor.argmax(dim=-1), cells_per_col).to(
            ctx_tensor.dtype
        )

    # 3) Triton-fused embedding gather + weighted sum
    retrieved = _FusedEngramKernel.apply(
        col_attn, cell_weights, topk_cols.to(torch.int64),
        cell_embed_weight, cells_per_col,
    )
    return retrieved


# --------------------------------------------------------------------------- #
# Module wrapper
# --------------------------------------------------------------------------- #
class FusedEngramGatherModule(nn.Module):
    """Drop-in replacement for the gather-retrieval portion of HTMEngram.

    forward(x) -> retrieved (B, T, D). The caller is responsible for the
    residual `x + g * out_proj(retrieved)` (gate + output projection are
    not part of the fused kernel — that's K2/K6 territory).
    """

    def __init__(self, d_model: int, n_columns: int = 4096,
                 cells_per_col: int = 8, k_active: int = 20,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.d_model = d_model
        self.n_columns = n_columns
        self.cells_per_col = cells_per_col
        self.k_active = k_active

        self.cell_embed = nn.Embedding(n_columns * cells_per_col, d_model)
        self.col_proj = nn.Linear(d_model, n_columns, bias=False)
        self.ctx_proj = nn.Linear(d_model, cells_per_col, bias=False)

        # Cast embedding table to target dtype upfront (production-only).
        self.cell_embed.weight.data = self.cell_embed.weight.data.to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        col_logits = self.col_proj(x)
        ctx = self.ctx_proj(x)
        return fused_engram_gather(
            col_logits, ctx, self.cell_embed.weight,
            self.k_active, self.cells_per_col, self.training,
        )

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, n_columns={self.n_columns}, "
                f"cells_per_col={self.cells_per_col}, k_active={self.k_active}")


def _next_pow2(x: int) -> int:
    return 1 << (max(1, x - 1)).bit_length()
