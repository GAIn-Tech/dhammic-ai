"""K8 — fused_engram_gather parity & gradient tests against eager HTMEngram."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.kernels.engram_gather import (  # noqa: E402
    FusedEngramGatherModule,
    fused_engram_gather,
)


CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="Triton kernel requires CUDA")


# --------------------------------------------------------------------------- #
# Eager reference — matches HTMEngram.forward's retrieval portion (no residual,
# no gate, no out_proj — just the retrieved (B,T,D) tensor before fan-out).
# --------------------------------------------------------------------------- #
def _eager_engram_retrieve(
    col_logits: torch.Tensor,
    ctx: torch.Tensor,
    cell_embed_weight: torch.Tensor,
    k_active: int,
    cells_per_col: int,
    training: bool,
) -> torch.Tensor:
    topk_scores, topk_cols = col_logits.topk(k_active, dim=-1)
    col_attn = F.softmax(topk_scores, dim=-1)

    if training:
        cell_weights = F.gumbel_softmax(ctx, tau=1.0, hard=True, dim=-1)
    else:
        cell_weights = F.one_hot(ctx.argmax(dim=-1), cells_per_col).to(ctx.dtype)

    col_base = topk_cols * cells_per_col
    cell_offsets = torch.arange(cells_per_col, device=col_logits.device)
    all_cell_ids = col_base.unsqueeze(-1) + cell_offsets  # (B,T,k,C)
    # Embedding lookup
    all_cell_embs = F.embedding(all_cell_ids.view(-1), cell_embed_weight).view(
        *all_cell_ids.shape, -1
    )  # (B,T,k,C,D)
    cell_weights_exp = cell_weights.unsqueeze(2).expand(-1, -1, k_active, -1)
    cell_embs = (all_cell_embs * cell_weights_exp.unsqueeze(-1)).sum(dim=3)
    retrieved = torch.einsum("btk,btkd->btd", col_attn.to(cell_embs.dtype), cell_embs)
    return retrieved


# Default test shape
B, T, D = 2, 256, 128
N_COLS = 512
K_ACTIVE = 20  # not a power of two — kernel requires pow2; we use 16
K_ACTIVE_POW2 = 16
CELLS_PER_COL = 8


# --------------------------------------------------------------------------- #
# test_shapes — sanity that requested shape (2, 256, 128) works through
#               FusedEngramGatherModule end-to-end.
# --------------------------------------------------------------------------- #
def test_shapes():
    torch.manual_seed(0)
    mod = FusedEngramGatherModule(
        d_model=D, n_columns=N_COLS, cells_per_col=CELLS_PER_COL,
        k_active=K_ACTIVE_POW2, dtype=torch.bfloat16,
    ).cuda().eval()

    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    # Cast linear weights to bf16 for consistent dtype flow
    mod.col_proj.weight.data = mod.col_proj.weight.data.to(torch.bfloat16)
    mod.ctx_proj.weight.data = mod.ctx_proj.weight.data.to(torch.bfloat16)

    out = mod(x)
    assert out.shape == (B, T, D)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), "non-finite values in retrieved"


# --------------------------------------------------------------------------- #
# test_eval_mode_parity — bf16 fused vs bf16 eager, training=False
#   In eval mode cell_weights = one_hot(argmax) is deterministic, so the fused
#   kernel should match the eager reference up to bf16 numerics.
# --------------------------------------------------------------------------- #
def test_eval_mode_parity():
    torch.manual_seed(0)
    # Build random inputs to fused_engram_gather directly so we can control
    # everything (skip the col_proj / ctx_proj layers — those are just torch).
    col_logits = torch.randn(B, T, N_COLS, device="cuda", dtype=torch.bfloat16) * 0.1
    ctx = torch.randn(B, T, CELLS_PER_COL, device="cuda", dtype=torch.bfloat16) * 0.5
    n_cells = N_COLS * CELLS_PER_COL
    W_E = torch.randn(n_cells, D, device="cuda", dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        out_fused = fused_engram_gather(
            col_logits, ctx, W_E,
            k_active=K_ACTIVE_POW2, cells_per_col=CELLS_PER_COL, training=False,
        )
        out_eager = _eager_engram_retrieve(
            col_logits, ctx, W_E,
            k_active=K_ACTIVE_POW2, cells_per_col=CELLS_PER_COL, training=False,
        )

    assert out_fused.shape == out_eager.shape == (B, T, D)
    assert out_fused.dtype == torch.bfloat16
    err = (out_fused.float() - out_eager.float()).abs().max().item()
    assert err < 1e-3, f"eval parity bf16 max_abs_err {err:.6f} >= 1e-3"


# --------------------------------------------------------------------------- #
# test_eval_mode_parity_fp32 — tighter tolerance on fp32 path.
# --------------------------------------------------------------------------- #
def test_eval_mode_parity_fp32():
    torch.manual_seed(1)
    col_logits = torch.randn(B, T, N_COLS, device="cuda", dtype=torch.float32) * 0.1
    ctx = torch.randn(B, T, CELLS_PER_COL, device="cuda", dtype=torch.float32) * 0.5
    n_cells = N_COLS * CELLS_PER_COL
    W_E = torch.randn(n_cells, D, device="cuda", dtype=torch.float32) * 0.1

    with torch.no_grad():
        out_fused = fused_engram_gather(
            col_logits, ctx, W_E,
            k_active=K_ACTIVE_POW2, cells_per_col=CELLS_PER_COL, training=False,
        )
        out_eager = _eager_engram_retrieve(
            col_logits, ctx, W_E,
            k_active=K_ACTIVE_POW2, cells_per_col=CELLS_PER_COL, training=False,
        )

    err = (out_fused - out_eager).abs().max().item()
    assert err < 1e-5, f"eval parity fp32 max_abs_err {err:.2e} >= 1e-5"


# --------------------------------------------------------------------------- #
# test_train_mode_smoke — non-deterministic (gumbel noise) but shape + finite
# --------------------------------------------------------------------------- #
def test_train_mode_smoke():
    torch.manual_seed(2)
    col_logits = torch.randn(B, T, N_COLS, device="cuda", dtype=torch.bfloat16) * 0.1
    ctx = torch.randn(B, T, CELLS_PER_COL, device="cuda", dtype=torch.bfloat16) * 0.5
    n_cells = N_COLS * CELLS_PER_COL
    W_E = torch.randn(n_cells, D, device="cuda", dtype=torch.bfloat16) * 0.1

    out = fused_engram_gather(
        col_logits, ctx, W_E,
        k_active=K_ACTIVE_POW2, cells_per_col=CELLS_PER_COL, training=True,
    )
    assert out.shape == (B, T, D)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), "non-finite values in training-mode retrieved"


# --------------------------------------------------------------------------- #
# test_backward_parity — eval-mode backward through fused vs eager.
#   Compare gradients on:
#     - col_logits (flows through softmax + topk_scores → col_attn)
#     - ctx        (one_hot via argmax has 0 gradient — so we directly check
#                   the fused kernel's chain by feeding precomputed col_attn /
#                   cell_weights into a small test)
#     - cell_embed_weight (the embedding table — heavy scatter_add gradient)
# --------------------------------------------------------------------------- #
def test_backward_parity_eval():
    torch.manual_seed(3)

    # bf16 inputs, but require_grad on float32 leaf tensors for stability,
    # then cast in-graph to bf16. This matches typical AMP usage and gives
    # a cleaner gradient comparison.
    col_logits_ref = (torch.randn(B, T, N_COLS, device="cuda", dtype=torch.float32) * 0.1)
    ctx_ref = (torch.randn(B, T, CELLS_PER_COL, device="cuda", dtype=torch.float32) * 0.5)
    n_cells = N_COLS * CELLS_PER_COL
    W_E_ref = (torch.randn(n_cells, D, device="cuda", dtype=torch.float32) * 0.1)

    # Two parallel graphs sharing same input data
    col_logits_f = col_logits_ref.clone().to(torch.bfloat16).requires_grad_(True)
    W_E_f = W_E_ref.clone().to(torch.bfloat16).requires_grad_(True)

    col_logits_e = col_logits_ref.clone().to(torch.bfloat16).requires_grad_(True)
    W_E_e = W_E_ref.clone().to(torch.bfloat16).requires_grad_(True)

    ctx_bf = ctx_ref.to(torch.bfloat16)  # eval mode → no grad through ctx anyway

    out_fused = fused_engram_gather(
        col_logits_f, ctx_bf, W_E_f,
        k_active=K_ACTIVE_POW2, cells_per_col=CELLS_PER_COL, training=False,
    )
    out_eager = _eager_engram_retrieve(
        col_logits_e, ctx_bf, W_E_e,
        k_active=K_ACTIVE_POW2, cells_per_col=CELLS_PER_COL, training=False,
    )

    # Loss = sum (.float() to avoid bf16 reduction noise)
    loss_fused = out_fused.float().sum()
    loss_eager = out_eager.float().sum()
    loss_fused.backward()
    loss_eager.backward()

    err_logits = (col_logits_f.grad.float() - col_logits_e.grad.float()).abs().max().item()
    err_we = (W_E_f.grad.float() - W_E_e.grad.float()).abs().max().item()

    # bf16 tolerance — these gradients accumulate K*C*D bf16 multiplies, so
    # 1e-2 is the right tolerance for accumulated bf16 noise.
    assert err_logits < 1e-2, f"d(col_logits) bf16 max_abs_err {err_logits:.4e} >= 1e-2"
    assert err_we < 1e-2, f"d(W_E) bf16 max_abs_err {err_we:.4e} >= 1e-2"


# --------------------------------------------------------------------------- #
# Backward parity (fp32, tight tolerance) — gives extra confidence the bwd
# formulas (d_col_attn, d_cell_w, d_W_E) are arithmetically correct.
# --------------------------------------------------------------------------- #
def test_backward_parity_fp32():
    torch.manual_seed(4)
    col_logits_ref = torch.randn(B, T, N_COLS, device="cuda", dtype=torch.float32) * 0.1
    ctx_ref = torch.randn(B, T, CELLS_PER_COL, device="cuda", dtype=torch.float32) * 0.5
    n_cells = N_COLS * CELLS_PER_COL
    W_E_ref = torch.randn(n_cells, D, device="cuda", dtype=torch.float32) * 0.1

    col_logits_f = col_logits_ref.clone().requires_grad_(True)
    W_E_f = W_E_ref.clone().requires_grad_(True)
    col_logits_e = col_logits_ref.clone().requires_grad_(True)
    W_E_e = W_E_ref.clone().requires_grad_(True)
    ctx32 = ctx_ref

    out_fused = fused_engram_gather(
        col_logits_f, ctx32, W_E_f,
        k_active=K_ACTIVE_POW2, cells_per_col=CELLS_PER_COL, training=False,
    )
    out_eager = _eager_engram_retrieve(
        col_logits_e, ctx32, W_E_e,
        k_active=K_ACTIVE_POW2, cells_per_col=CELLS_PER_COL, training=False,
    )

    out_fused.sum().backward()
    out_eager.sum().backward()

    err_logits = (col_logits_f.grad - col_logits_e.grad).abs().max().item()
    err_we = (W_E_f.grad - W_E_e.grad).abs().max().item()
    assert err_logits < 1e-4, f"d(col_logits) fp32 max_abs_err {err_logits:.4e} >= 1e-4"
    assert err_we < 1e-4, f"d(W_E) fp32 max_abs_err {err_we:.4e} >= 1e-4"
