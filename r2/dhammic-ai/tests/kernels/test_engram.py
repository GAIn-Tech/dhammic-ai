"""Faithful DeepSeek Engram — Triton kernel + module parity, gradient, and
memorization tests.

We test the *kernel* (fused_engram_lookup, src/kernels/engram.py) against a
pure-PyTorch reference, then the *module* (FaithfulEngram, src/engram_layer.py)
for gradient flow and a memorization-smoke test."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engram_layer import FaithfulEngram  # noqa: E402
from src.kernels.engram import fused_engram_lookup  # noqa: E402


CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="Triton kernel requires CUDA")


# --------------------------------------------------------------------------- #
# Pure-PyTorch reference for fused_engram_lookup.
# --------------------------------------------------------------------------- #
def _eager_engram_lookup(
    hash_ids: torch.Tensor,     # (B, T, N_GROUPS) int64
    offsets: torch.Tensor,      # (N_GROUPS,)      int64
    embed_weight: torch.Tensor, # (TOTAL_ROWS, D_HEAD)
    d_head: int,
    n_bands: int,
) -> torch.Tensor:
    """Reference: gather, then for each band b sum groups g with g % n_bands == b,
    concatenate bands to (B, T, n_bands * d_head)."""
    B, T, N_GROUPS = hash_ids.shape
    n_groups_per_band = N_GROUPS // n_bands

    # row = hash + offsets (per group), then F.embedding.
    rows = hash_ids + offsets  # (B, T, N_GROUPS), int64
    embs = F.embedding(rows.view(-1), embed_weight).view(
        B, T, N_GROUPS, d_head
    )  # (B, T, N_GROUPS, D_HEAD)

    # Reshape so band index is explicit. Group g maps to band (g % n_bands).
    # Equivalent: view as (B, T, n_groups_per_band, n_bands, d_head) only when
    # the storage order satisfies g = i * n_bands + band, i.e. groups for the
    # same band are contiguous after stride n_bands. Our kernel uses
    #     g = pid_d_band + i * n_bands
    # which is the same layout. So:
    embs_b = embs.view(B, T, n_groups_per_band, n_bands, d_head)
    summed = embs_b.sum(dim=2)  # (B, T, n_bands, d_head)
    return summed.reshape(B, T, n_bands * d_head)


# --------------------------------------------------------------------------- #
# Test shapes (small, fits in ~1.4 GB VRAM budget).
# --------------------------------------------------------------------------- #
B = 2
T = 256
D_HEAD = 16
N_BANDS = 8          # = n_heads_per_ngram in module-level test
N_GROUPS_PER_BAND = 2  # = max_ngram_size - 1
N_GROUPS = N_BANDS * N_GROUPS_PER_BAND  # 16
D_VALUE = N_BANDS * D_HEAD  # 128
PER_HEAD_VOCAB = 1024  # small for test
TOTAL_ROWS = PER_HEAD_VOCAB * N_GROUPS


def _make_offsets(per_head_vocab: int, n_groups: int) -> torch.Tensor:
    return torch.arange(0, n_groups * per_head_vocab, per_head_vocab,
                        dtype=torch.int64, device="cuda")


# --------------------------------------------------------------------------- #
# test_kernel_shapes — basic shape & dtype contract.
# --------------------------------------------------------------------------- #
def test_kernel_shapes():
    torch.manual_seed(0)
    hash_ids = torch.randint(
        0, PER_HEAD_VOCAB, (B, T, N_GROUPS), dtype=torch.int64, device="cuda"
    )
    offsets = _make_offsets(PER_HEAD_VOCAB, N_GROUPS)
    W = torch.randn(TOTAL_ROWS, D_HEAD, dtype=torch.bfloat16, device="cuda") * 0.1

    out = fused_engram_lookup(hash_ids, offsets, W, D_HEAD, N_BANDS)
    assert out.shape == (B, T, D_VALUE)
    assert out.dtype == torch.bfloat16
    assert out.is_cuda


# --------------------------------------------------------------------------- #
# test_kernel_parity_fp32 — fp32 parity vs eager, tight tol (no bf16 error).
# --------------------------------------------------------------------------- #
def test_kernel_parity_fp32():
    torch.manual_seed(1)
    hash_ids = torch.randint(
        0, PER_HEAD_VOCAB, (B, T, N_GROUPS), dtype=torch.int64, device="cuda"
    )
    offsets = _make_offsets(PER_HEAD_VOCAB, N_GROUPS)
    W = torch.randn(TOTAL_ROWS, D_HEAD, dtype=torch.float32, device="cuda") * 0.1

    out_fused = fused_engram_lookup(hash_ids, offsets, W, D_HEAD, N_BANDS)
    out_ref = _eager_engram_lookup(hash_ids, offsets, W, D_HEAD, N_BANDS)
    max_err = (out_fused - out_ref).abs().max().item()
    assert max_err < 1e-5, f"fp32 parity failed: max_abs_err={max_err}"


# --------------------------------------------------------------------------- #
# test_kernel_parity_bf16 — bf16 forward, fp32 accum, tol 1e-3.
# --------------------------------------------------------------------------- #
def test_kernel_parity_bf16():
    torch.manual_seed(2)
    hash_ids = torch.randint(
        0, PER_HEAD_VOCAB, (B, T, N_GROUPS), dtype=torch.int64, device="cuda"
    )
    offsets = _make_offsets(PER_HEAD_VOCAB, N_GROUPS)
    W = torch.randn(TOTAL_ROWS, D_HEAD, dtype=torch.bfloat16, device="cuda") * 0.1

    out_fused = fused_engram_lookup(hash_ids, offsets, W, D_HEAD, N_BANDS)
    out_ref = _eager_engram_lookup(hash_ids, offsets, W, D_HEAD, N_BANDS)
    # Cast both to fp32 for comparison.
    max_err = (out_fused.float() - out_ref.float()).abs().max().item()
    assert max_err < 5e-3, f"bf16 parity failed: max_abs_err={max_err}"


# --------------------------------------------------------------------------- #
# test_kernel_backward — gradient parity against eager autograd.
# --------------------------------------------------------------------------- #
def test_kernel_backward_parity():
    torch.manual_seed(3)
    hash_ids = torch.randint(
        0, PER_HEAD_VOCAB, (B, T, N_GROUPS), dtype=torch.int64, device="cuda"
    )
    offsets = _make_offsets(PER_HEAD_VOCAB, N_GROUPS)

    W = (torch.randn(TOTAL_ROWS, D_HEAD, dtype=torch.float32, device="cuda")
         * 0.1).requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)

    # Forward / backward through fused kernel.
    out_fused = fused_engram_lookup(hash_ids, offsets, W, D_HEAD, N_BANDS)
    loss_fused = out_fused.sum()
    loss_fused.backward()
    dW_fused = W.grad.detach().clone()

    # Forward / backward through eager reference.
    out_ref = _eager_engram_lookup(hash_ids, offsets, W_ref, D_HEAD, N_BANDS)
    loss_ref = out_ref.sum()
    loss_ref.backward()
    dW_ref = W_ref.grad.detach().clone()

    max_err = (dW_fused - dW_ref).abs().max().item()
    assert max_err < 1e-3, f"backward parity failed: max_abs_err={max_err}"


# --------------------------------------------------------------------------- #
# test_kernel_determinism — same input → same retrieval (eval mode is the
# kernel — no stochastic ops in either path).
# --------------------------------------------------------------------------- #
def test_kernel_determinism():
    torch.manual_seed(4)
    hash_ids = torch.randint(
        0, PER_HEAD_VOCAB, (B, T, N_GROUPS), dtype=torch.int64, device="cuda"
    )
    offsets = _make_offsets(PER_HEAD_VOCAB, N_GROUPS)
    W = torch.randn(TOTAL_ROWS, D_HEAD, dtype=torch.bfloat16, device="cuda") * 0.1

    o1 = fused_engram_lookup(hash_ids, offsets, W, D_HEAD, N_BANDS)
    o2 = fused_engram_lookup(hash_ids, offsets, W, D_HEAD, N_BANDS)
    assert torch.equal(o1, o2)


# --------------------------------------------------------------------------- #
# test_module_forward_shapes
# --------------------------------------------------------------------------- #
MOD_D_MODEL = 128
MOD_VOCAB = 1024


def test_module_forward_shapes():
    torch.manual_seed(10)
    mod = FaithfulEngram(
        d_model=MOD_D_MODEL,
        tokenizer_vocab_size=MOD_VOCAB,
        d_value=128,
        max_ngram_size=3,
        n_heads_per_ngram=8,
        engram_vocab_size_per_ngram=[512, 512],  # small for test
        dtype=torch.bfloat16,
    ).cuda()

    hidden = torch.randn(B, T, MOD_D_MODEL, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, MOD_VOCAB, (B, T), dtype=torch.int64, device="cuda")

    out = mod(hidden, ids)
    assert out.shape == (B, T, MOD_D_MODEL)
    assert out.dtype == torch.bfloat16


# --------------------------------------------------------------------------- #
# test_module_gradient_flow — all learnable params receive non-zero grads.
# --------------------------------------------------------------------------- #
def test_module_gradient_flow():
    torch.manual_seed(11)
    mod = FaithfulEngram(
        d_model=MOD_D_MODEL,
        tokenizer_vocab_size=MOD_VOCAB,
        d_value=128,
        max_ngram_size=3,
        n_heads_per_ngram=8,
        engram_vocab_size_per_ngram=[512, 512],
        dtype=torch.float32,
    ).cuda()

    hidden = torch.randn(B, T, MOD_D_MODEL, dtype=torch.float32, device="cuda",
                         requires_grad=True)
    ids = torch.randint(0, MOD_VOCAB, (B, T), dtype=torch.int64, device="cuda")

    out = mod(hidden, ids)
    loss = (out ** 2).mean()
    loss.backward()

    # Hidden state gradient — propagates through the gate (norm_q path).
    assert hidden.grad is not None
    assert hidden.grad.abs().sum().item() > 0, "no grad to hidden state"

    # Embedding bank — should have rows that received gradient.
    eg = mod.embed.weight.grad
    assert eg is not None, "embedding bank received no gradient"
    nz_rows = (eg.abs().sum(dim=-1) > 0).sum().item()
    assert nz_rows > 0, "embedding bank: no rows received gradient"

    # value_proj — must have grad.
    assert mod.value_proj.weight.grad is not None
    assert mod.value_proj.weight.grad.abs().sum().item() > 0

    # key_proj — must have grad (flows through the gate).
    assert mod.key_proj.weight.grad is not None
    assert mod.key_proj.weight.grad.abs().sum().item() > 0


# --------------------------------------------------------------------------- #
# test_module_eval_determinism — eval mode → same input twice → same output.
# --------------------------------------------------------------------------- #
def test_module_eval_determinism():
    torch.manual_seed(12)
    mod = FaithfulEngram(
        d_model=MOD_D_MODEL,
        tokenizer_vocab_size=MOD_VOCAB,
        d_value=128,
        max_ngram_size=3,
        n_heads_per_ngram=8,
        engram_vocab_size_per_ngram=[512, 512],
        dtype=torch.bfloat16,
    ).cuda().eval()

    hidden = torch.randn(B, T, MOD_D_MODEL, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, MOD_VOCAB, (B, T), dtype=torch.int64, device="cuda")

    o1 = mod(hidden, ids)
    o2 = mod(hidden, ids)
    assert torch.equal(o1, o2), "eval mode is not deterministic"


# --------------------------------------------------------------------------- #
# test_module_memorization_smoke — fit a single (id, target) example via GD;
# retrieval converges to the target. Demonstrates differentiable memorization.
#
# Use a *very* tiny model so that one token forces large gradient on the few
# rows it hits, and the value/key/norm don't fight the embedding update.
# --------------------------------------------------------------------------- #
def test_module_memorization_smoke():
    torch.manual_seed(13)
    d = 32
    mod = FaithfulEngram(
        d_model=d,
        tokenizer_vocab_size=128,
        d_value=d,
        max_ngram_size=2,            # 2-gram only to limit groups touched
        n_heads_per_ngram=4,
        engram_vocab_size_per_ngram=[256],
        dtype=torch.float32,
    ).cuda()

    # One fixed (B,T) input and one fixed target.
    hidden = torch.randn(1, 4, d, dtype=torch.float32, device="cuda")
    ids = torch.randint(0, 128, (1, 4), dtype=torch.int64, device="cuda")
    target = torch.randn(1, 4, d, dtype=torch.float32, device="cuda")

    # Freeze hidden (we want to learn the lookup table, not chase the gate).
    opt = torch.optim.SGD(mod.parameters(), lr=0.5, momentum=0.0)

    initial_loss: float = -1.0
    final_loss: float = -1.0
    for step in range(200):
        opt.zero_grad()
        out = mod(hidden, ids)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.5, (
        f"memorization didn't converge: initial={initial_loss:.4f}, "
        f"final={final_loss:.4f}"
    )
    # And the absolute residual should be small enough to call it 'memorized'.
    assert final_loss < 0.3, f"final loss too high: {final_loss:.4f}"


# --------------------------------------------------------------------------- #
# test_kernel_compare_module_pair — module's internal lookup matches eager
# reference (sanity that hash plumbing is right).
# --------------------------------------------------------------------------- #
def test_module_lookup_matches_eager_reference():
    torch.manual_seed(14)
    mod = FaithfulEngram(
        d_model=MOD_D_MODEL,
        tokenizer_vocab_size=MOD_VOCAB,
        d_value=128,
        max_ngram_size=3,
        n_heads_per_ngram=8,
        engram_vocab_size_per_ngram=[512, 512],
        dtype=torch.float32,
    ).cuda().eval()

    ids = torch.randint(0, MOD_VOCAB, (B, T), dtype=torch.int64, device="cuda")
    hash_ids = mod._compute_hashes(ids)

    fused_retrieved = fused_engram_lookup(
        hash_ids, mod.offsets, mod.embed.weight, mod.d_head, mod.n_bands
    )
    eager_retrieved = _eager_engram_lookup(
        hash_ids, mod.offsets, mod.embed.weight, mod.d_head, mod.n_bands
    )

    max_err = (fused_retrieved - eager_retrieved).abs().max().item()
    assert max_err < 1e-5, f"module-level kernel/eager mismatch: {max_err}"
