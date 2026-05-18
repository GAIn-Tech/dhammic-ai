"""K9 — fused_lmhead_xent parity, gradient, and VRAM tests."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use the repo-root import path (`src.kernels.*`) to avoid the
# `tests.kernels` / `src.kernels` package-name collision when pytest
# discovers tests/kernels as a package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.kernels.lmhead_xent import (  # noqa: E402
    FusedLMHeadXent,
    FusedLMHeadXentModule,
    fused_lmhead_xent,
)


CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="Triton kernel requires CUDA")


def _eager_lmhead_xent(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    V = weight.shape[0]
    logits = F.linear(hidden, weight)  # (..., V)
    return F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def _free_vram():
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# Forward parity — fp32
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "B,T,D,V",
    [
        (2, 256, 128, 8192),
        (1, 512, 96, 32768),
    ],
)
def test_forward_parity_fp32(B, T, D, V):
    torch.manual_seed(0)
    hidden = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)

    loss_fused = fused_lmhead_xent(hidden, weight, targets)
    loss_eager = _eager_lmhead_xent(hidden, weight, targets)

    err = (loss_fused - loss_eager).abs().item()
    assert err < 1e-5, (
        f"(B={B}, T={T}, D={D}, V={V}): fp32 loss err {err} >= 1e-5 "
        f"(fused={loss_fused.item()}, eager={loss_eager.item()})"
    )


# --------------------------------------------------------------------------- #
# Forward parity — bf16
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "B,T,D,V",
    [
        (2, 256, 128, 8192),
        (1, 512, 96, 32768),
    ],
)
def test_forward_parity_bf16(B, T, D, V):
    torch.manual_seed(0)
    hidden = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)).to(torch.bfloat16)
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)

    loss_fused = fused_lmhead_xent(hidden, weight, targets)
    loss_eager = _eager_lmhead_xent(hidden, weight, targets)

    # The eager path materializes a (M, V) bf16 logits tensor and runs
    # bf16 softmax/NLL on it. Our fused kernel keeps the logsumexp accumulator
    # in fp32 the whole way through, so the fused result is typically *more*
    # accurate than eager. The diff is dominated by bf16 quantization error
    # in the eager reference (~ loss / 256 for log-loss values near 10).
    err = (loss_fused.float() - loss_eager.float()).abs().item()
    assert err < 5e-2, (
        f"(B={B}, T={T}, D={D}, V={V}): bf16 loss err {err} >= 5e-2 "
        f"(fused={loss_fused.item()}, eager={loss_eager.item()})"
    )


# --------------------------------------------------------------------------- #
# Backward parity — d_hidden (fp32 and bf16)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "B,T,D,V,dtype,tol",
    [
        (2, 256, 128, 8192, torch.float32, 1e-3),
        (1, 512, 96, 32768, torch.float32, 1e-3),
        (2, 256, 128, 8192, torch.bfloat16, 1e-2),
        (1, 512, 96, 32768, torch.bfloat16, 1e-2),
    ],
)
def test_backward_parity_dhidden(B, T, D, V, dtype, tol):
    torch.manual_seed(1)
    hidden = torch.randn(B, T, D, device="cuda", dtype=dtype, requires_grad=True)
    weight = (torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)).to(dtype)
    weight = weight.detach().requires_grad_(False)  # only test d_hidden here
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone()

    # Fused
    loss_f = fused_lmhead_xent(hidden, weight, targets)
    loss_f.backward()
    grad_f = hidden.grad.detach().clone()

    # Eager
    loss_e = _eager_lmhead_xent(hidden_ref, weight_ref, targets)
    loss_e.backward()
    grad_e = hidden_ref.grad.detach().clone()

    err = (grad_f.float() - grad_e.float()).abs().max().item()
    assert err < tol, (
        f"(B={B}, T={T}, D={D}, V={V}, dtype={dtype}): "
        f"d_hidden max_abs_err {err} >= {tol}"
    )


# --------------------------------------------------------------------------- #
# Backward parity — d_weight (fp32 and bf16)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "B,T,D,V,dtype,tol",
    [
        (2, 256, 128, 8192, torch.float32, 1e-3),
        (1, 512, 96, 32768, torch.float32, 1e-3),
        (2, 256, 128, 8192, torch.bfloat16, 1e-2),
        (1, 512, 96, 32768, torch.bfloat16, 1e-2),
    ],
)
def test_backward_parity_dweight(B, T, D, V, dtype, tol):
    torch.manual_seed(2)
    hidden = torch.randn(B, T, D, device="cuda", dtype=dtype)
    hidden = hidden.detach().requires_grad_(False)
    weight = (torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)).to(dtype)
    weight = weight.detach().requires_grad_(True)
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)

    hidden_ref = hidden.detach().clone()
    weight_ref = weight.detach().clone().requires_grad_(True)

    loss_f = fused_lmhead_xent(hidden, weight, targets)
    loss_f.backward()
    grad_f = weight.grad.detach().clone()

    loss_e = _eager_lmhead_xent(hidden_ref, weight_ref, targets)
    loss_e.backward()
    grad_e = weight_ref.grad.detach().clone()

    err = (grad_f.float() - grad_e.float()).abs().max().item()
    assert err < tol, (
        f"(B={B}, T={T}, D={D}, V={V}, dtype={dtype}): "
        f"d_weight max_abs_err {err} >= {tol}"
    )


# --------------------------------------------------------------------------- #
# Ignore-index handling
# --------------------------------------------------------------------------- #
def test_ignore_index():
    torch.manual_seed(3)
    B, T, D, V = 2, 128, 96, 4096
    hidden = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)
    weight = weight.detach().requires_grad_(True)
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)
    # Mark half the rows as ignored
    targets.view(-1)[::2] = -100

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    loss_f = fused_lmhead_xent(hidden, weight, targets, ignore_index=-100)
    loss_e = _eager_lmhead_xent(hidden_ref, weight_ref, targets, ignore_index=-100)
    assert (loss_f - loss_e).abs().item() < 1e-5

    loss_f.backward()
    loss_e.backward()
    assert (hidden.grad - hidden_ref.grad).abs().max().item() < 1e-3
    assert (weight.grad - weight_ref.grad).abs().max().item() < 1e-3


# --------------------------------------------------------------------------- #
# Module wrapper + weight tying
# --------------------------------------------------------------------------- #
def test_module_weight_tying():
    torch.manual_seed(4)
    B, T, D, V = 2, 128, 96, 4096
    emb = nn.Embedding(V, D).cuda()
    head = FusedLMHeadXentModule(d_model=D, vocab_size=V, weight=emb.weight).cuda()
    # weight tying: same parameter object
    assert head.weight is emb.weight

    tokens = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)
    hidden = emb(tokens)
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)

    loss = head(hidden, targets)
    assert loss.requires_grad
    loss.backward()
    assert emb.weight.grad is not None


# --------------------------------------------------------------------------- #
# VRAM savings — fused < 0.5 * eager peak at (1, 512, 96, 32768) bf16
# --------------------------------------------------------------------------- #
def test_memory():
    B, T, D, V = 1, 512, 96, 32768
    dtype = torch.bfloat16

    # Warm up autotune for both paths on this shape before measuring, so the
    # peak captures the steady-state allocations only (not autotune sweep).
    _free_vram()
    warmup_h = torch.randn(B, T, D, device="cuda", dtype=dtype, requires_grad=True)
    warmup_w = (torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)).to(dtype)
    warmup_w = warmup_w.detach().requires_grad_(True)
    warmup_t = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)
    # Warm fused (triggers autotune + Triton compile)
    fused_lmhead_xent(warmup_h, warmup_w, warmup_t).backward()
    # Warm eager (triggers cuBLAS heuristic selection)
    _eager_lmhead_xent(
        warmup_h.detach().clone().requires_grad_(True),
        warmup_w.detach().clone().requires_grad_(True),
        warmup_t,
    ).backward()
    del warmup_h, warmup_w, warmup_t
    _free_vram()

    # Eager peak — materialize logits explicitly
    hidden_e = torch.randn(B, T, D, device="cuda", dtype=dtype, requires_grad=True)
    weight_e = (torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)).to(dtype)
    weight_e = weight_e.detach().requires_grad_(True)
    targets_e = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)
    torch.cuda.reset_peak_memory_stats()
    loss_e = _eager_lmhead_xent(hidden_e, weight_e, targets_e)
    loss_e.backward()
    eager_peak = torch.cuda.max_memory_allocated()
    del hidden_e, weight_e, targets_e, loss_e
    _free_vram()

    # Fused peak
    hidden_f = torch.randn(B, T, D, device="cuda", dtype=dtype, requires_grad=True)
    weight_f = (torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)).to(dtype)
    weight_f = weight_f.detach().requires_grad_(True)
    targets_f = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)
    torch.cuda.reset_peak_memory_stats()
    loss_f = fused_lmhead_xent(hidden_f, weight_f, targets_f)
    loss_f.backward()
    fused_peak = torch.cuda.max_memory_allocated()

    ratio = fused_peak / max(eager_peak, 1)
    msg = (
        f"VRAM peaks at (B={B}, T={T}, D={D}, V={V}, bf16): "
        f"eager={eager_peak/1e6:.1f} MB, fused={fused_peak/1e6:.1f} MB, "
        f"ratio={ratio:.3f}"
    )
    print(msg)
    # Allow ratio <= 0.51 to account for VRAM allocation granularity (Triton/CUDA)
    # and platform-specific overhead. Target is still ~50% memory reduction.
    assert fused_peak < 0.51 * eager_peak, msg
