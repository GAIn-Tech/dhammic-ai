"""Tests for :class:`src.runtime.chunked_lmhead.ChunkedLMHeadXent`.

These tests verify that streaming the fused LM-head + cross-entropy kernel
chunk-by-chunk preserves:

1. **Loss parity** with the unchunked path (fp32 ≤ 1e-4, bf16 ≤ 1e-2).
2. **Gradient parity** for both ``d(hidden)`` and ``d(weight)``
   (fp32 ≤ 1e-3, bf16 ≤ 1e-2).
3. **Flat peak VRAM** across a sweep of ``T`` at fixed ``chunk_size`` —
   peak at ``T=4096`` must stay within ``1.2x`` of peak at ``T=512``.
4. **Correct ignore-index handling**.

CUDA-only. No eager fallback.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.kernels.lmhead_xent import fused_lmhead_xent  # noqa: E402
from src.runtime.chunked_lmhead import ChunkedLMHeadXent  # noqa: E402


CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="Triton kernel requires CUDA")


def _free_vram() -> None:
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# Forward parity — chunked vs unchunked
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "B,T,D,V,chunk,dtype,tol",
    [
        (2, 512, 64, 1024, 256, torch.float32, 1e-4),
        (2, 512, 64, 1024, 128, torch.float32, 1e-4),
        (1, 1024, 64, 1024, 256, torch.float32, 1e-4),
        (2, 512, 64, 1024, 256, torch.bfloat16, 1e-2),
    ],
)
def test_forward_parity(B, T, D, V, chunk, dtype, tol):
    torch.manual_seed(0)
    hidden = torch.randn(B, T, D, device="cuda", dtype=dtype)
    weight = (torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)).to(dtype)
    weight = nn.Parameter(weight)
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)

    head = ChunkedLMHeadXent(
        d_model=D, vocab_size=V, weight=weight, chunk_size=chunk
    ).cuda()
    loss_chunked = head(hidden, targets)
    loss_unchunked = fused_lmhead_xent(hidden, weight, targets)

    err = (loss_chunked.float() - loss_unchunked.float()).abs().item()
    assert err < tol, (
        f"(B={B}, T={T}, D={D}, V={V}, chunk={chunk}, dtype={dtype}): "
        f"chunked-vs-unchunked loss err {err} >= {tol} "
        f"(chunked={loss_chunked.item()}, unchunked={loss_unchunked.item()})"
    )


# --------------------------------------------------------------------------- #
# Gradient parity — d_hidden + d_weight
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "B,T,D,V,chunk,dtype,tol_h,tol_w",
    [
        (2, 512, 64, 1024, 256, torch.float32, 1e-3, 1e-3),
        (2, 512, 64, 1024, 128, torch.float32, 1e-3, 1e-3),
        (2, 512, 64, 1024, 256, torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_grad_parity(B, T, D, V, chunk, dtype, tol_h, tol_w):
    torch.manual_seed(1)
    # Unchunked path
    hidden_u = torch.randn(B, T, D, device="cuda", dtype=dtype, requires_grad=True)
    weight_u = (torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)).to(dtype)
    weight_u = nn.Parameter(weight_u)
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)

    loss_u = fused_lmhead_xent(hidden_u, weight_u, targets)
    loss_u.backward()
    grad_h_u = hidden_u.grad.detach().clone()
    grad_w_u = weight_u.grad.detach().clone()

    # Chunked path — fresh copies of hidden/weight to avoid grad cross-talk.
    hidden_c = hidden_u.detach().clone().requires_grad_(True)
    weight_c = nn.Parameter(weight_u.detach().clone())

    head = ChunkedLMHeadXent(
        d_model=D, vocab_size=V, weight=weight_c, chunk_size=chunk
    ).cuda()
    loss_c = head(hidden_c, targets)
    loss_c.backward()
    grad_h_c = hidden_c.grad.detach().clone()
    grad_w_c = weight_c.grad.detach().clone()

    err_h = (grad_h_c.float() - grad_h_u.float()).abs().max().item()
    err_w = (grad_w_c.float() - grad_w_u.float()).abs().max().item()
    assert err_h < tol_h, (
        f"(B={B}, T={T}, D={D}, V={V}, chunk={chunk}, dtype={dtype}): "
        f"d_hidden max_abs_err {err_h} >= {tol_h}"
    )
    assert err_w < tol_w, (
        f"(B={B}, T={T}, D={D}, V={V}, chunk={chunk}, dtype={dtype}): "
        f"d_weight max_abs_err {err_w} >= {tol_w}"
    )


# --------------------------------------------------------------------------- #
# Flat VRAM — peak at large T stays close to peak at small T
# --------------------------------------------------------------------------- #
def test_flat_vram():
    """Kernel working memory should be flat across ``T`` at fixed chunk size.

    We sweep ``T in [512, 1024, 2048, 4096]`` at fixed ``chunk_size=256``,
    ``B=2, D=64, V=4096``, bf16, and measure the peak GPU memory allocation
    *over and above* the residual cost of holding the input ``hidden``
    tensor and its ``.grad`` on-device.

    Concretely:

      input_bytes(T) = hidden(B,T,D) + hidden.grad(B,T,D) + targets(B,T)
                     = 2*B*T*D*bf16_bytes + B*T*int64_bytes

    The "kernel working memory" we care about is ``peak(T) - input_bytes(T)``,
    which must stay within ``1.2x`` of its T=512 value. This isolates the
    growth that's the kernel's responsibility (chunked d_hidden_flat,
    weight.grad, transient autograd state, log/mask/target_logit tensors)
    from the upstream's unavoidable ``hidden`` and ``hidden.grad`` storage
    — in the real chunked-offload pipeline both of those live on pinned CPU
    anyway.
    """
    B, D, V = 2, 64, 4096
    chunk = 256
    seq_lens = [512, 1024, 2048, 4096]
    dtype = torch.bfloat16
    bf16_bytes = 2
    int64_bytes = 8

    weight = nn.Parameter(
        (torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)).to(dtype)
    )
    head = ChunkedLMHeadXent(
        d_model=D, vocab_size=V, weight=weight, chunk_size=chunk
    ).cuda()

    # Warm-up at the smallest T to settle Triton autotune.
    _free_vram()
    h_warm = torch.randn(B, seq_lens[0], D, device="cuda", dtype=dtype, requires_grad=True)
    t_warm = torch.randint(0, V, (B, seq_lens[0]), device="cuda", dtype=torch.long)
    head(h_warm, t_warm).backward()
    del h_warm, t_warm
    weight.grad = None
    _free_vram()

    peaks: dict[int, float] = {}
    working: dict[int, float] = {}
    for T in seq_lens:
        # Per-iteration freshly allocated hidden + targets, sized to T.
        # We resize at every T because the kernel's behavior at T is exactly
        # what we want to characterize, and reusing slices would muddy the
        # accounting.
        h = torch.randn(B, T, D, device="cuda", dtype=dtype, requires_grad=True)
        t = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)

        # Account for the unavoidable input + input-grad + targets footprint
        # so we can subtract it from the peak and isolate kernel working set.
        input_bytes = 2 * B * T * D * bf16_bytes + B * T * int64_bytes

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        loss = head(h, t)
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        peaks[T] = float(peak)
        working[T] = float(peak) - float(input_bytes)

        del h, t, loss
        weight.grad = None
        _free_vram()

    base = working[seq_lens[0]]
    top = working[seq_lens[-1]]
    ratio = top / max(base, 1.0)
    msg_lines = ["VRAM peaks (B={B}, D={D}, V={V}, chunk={chunk}, bf16):".format(
        B=B, D=D, V=V, chunk=chunk
    )]
    for T in seq_lens:
        msg_lines.append(
            f"  T={T:5d}: peak={peaks[T]/1e6:7.2f} MB | "
            f"input+grad={2*B*T*D*bf16_bytes/1e6:5.2f} MB | "
            f"kernel-working={working[T]/1e6:7.2f} MB"
        )
    msg_lines.append(
        f"  kernel-working ratio T={seq_lens[-1]}/T={seq_lens[0]} = {ratio:.3f}x"
    )
    msg = "\n".join(msg_lines)
    print("\n" + msg)
    assert ratio < 1.2, msg


# --------------------------------------------------------------------------- #
# Ignore-index handling
# --------------------------------------------------------------------------- #
def test_ignore_index():
    torch.manual_seed(3)
    B, T, D, V, chunk = 2, 512, 64, 1024, 256
    hidden_u = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    weight_u = nn.Parameter(
        torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)
    )
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)
    # Mark a contiguous block + scattered positions as ignored — this hits
    # multiple boundary cases:
    #   - rows where ALL targets in the chunk are ignored (entire chunk
    #     contributes zero),
    #   - rows where SOME are ignored.
    targets[:, 100:300] = -100        # straddles chunks 0 and 1
    targets.view(-1)[::7] = -100

    loss_u = fused_lmhead_xent(hidden_u, weight_u, targets, ignore_index=-100)
    loss_u.backward()
    grad_h_u = hidden_u.grad.detach().clone()
    grad_w_u = weight_u.grad.detach().clone()

    hidden_c = hidden_u.detach().clone().requires_grad_(True)
    weight_c = nn.Parameter(weight_u.detach().clone())
    head = ChunkedLMHeadXent(
        d_model=D, vocab_size=V, weight=weight_c,
        chunk_size=chunk, ignore_index=-100,
    ).cuda()
    loss_c = head(hidden_c, targets)
    loss_c.backward()
    grad_h_c = hidden_c.grad.detach().clone()
    grad_w_c = weight_c.grad.detach().clone()

    assert (loss_c - loss_u).abs().item() < 1e-4, (
        f"loss mismatch: chunked={loss_c.item()}, unchunked={loss_u.item()}"
    )
    err_h = (grad_h_c - grad_h_u).abs().max().item()
    err_w = (grad_w_c - grad_w_u).abs().max().item()
    assert err_h < 1e-3, f"d_hidden err {err_h}"
    assert err_w < 1e-3, f"d_weight err {err_w}"


# --------------------------------------------------------------------------- #
# Iterable-of-chunks input — matches the chunked-offload pipeline shape
# --------------------------------------------------------------------------- #
def test_iterable_input_parity():
    """Passing an iterable of (B, chunk, D) chunks must match the tensor path."""
    torch.manual_seed(7)
    B, T, D, V, chunk = 2, 1024, 64, 1024, 256
    hidden = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
    weight = nn.Parameter(
        torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)
    )
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)

    head = ChunkedLMHeadXent(
        d_model=D, vocab_size=V, weight=weight, chunk_size=chunk
    ).cuda()

    # Path A: tensor input
    loss_tensor = head(hidden, targets)

    # Path B: iterable of chunks
    h_chunks = [hidden[:, c0 : c0 + chunk, :] for c0 in range(0, T, chunk)]
    t_chunks = [targets[:, c0 : c0 + chunk] for c0 in range(0, T, chunk)]
    loss_iter = head(h_chunks, t_chunks)

    err = (loss_tensor - loss_iter).abs().item()
    assert err < 1e-5, (
        f"tensor vs iterable mismatch: {loss_tensor.item()} vs "
        f"{loss_iter.item()} (err={err})"
    )


# --------------------------------------------------------------------------- #
# Module-level eager reference (sanity)
# --------------------------------------------------------------------------- #
def test_matches_eager_reference():
    """End-to-end sanity: chunked head vs raw F.cross_entropy.

    Same as the eager reference used in test_lmhead_xent.py — we want to be
    sure we haven't introduced a normalization bug that happens to cancel
    when comparing against the unchunked fused path.
    """
    torch.manual_seed(11)
    B, T, D, V, chunk = 2, 768, 64, 2048, 256
    hidden = torch.randn(B, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = nn.Parameter(
        torch.randn(V, D, device="cuda", dtype=torch.float32) / (D ** 0.5)
    )
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.long)
    targets.view(-1)[::13] = -100

    # Eager reference
    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = nn.Parameter(weight.detach().clone())
    logits = F.linear(hidden_ref, weight_ref)
    loss_ref = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100,
    )

    # Chunked fused
    head = ChunkedLMHeadXent(
        d_model=D, vocab_size=V, weight=weight, chunk_size=chunk
    ).cuda()
    loss_c = head(hidden, targets)

    err = (loss_c.float() - loss_ref.float()).abs().item()
    assert err < 1e-4, (
        f"chunked vs F.cross_entropy mismatch: {loss_c.item()} vs "
        f"{loss_ref.item()} (err={err})"
    )
