"""Tests for ``src.runtime.chunked_offload.ChunkedOffloadRunner``.

Three properties are checked:
    1. Peak GPU memory stays roughly constant across a 16x seq_len sweep
       (the streaming pipeline keeps VRAM independent of ``seq_len``).
    2. The streamed output matches a non-chunked eager reference.
    3. The streamed gradient (w.r.t. input + params) matches the eager
       reference.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn

# Load src/runtime/chunked_offload.py directly to avoid the test-time
# name collision between ``tests/kernels`` and ``src/kernels`` (both ship
# an ``__init__.py``).  Mirrors the loader pattern used in test_rmsnorm.py.
_CO_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "runtime",
        "chunked_offload.py",
    )
)
_spec = importlib.util.spec_from_file_location("dhammic_runtime_chunked_offload", _CO_PATH)
_co_mod = importlib.util.module_from_spec(_spec)
sys.modules["dhammic_runtime_chunked_offload"] = _co_mod
_spec.loader.exec_module(_co_mod)  # type: ignore[union-attr]
ChunkedOffloadRunner = _co_mod.ChunkedOffloadRunner

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="ChunkedOffloadRunner requires CUDA")


# --------------------------------------------------------------------------- #
# Tiny test module — keeps the channel dim so chunking is trivially safe.
# --------------------------------------------------------------------------- #
class _TestMLP(nn.Module):
    def __init__(self, d: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _reset_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# 1. Peak VRAM stays flat across the seq_len sweep.
# --------------------------------------------------------------------------- #
def test_vram_flat() -> None:
    """Peak GPU memory at seq_len=4096 should be ≤ 1.5× peak at seq_len=256."""
    torch.manual_seed(0)
    B, D, hidden = 2, 64, 128
    chunk_size = 256

    module = _TestMLP(D, hidden).cuda().float()
    runner = ChunkedOffloadRunner()

    peaks: dict[int, int] = {}
    seq_lens = [256, 512, 1024, 2048, 4096]
    for T in seq_lens:
        _reset_peak()
        x = torch.randn(B, T, D, dtype=torch.float32, pin_memory=True)
        y = runner.forward(module, x, chunk_size)
        dy = torch.empty_like(y).copy_(torch.randn_like(y))
        _ = runner.backward(dy)
        torch.cuda.synchronize()
        peaks[T] = int(torch.cuda.max_memory_allocated())

    # The peaks must be flat — exact equality is too brittle (Triton/cuBLAS
    # caches), so we use a multiplicative bound from the plan: ≤ 1.5× the
    # smallest seq_len peak.
    base = peaks[seq_lens[0]]
    largest = peaks[seq_lens[-1]]
    ratio = largest / max(base, 1)
    assert ratio < 1.5, (
        "peak GPU memory grew with seq_len:\n"
        + "\n".join(
            f"  T={T:>5d}  peak={peaks[T] / (1024 * 1024):.2f} MiB"
            for T in seq_lens
        )
        + f"\nratio(4096/256) = {ratio:.3f} (must be < 1.5)"
    )

    # Expose the peaks for the final report.
    print("[test_vram_flat] peak VRAM per seq_len:")
    for T in seq_lens:
        print(f"  T={T:>5d}  peak={peaks[T] / (1024 * 1024):.2f} MiB")
    print(f"  ratio(4096/256) = {ratio:.3f}")


# --------------------------------------------------------------------------- #
# 2. Output correctness.
# --------------------------------------------------------------------------- #
def test_correctness() -> None:
    torch.manual_seed(1)
    B, T, D = 2, 1024, 64
    chunk_size = 256

    module = _TestMLP(D, 128).cuda().float()
    module_ref = _TestMLP(D, 128).cuda().float()
    module_ref.load_state_dict(module.state_dict())

    x = torch.randn(B, T, D, dtype=torch.float32, pin_memory=True)
    x_gpu = x.detach().to("cuda")

    runner = ChunkedOffloadRunner()
    y = runner.forward(module, x, chunk_size).detach()
    with torch.no_grad():
        y_ref = module_ref(x_gpu).cpu()

    err = (y - y_ref).abs().max().item()
    assert err < 1e-5, f"fp32 output mismatch: max_abs_err {err} >= 1e-5"


# --------------------------------------------------------------------------- #
# 3. Gradient correctness.
# --------------------------------------------------------------------------- #
def test_grad_correctness() -> None:
    torch.manual_seed(2)
    B, T, D = 2, 1024, 64
    chunk_size = 256

    module = _TestMLP(D, 128).cuda().float()
    module_ref = _TestMLP(D, 128).cuda().float()
    module_ref.load_state_dict(module.state_dict())

    x = torch.randn(B, T, D, dtype=torch.float32, pin_memory=True)

    # Streamed
    runner = ChunkedOffloadRunner()
    y = runner.forward(module, x, chunk_size)
    dy = torch.randn(y.shape, dtype=y.dtype, pin_memory=True)
    grad_x = runner.backward(dy)

    # Reference (non-chunked).
    x_ref = x.detach().clone().to("cuda").requires_grad_(True)
    y_ref = module_ref(x_ref)
    y_ref.backward(dy.to("cuda"))
    grad_x_ref = x_ref.grad.detach().cpu()

    err_x = (grad_x - grad_x_ref).abs().max().item()
    assert err_x < 1e-3, f"dx mismatch: max_abs_err {err_x} >= 1e-3"

    # Compare parameter grads.
    for (name, p), (name_r, p_r) in zip(
        module.named_parameters(), module_ref.named_parameters()
    ):
        assert p.grad is not None, f"streamed grad missing for {name}"
        assert p_r.grad is not None, f"ref grad missing for {name_r}"
        diff = (p.grad - p_r.grad).abs().max().item()
        assert diff < 1e-3, (
            f"param grad mismatch for {name}: max_abs_err {diff} >= 1e-3"
        )
