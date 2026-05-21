"""Tests for ``src.runtime.graph_cache.GraphCache``.

Three properties are checked:
    1. Capture-then-replay matches eager for many sequential calls.
    2. Changing the input shape transparently triggers re-capture.
    3. Post-capture replay is ≥ 2× faster than the eager equivalent.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

import pytest
import torch
import torch.nn as nn

_GC_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "runtime",
        "graph_cache.py",
    )
)
_spec = importlib.util.spec_from_file_location("dhammic_runtime_graph_cache", _GC_PATH)
_gc_mod = importlib.util.module_from_spec(_spec)
sys.modules["dhammic_runtime_graph_cache"] = _gc_mod
_spec.loader.exec_module(_gc_mod)  # type: ignore[union-attr]
GraphCache = _gc_mod.GraphCache

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="CUDA graphs require CUDA")


# --------------------------------------------------------------------------- #
# Simple test module — ensures captured kernel sequence is non-trivial.
# --------------------------------------------------------------------------- #
class _TinyBlock(nn.Module):
    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# --------------------------------------------------------------------------- #
# 1. Capture, replay 10×, compare to eager.
# --------------------------------------------------------------------------- #
def test_capture_then_replay() -> None:
    torch.manual_seed(0)
    B, T, D, H = 2, 256, 64, 128
    module = _TinyBlock(D, H).cuda().eval()
    cache = GraphCache(warmup_steps=3)

    key = ("tinyblock", B, T, D, torch.float32)
    eager_results = []
    cached_results = []
    for _ in range(10):
        x = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            eager_results.append(module(x).clone())
            cached_results.append(cache.run(module, x, key=key))

    for c, e in zip(cached_results, eager_results):
        err = (c - e).abs().max().item()
        assert err < 1e-5, f"cached vs eager mismatch: max_abs_err {err}"

    # We should have captured exactly once and replayed 10 times.
    assert cache.captures == 1, f"unexpected capture count: {cache.captures}"
    assert cache.replays == 10, f"unexpected replay count: {cache.replays}"


# --------------------------------------------------------------------------- #
# 2. Recapture on shape change.
# --------------------------------------------------------------------------- #
def test_recapture_on_shape_change() -> None:
    torch.manual_seed(1)
    D, H = 64, 128
    module = _TinyBlock(D, H).cuda().eval()
    cache = GraphCache(warmup_steps=2)

    # Phase 1: batch=2
    x1 = torch.randn(2, 128, D, device="cuda", dtype=torch.float32)
    _ = cache.run(module, x1, key=("blk", 128, torch.float32))

    # Phase 2: batch=4 — should retrigger capture.
    x2 = torch.randn(4, 128, D, device="cuda", dtype=torch.float32)
    out2 = cache.run(module, x2, key=("blk", 128, torch.float32))
    with torch.no_grad():
        out2_ref = module(x2)
    err = (out2 - out2_ref).abs().max().item()
    assert err < 1e-5, f"post-recapture mismatch: max_abs_err {err}"

    # Captures should have incremented to 2.
    assert cache.captures == 2, (
        f"expected 2 captures (one per shape) but got {cache.captures}"
    )


# --------------------------------------------------------------------------- #
# 3. Replay speedup vs eager.
# --------------------------------------------------------------------------- #
def test_speedup() -> None:
    """Captured replay should be at least 2× faster than the eager call."""
    torch.manual_seed(2)
    # Use a *tiny* model so that CPU launch overhead dominates — exactly the
    # regime where CUDA graphs help.  Many small ops amplify the difference.
    B, T, D, H = 1, 32, 16, 32

    class _Many(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_TinyBlock(D, H) for _ in range(8)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for ly in self.layers:
                x = ly(x) + x
            return x

    module = _Many().cuda().eval()
    cache = GraphCache(warmup_steps=3)
    key = ("many", B, T, D)

    x = torch.randn(B, T, D, device="cuda", dtype=torch.float32)

    # Prime cuBLAS / Triton for the eager path so the baseline is fair.
    for _ in range(20):
        with torch.no_grad():
            _ = module(x)
    torch.cuda.synchronize()

    # Prime the cache.
    _ = cache.run(module, x, key=key)
    torch.cuda.synchronize()

    N = 200
    # Eager timing.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.no_grad():
            _ = module(x)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) * 1000.0 / N

    # Cached replay timing.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        _ = cache.run(module, x, key=key)
    torch.cuda.synchronize()
    cached_ms = (time.perf_counter() - t0) * 1000.0 / N

    speedup = eager_ms / max(cached_ms, 1e-9)
    print(
        f"[test_speedup] eager={eager_ms:.3f}ms/call  "
        f"cached={cached_ms:.3f}ms/call  speedup={speedup:.2f}x"
    )
    assert speedup >= 2.0, (
        f"captured replay only {speedup:.2f}× faster than eager "
        f"(eager={eager_ms:.3f}ms, cached={cached_ms:.3f}ms)"
    )
