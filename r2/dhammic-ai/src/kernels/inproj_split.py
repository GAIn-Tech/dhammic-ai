"""K2: fused_inproj_split

Fuses ``nn.Linear(d_model, sum(split_sizes), bias=False)`` and the
subsequent ``torch.split`` into a single autograd op that writes
directly to N output tensors.

Reference pattern from ``src/mamba_base_engine.py``::

    proj = self.in_proj(x)
    z, x_ssm, B, C, dt, lam, theta = torch.split(
        proj,
        [d_inner, d_inner, d_state, d_state, n_heads, n_heads, theta_dim],
        dim=-1,
    )

Two strategies are implemented and a one-shot micro-benchmark picks
the faster path at construction time:

  (a) ``single_gemm``    one big GEMM into a contiguous buffer, then
                         return zero-copy views (the splits are
                         contiguous along the last dim, so this is
                         literally free).
  (b) ``multi_gemm``     N independent GEMMs with weight slices,
                         each writing straight into its own output
                         tensor; this avoids ever materialising the
                         full proj tensor (saves memory) but pays N
                         kernel launches.

On Ampere SM86 with large batches strategy (a) almost always wins
because cuBLAS likes big matmuls and the splits are zero-copy.
On small batch / small d_model the multi-GEMM path can edge ahead
because each sub-GEMM stays L2-resident.

Production-only: bf16 in/out, fp32 accumulation. No eager fallback.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Strategy (a): one big GEMM, return views.
# ---------------------------------------------------------------------------


class _FusedInProjSplitSingle(torch.autograd.Function):
    """One ``addmm``/``matmul`` followed by zero-copy ``narrow`` views."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, split_sizes: Tuple[int, ...]):
        # x:      (..., d_model)        bf16
        # weight: (sum_split, d_model)  bf16
        # proj:   (..., sum_split)      bf16
        proj = torch.nn.functional.linear(x, weight)
        ctx.save_for_backward(x, weight)
        ctx.split_sizes = split_sizes

        # narrow returns views — no copy
        outs = []
        offset = 0
        for s in split_sizes:
            outs.append(proj.narrow(-1, offset, s))
            offset += s
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grad_outs: Tensor):
        x, weight = ctx.saved_tensors
        split_sizes = ctx.split_sizes

        # Concatenate the incoming grads along the last dim into one
        # contiguous tensor whose layout matches ``proj``.  Using ``cat``
        # here is the right call: the upstream grads are independent
        # tensors with potentially different strides, so we cannot view
        # them as a single slab.
        grad_proj = torch.cat([g.contiguous() for g in grad_outs], dim=-1)

        # grad_x = grad_proj @ weight
        # grad_w = grad_proj.T @ x   (summed over batch dims)
        grad_x = grad_proj @ weight
        # collapse leading dims for the weight gradient
        gp2 = grad_proj.reshape(-1, grad_proj.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        grad_w = gp2.transpose(0, 1) @ x2
        return grad_x, grad_w, None


# ---------------------------------------------------------------------------
# Strategy (b): N independent GEMMs, no proj materialisation.
# ---------------------------------------------------------------------------


class _FusedInProjSplitMulti(torch.autograd.Function):
    """N independent ``linear`` calls with sliced weight rows."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, split_sizes: Tuple[int, ...]):
        outs: List[Tensor] = []
        offset = 0
        # Weight is (sum_split, d_model); each chunk's rows = (s, d_model)
        for s in split_sizes:
            w_slice = weight.narrow(0, offset, s)  # view
            outs.append(torch.nn.functional.linear(x, w_slice))
            offset += s
        ctx.save_for_backward(x, weight)
        ctx.split_sizes = split_sizes
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grad_outs: Tensor):
        x, weight = ctx.saved_tensors
        split_sizes = ctx.split_sizes

        # grad_x = sum_i grad_out_i @ weight_slice_i
        # grad_w_slice_i = grad_out_i.T @ x       (then concat along dim 0)
        grad_x = torch.zeros_like(x)
        grad_w_chunks: List[Tensor] = []
        offset = 0
        x2 = x.reshape(-1, x.shape[-1])
        for g, s in zip(grad_outs, split_sizes):
            w_slice = weight.narrow(0, offset, s)
            grad_x = grad_x + g @ w_slice
            g2 = g.reshape(-1, g.shape[-1])
            grad_w_chunks.append(g2.transpose(0, 1) @ x2)
            offset += s
        grad_w = torch.cat(grad_w_chunks, dim=0)
        return grad_x, grad_w, None


# ---------------------------------------------------------------------------
# Public functional wrapper (uses strategy single by default; the module
# below picks the faster strategy via micro-benchmark).
# ---------------------------------------------------------------------------


def fused_inproj_split(
    x: Tensor,
    weight: Tensor,
    split_sizes: List[int],
    *,
    strategy: str = "single_gemm",
) -> Tuple[Tensor, ...]:
    """Run the fused in-proj + split.

    Args:
        x:           shape (..., d_model), bf16 (or fp16/fp32).
        weight:      shape (sum(split_sizes), d_model), same dtype as x.
        split_sizes: how to slice the last dim of the implicit proj.
        strategy:    ``"single_gemm"`` or ``"multi_gemm"``.

    Returns:
        Tuple of tensors with shapes ``(..., split_sizes[i])``.
    """
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"x.shape[-1]={x.shape[-1]} must equal weight.shape[1]={weight.shape[1]}"
        )
    if sum(split_sizes) != weight.shape[0]:
        raise ValueError(
            f"sum(split_sizes)={sum(split_sizes)} must equal weight.shape[0]={weight.shape[0]}"
        )
    ss = tuple(int(s) for s in split_sizes)
    if strategy == "single_gemm":
        return _FusedInProjSplitSingle.apply(x, weight, ss)
    if strategy == "multi_gemm":
        return _FusedInProjSplitMulti.apply(x, weight, ss)
    raise ValueError(f"unknown strategy={strategy!r}")


# Alias kept for clarity / external use.
FusedInProjSplit = _FusedInProjSplitSingle


# ---------------------------------------------------------------------------
# Drop-in module.  Picks the faster of the two strategies once, at init.
# ---------------------------------------------------------------------------


def _benchmark_strategy(
    strategy: str,
    x: Tensor,
    weight: Tensor,
    split_sizes: Tuple[int, ...],
    iters: int,
) -> float:
    """Return mean per-iter wall time (s) on the CUDA stream."""
    if not x.is_cuda:
        # No micro-bench on CPU; just return a sentinel that prefers single.
        return float("inf") if strategy == "multi_gemm" else 0.0

    torch.cuda.synchronize()
    # warmup
    for _ in range(3):
        outs = fused_inproj_split(x, weight, list(split_sizes), strategy=strategy)
        # touch the outputs so the launch isn't optimised away
        _ = outs[0].sum()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        outs = fused_inproj_split(x, weight, list(split_sizes), strategy=strategy)
        _ = outs[0].sum()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters


class FusedInProjSplitModule(nn.Module):
    """Drop-in replacement for ``nn.Linear`` + ``torch.split``.

    The faster strategy is chosen once via a quick micro-benchmark on
    a representative shape (defaults to ``(2, 256, d_model)`` if not
    given via ``probe_shape``).
    """

    def __init__(
        self,
        d_model: int,
        split_sizes: List[int],
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        probe_shape: Tuple[int, int] | None = None,
        bench_iters: int = 30,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.split_sizes = tuple(int(s) for s in split_sizes)
        self.total = sum(self.split_sizes)

        self.weight = nn.Parameter(
            torch.empty(self.total, self.d_model, dtype=dtype, device=device)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # Pick the strategy.
        self.strategy = self._select_strategy(probe_shape, bench_iters, dtype, device)

    def _select_strategy(
        self,
        probe_shape: Tuple[int, int] | None,
        iters: int,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> str:
        if device is None or (isinstance(device, str) and not device.startswith("cuda")):
            return "single_gemm"
        if isinstance(device, torch.device) and device.type != "cuda":
            return "single_gemm"
        try:
            if probe_shape is None:
                B, T = 2, 256
            else:
                B, T = probe_shape
            x = torch.randn(B, T, self.d_model, dtype=dtype, device=device)
            t_single = _benchmark_strategy(
                "single_gemm", x, self.weight.detach(), self.split_sizes, iters
            )
            t_multi = _benchmark_strategy(
                "multi_gemm", x, self.weight.detach(), self.split_sizes, iters
            )
            self.bench_results = {"single_gemm": t_single, "multi_gemm": t_multi}
            return "single_gemm" if t_single <= t_multi else "multi_gemm"
        except Exception:
            # Fall back to single on any benchmarking hiccup; never crash __init__.
            self.bench_results = {}
            return "single_gemm"

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return fused_inproj_split(
            x, self.weight, list(self.split_sizes), strategy=self.strategy
        )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, split_sizes={list(self.split_sizes)}, "
            f"strategy={self.strategy}"
        )
