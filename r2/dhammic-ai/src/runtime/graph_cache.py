"""CUDA-graph capture/replay cache for fixed-shape kernel sequences.

Implements section 2.3 of ``docs/plans/fused-kernels-pretrain.md``.  The
chunked-offload runner above invokes the *same* kernel sequence with the
*same* tensor shapes for every chunk — that is the textbook case for
CUDA graphs.

Usage
-----

>>> cache = GraphCache()
>>> for c in range(n_chunks):
...     y = cache.run(layer, x_chunk, key=("chunk_fwd", chunk_size, B, dtype))

On the first call with a given ``key`` the cache:

1. Runs 3 *warmup* iterations in eager mode so cuBLAS heuristics settle and
   any Triton autotune machinery converges.
2. Allocates persistent static input/output buffers.
3. Captures a CUDA graph that does exactly ``layer_fn(static_inputs)``.

On every subsequent call with the same key the cache:

1. Copies the user's inputs into the static input buffers.
2. Replays the captured graph.
3. Copies the static outputs into freshly allocated output tensors and
   returns them.

If any field of ``key`` changes, the cache transparently re-captures (the old
graph is dropped — CUDA reclaims the memory once nothing references it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Per-key state
# --------------------------------------------------------------------------- #
@dataclass
class _CachedGraph:
    """Materialised state for one (key, shape) bucket."""

    graph: torch.cuda.CUDAGraph
    static_inputs: Tuple[Tensor, ...]
    static_outputs: Tuple[Tensor, ...]
    input_metas: Tuple[Tuple[Tuple[int, ...], torch.dtype], ...]


# --------------------------------------------------------------------------- #
# Public cache
# --------------------------------------------------------------------------- #
class GraphCache:
    """Cache + replay CUDA graphs keyed by user-supplied keys.

    Parameters
    ----------
    warmup_steps:
        Number of eager iterations to run before capture (default 3, per
        plan section 2.3).
    device:
        CUDA device.
    """

    def __init__(
        self,
        warmup_steps: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda")
        if device.type != "cuda":
            raise RuntimeError(
                f"GraphCache requires a CUDA device; got {device}."
            )
        self.device = device
        self.warmup_steps = warmup_steps

        self._cache: Dict[Hashable, _CachedGraph] = {}
        # Diagnostics for tests.
        self.captures: int = 0
        self.replays: int = 0

    # --------------------------------------------------------------------- #
    # API
    # --------------------------------------------------------------------- #
    def run(
        self,
        layer_fn: Callable[..., Any],
        *inputs: Tensor,
        key: Optional[Hashable] = None,
    ) -> Tensor | Tuple[Tensor, ...]:
        """Run ``layer_fn(*inputs)`` through a captured graph.

        ``key`` should encode anything that, when changed, requires a new
        graph: typically ``(chunk_size, batch_size, dtype, layer_id)``.
        If omitted, a key is derived from the input shape/dtype tuple — fine
        for simple cases but ambiguous when several different functions
        share the same shape.
        """
        for t in inputs:
            if not torch.is_tensor(t):
                raise TypeError(
                    "GraphCache.run only accepts torch.Tensor inputs"
                )
            if t.device.type != self.device.type:
                raise ValueError(
                    f"input tensor on {t.device} but cache device is "
                    f"{self.device}"
                )

        input_metas = tuple((tuple(t.shape), t.dtype) for t in inputs)
        if key is None:
            key = input_metas

        cached = self._cache.get(key)
        if cached is None or cached.input_metas != input_metas:
            # Recapture trigger.
            cached = self._capture(layer_fn, inputs, key, input_metas)

        # Copy user inputs into the static input buffers.
        for static_in, user_in in zip(cached.static_inputs, inputs):
            static_in.copy_(user_in)

        cached.graph.replay()
        self.replays += 1

        # Clone outputs out of the static buffers so the caller can keep
        # them across replays without corruption.
        outs = tuple(o.clone() for o in cached.static_outputs)
        return outs[0] if len(outs) == 1 else outs

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #
    def _capture(
        self,
        layer_fn: Callable[..., Any],
        inputs: Tuple[Tensor, ...],
        key: Hashable,
        input_metas: Tuple[Tuple[Tuple[int, ...], torch.dtype], ...],
    ) -> _CachedGraph:
        # Allocate / refresh static buffers.
        static_inputs = tuple(
            torch.empty(meta[0], dtype=meta[1], device=self.device)
            for meta in input_metas
        )
        for si, ui in zip(static_inputs, inputs):
            si.copy_(ui)

        # Use a dedicated capture stream as PyTorch recommends.
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))

        with torch.cuda.stream(capture_stream):
            # ---- Warmup ----
            for _ in range(self.warmup_steps):
                _ = layer_fn(*static_inputs)
            capture_stream.synchronize()

            # ---- Capture ----
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                captured_out = layer_fn(*static_inputs)

        # Ensure the capture is complete before the main stream uses it.
        torch.cuda.current_stream(self.device).wait_stream(capture_stream)

        # Normalise outputs to a tuple of tensors.
        if isinstance(captured_out, tuple):
            static_outputs: Tuple[Tensor, ...] = captured_out
        elif isinstance(captured_out, list):
            static_outputs = tuple(captured_out)
        elif isinstance(captured_out, Tensor):
            static_outputs = (captured_out,)
        else:
            raise TypeError(
                "layer_fn must return a Tensor or a tuple/list of Tensors; "
                f"got {type(captured_out).__name__}"
            )

        cached = _CachedGraph(
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=static_outputs,
            input_metas=input_metas,
        )
        self._cache[key] = cached
        self.captures += 1
        return cached

    # --------------------------------------------------------------------- #
    # Introspection
    # --------------------------------------------------------------------- #
    def keys(self) -> List[Hashable]:
        return list(self._cache.keys())

    def clear(self) -> None:
        """Drop all cached graphs and free their static buffers."""
        self._cache.clear()
