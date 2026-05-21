"""Chunked-offload streaming runtime.

Implements a forward+backward pipeline that processes a long sequence one
``chunk_size``-sized slice at a time.  Activations stored for the backward
pass are staged on pinned CPU memory; only a small GPU ring buffer of the
chunks that are *actually* being computed on right now stays resident.

The design follows section 2.2 of ``docs/plans/fused-kernels-pretrain.md``:

* compute and copy run on two separate CUDA streams,
* the H2D for chunk ``c+1`` overlaps the compute on chunk ``c``,
* the D2H of chunk ``c-2`` runs concurrently with the same compute,
* SSM-style chunk-to-chunk state (small) stays on GPU forever.

The result is that **GPU resident memory is independent of ``seq_len``**.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pinned_empty(shape, dtype) -> Tensor:
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


def _ensure_pinned(x: Tensor) -> Tensor:
    """Move ``x`` to pinned CPU memory if it is not already there."""
    if x.device.type == "cpu" and x.is_pinned():
        return x
    pinned = _pinned_empty(x.shape, x.dtype)
    pinned.copy_(x.detach())
    return pinned


# --------------------------------------------------------------------------- #
# Custom autograd Function — owns the offload / prefetch logic for backward
# --------------------------------------------------------------------------- #
class _ChunkedOffloadFunction(torch.autograd.Function):
    """Forward+backward pass over a sequence in chunks with CPU offload.

    Forward
    -------
    * GPU ring buffer of size 3 for input chunks,
    * runs ``layer_fn`` on every input chunk under ``no_grad`` (we recompute
      under grad enable during backward — same idea as activation
      checkpointing, but with pinned-CPU staging instead of plain recomputation
      from the layer's inputs because we already stream those),
    * stages every chunk's output back to pinned CPU memory and assembles
      ``y_full_cpu``.

    Backward
    --------
    * walks chunks in reverse,
    * prefetches the chunk's input and output-gradient back to GPU on the copy
      stream,
    * recomputes the chunk under ``enable_grad`` to rebuild the autograd tape,
    * runs ``torch.autograd.grad`` and stages the input gradient back to a
      pinned-CPU accumulator.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x_full_cpu: Tensor,
        chunk_size: int,
        layer_fn: Callable[[Tensor], Tensor],
        compute_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        out_dtype: torch.dtype,
        *layer_params: Tensor,
    ) -> Tensor:
        assert x_full_cpu.device.type == "cpu", "x_full must live on CPU"
        assert x_full_cpu.is_pinned(), "x_full must be in pinned memory"
        B, T, D = x_full_cpu.shape
        if T % chunk_size != 0:
            raise ValueError(
                f"seq_len ({T}) must be a multiple of chunk_size ({chunk_size})"
            )
        n_chunks = T // chunk_size
        device = torch.device("cuda")

        # CPU buffer where the assembled output lives.  We will allocate the
        # final shape lazily after the first chunk runs, since ``layer_fn``
        # may change the channel dimension.
        y_full_cpu: Optional[Tensor] = None
        D_out = D  # default; updated after first chunk

        x_chunks_cpu: List[Tensor] = [None] * n_chunks  # type: ignore[list-item]

        # Ring of GPU input chunks.
        gpu_inputs: dict[int, Tensor] = {}

        def _prefetch(c: int) -> None:
            if c >= n_chunks or c in gpu_inputs:
                return
            src = x_full_cpu[:, c * chunk_size : (c + 1) * chunk_size, :]
            with torch.cuda.stream(copy_stream):
                dst = torch.empty(src.shape, dtype=src.dtype, device=device)
                dst.copy_(src, non_blocking=True)
            gpu_inputs[c] = dst

        _prefetch(0)
        _prefetch(1)

        for c in range(n_chunks):
            compute_stream.wait_stream(copy_stream)
            if c + 2 < n_chunks:
                _prefetch(c + 2)

            x_c_gpu = gpu_inputs[c]

            with torch.cuda.stream(compute_stream):
                with torch.no_grad():
                    y_c_gpu = layer_fn(x_c_gpu)

            # Lazily allocate y_full_cpu now that we know the output channel
            # count and dtype.
            if y_full_cpu is None:
                D_out = y_c_gpu.shape[-1]
                y_full_cpu = _pinned_empty((B, T, D_out), out_dtype)

            # Stage the input chunk back to CPU for the backward pass.
            x_c_cpu = _pinned_empty(x_c_gpu.shape, x_c_gpu.dtype)
            copy_stream.wait_stream(compute_stream)
            with torch.cuda.stream(copy_stream):
                x_c_cpu.copy_(x_c_gpu, non_blocking=True)
            x_chunks_cpu[c] = x_c_cpu

            # Stage the output chunk back to CPU and into y_full_cpu.
            with torch.cuda.stream(copy_stream):
                y_full_cpu[
                    :, c * chunk_size : (c + 1) * chunk_size, :
                ].copy_(y_c_gpu, non_blocking=True)

            # Drop the GPU input we no longer need.
            if c - 1 in gpu_inputs:
                gpu_inputs[c - 1].record_stream(copy_stream)
                del gpu_inputs[c - 1]

        copy_stream.synchronize()
        compute_stream.synchronize()
        gpu_inputs.clear()
        assert y_full_cpu is not None

        # Stash for backward.
        ctx.save_for_backward(*layer_params)
        ctx.x_chunks_cpu = x_chunks_cpu
        ctx.chunk_size = chunk_size
        ctx.n_chunks = n_chunks
        ctx.layer_fn = layer_fn
        ctx.compute_stream = compute_stream
        ctx.copy_stream = copy_stream
        ctx.input_shape = (B, T, D)
        ctx.input_dtype = x_full_cpu.dtype
        ctx.output_shape = (B, T, D_out)
        ctx.output_dtype = out_dtype

        return y_full_cpu

    @staticmethod
    def backward(ctx, grad_y_full_cpu: Tensor):  # type: ignore[override]
        x_chunks_cpu: List[Tensor] = ctx.x_chunks_cpu
        chunk_size: int = ctx.chunk_size
        n_chunks: int = ctx.n_chunks
        layer_fn: Callable[[Tensor], Tensor] = ctx.layer_fn
        compute_stream: torch.cuda.Stream = ctx.compute_stream
        copy_stream: torch.cuda.Stream = ctx.copy_stream
        B, T, D = ctx.input_shape
        layer_params = ctx.saved_tensors

        device = torch.device("cuda")
        grad_y_full_cpu = _ensure_pinned(grad_y_full_cpu.contiguous())

        # Accumulator for input gradient.
        grad_x_full_cpu = _pinned_empty((B, T, D), ctx.input_dtype)

        gpu_inputs: dict[int, Tensor] = {}
        gpu_grad_outs: dict[int, Tensor] = {}

        def _prefetch_back(c: int) -> None:
            if c < 0 or c in gpu_inputs:
                return
            with torch.cuda.stream(copy_stream):
                x_c = torch.empty(
                    x_chunks_cpu[c].shape,
                    dtype=x_chunks_cpu[c].dtype,
                    device=device,
                )
                x_c.copy_(x_chunks_cpu[c], non_blocking=True)
                g_src = grad_y_full_cpu[
                    :, c * chunk_size : (c + 1) * chunk_size, :
                ].contiguous()
                g_c = torch.empty(g_src.shape, dtype=g_src.dtype, device=device)
                g_c.copy_(g_src, non_blocking=True)
            gpu_inputs[c] = x_c
            gpu_grad_outs[c] = g_c

        _prefetch_back(n_chunks - 1)
        _prefetch_back(n_chunks - 2)

        # Accumulate parameter gradients across chunks.
        param_grads: List[Optional[Tensor]] = [None] * len(layer_params)
        # Which params actually require grad?
        params_for_grad: List[Tensor] = [p for p in layer_params if p.requires_grad]
        params_for_grad_idx: List[int] = [
            i for i, p in enumerate(layer_params) if p.requires_grad
        ]

        for c in range(n_chunks - 1, -1, -1):
            compute_stream.wait_stream(copy_stream)
            if c - 2 >= 0:
                _prefetch_back(c - 2)

            x_c_gpu = gpu_inputs[c].detach().requires_grad_(True)
            grad_y_c_gpu = gpu_grad_outs[c]

            with torch.cuda.stream(compute_stream):
                with torch.enable_grad():
                    y_c = layer_fn(x_c_gpu)
                inputs_for_grad: Tuple[Tensor, ...] = (x_c_gpu,) + tuple(params_for_grad)
                grads = torch.autograd.grad(
                    outputs=(y_c,),
                    inputs=inputs_for_grad,
                    grad_outputs=(grad_y_c_gpu,),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )

            grad_x_c_gpu = grads[0]
            grad_params_chunk = grads[1:]

            # Stage grad_x_c back to CPU and into the accumulator.
            copy_stream.wait_stream(compute_stream)
            with torch.cuda.stream(copy_stream):
                grad_x_full_cpu[
                    :, c * chunk_size : (c + 1) * chunk_size, :
                ].copy_(grad_x_c_gpu, non_blocking=True)

            # Accumulate parameter gradients (kept on GPU; they're small).
            for j, idx in enumerate(params_for_grad_idx):
                g = grad_params_chunk[j]
                if g is None:
                    continue
                if param_grads[idx] is None:
                    param_grads[idx] = g.detach().clone()
                else:
                    param_grads[idx] = param_grads[idx] + g.detach()

            if c + 1 in gpu_inputs:
                gpu_inputs[c + 1].record_stream(copy_stream)
                gpu_grad_outs[c + 1].record_stream(copy_stream)
                del gpu_inputs[c + 1]
                del gpu_grad_outs[c + 1]

        copy_stream.synchronize()
        compute_stream.synchronize()

        param_grad_outs = tuple(param_grads)
        # forward args: x_full_cpu, chunk_size, layer_fn, compute_stream,
        # copy_stream, out_dtype, *layer_params  → 6 non-param + N param.
        return (grad_x_full_cpu, None, None, None, None, None) + param_grad_outs


# --------------------------------------------------------------------------- #
# Public runner
# --------------------------------------------------------------------------- #
class ChunkedOffloadRunner:
    """Drives the chunked-offload streaming pipeline."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = torch.device("cuda")
        if device.type != "cuda":
            raise RuntimeError(
                "ChunkedOffloadRunner requires a CUDA device; got "
                f"{device}."
            )
        self.device = device
        self.compute_stream = torch.cuda.current_stream(device)
        self.copy_stream = torch.cuda.Stream(device=device)

        self._last_y: Optional[Tensor] = None
        self._last_x: Optional[Tensor] = None

    def forward(
        self,
        layer_fn: Callable[[Tensor], Tensor],
        x_full: Tensor,
        chunk_size: int,
    ) -> Tensor:
        """Run a streaming forward pass.

        Parameters
        ----------
        layer_fn:
            Callable mapping ``(B, chunk_size, D_in)`` → ``(B, chunk_size, D_out)``.
            For an ``nn.Module`` pass ``module``; bound parameters will be
            picked up automatically for grad accumulation.
        x_full:
            ``(B, T, D_in)`` on CPU.  ``T`` must be divisible by ``chunk_size``.
        chunk_size:
            Number of timesteps per chunk.

        Returns
        -------
        Tensor
            ``(B, T, D_out)`` on pinned CPU with autograd tracking.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if x_full.shape[1] % chunk_size != 0:
            raise ValueError(
                f"seq_len ({x_full.shape[1]}) must be a multiple of "
                f"chunk_size ({chunk_size})"
            )

        # Make sure the input is pinned and re-enable grad tracking on it
        # (we need ``requires_grad=True`` for .grad to be populated by the
        # custom Function backward).
        x_pinned = _ensure_pinned(x_full.contiguous())
        x_pinned.requires_grad_(True)

        # Collect parameters that participated in layer_fn so the custom
        # Function backward can return parameter gradients.
        layer_params: Tuple[Tensor, ...] = ()
        if isinstance(layer_fn, torch.nn.Module):
            layer_params = tuple(layer_fn.parameters())
            callable_fn: Callable[[Tensor], Tensor] = lambda xc: layer_fn(xc)
        else:
            owner = getattr(layer_fn, "__self__", None)
            if owner is not None and hasattr(owner, "parameters"):
                layer_params = tuple(owner.parameters())
            callable_fn = layer_fn

        # Determine output dtype by running a single dummy chunk on GPU.
        # Cheap probe: a chunk_size slice of the input.
        probe = x_pinned[:, :chunk_size, :].to(self.device, non_blocking=False)
        with torch.no_grad():
            probe_out = callable_fn(probe)
        out_dtype = probe_out.dtype
        del probe, probe_out

        y = _ChunkedOffloadFunction.apply(
            x_pinned,
            chunk_size,
            callable_fn,
            self.compute_stream,
            self.copy_stream,
            out_dtype,
            *layer_params,
        )

        self._last_y = y
        self._last_x = x_pinned
        return y

    def backward(self, dy_full: Tensor) -> Tensor:
        """Back-propagate ``dy_full`` through the most recent forward pass.

        Returns the input gradient as a pinned-CPU tensor.  Parameter
        gradients are written into ``.grad`` of every parameter that
        participated in the forward pass.
        """
        if self._last_y is None or self._last_x is None:
            raise RuntimeError(
                "ChunkedOffloadRunner.backward called before forward"
            )
        if dy_full.shape != self._last_y.shape:
            raise ValueError(
                f"dy_full shape {tuple(dy_full.shape)} does not match the "
                f"forward output shape {tuple(self._last_y.shape)}"
            )
        self._last_y.backward(dy_full)
        grad_in = self._last_x.grad
        assert grad_in is not None, (
            "Input gradient was not populated; this should not happen."
        )
        # Detach for safety so the caller can't accidentally extend the tape.
        return grad_in.detach()
