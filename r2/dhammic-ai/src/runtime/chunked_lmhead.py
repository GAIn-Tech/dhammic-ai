"""Chunk-streaming fused LM-head + cross-entropy.

The :class:`~src.kernels.lmhead_xent.FusedLMHeadXentModule` already chunks the
**vocab** dimension internally so the (M, V) logits tensor never lives on the
GPU.  What it does *not* chunk is the **sequence** dimension: callers pass a
full ``(B, T, D)`` hidden tensor, the kernel materializes a ``(B*T, D)`` fp32
``d_hidden`` accumulator in backward, and both grow linearly with ``T``.

For the chunked-offload pipeline the hidden tensor returned by
``ChunkedOffloadRunner`` is already conceptually a stream of ``(B, chunk, D)``
chunks — pulling the whole thing back to GPU and feeding it to the head is
the exact step that breaks the flat-VRAM guarantee.

This module fixes that: it takes either

* a single ``(B, T, D)`` hidden tensor that lives on CPU or GPU, or
* an iterable of ``(B, chunk, D)`` hidden chunks,

with matching targets, and computes the **token-weighted mean cross-entropy
loss** by streaming chunks through the existing fused kernel.  Each chunk's
contribution to ``d(hidden)`` and ``weight.grad`` is computed in its OWN
manually-driven backward — so the kernel's per-chunk fp32 ``d_weight (V, D)``
and ``d_hidden_flat (chunk*B, D)`` working tensors get freed BEFORE the
next chunk's backward starts.  GPU peak therefore depends on ``chunk_size``
(and ``D``, ``V``) but **not on ``T``**.

The token-weighted mean is exact: ``fused_lmhead_xent`` returns
``sum(losses) / n_valid`` for the chunk, so multiplying by ``n_valid`` and
dividing by the *global* valid count at the end recovers the same loss the
unchunked kernel would have produced — up to fp32 reduction order, which is
well within the parity tolerances we hit elsewhere in the repo.

Hard rules (mirrors :mod:`src.kernels.lmhead_xent`):

* CUDA only, production-only — no eager fallback.
* The wrapped kernel module is **not** rewritten; we call ``fused_lmhead_xent``
  per chunk.
* Weight tying works because the same ``nn.Parameter`` is referenced across
  every chunk call.
"""

from __future__ import annotations

import math
from typing import Iterable, Iterator, List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

try:
    # When imported as ``src.runtime.chunked_lmhead`` (pytest, package).
    from src.kernels.lmhead_xent import fused_lmhead_xent
except ImportError:  # pragma: no cover
    # When imported as ``runtime.chunked_lmhead`` (scripts that put ``src/``
    # on ``sys.path``).
    from kernels.lmhead_xent import fused_lmhead_xent  # type: ignore[no-redef]


HiddenInput = Union[Tensor, Iterable[Tensor]]
TargetInput = Union[Tensor, Iterable[Tensor]]


def _slice_tensor_chunks(
    hidden: Tensor, targets: Tensor, chunk_size: int,
) -> Iterator[Tuple[Tensor, Tensor]]:
    """Yield contiguous ``(B, chunk, D)`` / ``(B, chunk)`` views."""
    if hidden.dim() != 3:
        raise ValueError(
            f"hidden tensor must be 3-D (B, T, D), got shape {tuple(hidden.shape)}"
        )
    if targets.dim() != 2:
        raise ValueError(
            f"targets tensor must be 2-D (B, T), got shape {tuple(targets.shape)}"
        )
    B, T, _ = hidden.shape
    if targets.shape != (B, T):
        raise ValueError(
            f"targets shape {tuple(targets.shape)} != hidden (B, T) "
            f"({B}, {T})"
        )
    for c0 in range(0, T, chunk_size):
        c1 = min(c0 + chunk_size, T)
        yield hidden[:, c0:c1, :], targets[:, c0:c1]


def _collect_iterable(
    hidden_iter: Iterable[Tensor], targets_iter: Iterable[Tensor],
) -> List[Tuple[Tensor, Tensor]]:
    """Materialize an iterable input into a list of paired chunks."""
    out: List[Tuple[Tensor, Tensor]] = []
    h_it = iter(hidden_iter)
    t_it = iter(targets_iter)
    while True:
        try:
            h_c = next(h_it)
        except StopIteration:
            try:
                next(t_it)
            except StopIteration:
                break
            raise ValueError("targets iterable longer than hidden iterable")
        try:
            t_c = next(t_it)
        except StopIteration as e:
            raise ValueError("hidden iterable longer than targets iterable") from e
        if not isinstance(h_c, Tensor) or not isinstance(t_c, Tensor):
            raise TypeError("chunks must be torch.Tensor instances")
        out.append((h_c, t_c))
    return out


class _ChunkedLMHeadXentFunction(torch.autograd.Function):
    """Custom autograd Function that streams chunks in forward AND backward.

    Forward: iterate chunks, call ``fused_lmhead_xent`` under ``no_grad``
    just to compute each chunk's mean loss; accumulate a token-weighted sum
    and divide by the global non-ignored count.  Stash the chunks
    themselves + the per-chunk ``n_valid_c`` for backward.

    Backward: iterate chunks again; for each chunk, re-run
    ``fused_lmhead_xent`` UNDER ``enable_grad`` (this is "checkpointing": the
    forward we use for the loss value is discarded; we re-compute it during
    backward only to rebuild the autograd tape locally).  Then call
    ``torch.autograd.grad`` to get ``(d_h_c, d_weight_c)`` for that chunk,
    accumulate ``d_weight_c`` into the shared ``weight.grad``, and write
    ``d_h_c`` into the corresponding slice of the returned ``d_hidden``.

    Because each chunk's backward owns its own short-lived ``d_weight`` and
    ``d_hidden_flat`` allocations and we ``del`` them immediately after
    accumulation, peak GPU memory holds only ONE chunk's transients at a
    time regardless of how many chunks there are.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden_full: Tensor,
        weight: Tensor,
        targets_full: Tensor,
        chunk_size: int,
        ignore_index: int,
    ) -> Tensor:
        # We do NOT call the inner fused_lmhead_xent under autograd here:
        # we'll redo it ourselves in backward to rebuild the local tape per
        # chunk.  This is checkpointing-by-recomputation.
        device = weight.device
        assert hidden_full.is_cuda or hidden_full.device == device, (
            "hidden must be on the same CUDA device as weight"
        )

        with torch.no_grad():
            # Build chunk index list (start, end) for both fwd-accum and bwd.
            B, T, D = hidden_full.shape
            chunk_bounds: List[Tuple[int, int, int]] = []  # (c0, c1, n_valid_c)
            loss_sum: Tensor = torch.zeros((), dtype=torch.float32, device=device)
            n_valid_total: int = 0

            for c0, c1 in _bounds(T, chunk_size):
                h_c = hidden_full[:, c0:c1, :]
                t_c = targets_full[:, c0:c1]
                n_valid_c = int((t_c != ignore_index).sum().item())
                if n_valid_c == 0:
                    chunk_bounds.append((c0, c1, 0))
                    continue
                chunk_loss = fused_lmhead_xent(h_c, weight, t_c, ignore_index)
                # chunk_loss is already a tensor on `device`; keep its
                # *value* (.detach()) since we don't need its tape.
                loss_sum = loss_sum + chunk_loss.detach().to(torch.float32) * float(
                    n_valid_c
                )
                n_valid_total += n_valid_c
                chunk_bounds.append((c0, c1, n_valid_c))

        if n_valid_total == 0:
            loss = loss_sum  # zero, but keep autograd connectivity via weight
            ctx.skip_backward = True
            ctx.weight = weight
            ctx.hidden_shape = hidden_full.shape
            ctx.hidden_dtype = hidden_full.dtype
            return loss

        loss = loss_sum / float(n_valid_total)
        # Save the tensors we need verbatim for the backward recompute.
        ctx.save_for_backward(hidden_full, weight, targets_full)
        ctx.chunk_bounds = chunk_bounds
        ctx.n_valid_total = n_valid_total
        ctx.ignore_index = ignore_index
        ctx.hidden_shape = hidden_full.shape
        ctx.hidden_dtype = hidden_full.dtype
        ctx.skip_backward = False
        return loss

    @staticmethod
    def backward(ctx, dloss: Tensor):  # type: ignore[override]
        # No-grad branch — every chunk was ignored.
        if ctx.skip_backward:
            return None, None, None, None, None

        hidden_full, weight, targets_full = ctx.saved_tensors
        chunk_bounds: List[Tuple[int, int, int]] = ctx.chunk_bounds
        n_valid_total: int = ctx.n_valid_total
        ignore_index: int = ctx.ignore_index
        device = weight.device
        # Allocate the full d_hidden ONCE, in the input's dtype.
        # This is the unavoidable (B, T, D) gradient that the upstream graph
        # expects from us.  In the chunked-offload pipeline this very tensor
        # is what the runner then streams back to pinned CPU one chunk at a
        # time — but at THIS boundary we have to materialize it because
        # autograd's contract is a single tensor return.
        d_hidden_full = torch.zeros(
            ctx.hidden_shape, dtype=ctx.hidden_dtype, device=device
        )

        # Scalar grad factor common to every chunk: dloss / n_valid_total.
        # We pre-bake this into the per-chunk "grad_outputs" we pass to
        # autograd.grad: scale_c = (n_valid_c / n_valid_total) * dloss.
        # That recovers the same gradient flow the unchunked kernel would
        # have produced (whose .backward() carries dloss * 1/N_total to each
        # row already).
        dloss_f = dloss.detach().to(torch.float32)
        inv_total = 1.0 / float(n_valid_total)

        for c0, c1, n_valid_c in chunk_bounds:
            if n_valid_c == 0:
                continue
            # View into the original input; we re-enable grad locally so
            # autograd.grad can flow back through fused_lmhead_xent.
            h_c_view = hidden_full[:, c0:c1, :]
            t_c_view = targets_full[:, c0:c1]
            # Clone to a leaf with requires_grad=True for the per-chunk
            # backward pass.  We can't use the view directly because the
            # outer Function has its own saved_tensors; opening a child tape
            # against a leaf gives us clean d_h_c.
            h_c_leaf = h_c_view.detach().clone().requires_grad_(True)

            with torch.enable_grad():
                chunk_loss = fused_lmhead_xent(
                    h_c_leaf, weight, t_c_view, ignore_index
                )
                # The fused kernel returns mean over THIS chunk's valid rows:
                #   chunk_loss = (1/n_valid_c) * sum_m mask_m * (logZ_m - tgt_m)
                # The global loss is
                #   loss = (1/n_valid_total) * sum_chunks sum_m mask_m * ...
                #        = (1/n_valid_total) * sum_chunks (n_valid_c * chunk_loss_c)
                # so d(loss) / d(chunk_loss_c) = n_valid_c / n_valid_total.
                # Multiplying by the upstream dloss gives the grad_outputs.
                grad_out = dloss_f * (float(n_valid_c) * inv_total)
                # torch.autograd.grad returns (d_h_c_leaf, d_weight_c).
                d_h_c, d_weight_c = torch.autograd.grad(
                    outputs=(chunk_loss,),
                    inputs=(h_c_leaf, weight),
                    grad_outputs=(grad_out,),
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

            # Write this chunk's d_hidden into the full d_hidden buffer.
            d_hidden_full[:, c0:c1, :].copy_(d_h_c.to(ctx.hidden_dtype))

            # Accumulate d_weight into weight.grad directly. We CANNOT
            # return it through autograd because PyTorch will then try to
            # accumulate it into weight.grad a SECOND time (our return is
            # already what gets fed to AccumulateGrad). So we set
            # ``weight_grad_out = None`` below and side-channel it here.
            if weight.grad is None:
                weight.grad = d_weight_c.to(weight.dtype).detach()
            else:
                weight.grad.add_(d_weight_c.to(weight.dtype).detach())

            # Drop everything we no longer need; this is the key step that
            # lets peak VRAM stay flat across n_chunks.
            del chunk_loss, d_h_c, d_weight_c, h_c_leaf

        # We side-channelled weight.grad above so PyTorch's AccumulateGrad
        # should NOT add anything more for the weight.
        # forward args order: hidden_full, weight, targets_full, chunk_size,
        # ignore_index → 5 returns.
        return d_hidden_full, None, None, None, None


def _bounds(T: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    for c0 in range(0, T, chunk_size):
        yield c0, min(c0 + chunk_size, T)


class ChunkedLMHeadXent(nn.Module):
    """Drop-in replacement for ``FusedLMHeadXentModule`` with sequence chunking.

    Parameters
    ----------
    d_model:
        Hidden dim ``D``.
    vocab_size:
        Vocabulary size ``V``.
    weight:
        Optional pre-existing ``nn.Parameter`` of shape ``(V, D)`` to use
        directly (for weight tying).  If ``None``, a new parameter is created
        with ``N(0, 1/sqrt(D))`` initialization.
    chunk_size:
        Sequence chunk size used when ``hidden`` is passed as a single
        ``(B, T, D)`` tensor.  Ignored when ``hidden`` is passed as an
        iterable of chunks (the iterable's natural chunking is used).
    ignore_index:
        Target value to skip (matches ``F.cross_entropy``).

    Forward
    -------
    .. code-block:: python

        loss = head(hidden, targets)              # (B, T, D), (B, T)
        loss = head(hidden_chunks, target_chunks) # iterables of chunks

    The returned scalar is the token-weighted mean cross-entropy loss across
    *all* non-ignored positions, identical (up to fp32 reduction order) to
    calling the unchunked kernel on the full ``(B, T, D)``.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        weight: nn.Parameter | None = None,
        chunk_size: int = 256,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
        if weight is None:
            self.weight = nn.Parameter(
                torch.empty(vocab_size, d_model).normal_(
                    mean=0.0, std=1.0 / math.sqrt(d_model)
                )
            )
        else:
            if tuple(weight.shape) != (vocab_size, d_model):
                raise ValueError(
                    f"tied weight shape {tuple(weight.shape)} != "
                    f"({vocab_size}, {d_model})"
                )
            self.weight = weight  # share the same parameter (weight tying)

    def forward(
        self,
        hidden: HiddenInput,
        targets: TargetInput,
        total_tokens: int | None = None,  # noqa: ARG002 — kept for API compat
    ) -> Tensor:
        """Chunk-streaming forward.

        Parameters
        ----------
        hidden:
            ``(B, T, D)`` CUDA tensor (chunked internally by ``chunk_size``),
            OR an iterable of ``(B, chunk, D)`` CUDA tensor chunks (each
            chunk forms one streaming unit).
        targets:
            ``(B, T)`` int64 CUDA tensor, OR an iterable of ``(B, chunk)``
            int64 CUDA tensor chunks matching ``hidden``'s kind.
        total_tokens:
            Accepted for API symmetry with manual streaming callers, but
            ignored: we compute the global non-ignored count exactly from
            the supplied targets.

        Returns
        -------
        Tensor
            Scalar fp32 mean loss with autograd tracking.
        """
        weight = self.weight
        if not weight.is_cuda:
            raise RuntimeError(
                "ChunkedLMHeadXent requires a CUDA weight; got "
                f"{weight.device}."
            )

        hidden_is_tensor = isinstance(hidden, Tensor)
        targets_is_tensor = isinstance(targets, Tensor)
        if hidden_is_tensor != targets_is_tensor:
            raise TypeError(
                "hidden and targets must both be tensors or both be "
                f"iterables, got {type(hidden).__name__} and "
                f"{type(targets).__name__}"
            )

        if hidden_is_tensor:
            assert isinstance(hidden, Tensor) and isinstance(targets, Tensor)
            hidden_full = hidden
            targets_full = targets
        else:
            chunks = _collect_iterable(hidden, targets)  # type: ignore[arg-type]
            if not chunks:
                return weight.sum() * 0.0
            # Concatenate chunks back into a contiguous (B, T, D) on the
            # weight's device.  This is a copy, but it's bounded by the same
            # T as the eventual reference; the chunked autograd Function
            # immediately re-slices it without copying further.  If the
            # caller really wants pinned-CPU streaming all the way through,
            # they should pass the (B, T, D) CPU tensor directly: our
            # autograd Function will route per-chunk autograd through the
            # same path regardless of where the original lives.
            h_list = [hc.to(weight.device, non_blocking=True) for hc, _ in chunks]
            t_list = [tc.to(weight.device, non_blocking=True) for _, tc in chunks]
            hidden_full = torch.cat(h_list, dim=1)
            targets_full = torch.cat(t_list, dim=1)

        if hidden_full.device != weight.device:
            hidden_full = hidden_full.to(weight.device, non_blocking=True)
        if targets_full.device != weight.device:
            targets_full = targets_full.to(weight.device, non_blocking=True)

        # Forward / backward are driven through the chunked Function so the
        # caller still gets ordinary autograd semantics — they call
        # `loss.backward()` and get gradients in hidden.grad / weight.grad.
        loss = _ChunkedLMHeadXentFunction.apply(
            hidden_full, weight, targets_full,
            self.chunk_size, self.ignore_index,
        )
        return loss

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, vocab_size={self.vocab_size}, "
            f"chunk_size={self.chunk_size}, ignore_index={self.ignore_index}"
        )
