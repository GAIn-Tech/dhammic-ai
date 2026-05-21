"""
K5: Fused SSD Scan — Structured State Space Duality, chunked parallel.

Hybrid Triton + cuBLAS strategy:
  - Triton kernels: segsum (stable segment-sum + cumsum + clamp), and the
    exp-clamp tensor builders (L, decay_states, state_decay_out, decay_chunk).
    These are the elementwise/reduction-heavy pieces.
  - cuBLAS (via torch.einsum -> bmm): the 4 main contractions in steps 1-4.
    BF16 inputs with tensor-core matmul + fp32 accumulate.

Backward strategy:
  Custom Triton bwd for the elementwise ops (segsum, exp+clamp combine).
  For the four einsums we re-run the forward path inside ``torch.enable_grad()``
  and let PyTorch's native autograd flow gradients through the bmm calls --
  that is, the outer ``FusedSSDScan`` uses gradient *checkpointing* to dispatch
  backward through the standard bmm grads. This is the pragmatic split: full
  autograd correctness via cuBLAS bmm grads + manual grad for the Triton
  elementwise pieces (which are wrapped in their own autograd.Functions).

The eager reference is ``src/mamba_base_engine.py:ssd()``. Algorithm (per the
eager docstring) is the standard SSD chunked-parallel scan:
  Step 1: intra-chunk diagonals via einsum(C, B, L, x) where L = exp(segsum(A))
  Step 2: per-chunk states via einsum(B, decay_states, x)
  Step 3: inter-chunk recurrence via einsum(decay_chunk, states)
  Step 4: state-to-output via einsum(C, states, state_decay_out)

We preserve the eager signature so the Mamba3Block's two-SSD decomposition
(which shares cache between the gamma- and beta-paths) works unchanged.

Shapes (after chunking):
    x:  (b, c, l, h, p)           where c = n_chunks, l = chunk_size
    A:  (b, h, c, l)              log-decay (post-permute, post-cumsum)
    B:  (b, c, l, h, n)
    C:  (b, c, l, h, n)

Tolerances:
  fp32 forward parity: max_abs_err < 1e-4
  bf16 forward parity: max_abs_err < 5e-2  (SSD has deep accumulation chains)
  fp32 gradient parity: max_abs_err < 1e-3
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

# Clamp constants must match the eager reference exactly.
# These are exposed as tl.constexpr so the Triton @jit'ed kernels can use them
# as global constants (Triton 3.x disallows free-Python globals).
_SEGSUM_CLAMP_MIN = tl.constexpr(-20.0)
_SEGSUM_CLAMP_MAX = tl.constexpr(20.0)
_SEGSUM_OFF_DIAG_FILL = tl.constexpr(-1e4)  # upper-triangular fill value in segsum


# --------------------------------------------------------------------------- #
# Triton kernels
# --------------------------------------------------------------------------- #
# segsum:
#   input  x has shape (..., T)
#   output y has shape (..., T, T) with
#     y[..., i, j] = clamp(cumsum_strict_lower(x)[..., i, j], -20, 20)  if i >= j
#     y[..., i, j] = -1e4                                                if i <  j
#   where cumsum_strict_lower(x)[..., i, j] = sum_{k=j+1..i} x[..., k]   (note: NOT inclusive of j)
#
# This matches the eager implementation:
#     x = x.unsqueeze(-1).expand(..., T, T) * strict_lower   # x[..., i, j] = x[..., i] if i > j else 0
#     x_segsum = cumsum(x, dim=-2)                            # cumsum over row index i
#     upper_mask = (1 - lower) * -1e4                         # i < j -> -1e4
#     x_segsum = clamp(x_segsum + upper_mask, -20, 20)
#
# We parallelise one program per (..., col_block) and compute the column j over
# all rows i in fp32 using a serial cumsum within the kernel.


@triton.jit
def _segsum_fwd_kernel(
    X_ptr, Y_ptr,
    M, T,                       # M = prod(leading dims), T = seq dim
    stride_x_m, stride_x_t,
    stride_y_m, stride_y_i, stride_y_j,
    BLOCK_T: tl.constexpr,
):
    """Compute y[m, i, j] for one (m, j) pair across all rows i = 0..T-1.

    grid = (M, T)
    """
    m = tl.program_id(0)
    j = tl.program_id(1)

    # Load the full row x[m, :] into registers (T <= BLOCK_T expected by autotune key)
    rows = tl.arange(0, BLOCK_T)
    row_mask = rows < T
    x_row_ptrs = X_ptr + m * stride_x_m + rows * stride_x_t
    x = tl.load(x_row_ptrs, mask=row_mask, other=0.0).to(tl.float32)

    # Build the lane mask: only positions with i > j contribute to the cumsum.
    # We then accumulate prefix-sum over i. Position i=j gets value 0 (i.e. no
    # entry of x), which encodes y[i==j] = clamp(0, ..) = 0 - matching the
    # eager spec (strict_lower mask at i==j is 0, but lower mask is 1, so the
    # upper_mask term doesn't fire and we get 0).
    contrib_mask = rows > j
    x_masked = tl.where(contrib_mask, x, 0.0)

    # Serial associative scan: y_i = sum_{k <= i} x_masked_k
    # tl.cumsum is available in Triton >= 2.1 / 3.x.
    y_col = tl.cumsum(x_masked, axis=0)

    # Apply upper-triangular fill: positions with i < j are -1e4
    upper = rows < j
    y_col = tl.where(upper, _SEGSUM_OFF_DIAG_FILL, y_col)

    # Clamp to [-20, 20]
    y_col = tl.minimum(tl.maximum(y_col, _SEGSUM_CLAMP_MIN), _SEGSUM_CLAMP_MAX)

    # Store y[m, :, j]
    y_ptrs = Y_ptr + m * stride_y_m + rows * stride_y_i + j * stride_y_j
    tl.store(y_ptrs, y_col.to(Y_ptr.dtype.element_ty), mask=row_mask)


@triton.jit
def _segsum_bwd_kernel(
    X_ptr, Y_ptr, DY_ptr, DX_ptr,
    M, T,
    stride_x_m, stride_x_t,
    stride_y_m, stride_y_i, stride_y_j,
    BLOCK_T: tl.constexpr,
):
    """Backward for segsum.

    Forward (un-clamped, un-masked):
        y[i, j] = sum_{k = j+1}^{i} x[k]   for i >= j  (otherwise -1e4)
    After clamp:
        y_out = clamp(y, lo, hi)
    Therefore:
        dy_out -> dy = dy_out * (lo < y < hi)
    The off-diagonal (i<j) gradient is zero by construction (forward emits constant -1e4).

    Now d/dx[t] of  sum_{k=j+1}^{i} x[k]  =  1 iff j < t <= i, else 0.

    => dL/dx[t] = sum_{j < t} sum_{i >= t} dy[i, j] * indicator(in_band)
                 = sum_{j=0}^{t-1} sum_{i=t}^{T-1} dy[i, j]

    We compute one program per (m, t) and reduce by reading the relevant slab
    of dy. For T <= 512 this is a small reduction (~T^2/4 ops) and stays
    bandwidth-bound.
    """
    m = tl.program_id(0)
    t = tl.program_id(1)

    # We'll loop over j in [0, t) and accumulate sum over i in [t, T) of dy[i, j].
    # Use a single 1-D reduction over i for fixed j via a small loop on j.

    rows = tl.arange(0, BLOCK_T)
    row_mask = rows < T

    # Reload y to recompute the clamp mask (so we don't have to save it).
    # We need y[i, j] for the band i >= t and 0 <= j < t.
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over j = 0..t-1
    # The compiler should unroll moderate t; for larger T we accept loop cost.
    # Triton doesn't support dynamic-bound for loops well in jit, so we mask.
    j_idx = tl.arange(0, BLOCK_T)
    j_mask = j_idx < t  # only j < t contributes

    # For each (i, j) load y and dy, build clamp mask, accumulate.
    # We re-shape this as a 2D reduction: dy_band[i, j] * (lo < y[i, j] < hi) over (i >= t, j < t).
    # Loading the full (T, T) tile fits when T <= 512 (256KB for fp32).
    # Build a (BLOCK_T, BLOCK_T) tile, then mask + reduce.
    i_idx = rows[:, None]
    j_idx2 = j_idx[None, :]
    tile_mask = (i_idx < T) & (j_idx2 < T) & (i_idx >= t) & (j_idx2 < t)

    y_ptrs = (
        Y_ptr + m * stride_y_m
        + i_idx * stride_y_i
        + j_idx2 * stride_y_j
    )
    dy_ptrs = (
        DY_ptr + m * stride_y_m
        + i_idx * stride_y_i
        + j_idx2 * stride_y_j
    )

    y_val = tl.load(y_ptrs, mask=tile_mask, other=0.0).to(tl.float32)
    dy_val = tl.load(dy_ptrs, mask=tile_mask, other=0.0).to(tl.float32)

    # Clamp gradient mask: pass only when strictly inside [lo, hi]
    clamp_pass = (y_val > _SEGSUM_CLAMP_MIN) & (y_val < _SEGSUM_CLAMP_MAX)
    contrib = tl.where(tile_mask & clamp_pass, dy_val, 0.0)
    acc = tl.sum(contrib)

    # Write dx[m, t]
    dx_ptr = DX_ptr + m * stride_x_m + t * stride_x_t
    tl.store(dx_ptr, acc.to(DX_ptr.dtype.element_ty))


@triton.jit
def _exp_clamp_fwd_kernel(
    X_ptr, Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """y = exp(clamp(x, -20, 20))   (elementwise)"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x_c = tl.minimum(tl.maximum(x, _SEGSUM_CLAMP_MIN), _SEGSUM_CLAMP_MAX)
    y = tl.exp(x_c)
    tl.store(Y_ptr + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _exp_clamp_bwd_kernel(
    X_ptr, Y_ptr, DY_ptr, DX_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """dx = dy * exp(clamp(x)) * (lo < x < hi).  Using the cached y = exp(clamp(x))."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    clamp_pass = (x > _SEGSUM_CLAMP_MIN) & (x < _SEGSUM_CLAMP_MAX)
    dx = tl.where(clamp_pass, dy * y, 0.0)
    tl.store(DX_ptr + offs, dx.to(DX_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# autograd.Function wrappers for the Triton elementwise/segsum pieces
# --------------------------------------------------------------------------- #
class _Segsum(torch.autograd.Function):
    """Stable segment sum matching eager segsum() exactly.

    Forward: input (..., T) -> output (..., T, T).
    Backward: standard chain through cumsum + clamp.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        assert x.is_cuda
        x = x.contiguous()
        *leading, T = x.shape
        M = 1
        for d in leading:
            M *= d
        # Output tensor
        y = torch.empty(*leading, T, T, dtype=x.dtype, device=x.device)
        # Power-of-2 block size >= T
        block_t = max(16, triton.next_power_of_2(T))
        grid = (M, T)
        x_flat = x.view(M, T)
        y_flat = y.view(M, T, T)
        _segsum_fwd_kernel[grid](
            x_flat, y_flat,
            M, T,
            x_flat.stride(0), x_flat.stride(1),
            y_flat.stride(0), y_flat.stride(1), y_flat.stride(2),
            BLOCK_T=block_t,
        )
        ctx.save_for_backward(x_flat, y_flat)
        ctx.leading = leading
        ctx.T = T
        ctx.M = M
        ctx.block_t = block_t
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        x_flat, y_flat = ctx.saved_tensors
        T = ctx.T
        M = ctx.M
        dy_flat = dy.contiguous().view(M, T, T)
        dx_flat = torch.empty_like(x_flat)
        grid = (M, T)
        _segsum_bwd_kernel[grid](
            x_flat, y_flat, dy_flat, dx_flat,
            M, T,
            x_flat.stride(0), x_flat.stride(1),
            y_flat.stride(0), y_flat.stride(1), y_flat.stride(2),
            BLOCK_T=ctx.block_t,
        )
        return dx_flat.view(*ctx.leading, T)


def _segsum_triton(x: Tensor) -> Tensor:
    return _Segsum.apply(x)


class _ExpClamp(torch.autograd.Function):
    """y = exp(clamp(x, -20, 20))  with Triton fwd+bwd."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        assert x.is_cuda
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _exp_clamp_fwd_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        x, y = ctx.saved_tensors
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        N = x.numel()
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _exp_clamp_bwd_kernel[grid](x, y, dy, dx, N, BLOCK_SIZE=BLOCK)
        return dx


def _exp_clamp_triton(x: Tensor) -> Tensor:
    return _ExpClamp.apply(x)


# --------------------------------------------------------------------------- #
# Core fused SSD scan implementation
# --------------------------------------------------------------------------- #
def _ssd_forward_impl(
    x: Tensor, A: Tensor, B: Tensor, C: Tensor,
    chunk_size: int,
    initial_states: Tensor | None,
    L_cache: Tensor | None,
    decay_chunk_cache: Tensor | None,
    A_cumsum_cache: Tensor | None,
):
    """Pure-functional SSD forward.

    This function uses Triton kernels for segsum and exp+clamp, and torch.einsum
    for the 4 contractions. When called inside ``torch.enable_grad()`` and with
    ``requires_grad`` inputs, PyTorch's native autograd handles backward through
    the einsums (via batched matmul gradients on tensor cores), and the
    ``_Segsum`` / ``_ExpClamp`` autograd.Functions handle their own pieces.

    Returns (Y, final_state, cache) matching eager ``ssd()``.
    """
    assert x.shape[1] % chunk_size == 0, (
        f"sequence length {x.shape[1]} not divisible by chunk_size {chunk_size}"
    )

    b, seqlen = x.shape[0], x.shape[1]
    n_chunks = seqlen // chunk_size

    def _chunk(m: Tensor) -> Tensor:
        return m.view(b, n_chunks, chunk_size, *m.shape[2:])

    x_c = _chunk(x)
    A_c = _chunk(A)
    B_c = _chunk(B)
    C_c = _chunk(C)

    # A_cumsum: (b, h, c, l)
    if A_cumsum_cache is not None:
        A_cumsum = A_cumsum_cache
    else:
        A_perm = A_c.permute(0, 3, 1, 2).contiguous()  # b c l h -> b h c l
        A_cumsum = torch.cumsum(A_perm, dim=-1)

    # Step 1: intra-chunk diagonals. L = exp(segsum(A_perm)) at shape (b, h, c, l, l)
    if L_cache is not None:
        L = L_cache
    else:
        # We need to call segsum on a tensor of shape (..., l). The eager
        # implementation uses segsum on `A` (the post-permute A, which is the
        # un-cumsummed log-rates). We reconstruct that here from A_cumsum if
        # we computed it ourselves -- but actually, eager passes the post-
        # permute A directly, so we re-derive it.
        # Simpler: use the same permuted A.
        A_perm_for_segsum = A_c.permute(0, 3, 1, 2).contiguous()  # b h c l
        L_log = _segsum_triton(A_perm_for_segsum)  # (b, h, c, l, l), clamped
        L = _exp_clamp_triton(L_log)  # exp(clamp(...)) - clamp is idempotent w/ segsum's clamp

    # Step 1 contraction: Y_diag = einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_c, B_c, L, x_c)

    # Step 2: per-chunk states
    # decay_states = exp(clamp(A_cumsum[..., -1:] - A_cumsum, -20, 20))
    decay_states_input = A_cumsum[:, :, :, -1:] - A_cumsum  # (b, h, c, l)
    decay_states = _exp_clamp_triton(decay_states_input)
    # states = einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B_c, decay_states, x_c)

    # Step 3: inter-chunk recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states_full = torch.cat([initial_states, states], dim=1)

    if decay_chunk_cache is not None:
        decay_chunk = decay_chunk_cache
    else:
        # decay_chunk = exp(segsum(F.pad(A_cumsum[..., -1], (1, 0))))
        last_cum = A_cumsum[:, :, :, -1]  # (b, h, c)
        last_cum_padded = F.pad(last_cum, (1, 0))  # (b, h, c+1)
        decay_chunk_log = _segsum_triton(last_cum_padded)  # (b, h, c+1, c+1)
        decay_chunk = _exp_clamp_triton(decay_chunk_log)

    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_full)
    states_out = new_states[:, :-1]
    final_state = new_states[:, -1]

    # Step 4: state-to-output
    # state_decay_out = exp(clamp(A_cumsum, -20, 20))
    state_decay_out = _exp_clamp_triton(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C_c, states_out, state_decay_out)

    combined = Y_diag + Y_off
    Y = combined.reshape(b, -1, combined.shape[3], combined.shape[4])

    cache = {"L": L, "decay_chunk": decay_chunk, "A_cumsum": A_cumsum}
    return Y, final_state, cache


# --------------------------------------------------------------------------- #
# Outer autograd.Function: gradient checkpointing wrapper
# --------------------------------------------------------------------------- #
class FusedSSDScan(torch.autograd.Function):
    """Whole-pipeline fused SSD scan with Triton + cuBLAS hybrid.

    Forward: runs _ssd_forward_impl with the Triton-backed helpers (no grad).
    Backward: re-runs _ssd_forward_impl inside torch.enable_grad() so PyTorch
    can flow gradients back through the 4 einsums (bmm) and the Triton
    autograd.Functions handle the segsum/exp_clamp gradients.

    This is the standard gradient-checkpointing composition pattern. It costs
    one extra forward at backward time, but it lets us reuse cuBLAS bmm
    gradients (fast, fp32-accumulated) without re-deriving 4 einsum grads.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor, A: Tensor, B: Tensor, C: Tensor,
        chunk_size: int,
        initial_states: Tensor | None,
        L_cache_in: Tensor | None,
        decay_chunk_cache_in: Tensor | None,
        A_cumsum_cache_in: Tensor | None,
    ):
        # Run forward with no autograd tracking on the einsums.
        with torch.no_grad():
            Y, final_state, cache = _ssd_forward_impl(
                x, A, B, C, chunk_size,
                initial_states,
                L_cache_in, decay_chunk_cache_in, A_cumsum_cache_in,
            )

        ctx.save_for_backward(x, A, B, C, initial_states if initial_states is not None else torch.empty(0, device=x.device))
        ctx.chunk_size = chunk_size
        ctx.has_initial_states = initial_states is not None
        # Pass cached tensors through as non-tensor ctx attributes
        ctx.L_cache_in = L_cache_in
        ctx.decay_chunk_cache_in = decay_chunk_cache_in
        ctx.A_cumsum_cache_in = A_cumsum_cache_in
        return Y, final_state, cache["L"], cache["decay_chunk"], cache["A_cumsum"]

    @staticmethod
    def backward(ctx, dY, dfinal_state, dL_unused, ddecay_chunk_unused, dA_cumsum_unused):
        x, A, B, C, initial_states_saved = ctx.saved_tensors
        initial_states = initial_states_saved if ctx.has_initial_states else None

        # Re-run forward inside enable_grad with leaves that track gradients.
        # We detach + clone to get fresh leaves; PyTorch will then track the
        # einsum + autograd.Function ops, and we use torch.autograd.grad to
        # extract input grads in one shot.
        x_l = x.detach().requires_grad_(x.requires_grad)
        A_l = A.detach().requires_grad_(A.requires_grad)
        B_l = B.detach().requires_grad_(B.requires_grad)
        C_l = C.detach().requires_grad_(C.requires_grad)
        if initial_states is not None and initial_states.requires_grad:
            is_l = initial_states.detach().requires_grad_(True)
        else:
            is_l = initial_states  # may be None or non-grad-tracking

        # Build the list of inputs that actually need grads.
        inputs_for_grad = []
        if x_l.requires_grad: inputs_for_grad.append(x_l)
        if A_l.requires_grad: inputs_for_grad.append(A_l)
        if B_l.requires_grad: inputs_for_grad.append(B_l)
        if C_l.requires_grad: inputs_for_grad.append(C_l)
        if is_l is not None and is_l.requires_grad: inputs_for_grad.append(is_l)

        with torch.enable_grad():
            Y_rerun, final_state_rerun, _cache_rerun = _ssd_forward_impl(
                x_l, A_l, B_l, C_l, ctx.chunk_size,
                is_l,
                ctx.L_cache_in,
                ctx.decay_chunk_cache_in,
                ctx.A_cumsum_cache_in,
            )

        # Compose the output gradient: dY contributes to Y; dfinal_state to final_state.
        # If dfinal_state is all-zeros (or None semantics from caller), skip it.
        outputs = [Y_rerun]
        grad_outs = [dY]
        if dfinal_state is not None and dfinal_state.abs().sum() > 0:
            outputs.append(final_state_rerun)
            grad_outs.append(dfinal_state)

        if not inputs_for_grad:
            # Nothing requires gradient; return zeros / None for each input.
            zeros = lambda t: torch.zeros_like(t) if t is not None else None
            return (zeros(x), zeros(A), zeros(B), zeros(C), None, None, None, None, None)

        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs_for_grad,
            grad_outputs=grad_outs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # Map grads back into the slot for each forward arg in order.
        grad_iter = iter(grads)
        dx = next(grad_iter) if x_l.requires_grad else None
        dA = next(grad_iter) if A_l.requires_grad else None
        dB = next(grad_iter) if B_l.requires_grad else None
        dC = next(grad_iter) if C_l.requires_grad else None
        dis = next(grad_iter) if (is_l is not None and is_l.requires_grad) else None

        # forward signature: x, A, B, C, chunk_size, initial_states, L_cache_in, decay_chunk_cache_in, A_cumsum_cache_in
        return (dx, dA, dB, dC, None, dis, None, None, None)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def fused_ssd_scan(
    x: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    chunk_size: int,
    initial_states: Tensor | None = None,
    cache: dict | None = None,
):
    """Fused SSD chunked-parallel scan.

    Matches the eager ``ssd()`` signature so it can be a drop-in replacement
    inside ``Mamba3Block``. The two-SSD decomposition shares cache between the
    gamma- and beta-paths -- pass the dict returned from the first call in as
    ``cache`` for the second call.

    Args:
        x:               (b, seqlen, n_heads, d_head), bf16 or fp32, CUDA.
        A:               (b, seqlen, n_heads), bf16 or fp32, CUDA.
        B, C:            (b, seqlen, n_heads, d_state), bf16 or fp32, CUDA.
        chunk_size:      seqlen must be divisible by this.
        initial_states:  optional (b, 1, n_heads, d_head, d_state); zeros if None.
        cache:           optional dict with keys L, decay_chunk, A_cumsum from
                         a prior call (reuses these intermediates).

    Returns:
        Y:           (b, seqlen, n_heads, d_head)
        final_state: (b, n_heads, d_head, d_state)
        cache:       dict {L, decay_chunk, A_cumsum}
    """
    if cache is not None:
        L_in = cache.get("L")
        dc_in = cache.get("decay_chunk")
        ac_in = cache.get("A_cumsum")
    else:
        L_in = dc_in = ac_in = None

    Y, final_state, L_out, dc_out, ac_out = FusedSSDScan.apply(
        x, A, B, C,
        int(chunk_size),
        initial_states,
        L_in, dc_in, ac_in,
    )
    out_cache = {"L": L_out, "decay_chunk": dc_out, "A_cumsum": ac_out}
    return Y, final_state, out_cache
