"""Faithful DeepSeek Engram module — production differentiable conditional memory.

Implements the core lookup path of "Conditional Memory via Scalable Lookup"
(DeepSeek-AI, arXiv:2601.07372, Jan 2026, github.com/deepseek-ai/Engram).

What is faithful:
  - Token-id-based N-gram hashing (2,3-gram by default), per the paper.
  - Multi-head hash with prime-modular mixing: H independent hash heads per
    n-gram size reduce collision noise; head embeddings are *concatenated*
    along the feature axis.
  - Distinct (n-gram size, head) embedding sub-tables packed into a single
    combined `nn.Embedding(TOTAL_ROWS, d_head)` for one fused gather.
  - End-to-end differentiable: embedding table is a learnable parameter;
    gradients flow back into rows that were hit (atomic-add in Triton).
  - Hidden-state-conditioned gate decides whether to trust the retrieved
    memory, exactly per the paper:
        gate = sigmoid( sqrt(|q·k|) * sign(q·k) )

What is simplified (vs. the demo at github.com/deepseek-ai/Engram):
  - No compressed tokenizer normalization. We treat input ids as already-
    compressed token ids in [0, tokenizer_vocab_size). Production callers
    should pre-compress with their tokenizer.
  - No ShortConv post-processing branch (the demo applies depth-wise conv
    after the value projection; that's a K6/SwiGLU-style fused op, handled
    elsewhere in the stack).
  - No Hyper-Connection (HC) multi-branch wrapping. Forward returns a single
    (B, T, d_value) tensor for residual add. Wrap with HC at the call site.
  - One Engram per layer; no cross-layer prime de-duplication across modules.

Hot path (Triton-fused, see src/kernels/engram.py):
    hash_ids (B,T,N_GROUPS) ──gather────► retrieved (B, T, n_embed_per_ngram)
                              concat-by-head
                              sum-over-ngram
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import isprime
from torch import Tensor

try:
    from src.kernels.engram import fused_engram_lookup
except ImportError:
    from kernels.engram import fused_engram_lookup


# --------------------------------------------------------------------------- #
# Hash table planning — prime-modular vocab sizes per (ngram_size, head),
# matching deepseek-ai/Engram demo (engram_demo_v1.py).
# --------------------------------------------------------------------------- #
def _find_next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen:
            return candidate
        candidate += 1


def _plan_hash_tables(
    engram_vocab_size_per_ngram: list[int],
    max_ngram_size: int,
    n_heads_per_ngram: int,
) -> list[list[int]]:
    """Return head_vocab_sizes[ngram_index][head_index] (distinct primes)."""
    seen: set[int] = set()
    out: list[list[int]] = []
    for ngram_idx, base in enumerate(engram_vocab_size_per_ngram):
        per_head: list[int] = []
        start = base - 1
        for _ in range(n_heads_per_ngram):
            p = _find_next_prime(start, seen)
            seen.add(p)
            per_head.append(p)
            start = p
        out.append(per_head)
    return out


# --------------------------------------------------------------------------- #
# FaithfulEngram
# --------------------------------------------------------------------------- #
class FaithfulEngram(nn.Module):
    """DeepSeek-Engram-style conditional memory layer.

    Parameters
    ----------
    d_model : int
        Hidden state dimension of the backbone (query/value projection target).
    d_value : int, optional
        n_embed_per_ngram in the paper. Output of the gather, before value_proj.
        Defaults to ``d_model``. Must be divisible by ``n_heads_per_ngram``.
    tokenizer_vocab_size : int
        Size of the *compressed* token vocabulary (input ids must be in
        [0, tokenizer_vocab_size)).
    max_ngram_size : int, default=3
        Hash 2-grams .. max_ngram_size-grams.
    n_heads_per_ngram : int, default=8
        Hash heads per n-gram size.
    engram_vocab_size_per_ngram : list[int] | None
        Per-n-gram base vocab size (final per-head vocab is the next prime
        from this lower bound, distinct across heads/n-grams). If None,
        defaults to ``[64 * tokenizer_vocab_size] * (max_ngram_size - 1)``,
        matching the paper's ratio for small-scale tests scaled down.
    pad_id : int, default=0
    seed : int, default=0
    dtype : torch.dtype, default=torch.bfloat16

    Forward
    -------
    forward(hidden_state: (B,T,d_model), input_ids: (B,T) int64)
        -> retrieved: (B, T, d_model)

    The retrieved tensor is gated and value-projected, ready to be *added
    residually* to ``hidden_state`` by the caller.
    """

    def __init__(
        self,
        d_model: int,
        tokenizer_vocab_size: int,
        d_value: Optional[int] = None,
        max_ngram_size: int = 3,
        n_heads_per_ngram: int = 8,
        engram_vocab_size_per_ngram: Optional[list[int]] = None,
        pad_id: int = 0,
        seed: int = 0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if max_ngram_size < 2:
            raise ValueError(f"max_ngram_size must be >= 2, got {max_ngram_size}")

        self.d_model = d_model
        self.d_value = d_value if d_value is not None else d_model
        if self.d_value % n_heads_per_ngram != 0:
            raise ValueError(
                f"d_value ({self.d_value}) must be divisible by "
                f"n_heads_per_ngram ({n_heads_per_ngram})"
            )
        self.d_head = self.d_value // n_heads_per_ngram
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_heads_per_ngram = n_heads_per_ngram
        self.pad_id = pad_id

        n_ngrams = max_ngram_size - 1  # 2..max_ngram_size inclusive
        if engram_vocab_size_per_ngram is None:
            engram_vocab_size_per_ngram = [64 * tokenizer_vocab_size] * n_ngrams
        if len(engram_vocab_size_per_ngram) != n_ngrams:
            raise ValueError(
                f"engram_vocab_size_per_ngram length {len(engram_vocab_size_per_ngram)} "
                f"!= n_ngrams {n_ngrams}"
            )

        # ----------------------------------------------------------------- #
        # Hash table layout
        # head_vocab_sizes[ngram_idx][head_idx] = prime modulus.
        # Group ordering for the kernel: g iterates as
        #   for head in 0..H: for ngram_idx in 0..n_ngrams:
        # so that g % n_bands == head — groups in the same band are SUMMED,
        # heads define bands which are CONCATENATED in the output.
        # ----------------------------------------------------------------- #
        head_vocab_sizes = _plan_hash_tables(
            engram_vocab_size_per_ngram, max_ngram_size, n_heads_per_ngram
        )
        n_groups = n_heads_per_ngram * n_ngrams
        sizes_flat: list[int] = []  # per-group vocab size in (head, ngram) order
        for head in range(n_heads_per_ngram):
            for ngram_idx in range(n_ngrams):
                sizes_flat.append(head_vocab_sizes[ngram_idx][head])
        offsets_flat = [0]
        for s in sizes_flat[:-1]:
            offsets_flat.append(offsets_flat[-1] + s)
        total_rows = sum(sizes_flat)

        self.n_groups = n_groups
        self.n_bands = n_heads_per_ngram  # output is concat over heads
        self.register_buffer(
            "offsets",
            torch.tensor(offsets_flat, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "head_vocab_sizes",
            torch.tensor(sizes_flat, dtype=torch.int64),
            persistent=False,
        )

        # Single packed embedding table, learnable.
        self.embed = nn.Embedding(total_rows, self.d_head)
        # Initialize small (default nn.Embedding is N(0,1) which is large for d_head).
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(self.d_head))
        # Cast embedding to target dtype upfront (production-only).
        self.embed.weight.data = self.embed.weight.data.to(dtype)

        # ----------------------------------------------------------------- #
        # Hash multipliers — fixed per-layer odd integers, per the paper.
        # We use the same multiplier per ngram-position across heads (matches
        # the demo). Heads diverge via their distinct prime moduli, not
        # multipliers.
        # ----------------------------------------------------------------- #
        rng = np.random.default_rng(seed)
        max_long = np.iinfo(np.int64).max
        m_max = int(max_long // tokenizer_vocab_size)
        half_bound = max(1, m_max // 2)
        mults = rng.integers(low=0, high=half_bound, size=(max_ngram_size,),
                             dtype=np.int64) * 2 + 1  # odd
        self.register_buffer(
            "multipliers", torch.from_numpy(mults), persistent=False
        )

        # ----------------------------------------------------------------- #
        # Projections + gating, faithful to the paper.
        # key_proj is also learnable per the demo, but with a single HC branch
        # we keep it singular here.
        # All projections cast to target dtype upfront (production-only,
        # matches the embedding bank).
        # ----------------------------------------------------------------- #
        self.value_proj = nn.Linear(self.d_value, d_model, bias=False)
        self.key_proj = nn.Linear(self.d_value, d_model, bias=False)
        # RMSNorms for the gate's (q, k) inner product, per the paper.
        self.norm_q = nn.RMSNorm(d_model)
        self.norm_k = nn.RMSNorm(d_model)
        self._dtype = dtype
        self.to(dtype=dtype)

    # --------------------------------------------------------------------- #
    # Hash computation — vectorized in PyTorch on the device. Cheap enough
    # not to need fusion (just integer arithmetic over (B,T,N_GROUPS)).
    # --------------------------------------------------------------------- #
    def _compute_hashes(self, input_ids: Tensor) -> Tensor:
        """Compute (B, T, N_GROUPS) int64 hash ids.

        Mixing matches the DeepSeek demo:
            mix(n) = (x[t]   * m[0])
                  ^ (x[t-1] * m[1])
                  ...
                  ^ (x[t-(n-1)] * m[n-1])
            hash_g = mix(n_g) % head_vocab_size[g]

        where ``g`` indexes (head, ngram) — heads share multipliers but use
        distinct prime moduli.
        """
        B, T = input_ids.shape
        device = input_ids.device
        x = input_ids.to(torch.int64)
        max_n = self.max_ngram_size

        # Precompute shift_k tensors for k = 0..max_n-1.
        # shift_k[t] = x[t - k], with pad_id for t < k.
        shifts = []
        for k in range(max_n):
            if k == 0:
                shifts.append(x)
            else:
                pad = torch.full(
                    (B, k), self.pad_id, dtype=torch.int64, device=device
                )
                shifts.append(torch.cat([pad, x[:, : T - k]], dim=1))

        n_ngrams = max_n - 1
        # per-ngram mix tensor: (B, T)
        ngram_mix = []
        mults = self.multipliers.to(device)
        for n in range(2, max_n + 1):
            ngram_idx = n - 2
            mix = shifts[0] * mults[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, shifts[k] * mults[k])
            ngram_mix.append(mix)
        # ngram_mix: list of length n_ngrams of (B,T)

        # Group-order: for head h in 0..H, for ngram_idx in 0..n_ngrams,
        # giving N_GROUPS = H * n_ngrams. This places head h in band h.
        all_hashes = []
        head_sizes = self.head_vocab_sizes.view(self.n_bands, n_ngrams)  # (H, n_ngrams)
        for h in range(self.n_bands):
            for ngram_idx in range(n_ngrams):
                mod = int(head_sizes[h, ngram_idx].item())
                hg = torch.remainder(ngram_mix[ngram_idx], mod)
                all_hashes.append(hg)
        # Stack on last axis to (B, T, N_GROUPS).
        return torch.stack(all_hashes, dim=-1).contiguous()

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(self, hidden_state: Tensor, input_ids: Tensor) -> Tensor:
        """Return gated retrieved memory (B, T, d_model) for residual add.

        Args
        ----
        hidden_state : (B, T, d_model), bf16/fp32 — current backbone state.
        input_ids    : (B, T) int64 — *compressed* token ids in
                       [0, tokenizer_vocab_size).
        """
        assert input_ids.dtype == torch.int64, (
            f"input_ids must be int64 (got {input_ids.dtype})"
        )
        assert input_ids.is_cuda, "input_ids must be on CUDA"
        if (input_ids.max() >= self.tokenizer_vocab_size) or (input_ids.min() < 0):
            raise ValueError(
                f"input_ids out of range [0, {self.tokenizer_vocab_size})"
            )

        hash_ids = self._compute_hashes(input_ids)  # (B, T, N_GROUPS)

        # Triton-fused gather + sum-over-ngram + concat-over-heads.
        retrieved = fused_engram_lookup(
            hash_ids, self.offsets, self.embed.weight,
            self.d_head, self.n_bands,
        )  # (B, T, d_value), same dtype as embed.weight.

        # value & key projections.
        # Under CUDA autocast, F.linear runs with autocast-cached bf16 weights even
        # when module parameters are stored as fp32.  The fused lookup returns fp32,
        # and hidden_state can also still be fp32 early in the pipeline, so choosing
        # hidden_state.dtype here can create a mat1=float32 vs mat2=bf16 mismatch.
        # Match projection parameter dtype. FaithfulEngram casts itself to bf16
        # during init, so this must be bf16 even outside an autocast context
        # (e.g. the train.py VRAM probe before the main AMP loop).
        proj_dtype = self.value_proj.weight.dtype
        retrieved_for_proj = retrieved.to(proj_dtype)
        v = self.value_proj(retrieved_for_proj)  # (B, T, d_model)
        k = self.key_proj(retrieved_for_proj)    # (B, T, d_model)

        # Gate (paper Eq. 5):
        #   g = sigmoid( sqrt(|q·k|) * sign(q·k) )
        # where q = norm_q(hidden_state), k = norm_k(k).
        gate_input = hidden_state.to(k.dtype)
        q_n = self.norm_q(gate_input)
        k_n = self.norm_k(k)
        qk = (q_n * k_n).sum(dim=-1, keepdim=True) / math.sqrt(self.d_model)
        gate = (qk.abs().clamp_min(1e-6).sqrt() * qk.sign()).sigmoid()
        return (gate * v).to(hidden_state.dtype)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_value={self.d_value}, "
            f"d_head={self.d_head}, max_ngram_size={self.max_ngram_size}, "
            f"n_heads_per_ngram={self.n_heads_per_ngram}, "
            f"n_groups={self.n_groups}, "
            f"total_rows={int(self.head_vocab_sizes.sum().item())}"
        )
