# FaithfulEngram — Fidelity Notes

## Source

This module follows DeepSeek-AI's **"Conditional Memory via Scalable Lookup:
A New Axis of Sparsity for Large Language Models"** (Cheng et al., Jan 2026):

- arXiv: <https://arxiv.org/abs/2601.07372>
- Reference code: <https://github.com/deepseek-ai/Engram> (file
  `engram_demo_v1.py`)

It supersedes the original `src/deepseek_engram_memory.py`, which is an
MD5-hash-based N-gram lookup over *feature vectors* and is **not
differentiable** (the embedding bank and label table are
`requires_grad=False`). The new module is a token-id-based, prime-modular
multi-head hash lookup with a fully learnable embedding bank — the same
shape as the published DeepSeek Engram.

## Where this implementation lives

- Triton kernel: `src/kernels/engram.py` — fused (gather + sum-over-ngram +
  concat-over-heads) with `tl.atomic_add` backward into the embedding bank.
- nn.Module:    `src/engram_layer.py` — hash computation, value/key
  projection, gated residual contribution.
- Tests:        `tests/kernels/test_engram.py` — 10 tests.

## What is faithful (matches the paper / reference repo)

| Aspect                       | Faithful? | Notes |
|------------------------------|-----------|-------|
| Token-id N-gram hashing      | ✅        | 2-gram and 3-gram by default. |
| Prime-modular multi-head hash| ✅        | `mix = (x_t * m_0) XOR (x_{t-1} * m_1) XOR ...`, modulo a distinct prime per (ngram, head). Multipliers are odd integers from a seeded RNG (matches demo). |
| H independent hash heads      | ✅        | Embeddings from heads are **concatenated** along the feature axis. Embeddings across n-gram sizes within a head are **summed** (so output dim is fixed at `n_heads * d_head` regardless of `max_ngram_size`). |
| Single packed embedding bank | ✅        | One `nn.Embedding(total_rows, d_head)` packs every (ngram, head) sub-table behind a per-group offset, enabling one fused gather. |
| Learnable / differentiable   | ✅        | Embedding bank is a `nn.Parameter`; backward atomically scatters into it. `value_proj`, `key_proj`, RMSNorms also learnable. |
| Sigmoid–sqrt gate (Eq. 5)    | ✅        | `gate = sigmoid(sqrt(|q·k|) * sign(q·k))` with `q = RMSNorm(hidden)`, `k = RMSNorm(key_proj(retrieved))`, scaled by `1/sqrt(d_model)`. |

## What is simplified

| Simplification                       | Why |
|--------------------------------------|-----|
| No `CompressedTokenizer`             | The demo collapses casing / whitespace by re-mapping tokenizer ids via a lookup table. We assume the caller pre-compresses, treating input ids as already in `[0, tokenizer_vocab_size)`. |
| No ShortConv post-process            | The demo appends a depth-wise `Conv1d` on the gated value. ShortConv is a separate op (a K6/SwiGLU-class fused kernel) and not part of the Engram-faithfulness question. |
| No hyper-connection (HC)             | The demo wraps Engram in 4-branch HC. Our forward returns a single `(B,T,d_model)` tensor for residual add; HC composition is the caller's responsibility. |
| One Engram per layer (no cross-layer prime de-duplication) | The demo uses `_find_next_prime` with a *shared* `seen` set across all Engram layers in the model. For a single-module test we keep `seen` local; multi-layer wiring will need the integration agent to thread a shared set. |

## Why this differentiable variant (vs. the MD5-hash variant)

The pre-existing `src/deepseek_engram_memory.py`:

1. **Is not differentiable.** `conceptual_labels` is `requires_grad=False`,
   and `learn_pattern` is a non-gradient `with torch.no_grad()` write. There
   is no gradient path into the bank from training loss — the embedding
   table is set by explicit `learn_pattern` calls, not by the LM loss.
2. **Hashes feature vectors via MD5.** Discretized features → byte buffer →
   MD5 → table index. Useful for content-addressable lookup but not for
   training a memory **alongside** an LM head, because (a) features change
   during training (discretized neighbourhood drifts), and (b) gradients
   cannot reach the rows the lookup hit.
3. **Has collisions silently dropped.** `if current_label == -1 or
   current_label == label_idx` — different patterns colliding to the same
   slot keep the first; no probing or chaining.

For the dhammic-ai pretrain, the **differentiable bank with prime-modular
n-gram hashing is the correct variant**: it gives the LM the ability to
*write* what it wants to remember via gradient descent, exactly as in
DeepSeek's published work. The new module is therefore a drop-in
*replacement* (not an addition) for any caller that previously instantiated
the MD5 variant for training use.

## Test summary

`uv run pytest tests/kernels/test_engram.py -v` — 10/10 passing.

- `test_kernel_shapes`, `test_kernel_parity_fp32` (max_abs_err < 1e-5),
  `test_kernel_parity_bf16` (max_abs_err < 5e-3), `test_kernel_determinism`,
  `test_kernel_backward_parity` (grad max_abs_err < 1e-3 vs eager autograd).
- `test_module_forward_shapes`, `test_module_gradient_flow` (verifies
  non-zero grads on embedding bank, value_proj, key_proj, hidden input),
  `test_module_eval_determinism`, `test_module_lookup_matches_eager_reference`.
- `test_module_memorization_smoke`: SGD lr=0.5, 200 steps over one
  `(B=1, T=4)` example with d=32 — loss converges from ~0.91 to <0.001,
  demonstrating the bank learns the target via gradient descent.

## Autotune gotcha (resolved)

`@triton.autotune` re-runs each candidate config for benchmarking. The
backward kernel scatters via `tl.atomic_add` into the gradient bank — if
the bank is not zeroed between trials, the autotune phase produces a
gradient that is `N_trials * true_grad`. Fixed by `reset_to_zero=["DW_ptr"]`
on the bwd autotune config.
