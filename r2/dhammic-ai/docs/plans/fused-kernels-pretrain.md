# Fully-Fused Kernels + Chunked-Offload Pretrain Plan

**Target:** RTX 3060 (6 GB, SM86) · seq_len 2048+ · production-only code · pretrain to convergence
**Stack:** Triton 3.5.1 · PyTorch 2.9.1 (cu128) · custom `torch.autograd.Function` wrappers · `torch.cuda.CUDAGraph`
**Baselines to beat:** Gen 76 local val_bpb 1.777 (5-min budget) · Run 47 cloud 1.279

---

## 1. Kernel surface

One Triton kernel per row. Each kernel ships fwd + bwd as a `torch.autograd.Function`. Each kernel has a parity test vs the eager reference (rtol 1e-3 fp32, 1e-2 bf16) and a gradient check.

| # | Kernel | Fuses today's ops | Eager source |
|---|---|---|---|
| K1 | `fused_rmsnorm` | `pow + mean + rsqrt + mul + weight` | `mamba_base_engine.RMSNorm` |
| K2 | `fused_inproj_split` | `nn.Linear(d_model → 7-way) + torch.split` | `Mamba3Block.in_proj` |
| K3 | `fused_rope` | `cos + sin + interleave + stack/flatten` (BC + cumsum angles) | `apply_rope` |
| K4 | `fused_trapezoidal_coefs` | `softplus + sigmoid + exp(dA) + (1-λ)·dt·α + λ·dt` → α, β, γ | inline in `Mamba3Block.forward` |
| K5 | **`fused_ssd_scan`** | Whole chunked SSD: segsum, intra/inter einsums, state recurrence, fwd + bwd | `ssd()` |
| K6 | `fused_swiglu` | `mlp_up → chunk(2) → silu(g)·v → mlp_down` (or fuse the act+mul only and keep mm) | `MambaBlock` MLP path |
| K7 | `fused_sdr_topk_ste` | `gumbel + topk + scatter + STE` (training); `topk + scatter` (eval) | `SDREmbedding` |
| K8 | `fused_engram_gather` | `topk(col) + softmax + gumbel_softmax(cell) + gather + einsum` | `HTMEngram` |
| K9 | `fused_lmhead_xent` | `lm_head matmul + log_softmax + NLL`, only logits at gradient time | `train.py` loss line |

**Production-only**: delete `use_vectorized` branches, `use_grad_checkpoint` fallbacks for these layers, the Python-loop fallback in `NumentaSDRTokenizer`. One path per layer.

K5 is the load-bearing kernel and the longest item. Strategy:
- Step 1: keep eager `ssd()` but swap segsum + decay paths to Triton (cheap wins, no autograd headache).
- Step 2: write a Triton fwd+bwd that fuses intra-chunk einsums and decay multiplications, keeping `B @ x_chunk` and `C @ states` as `bmm` calls (cuBLAS handles tensor-core matmul better than handwritten Triton matmul on Ampere).
- This is a **hybrid fused kernel** — not all-Triton, but every elementwise/reduction op fused, matmuls via cuBLAS through tensor cores. That's how mamba-ssm and FlashAttention-3 do it.

---

## 2. Chunked offload + CUDA graph recapture

### 2.1 Why VRAM scales today

Activations stored for backward = `O(n_layers · batch · seq_len · d_model)`. At seq_len 2048, d=128, n_layers=4, b=4 that's ~16 MB per layer just for residual — fine. The killer is the in_proj split (d_inner=384 + 384 + 16 + 16 + 8 + 8 + 8 = 824 dims) and the SSD A_cumsum / decay tensors which scale as `n_chunks · n_heads · chunk_size`. On seq_len 8192+, VRAM goes from "tight" to "OOM."

### 2.2 Chunked-offload pipeline

Fixed `chunk_size=256` (matches SSD). For `seq_len=N`, `n_chunks=N/256`.

```
Ring buffer on GPU (always-resident): 3 chunks per tensor stream
  • chunk[c-1]: live (its grads needed by chunk c's bwd)
  • chunk[c]:   compute target
  • chunk[c+1]: prefetched async via copy stream
All other chunks: pinned CPU memory.

Forward:
  for c in 0..n_chunks:
    if c+1 < n_chunks: launch H2D copy chunk[c+1] on copy_stream
    on compute_stream: run fused kernels for chunk c
    if c-2 >= 0: D2H chunk[c-2] activations on copy_stream
    state = ssd_chunk_step(state, chunk_inputs_c)  # state stays GPU (small)

Backward (reverse order):
  for c in n_chunks-1..0:
    if c-1 >= 0: H2D chunk[c-1] activations
    backward through chunk c on compute_stream
    free chunk[c+1] from GPU
```

GPU resident memory: `3 · chunk_size · per_chunk_bytes` (fixed) + per-layer SSM state `(b · n_heads · d_head · d_state)` (fixed, small). **VRAM is independent of `seq_len`.**

### 2.3 CUDA graph capture/replay

Every chunk hits the *same* kernel sequence with *same* shapes — perfect for graph capture.

- Warm up 3 steps in eager mode (cuBLAS heuristic stabilizes, Triton autotunes).
- On step 4: `with torch.cuda.graph(g):` record one full `chunk_forward` + one `chunk_backward` graph per layer (Mamba block + MLP block; engram and hebbian get separate graphs because they only run at one layer).
- Replay graphs for every chunk in every subsequent step.
- **Recapture trigger:** chunk_size, batch_size, dtype, or num_chunks-on-GPU changes. Tracked via a `(chunk_size, batch_size, dtype)` key; cache holds the graph + its input/output static buffers.
- Static input/output buffers: pre-allocate once per shape key. Copy data in/out per replay.

Refs we'll follow: PyTorch `make_graphed_callables` source, NVIDIA Apex GraphedTransformerLayer.

---

## 3. Validation strategy

**Per-kernel** (each parallel agent ships these alongside the kernel):
- Numerical parity: `max_abs_err` and `max_rel_err` < 1e-3 in fp32, < 1e-2 in bf16 vs eager reference on random inputs across realistic shapes.
- Gradient parity: forward, then `loss = out.sum()`, then `loss.backward()` — compare all param/input grads.
- Benchmark: tok/s vs eager on RTX 3060.

**End-to-end** (after wiring):
- 100 steps of training: fused vs eager loss curve within ±5%.
- Peak VRAM at seq_len 256, 512, 1024, 2048, 4096 — must be approximately flat (chunked-offload working).
- `torch.cuda.graph` replay timing: should drop CPU overhead to near-zero.

---

## 4. Pretrain harness

`train.py` updates:
- Stripped of `use_vectorized`, eager-fallback branches.
- New `--chunk_seq_len N` flag (default 2048, can push to 8192).
- Monitor every 10 steps: `loss, lr, tok/s, peak_vram_mb`.
- Eval every 200 steps: `val_loss, val_bpb`.
- Factual eval every 1000 steps: accuracy + perplexity.
- Checkpoint every 500 steps. Resume support.
- Stopping: convergence detected (val_bpb plateau: rolling std < 0.005 over 2000 steps) **or** wall-time budget hit.

Logging output written to `runs/<timestamp>/train.log` + machine-readable `runs/<timestamp>/metrics.jsonl` (one line per eval). I'll tail this to the chat as we go.

---

## 5. Parallel-agent decomposition (ultrapilot)

Nine kernel agents + 1 runtime agent + 1 integration agent, all kicked off in parallel after this plan is approved. Each owns its file(s) and a parity test. No two agents touch the same file.

| Agent | Owns | Deliverable |
|---|---|---|
| A1 | `src/kernels/rmsnorm.py` + test | K1 + parity test |
| A2 | `src/kernels/inproj_split.py` + test | K2 + parity test |
| A3 | `src/kernels/rope.py` + test | K3 + parity test |
| A4 | `src/kernels/trapezoidal.py` + test | K4 + parity test |
| A5 | `src/kernels/ssd_scan.py` + test | K5 hybrid (load-bearing) + parity + grad-check |
| A6 | `src/kernels/swiglu.py` + test | K6 + parity test |
| A7 | `src/kernels/sdr_topk.py` + test | K7 + parity test |
| A8 | `src/kernels/engram_gather.py` + test | K8 + parity test |
| A9 | `src/kernels/lmhead_xent.py` + test | K9 + parity test |
| A10 | `src/runtime/chunked_offload.py` + `src/runtime/graph_cache.py` | Ring-buffer streaming + graph capture/replay |
| A11 | `src/citta_vithi_pipeline.py` rewrite + `train.py` update | Wire kernels + runtime, strip eager paths |

After all 11 ship: I run end-to-end parity (1.) and short pretrain smoke test, then launch the long pretrain.

---

## 6. Risk register

- **R1: SSD bwd correctness.** SSD has nested einsums; manual bwd is error-prone. Mitigation: build it as `torch.autograd.Function` with sanity-checked manual gradients **and** keep eager `ssd()` as ground truth for `torch.autograd.gradcheck`. If grad-check fails, fall back to PyTorch autograd through Triton ops (slightly slower, still fused).
- **R2: Graph capture + dynamic shapes.** If the dataloader yields odd seq_lens, pad to multiple of chunk_size. Already enforced in train.py — keep enforcing.
- **R3: 1.4 GB free VRAM right now.** I'll size starting config so probe pass fits, scale up as kernels reduce memory. Plan starts at `b=2, seq_len=512`, scales after kernel landing.
- **R4: Eager fallback removal.** LOCKED: full strip. No eager paths anywhere. If a kernel regresses, fix it and redeploy. No runtime escape hatch. Parity tests on every kernel are the safety net.

**Locked decisions:** K5 = hybrid Triton (decay/segsum fused) + cuBLAS `bmm` (tensor-core matmuls). Pretrain runs until val_bpb plateau auto-stop. Eager fallback = fully stripped.

---

## 7. Success criteria

- All 9 parity tests pass.
- Peak VRAM at seq_len 2048 ≤ peak VRAM at seq_len 512 + 200 MB (chunked-offload working).
- Pretrain reaches **val_bpb < 1.5** on the FineWeb subset (clearly beating Gen 76's 1.777).
- Stretch: val_bpb < 1.3 (approaches Run 47's 1.279).
- Throughput at seq_len 2048: stretch target 50k tok/s on RTX 3060 (Run 47 hit 113k on A10G; 3060 is ~3× slower so 30–50k is realistic).
