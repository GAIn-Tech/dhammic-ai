# Scaling Sweep — Chunked Offload

Counterpart to `scaling_sweep.md`. Same architecture and the same (seq_len, batch) grid, but the backbone + `final_norm` are run through `ChunkedOffloadRunner` so their activations are staged on pinned-CPU between chunks.

- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- VRAM total / free at start: 6144 / 5122 MB
- Architecture: d=64 l=2 ds=8 exp=2 heads=4 V=1024 chunk=256
- Wrap strategy: (b) — single `runner.forward()` over the whole backbone + `final_norm`. The embedding is eager (token IDs are tiny) and the head is eager (its fused kernel already chunks vocab internally).

## Sweep 1: seq_len at batch=2 — chunked vs eager

| seq_len | eager VRAM (MB) | chunked VRAM (MB) | ratio eager/chunked | chunked/chunked@256 | chunked wall (ms) | chunked tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 31.9 | 31.2 | 1.02x | 1.00x | 972.3 | 527 |
| 512 | 46.2 | 32.5 | 1.42x | 1.04x | 2648.0 | 387 |
| 1024 | 77.3 | 33.3 | 2.32x | 1.07x | 3934.1 | 521 |
| 2048 | 132.1 | 40.3 | 3.28x | 1.29x | 4860.4 | 843 |
| 4096 | 249.6 | 63.5 | 3.93x | 2.04x | 16579.5 | 494 |
| 8192 | 475.6 | 110.4 | 4.31x | 3.54x | 31076.4 | 527 |

## Sweep 2: batch at seq=2048 — chunked vs eager

| batch | eager VRAM (MB) | chunked VRAM (MB) | ratio eager/chunked | chunked wall (ms) | chunked tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 77.3 | 28.7 | 2.69x | 5273.0 | 388 |
| 2 | 132.1 | 40.3 | 3.28x | 4860.4 | 843 |
| 4 | 249.6 | 63.5 | 3.93x | 7943.4 | 1031 |

## Key assertions

- **Flat-VRAM:** chunked@8192 / chunked@256 = **3.54x** (110.4 MB / 31.2 MB) — target ≤ 1.5x — **FAIL**.
- **chunked < eager at seq ≥ 1024:** **PASS**.
  - seq=1024: eager=77.3 MB vs chunked=33.3 MB ✓
  - seq=2048: eager=132.1 MB vs chunked=40.3 MB ✓
  - seq=4096: eager=249.6 MB vs chunked=63.5 MB ✓
  - seq=8192: eager=475.6 MB vs chunked=110.4 MB ✓

**Flat-VRAM claim NOT yet proven**. See the raw numbers above. Probable suspects:
- the runner's ring-buffer GPU input chunks (3-deep, size B×chunk×D) — these scale with B and D, NOT T, so should be flat across the sweep;
- the head's eager (B, T, D) input — this is the biggest known seq-proportional residual; chunked head-loss would be a future optimization;
- transient torch / cublas workspace allocations.
