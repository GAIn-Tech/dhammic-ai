# Scaling Sweep — Chunked Offload v2 (chunk-streaming LM head)

V2 of the chunked-offload sweep. Same architecture and (seq_len, batch) grid as `scaling_sweep_chunked.md`, but the LM head is `ChunkedLMHeadXent` so the head's input is streamed chunk-by-chunk and the kernel's per-chunk `d_weight` / `d_hidden_flat` transients are freed before the next chunk runs.

- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- VRAM total / free at start: 6144 / 5122 MB
- Architecture: d=64 l=2 ds=8 exp=2 heads=4 V=1024 chunk=256

## Sweep 1: seq_len at batch=2 — eager vs chunked v1 vs chunked v2

| seq_len | eager VRAM (MB) | chunked v1 (MB) | chunked v2 (MB) | v2/v2@256 | v2 wall (ms) | v2 tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 31.9 | 31.2 | 31.0 | 1.00x | 2264.5 | 226 |
| 512 | 46.2 | 32.5 | 32.2 | 1.04x | 2143.2 | 478 |
| 1024 | 77.3 | 33.3 | 32.8 | 1.06x | 1972.9 | 1038 |
| 2048 | 132.1 | 40.3 | 32.8 | 1.06x | 7668.1 | 534 |
| 4096 | 249.6 | 63.5 | 32.9 | 1.06x | 12991.3 | 631 |
| 8192 | 475.6 | 110.4 | 33.0 | 1.06x | 23193.0 | 706 |

## Sweep 2: batch at seq=2048 — eager vs chunked v1 vs chunked v2

| batch | eager VRAM (MB) | chunked v1 (MB) | chunked v2 (MB) | v2 wall (ms) | v2 tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 77.3 | 28.7 | 25.6 | 6376.9 | 321 |
| 2 | 132.1 | 40.3 | 32.8 | 7668.1 | 534 |
| 4 | 249.6 | 63.5 | 47.2 | 7833.1 | 1046 |

## Key assertions

- **Flat-VRAM (v2):** chunked-v2@8192 / chunked-v2@256 = **1.06x** (33.0 MB / 31.0 MB) — target ≤ 1.5x — **PASS**.
- For comparison: v1 ratio was **3.54x** (110.4 MB / 31.2 MB) — v2 improves the seq-scale residual by **3.33x**.
