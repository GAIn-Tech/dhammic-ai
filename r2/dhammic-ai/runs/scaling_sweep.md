# Scaling Sweep — Fused Dhammic-AI Pipeline

- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- VRAM total / free at start: 6144 / 5122 MB
- Architecture: d=64 l=2 ds=8 exp=2 heads=4 V=1024 chunk=256

## Sweep 1: seq_len at batch=2

| seq_len | peak VRAM (MB) | wall (ms) | tok/s | status |
|---:|---:|---:|---:|:---|
| 256 | 31.9 | 812.6 | 630 | OK |
| 512 | 46.2 | 368.0 | 2783 | OK |
| 1024 | 77.3 | 478.6 | 4280 | OK |
| 2048 | 132.1 | 589.1 | 6954 | OK |
| 2048 | 132.1 | 1232.6 | 3323 | OK |
| 4096 | 249.6 | 1135.1 | 7217 | OK |
| 8192 | 475.6 | 2085.4 | 7856 | OK |

## Sweep 2: batch at seq_len=2048

| batch | peak VRAM (MB) | wall (ms) | tok/s | status |
|---:|---:|---:|---:|:---|
| 1 | 77.3 | 947.8 | 2161 | OK |
| 2 | 132.1 | 589.1 | 6954 | OK |
| 2 | 132.1 | 1232.6 | 3323 | OK |
| 4 | 249.6 | 1391.2 | 5888 | OK |

## Key assertions

- VRAM ratio (seq=8192 / seq=256): **14.91x** (475.6 MB / 31.9 MB)
- Target: ratio ≤ 1.5x for chunked-offload to be "working" — observed 14.91x.
- VRAM batch=2 / batch=1 ratio: 1.71x (target ≈ 2x — roughly linear)
- VRAM batch=4 / batch=2 ratio: 1.89x (target ≈ 2x — roughly linear)
- Throughput seq=8192 vs seq=256: 7856 vs 630 tok/s (target: roughly flat as seq grows)
