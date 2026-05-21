# Seq-coupling audit V2 — Python-stack frames

Each allocation observed at peak fwd+bwd after a warmup pass to absorb
Triton autotune compile buffers. Frames resolved to actual Python source.

## Peak by seq_len

- seq=2048: peak=39.84 MB
- seq=4096: peak=62.00 MB
- seq=8192: peak=107.12 MB

**Linear fit**: peak(T) ≈ 16.87 MB + 11016.00 bytes/token

## Top 30 allocations at seq=8192

| size_MB | @2k | @4k | @8k | total@8k_MB | growth | frame |
|---:|---:|---:|---:|---:|---:|:---|
| 8.520 | 2 | 2 | 2 | 17.04 | 1.00x | `<no frame>` |

## Identified seq-coupled allocations

None found above 1 MB threshold.
