# Seq-coupling diff audit

Each allocation observed at peak fwd+bwd, grouped by (size, top frame).
If `total@8k / total@2k ~ 4x`, that allocation is seq-coupled.
If the ratio is ~1x, that allocation is constant-size in T.

| size_MB | count@2k | count@4k | count@8k | total@8k_MB | growth(8k/2k) | frame |
|---:|---:|---:|---:|---:|---:|:---|
| 8.520 | 2 | 2 | 2 | 17.04 | 1.00 | `torch::unwind::unwind()` |
| 16.780 | 0 | 0 | 1 | 16.78 | inf | `torch::unwind::unwind()` |
| 8.390 | 0 | 1 | 1 | 8.39 | inf | `torch::unwind::unwind()` |
| 2.100 | 1 | 0 | 2 | 4.20 | 2.00 | `torch::unwind::unwind()` |
| 0.130 | 2 | 2 | 4 | 0.52 | 2.00 | `torch::unwind::unwind()` |
| 0.520 | 2 | 0 | 0 | 0.00 | 0.00 | `torch::unwind::unwind()` |
| 4.190 | 1 | 1 | 0 | 0.00 | 0.00 | `torch::unwind::unwind()` |
| 0.070 | 0 | 2 | 0 | 0.00 | 0.00 | `torch::unwind::unwind()` |
| 1.050 | 0 | 2 | 0 | 0.00 | 0.00 | `torch::unwind::unwind()` |


## Peak

- seq=2048: peak=300.86 MB
- seq=4096: peak=62.00 MB
- seq=8192: peak=107.12 MB

Peak growth 8k/2k = 0.36x.
