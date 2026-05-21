# Offload Audit — Fused Dhammic-AI Pipeline

- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- Probe config: batch=2, seq_len=2048, d_model=64, n_layers=2, vocab=1024, chunk_size=256
- Peak VRAM during fwd+bwd: **131.88 MB**
- Free / total VRAM at audit start: 4748 / 6144 MB

## Parameter (weight) memory by class

| class | bytes | MB | % of weights |
|:---|---:|---:|---:|
| W:emb | 425984 | 0.406 | 52.0% |
| W:linear | 385536 | 0.368 | 47.1% |
| W:other | 4128 | 0.004 | 0.5% |
| W:norm/bias | 2984 | 0.003 | 0.4% |
| W:ssm-param | 32 | 0.000 | 0.0% |
| **TOTAL** | 818664 | 0.781 | 100.0% |

## Top 15 parameter tensors by size

| name | bytes | MB |
|:---|---:|---:|
| `embedding.dense_embed.weight` | 262144 | 0.250 |
| `backbone.0.mlp.mlp_up.weight` | 81920 | 0.078 |
| `backbone.1.mlp.mlp_up.weight` | 81920 | 0.078 |
| `embedding.to_sdr.weight` | 65536 | 0.062 |
| `embedding.from_sdr.weight` | 65536 | 0.062 |
| `backbone.0.mlp.mlp_down.weight` | 40960 | 0.039 |
| `backbone.1.mlp.mlp_down.weight` | 40960 | 0.039 |
| `backbone.0.mamba.block.in_proj.weight` | 36352 | 0.035 |
| `backbone.1.mamba.block.in_proj.weight` | 36352 | 0.035 |
| `backbone.1.engram.gather.cell_embed.weight` | 32768 | 0.031 |
| `backbone.0.mamba.block.out_proj.weight` | 16384 | 0.016 |
| `backbone.1.mamba.block.out_proj.weight` | 16384 | 0.016 |
| `backbone.1.engram.gather.col_proj.weight` | 16384 | 0.016 |
| `backbone.1.engram.out_proj.weight` | 16384 | 0.016 |
| `backbone.1.hebbian.W_A` | 2048 | 0.002 |

## Top 20 live allocations (> 1 MB)

| MB | top frame |
|---:|:---|
| 8.12 | `<no frame>` |
| 8.12 | `<no frame>` |
| 8.00 | `torch::unwind::unwind() @ ??:0` |
| 8.00 | `torch::unwind::unwind() @ ??:0` |
| 1.00 | `torch::unwind::unwind() @ ??:0` |

## Optimisation opportunities

- SDR projections account for 0.12 MB (16.0% of weights). At larger vocab the dense_embed dwarfs this; at small vocab consider shrinking ``sdr_dim`` further.
- Peak (131.9 MB) is well under the 1.4 GB budget for the chosen probe shape; the budget is already comfortable at this config. The chunked-offload runtime is the right tool for longer-seq scaling rather than this probe.

