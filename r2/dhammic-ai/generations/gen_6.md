# Generation 6: Crossover of Gen 5 Winners

Winners: pretrain_d192_10k, pretrain_champion_20k

## Mutations

| # | Name | Config Override |
|---|------|----------------|
| 1 | pretrain_d192_10k_elite | `d_model=192, n_layers=4, d_state=16, mamba_expand=3, n_heads=8, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16` |
| 2 | pretrain_champion_20k_elite | `d_model=128, n_layers=4, d_state=16, mamba_expand=3, n_heads=8, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16` |
| 3 | pretrain_d192_10k+pretrain_champion_20k | `d_model=192, n_layers=4, d_state=16, mamba_expand=3, n_heads=8, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
