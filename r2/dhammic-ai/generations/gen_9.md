# Generation 9: Crossover of Gen 8 Winners

Winners: g7_d128_l4_lr1.5e2+g7_d192_l4_lr1.5e2, g7_d128_l4_lr1.5e2+g7_d160_l4_lr1.5e2, g7_d128_l4_lr1.5e2+g7_d176_l4_lr1.5e2, g7_d128_l4_lr1.5e2+g7_d160_l4_ds32, g7_d128_l4_lr1.5e2+g7_d160_l4_lr8e3, g7_d128_l4_lr1.5e2_elite, g7_d160_l4_lr1.5e2_elite, g7_d128_l4_lr1.5e2+g7_d160_l4_eng4k, g7_d192_l4_lr1.5e2_elite

## Mutations

| # | Name | Config Override |
|---|------|----------------|
| 1 | g7_d128_l4_lr1.5e2+g7_d192_l4_lr1.5e2_elite | `d_model=128, n_layers=4, d_state=16, n_heads=8, mamba_expand=3, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 2 | g7_d128_l4_lr1.5e2+g7_d160_l4_lr1.5e2_elite | `d_model=128, n_layers=4, d_state=16, n_heads=8, mamba_expand=3, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 3 | g7_d128_l4_lr1.5e2+g7_d176_l4_lr1.5e2_elite | `d_model=128, n_layers=4, d_state=16, n_heads=8, mamba_expand=3, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 4 | g7_d128_l4_lr1.5e2+g7_d192_l4_lr1.5e2+g7_d128_l4_lr1.5e2+g7_d160_l4_lr1.5e2 | `d_model=128, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 5 | g7_d128_l4_lr1.5e2+g7_d192_l4_lr1.5e2+g7_d128_l4_lr1.5e2+g7_d176_l4_lr1.5e2 | `d_model=128, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 6 | g7_d128_l4_lr1.5e2+g7_d192_l4_lr1.5e2+g7_d128_l4_lr1.5e2+g7_d160_l4_ds32 | `d_model=128, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 7 | g7_d128_l4_lr1.5e2+g7_d192_l4_lr1.5e2+g7_d128_l4_lr1.5e2+g7_d160_l4_lr8e3 | `d_model=128, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.008` |
| 8 | g7_d128_l4_lr1.5e2+g7_d192_l4_lr1.5e2+g7_d128_l4_lr1.5e2_elite | `d_model=128, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 9 | g7_d128_l4_lr1.5e2+g7_d192_l4_lr1.5e2+g7_d160_l4_lr1.5e2_elite | `d_model=128, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 10 | g7_d128_l4_lr1.5e2+g7_d192_l4_lr1.5e2+g7_d128_l4_lr1.5e2+g7_d160_l4_eng4k | `d_model=128, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=4096, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
