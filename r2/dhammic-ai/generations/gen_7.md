# Generation 7: Crossover of Gen 6 Winners

Winners: g6_d160_l4_lr1e2, g6_d192_l4_lr2e2, g6_d256_l4_lr1e2, g6_d192_l6_lr1e2, g6_d192_l4_h16, g6_d128_l4_eng4k, g6_d192_l4_lr5e3, g6_d128_l4_lr5e3, g6_d128_l6_lr1e2, pretrain_d192_10k_elite, pretrain_champion_20k_elite

## Mutations

| # | Name | Config Override |
|---|------|----------------|
| 1 | g6_d160_l4_lr1e2_elite | `d_model=160, n_layers=4, d_state=16, n_heads=8, mamba_expand=3, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 2 | g6_d192_l4_lr2e2_elite | `d_model=192, n_layers=4, d_state=16, n_heads=8, mamba_expand=3, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.02` |
| 3 | g6_d256_l4_lr1e2_elite | `d_model=256, n_layers=4, d_state=16, n_heads=8, mamba_expand=3, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 4 | g6_d160_l4_lr1e2+g6_d192_l4_lr2e2 | `d_model=160, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.02` |
| 5 | g6_d160_l4_lr1e2+g6_d256_l4_lr1e2 | `d_model=160, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 6 | g6_d160_l4_lr1e2+g6_d192_l6_lr1e2 | `d_model=160, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 7 | g6_d160_l4_lr1e2+g6_d192_l4_h16 | `d_model=160, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 8 | g6_d160_l4_lr1e2+g6_d128_l4_eng4k | `d_model=160, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=4096, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 9 | g6_d160_l4_lr1e2+g6_d192_l4_lr5e3 | `d_model=160, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.005` |
| 10 | g6_d160_l4_lr1e2+g6_d128_l4_lr5e3 | `d_model=160, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.005` |
