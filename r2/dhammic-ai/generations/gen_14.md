# Generation 14: Crossover of Gen 13 Winners

Winners: g13_d512_l4_exp2, g13_d512_l4_exp3, g13_d768_l4, g13_d256_l8, g13_d384_l4_h16, g13_d512_l6, g13_d512_l4_ds32, g13_d384_l6, g13_d256_l12

## Mutations

| # | Name | Config Override |
|---|------|----------------|
| 1 | g14_elite1_d512_l4_lr8e-03 | `d_model=512, n_layers=4, d_state=16, n_heads=8, mamba_expand=2, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.008` |
| 2 | g14_elite2_d512_l4_lr8e-03 | `d_model=512, n_layers=4, d_state=16, n_heads=8, mamba_expand=3, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.008` |
| 3 | g14_elite3_d768_l4_lr5e-03 | `d_model=768, n_layers=4, d_state=16, n_heads=8, mamba_expand=2, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.005` |
| 4 | g14_cross1_d512_l4_lr5e-03 | `d_model=512, n_layers=4, mamba_expand=2, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.005` |
| 5 | g14_cross2_d512_l4_lr1e-02 | `d_model=512, n_layers=4, mamba_expand=2, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 6 | g14_cross3_d512_l4_lr5e-03 | `d_model=512, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.005` |
| 7 | g14_cross4_d512_l4_lr1e-02 | `d_model=512, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 8 | g14_cross5_d768_l4_lr1e-02 | `d_model=768, n_layers=4, mamba_expand=2, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 9 | g14_cross6_d768_l4_lr8e-03 | `d_model=768, n_layers=4, mamba_expand=2, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.008` |
| 10 | g14_cross7_d256_l8_lr1e-02 | `d_model=256, n_layers=8, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
