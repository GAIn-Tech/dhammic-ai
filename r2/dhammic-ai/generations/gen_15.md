# Generation 15: Crossover of Gen 14 Winners

Winners: g14_elite1_d512_l4_lr8e-03, g14_cross1_d512_l4_lr5e-03, g14_cross2_d512_l4_lr1e-02, g14_elite2_d512_l4_lr8e-03, g14_cross7_d256_l8_lr1e-02, g14_cross4_d512_l4_lr1e-02, g14_cross5_d768_l4_lr1e-02, g14_cross6_d768_l4_lr8e-03

## Mutations

| # | Name | Config Override |
|---|------|----------------|
| 1 | g15_elite1_d512_l4_lr8e-03 | `d_model=512, n_layers=4, d_state=16, n_heads=8, mamba_expand=2, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.008` |
| 2 | g15_elite2_d512_l4_lr5e-03 | `d_model=512, n_layers=4, d_state=16, n_heads=8, mamba_expand=2, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.005` |
| 3 | g15_elite3_d512_l4_lr1e-02 | `d_model=512, n_layers=4, d_state=16, n_heads=8, mamba_expand=2, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 4 | g15_cross1_d512_l4_lr1e-02 | `d_model=512, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 5 | g15_cross2_d512_l4_lr8e-03 | `d_model=512, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.008` |
| 6 | g15_cross3_d256_l8_lr1e-02 | `d_model=256, n_layers=8, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.01` |
| 7 | g15_cross4_d256_l8_lr8e-03 | `d_model=256, n_layers=8, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.008` |
| 8 | g15_cross5_d768_l4_lr8e-03 | `d_model=768, n_layers=4, mamba_expand=2, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.008` |
