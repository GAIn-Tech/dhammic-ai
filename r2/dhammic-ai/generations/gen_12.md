# Generation 12: Crossover of Gen 11 Winners

Winners: g11_d128_l4_lora32, g11_d128_l4_lr3e2, g11_d128_l5, g11_d128_l4_exp4, g11_d144_l4, g11_d128_l4_ds64, g11_d128_l4_eng4k, g11_d128_l4_ds32

## Mutations

| # | Name | Config Override |
|---|------|----------------|
| 1 | g12_elite1_d128_l4_lr1e-02 | `d_model=128, n_layers=4, d_state=16, n_heads=8, mamba_expand=3, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=32, lr=0.015` |
| 2 | g12_elite2_d128_l4_lr3e-02 | `d_model=128, n_layers=4, d_state=16, n_heads=8, mamba_expand=3, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.03` |
| 3 | g12_elite3_d128_l5_lr1e-02 | `d_model=128, n_layers=5, d_state=16, n_heads=8, mamba_expand=3, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 4 | g12_cross1_d128_l4_lr1e-02 | `d_model=128, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 5 | g12_cross2_d128_l4_lr1e-02_eng4096 | `d_model=128, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=4096, engram_cells_per_col=16, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 6 | g12_cross3_d128_l5_lr1e-02_eng4096 | `d_model=128, n_layers=5, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=4096, engram_cells_per_col=16, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 7 | g12_cross4_d128_l4_lr1e-02 | `d_model=128, n_layers=4, mamba_expand=4, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 8 | g12_cross5_d128_l4_lr1e-02_eng4096 | `d_model=128, n_layers=4, mamba_expand=4, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=4096, engram_cells_per_col=16, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 9 | g12_cross6_d144_l4_lr1e-02 | `d_model=144, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=2048, engram_cells_per_col=8, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
| 10 | g12_cross7_d144_l4_lr1e-02_eng4096 | `d_model=144, n_layers=4, mamba_expand=3, n_heads=8, d_state=16, sdr_dim=2048, sdr_k_active=40, engram_n_columns=4096, engram_cells_per_col=16, engram_k_active=20, engram_layer=3, lora_rank=16, lr=0.015` |
