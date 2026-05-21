# HTM Fidelity: What's Faithful, What's Simplified

This document records the design choices made in `src/htm_layer.py` /
`src/kernels/htm.py` for adapting Numenta's HTM into a differentiable layer
suitable for LM pretraining.

## Faithful to Numenta HTM

The following pieces match the published Numenta algorithm
(Hawkins & Ahmad 2016; BAMI book; `htm.core` C++/Python community fork):

| HTM concept | Where it lives | Reference |
|---|---|---|
| Potential-synapse permanence | `FaithfulHTM.sp_permanence` (n_columns × d_input) | `htm.core/py/htm/algorithms/spatial_pooler.py` — `_potentialPools` / `_permanences` |
| Column overlap | `_sp_overlap_fwd_kernel` (sum over d of gate(perm) × input) | `SpatialPooler.compute()` (overlap computation) |
| Boost factor homeostasis | `FaithfulHTM.boost` buffer + `_update_boost()` EMA; formula `exp(boost_strength · (target_density − duty_cycle))` | `SpatialPooler._updateBoostFactors` |
| k-Winners-Take-All over columns | `FaithfulHTM._kwta` (top-k + scatter mask) | `SpatialPooler._inhibitColumns` |
| Per-column cells (mini-column structure) | `n_cells = n_columns × cells_per_column` | `TemporalMemory` cell layout |
| Per-cell dendritic segments with synapses to a sample of other cells | `tm_permanence` (n_cells × n_segs × n_syns) + static `seg_idx` topology | `TemporalMemory._segments`, `Synapse.presynapticCell` |
| Predicted-cell scoring (segment activation) | `_tm_predict_fwd_kernel` — soft-max-pooled over segments | `TemporalMemory._activatePredictedColumn` predicted-cell detection |
| Predicted-vs-burst column dynamics | `forward()` — `active_cells = sp_gated × (any_pred · cell_pred + (1−any_pred) · uniform_burst)` | `TemporalMemory._burstColumn` / `_activatePredictedColumn` branching |
| Hebbian permanence increment direction | `_tm_learn_kernel` (`tm_hebbian_delta`) — `inc·prev − dec·(1−prev) − decay·perm` | `TemporalMemory._adaptSegment` |

## Simplified for differentiability

The differences from raw Numenta are explicitly the discrete-vs-continuous
gap, kept as small as possible:

| Numenta (discrete) | This implementation (continuous) | Why |
|---|---|---|
| `perm > threshold` step function | `sigmoid((perm − thr) / tau)` soft synapse | Allows gradient to flow through permanences so LM loss can train them. The same operator is used at train and eval — no Python dual-branch. |
| Hard argmax over segments to pick "max segment activation" | `logsumexp(β · seg_act) / β` (softmax pool with sharpness β) | Smooth max, gradient flows through *all* segments weighted by softmax mass. |
| Hard "any predicted in column → don't burst" branch | `any_pred = sigmoid(8·(max_col_pred − 0.5))` (sharp sigmoid), `burst = (1 − any_pred)/cells_per_col` | Same logical effect: when any cell scores > 0.5 the column suppresses bursting, but smooth so it's differentiable. |
| Discrete Hebbian permanence increment applied each step | Exposed as the kernel `tm_hebbian_delta`, but applied as an **auxiliary regulariser** when `hebbian_lr > 0`. The primary learning signal is the LM-loss gradient. | Combining hard Hebbian increments with SGD on the same tensor is unstable; this keeps the gradient path as the source of truth. |

## Tests that confirm fidelity

`tests/kernels/test_htm.py`:

- `test_sp_sparsity_exact` — k-WTA produces *exactly* `k_active_columns` active columns per row (matches HTM's hard sparsity guarantee).
- `test_sp_parity_fp32` — fused SP kernel matches eager sigmoid-overlap reference within `~3e-6` fp32 max abs error.
- `test_tm_parity_fp32` — fused TM predict kernel matches the eager softmax-pool-over-segments reference within `~5e-7` fp32.
- `test_tm_learns_repeating_sequence` — feeding a cyclic 3-token sequence and training with Adam → loss decreases (HTM is finding the repeat structure).
- `test_gradient_flow_all_params` — backward reaches `sp_permanence`, `tm_permanence`, and `cell_proj.weight` with finite, non-zero grads.
- `test_shape_2_256_128` — end-to-end on the requested (B,T,d) = (2,256,128) with n_columns=256, cells_per_column=8, k=20.
- `test_hebbian_delta_shape_and_sign` — Hebbian kernel emits `+inc` everywhere when pre=post=1, decay=0 (correct sign and magnitude).
- `test_htm_output_is_sparse` — actual HTM column mask is k-sparse end-to-end.

## References

- `htm.core` — github.com/htm-community/htm.core — community-maintained fork of Numenta's NuPIC. Key files studied:
  - `py/htm/algorithms/spatial_pooler.py`
  - `py/htm/algorithms/temporal_memory.py`
- Hawkins, J. & Ahmad, S. (2016) *Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex*. Frontiers in Neural Circuits.
- Numenta (2019) *Biological and Machine Intelligence (BAMI)* — public textbook on HTM theory.
- Cui, Y., Ahmad, S. & Hawkins, J. (2016) *Continuous Online Sequence Learning with an Unsupervised Neural Network Model*. Neural Computation.
