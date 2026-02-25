# Mamba-2 SSD Algorithm - Mathematical Pseudocode

Reference: Tri Dao and Albert Gu, 2024, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2).

## Inputs
- `x in R^{B x T x H}`: per-head input signal (here `H = n_heads`, or flattened `H = n_heads * head_dim` in implementation)
- `A in R^H`: continuous-time diagonal state matrix entries (negative for stability)
- `B in R^{B x T x H x N}`: input projection into state channels (`N = d_state`)
- `C in R^{B x T x H x N}`: readout projection from state channels
- `dt in R^{B x T x H}`: pre-activation step sizes
- `dt_bias in R^H` (optional): learned bias for `dt`
- `D in R^H`: skip/residual gain
- `chunk_len in N`: chunk size for chunked SSD scan

## State-Space Form
For each batch `b`, time `t`, head `h`, state channel `n`:

`s_{t,h,n} = exp(Delta_{t,h} A_h) s_{t-1,h,n} + Delta_{t,h} B_{t,h,n} x_{t,h}`

`y_{t,h} = sum_{n=1..N} C_{t,h,n} s_{t,h,n} + D_h x_{t,h}`

with

`Delta_{t,h} = softplus(dt_{t,h} + dt_bias_h)` (or `softplus(dt_{t,h})` if no bias).

Define

`dA_{t,h} = exp(Delta_{t,h} A_h)`,

`dB_{t,h,n} = Delta_{t,h} B_{t,h,n}`.

Then recurrence is

`s_t = dA_t odot s_{t-1} + dB_t odot x_t`,

where `odot` is elementwise multiplication with broadcast over `n`.

## Algorithm (Chunked Selective Scan)
Let sequence be partitioned into `K = ceil(T / chunk_len)` contiguous chunks `c = 0..K-1`, each covering indices `[t0, t1)`.

1. **Discretize (pointwise over `b,t,h`)**
   - `Delta = softplus(dt + dt_bias)`
   - `dA = exp(Delta * A)`
   - `dB = Delta[..., None] * B`

2. **Intra-chunk scan (local recurrence)**
   For each chunk `c`, given chunk-entry state `s_in^{(c)}`:
   - For `t in [t0, t1)`:
     - `s_t = dA_t odot s_{t-1} + dB_t odot x_t`
   - This is an associative scan over affine maps `(alpha, beta)` with composition
     - `(alpha2, beta2) circ (alpha1, beta1) = (alpha2 odot alpha1, beta2 + alpha2 odot beta1)`
     - where `alpha = dA`, `beta = dB odot x`

3. **Inter-chunk state propagation**
   Let chunk transition be
   - `s_out^{(c)} = Alpha^{(c)} odot s_in^{(c)} + Beta^{(c)}`
   where `(Alpha^{(c)}, Beta^{(c)})` is the composed affine map for chunk `c`.
   Then
   - `s_in^{(0)} = 0`
   - `s_in^{(c+1)} = s_out^{(c)}`
   Equivalent to a second scan over chunk-level affine transitions.

4. **Readout and skip**
   - `y_t = <C_t, s_t>_N + D odot x_t`
   - `(<C_t, s_t>_N)_{b,t,h} = sum_n C_{b,t,h,n} s_{b,t,h,n}`

## Outputs
- `y in R^{B x T x H}`

## Notes on Numerical Precision
- Reference/golden generation should use `float32` end-to-end.
- Stability relies on negative `A` and positive `Delta` after softplus.
- The sequential recurrence and chunked SSD are mathematically equivalent; chunking changes compute schedule, not the operator.
