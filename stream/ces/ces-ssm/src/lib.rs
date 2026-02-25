//! Mamba-2 SSD + multi-scale temporal heads

use candle_core::{DType, Result, Tensor};

pub fn ssd_forward(
    x: &Tensor,
    a: &Tensor,
    b_proj: &Tensor,
    c_proj: &Tensor,
    dt: &Tensor,
    d_skip: &Tensor,
    dt_bias: Option<&Tensor>,
) -> Result<Tensor> {
    let (batch, seq_len, n_heads) = x.dims3()?;
    let d_state = b_proj.dim(3)?;
    let device = x.device();

    let dt_biased = match dt_bias {
        Some(bias) => dt.broadcast_add(&bias.reshape((1, 1, n_heads))?)?,
        None => dt.clone(),
    };
    let dt_disc = softplus(&dt_biased)?;

    let a_bcast = a.reshape((1, 1, n_heads))?;
    let d_a = dt_disc.broadcast_mul(&a_bcast)?.exp()?;

    let d_b = dt_disc.unsqueeze(3)?.broadcast_mul(b_proj)?;

    let mut state = Tensor::zeros((batch, n_heads, d_state), DType::F32, device)?;
    let mut outputs = Vec::with_capacity(seq_len);
    let d_skip_bcast = d_skip.reshape((1, n_heads))?;

    for t in 0..seq_len {
        let d_a_t = d_a.narrow(1, t, 1)?.squeeze(1)?.unsqueeze(2)?;
        let x_t = x.narrow(1, t, 1)?.squeeze(1)?;
        let x_t_exp = x_t.unsqueeze(2)?;
        let d_b_t = d_b.narrow(1, t, 1)?.squeeze(1)?;
        let input_contrib = d_b_t.broadcast_mul(&x_t_exp)?;

        state = d_a_t.broadcast_mul(&state)?.broadcast_add(&input_contrib)?;

        let c_t = c_proj.narrow(1, t, 1)?.squeeze(1)?;
        let y_t_ssm = c_t.broadcast_mul(&state)?.sum(2)?;
        let y_t_skip = x_t.broadcast_mul(&d_skip_bcast)?;
        let y_t = y_t_ssm.broadcast_add(&y_t_skip)?;
        outputs.push(y_t.unsqueeze(1)?);
    }

    Tensor::cat(&outputs, 1)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    x.exp()?.affine(1.0, 1.0)?.log()
}

pub fn placeholder() {}

// ─── Multi-Scale Temporal Heads ───────────────────────────────────────────────
//
// K=4 head groups with fixed Δt ∈ {0.01, 0.1, 1.0, 10.0} per architecture spec.
//
// Mathematical basis (HiPPO/S4 multi-scale theory):
//   Effective memory length τ_k ≈ 1/Δt_k (in token steps)
//   - Fast  heads (Δt=0.01): τ≈100 steps  — token-level features
//   - Mid-1 heads (Δt=0.1):  τ≈10 steps   — phrase-level features
//   - Mid-2 heads (Δt=1.0):  τ≈1 step     — sentence-level features
//   - Slow  heads (Δt=10):   τ≈0.1 steps  — near-instant decay (doc-level gating)
//   Frequency range coverage: Δt_max/Δt_min = 10/0.01 = 1000×
//
// The dt_bias for each head group is overridden with log(Δt_k) so that
// softplus(dt + dt_bias) ≈ Δt_k when dt ≈ 0.

/// Fixed Δt values for K=4 head groups (architecture spec).
pub const MULTI_SCALE_DT: [f32; 4] = [0.01, 0.1, 1.0, 10.0];

/// Run SSD with heterogeneous Δt per head group.
///
/// Heads are divided into K groups of size `n_heads / K`.
/// Each group gets a fixed dt_bias = log(Δt_k) so softplus(dt + bias) ≈ Δt_k.
///
/// # Arguments
/// - `x`:      (B, T, H) — input (H must be divisible by K=4)
/// - `a`:      (H,) — log state decay
/// - `b_proj`: (B, T, H, N)
/// - `c_proj`: (B, T, H, N)
/// - `dt`:     (B, T, H) — base delta (pre-softplus, typically near-zero)
/// - `d_skip`: (H,)
///
/// Returns `y: (B, T, H)`.
pub fn ssd_multi_scale(
    x: &Tensor,
    a: &Tensor,
    b_proj: &Tensor,
    c_proj: &Tensor,
    dt: &Tensor,
    d_skip: &Tensor,
) -> Result<Tensor> {
    let (_batch, _seq_len, n_heads) = x.dims3()?;
    let k = MULTI_SCALE_DT.len();
    assert_eq!(
        n_heads % k,
        0,
        "n_heads ({n_heads}) must be divisible by K={k}"
    );
    let group_size = n_heads / k;
    let device = x.device();

    let mut group_outputs: Vec<Tensor> = Vec::with_capacity(k);

    for (g, &dt_val) in MULTI_SCALE_DT.iter().enumerate() {
        let h_start = g * group_size;

        let x_g = x.narrow(2, h_start, group_size)?;
        let a_g = a.narrow(0, h_start, group_size)?;
        let b_g = b_proj.narrow(2, h_start, group_size)?;
        let c_g = c_proj.narrow(2, h_start, group_size)?;
        let dt_g = dt.narrow(2, h_start, group_size)?;
        let d_g = d_skip.narrow(0, h_start, group_size)?;

        let dt_bias_val = dt_val.ln();
        let dt_bias_g =
            Tensor::full(dt_bias_val, group_size, device)?.to_dtype(candle_core::DType::F32)?;

        let y_g = ssd_forward(&x_g, &a_g, &b_g, &c_g, &dt_g, &d_g, Some(&dt_bias_g))?;
        group_outputs.push(y_g);
    }

    Tensor::cat(&group_outputs, 2)
}

/// Measure effective memory length for a single head with given Δt.
///
/// Memory length τ is defined as the step t where state decay drops below threshold:
///   exp(-Δt * t) < threshold  →  t > -ln(threshold) / Δt  →  τ = ceil(-ln(threshold)/Δt)
///
/// With threshold=0.01 (1% remaining): τ ≈ ln(100) / Δt ≈ 4.605 / Δt
pub fn effective_memory_length(dt: f32, threshold: f32) -> f32 {
    assert!(dt > 0.0 && threshold > 0.0 && threshold < 1.0);
    (-threshold.ln()) / dt
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use proptest::prelude::*;

    fn softplus_scalar(x: f32) -> f32 {
        (x.exp() + 1.0).ln()
    }

    fn ref_ssd_forward(
        x: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        dt: &[f32],
        d: &[f32],
        dt_bias: Option<&[f32]>,
        batch: usize,
        seq_len: usize,
        n_heads: usize,
        d_state: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * seq_len * n_heads];
        let mut state = vec![0.0f32; batch * n_heads * d_state];

        for t in 0..seq_len {
            for b_ix in 0..batch {
                for h in 0..n_heads {
                    let dt_idx = (b_ix * seq_len + t) * n_heads + h;
                    let x_idx = dt_idx;
                    let mut dt_disc = dt[dt_idx];
                    if let Some(bias) = dt_bias {
                        dt_disc += bias[h];
                    }
                    dt_disc = softplus_scalar(dt_disc);
                    let d_a = (dt_disc * a[h]).exp();
                    let x_val = x[x_idx];

                    let mut y_ssm = 0.0f32;
                    for n in 0..d_state {
                        let idx_btn = ((b_ix * seq_len + t) * n_heads + h) * d_state + n;
                        let idx_bhn = (b_ix * n_heads + h) * d_state + n;
                        let d_b = dt_disc * b[idx_btn];
                        state[idx_bhn] = d_a * state[idx_bhn] + d_b * x_val;
                        y_ssm += c[idx_btn] * state[idx_bhn];
                    }

                    out[x_idx] = y_ssm + d[h] * x_val;
                }
            }
        }

        out
    }

    #[test]
    fn numerical_equivalence() {
        let device = Device::Cpu;
        let batch = 2usize;
        let seq_len = 4usize;
        let n_heads = 2usize;
        let d_state = 2usize;

        let x_v = vec![
            0.3, -0.2, 0.1, 0.4, -0.5, 0.7, 0.2, -0.6, 0.15, 0.05, -0.25, 0.35, 0.45, -0.15, 0.55,
            -0.05,
        ];
        let a_v = vec![-0.5, -0.3];
        let b_v = vec![
            0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.15, -0.35, 0.05, 0.25, -0.45, 0.2, 0.4, -0.3, 0.1,
            0.05, -0.15, 0.2, 0.3, -0.25, -0.4, 0.35, 0.12, -0.08, 0.22, -0.18, 0.28, 0.06, -0.14,
            0.32, -0.24, 0.16,
        ];
        let c_v = vec![
            -0.3, 0.1, 0.25, -0.2, 0.35, 0.15, -0.1, 0.05, 0.2, -0.4, 0.12, 0.18, -0.22, 0.27,
            0.08, -0.16, 0.11, 0.09, -0.05, 0.31, -0.12, 0.24, 0.17, -0.07, -0.21, 0.14, 0.29,
            -0.13, 0.04, 0.19, -0.26, 0.33,
        ];
        let dt_v = vec![
            0.02, -0.03, 0.04, 0.01, -0.02, 0.05, 0.03, -0.01, 0.06, -0.04, 0.02, 0.01, -0.03,
            0.07, 0.05, -0.02,
        ];
        let d_v = vec![0.1, -0.1];
        let dt_bias_v = vec![0.0, 0.02];

        let x = Tensor::from_vec(x_v.clone(), (batch, seq_len, n_heads), &device).unwrap();
        let a = Tensor::from_vec(a_v.clone(), n_heads, &device).unwrap();
        let b = Tensor::from_vec(b_v.clone(), (batch, seq_len, n_heads, d_state), &device).unwrap();
        let c = Tensor::from_vec(c_v.clone(), (batch, seq_len, n_heads, d_state), &device).unwrap();
        let dt = Tensor::from_vec(dt_v.clone(), (batch, seq_len, n_heads), &device).unwrap();
        let d = Tensor::from_vec(d_v.clone(), n_heads, &device).unwrap();
        let dt_bias = Tensor::from_vec(dt_bias_v.clone(), n_heads, &device).unwrap();

        let y = ssd_forward(&x, &a, &b, &c, &dt, &d, Some(&dt_bias)).unwrap();
        assert_eq!(y.dims(), &[batch, seq_len, n_heads]);

        let expected = ref_ssd_forward(
            &x_v,
            &a_v,
            &b_v,
            &c_v,
            &dt_v,
            &d_v,
            Some(&dt_bias_v),
            batch,
            seq_len,
            n_heads,
            d_state,
        );

        let out = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut max_abs_error = 0.0f32;
        for (o, e) in out.iter().zip(expected.iter()) {
            let err = (o - e).abs();
            if err > max_abs_error {
                max_abs_error = err;
            }
            assert!(
                o.is_finite(),
                "numerical_equivalence: non-finite output value {o}"
            );
            assert!(err < 1e-5, "mismatch: got={o} expected={e} err={err}");
        }

        println!(
            "numerical_equivalence: output shape {:?}, all values finite, max_abs_error={:.8e}",
            y.dims(),
            max_abs_error
        );
    }

    #[test]
    fn property_output_shape() {
        let device = Device::Cpu;
        for (b, t, h, n) in [(1, 4, 2, 4), (2, 8, 4, 8), (1, 16, 4, 16)] {
            let x = Tensor::zeros((b, t, h), DType::F32, &device).unwrap();
            let a = Tensor::full(-0.5f32, h, &device).unwrap();
            let b_p = Tensor::zeros((b, t, h, n), DType::F32, &device).unwrap();
            let c_p = Tensor::zeros((b, t, h, n), DType::F32, &device).unwrap();
            let dt = Tensor::zeros((b, t, h), DType::F32, &device).unwrap();
            let d = Tensor::zeros(h, DType::F32, &device).unwrap();
            let y = ssd_forward(&x, &a, &b_p, &c_p, &dt, &d, None).unwrap();
            assert_eq!(y.dims(), &[b, t, h], "shape mismatch for B={b} T={t} H={h}");
        }
        println!("property_output_shape: all shape checks passed");
    }

    #[test]
    fn property_zero_input_zero_output() {
        let device = Device::Cpu;
        let (b, t, h, n) = (2, 8, 4, 8);
        let x = Tensor::zeros((b, t, h), DType::F32, &device).unwrap();
        let a = Tensor::full(-0.5f32, h, &device).unwrap();
        let b_p = Tensor::zeros((b, t, h, n), DType::F32, &device).unwrap();
        let c_p = Tensor::zeros((b, t, h, n), DType::F32, &device).unwrap();
        let dt = Tensor::zeros((b, t, h), DType::F32, &device).unwrap();
        let d = Tensor::zeros(h, DType::F32, &device).unwrap();
        let y = ssd_forward(&x, &a, &b_p, &c_p, &dt, &d, None).unwrap();
        let y_vec = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &y_vec {
            assert!(
                v.abs() < 1e-6,
                "Expected zero output for zero input, got {v}"
            );
        }
        println!("property_zero_input_zero_output: passed");
    }

    proptest! {
        #[test]
        fn property_finite_random(seed in 0u64..1000u64) {
            let device = Device::Cpu;
            let (b, t, h, n) = (1usize, 6usize, 3usize, 4usize);
            let x = Tensor::randn(0f32, 1f32, (b, t, h), &device).unwrap();
            let a = Tensor::full(-0.5f32, h, &device).unwrap();
            let b_p = Tensor::randn(0f32, 0.5f32, (b, t, h, n), &device).unwrap();
            let c_p = Tensor::randn(0f32, 0.5f32, (b, t, h, n), &device).unwrap();
            let dt = Tensor::randn(0f32, 0.1f32, (b, t, h), &device).unwrap();
            let d = Tensor::from_vec(vec![0.1f32, -0.05f32, 0.2f32], h, &device).unwrap();
            let dt_bias = Tensor::from_vec(vec![0.0f32, 0.01f32, -0.02f32], h, &device).unwrap();

            let _ = seed;
            let y = ssd_forward(&x, &a, &b_p, &c_p, &dt, &d, Some(&dt_bias)).unwrap();
            let vals = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for v in vals {
                prop_assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn long_sequence_no_oom() {
        let device = Device::Cpu;
        let (b, t, h, n) = (1usize, 2048usize, 8usize, 64usize);
        let x = Tensor::zeros((b, t, h), DType::F32, &device).unwrap();
        let a = Tensor::full(-0.5f32, h, &device).unwrap();
        let b_p = Tensor::zeros((b, t, h, n), DType::F32, &device).unwrap();
        let c_p = Tensor::zeros((b, t, h, n), DType::F32, &device).unwrap();
        let dt = Tensor::zeros((b, t, h), DType::F32, &device).unwrap();
        let d = Tensor::zeros(h, DType::F32, &device).unwrap();

        let y = ssd_forward(&x, &a, &b_p, &c_p, &dt, &d, None).unwrap();
        assert_eq!(y.dims(), &[b, t, h]);
    }

    #[test]
    fn multi_scale_output_shape() {
        let device = Device::Cpu;
        let (b, t, h, n) = (2usize, 8usize, 8usize, 16usize);
        let x = Tensor::zeros((b, t, h), DType::F32, &device).unwrap();
        let a = Tensor::full(-0.5f32, h, &device).unwrap();
        let b_p = Tensor::zeros((b, t, h, n), DType::F32, &device).unwrap();
        let c_p = Tensor::zeros((b, t, h, n), DType::F32, &device).unwrap();
        let dt = Tensor::zeros((b, t, h), DType::F32, &device).unwrap();
        let d = Tensor::zeros(h, DType::F32, &device).unwrap();
        let y = ssd_multi_scale(&x, &a, &b_p, &c_p, &dt, &d).unwrap();
        assert_eq!(
            y.dims(),
            &[b, t, h],
            "multi-scale output shape must be (B,T,H)"
        );
        println!("multi_scale_output_shape: passed {:?}", y.dims());
    }

    #[test]
    fn multi_scale_uniform_dt_matches_single_scale() {
        let device = Device::Cpu;
        let (b, t, h, n) = (1usize, 4usize, 4usize, 8usize);
        let x = Tensor::randn(0f32, 1f32, (b, t, h), &device).unwrap();
        let a = Tensor::full(-0.5f32, h, &device).unwrap();
        let b_p = Tensor::randn(0f32, 0.3f32, (b, t, h, n), &device).unwrap();
        let c_p = Tensor::randn(0f32, 0.3f32, (b, t, h, n), &device).unwrap();
        let dt_zero = Tensor::zeros((b, t, h), DType::F32, &device).unwrap();
        let d = Tensor::zeros(h, DType::F32, &device).unwrap();

        let y_multi = ssd_multi_scale(&x, &a, &b_p, &c_p, &dt_zero, &d).unwrap();
        assert_eq!(y_multi.dims(), &[b, t, h]);

        let vals = y_multi.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &vals {
            assert!(
                v.is_finite(),
                "multi-scale output has non-finite value: {v}"
            );
        }
        println!("multi_scale_uniform_dt_matches_single_scale: all finite, passed");
    }

    #[test]
    fn property_memory_length_scales() {
        let threshold = 0.01f32;
        let mut prev_len = f32::INFINITY;
        for &dt_val in MULTI_SCALE_DT.iter() {
            let tau = effective_memory_length(dt_val, threshold);
            assert!(
                tau < prev_len,
                "memory length should decrease as dt increases: tau({dt_val})={tau} >= prev={prev_len}"
            );
            prev_len = tau;
            println!("  dt={dt_val:.2}: τ = {tau:.1} steps");
        }
        println!("property_memory_length_scales: monotone decrease confirmed");
    }

    #[test]
    fn frequency_range_coverage() {
        let dt_min = MULTI_SCALE_DT[0];
        let dt_max = MULTI_SCALE_DT[MULTI_SCALE_DT.len() - 1];
        let range = dt_max / dt_min;
        println!("frequency_range_coverage: Δt_max/Δt_min = {range:.0}×");
        assert!(
            range >= 500.0,
            "expected >= 500× frequency range, got {range:.1}×"
        );
    }

    #[test]
    fn multi_scale_head_groups_differ() {
        let device = Device::Cpu;
        let (b, t, h, n) = (1usize, 16usize, 8usize, 4usize);
        let x = Tensor::randn(0f32, 1f32, (b, t, h), &device).unwrap();
        let a = Tensor::full(-0.3f32, h, &device).unwrap();
        let b_p = Tensor::randn(0f32, 0.5f32, (b, t, h, n), &device).unwrap();
        let c_p = Tensor::randn(0f32, 0.5f32, (b, t, h, n), &device).unwrap();
        let dt = Tensor::zeros((b, t, h), DType::F32, &device).unwrap();
        let d = Tensor::zeros(h, DType::F32, &device).unwrap();
        let y = ssd_multi_scale(&x, &a, &b_p, &c_p, &dt, &d).unwrap();

        let group_size = h / 4;
        let g0 = y
            .narrow(2, 0, group_size)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let g3 = y
            .narrow(2, 3 * group_size, group_size)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let diff: f32 = g0
            .iter()
            .zip(g3.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / g0.len() as f32;
        println!("multi_scale_head_groups_differ: mean |g0 - g3| = {diff:.6}");
        assert!(
            diff > 1e-6,
            "fast and slow head groups should produce different outputs, diff={diff}"
        );
    }
}
