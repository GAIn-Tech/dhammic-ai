//! Cortical Column MoE routing
//!
//! Top-k routing (k=2 of n=8 experts) with voting-based confidence weighting.
//! Each expert computes a forward pass and emits a confidence scalar.
//! Final output = weighted average of top-k expert outputs, weights = softmax(confidence scores).
//!
//! Architecture: n_experts=8, top_k=2, d_model=768, d_expert=384

use thiserror::Error;

// ─── Error Types ─────────────────────────────────────────────────────────────

#[derive(Error, Debug)]
pub enum MoeError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

// ─── Math Helpers ────────────────────────────────────────────────────────────

/// Deterministic pseudo-random init (Lehmer LCG, same pattern as ces-hebbian).
fn lehmer_rng(mut s: u64) -> u32 {
    s = s.wrapping_mul(0x5DEECE66Du64).wrapping_add(0xBu64) & ((1u64 << 32) - 1);
    s as u32
}

/// Row-major matrix-vector multiply: M (rows×cols) · x (cols) → y (rows).
pub fn mat_vec_mul(m: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(m.len(), rows * cols);
    assert_eq!(x.len(), cols);
    let mut y = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += m[r * cols + c] * x[c];
        }
        y[r] = acc;
    }
    y
}

/// Element-wise ReLU.
pub fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// Dot product of two vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// Softmax over only the provided values (top-k scores).
/// Numerically stable via max subtraction.
pub fn softmax_topk(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-12 {
        // Uniform fallback for degenerate case
        let n = scores.len() as f32;
        return vec![1.0 / n; scores.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// L2 norm of a vector.
pub fn vec_norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

// ─── ColumnExpert ────────────────────────────────────────────────────────────

/// A single expert in the cortical column.
///
/// Forward: `h = relu(W·x + timescale_bias)`
/// Confidence: `dot(timescale_vec, x).tanh().abs()`
#[derive(Clone, Debug)]
pub struct ColumnExpert {
    pub d_in: usize,
    pub d_out: usize,
    pub timescale_bias: f32,
    /// Weight matrix W: d_out × d_in (row-major)
    pub weight: Vec<f32>,
    /// Timescale weight vector: length d_in
    pub timescale_vec: Vec<f32>,
}

impl ColumnExpert {
    pub fn new(d_in: usize, d_out: usize, seed: u64) -> Self {
        let inv_sqrt = 1.0 / (d_in as f32).sqrt();
        let weight: Vec<f32> = (0..d_out * d_in)
            .map(|i| {
                let v = lehmer_rng(seed + i as u64 + 1) as f32 / u32::MAX as f32 * 2.0 - 1.0;
                v * inv_sqrt
            })
            .collect();
        let offset = (d_out * d_in) as u64;
        let timescale_vec: Vec<f32> = (0..d_in)
            .map(|i| {
                let v =
                    lehmer_rng(seed + offset + i as u64 + 1) as f32 / u32::MAX as f32 * 2.0 - 1.0;
                v * inv_sqrt
            })
            .collect();
        let bias_raw = lehmer_rng(seed + offset + d_in as u64 + 1) as f32 / u32::MAX as f32;
        let timescale_bias = (bias_raw - 0.5) * 0.1; // Small bias in [-0.05, 0.05]

        Self {
            d_in,
            d_out,
            timescale_bias,
            weight,
            timescale_vec,
        }
    }

    /// Forward pass: h = relu(W·x + timescale_bias)
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.d_in);
        let wx = mat_vec_mul(&self.weight, x, self.d_out, self.d_in);
        wx.iter()
            .map(|&v| (v + self.timescale_bias).max(0.0))
            .collect()
    }

    /// Confidence score: dot(timescale_vec, x).tanh().abs()
    ///
    /// Measures how well the expert's frequency/timescale matches the current input.
    pub fn confidence(&self, x: &[f32]) -> f32 {
        assert_eq!(x.len(), self.d_in);
        dot(&self.timescale_vec, x).tanh().abs()
    }
}

// ─── LoadTracker ─────────────────────────────────────────────────────────────

/// Tracks expert activation counts for load balancing.
///
/// Computes coefficient of variation (CV = std/mean) across experts.
/// Lower CV = more balanced routing.
#[derive(Clone, Debug)]
pub struct LoadTracker {
    pub counts: Vec<u64>,
    pub n_experts: usize,
}

impl LoadTracker {
    pub fn new(n_experts: usize) -> Self {
        Self {
            counts: vec![0u64; n_experts],
            n_experts,
        }
    }

    /// Record that experts at given indices were activated.
    pub fn record(&mut self, expert_indices: &[usize]) {
        for &idx in expert_indices {
            if idx < self.n_experts {
                self.counts[idx] += 1;
            }
        }
    }

    /// Coefficient of variation: std / mean. Lower = more balanced.
    pub fn coefficient_of_variation(&self) -> f32 {
        let n = self.n_experts as f32;
        let mean = self.counts.iter().sum::<u64>() as f32 / n;
        if mean < 1e-12 {
            return 0.0;
        }
        let var = self
            .counts
            .iter()
            .map(|&c| {
                let diff = c as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / n;
        var.sqrt() / mean
    }

    pub fn total_activations(&self) -> u64 {
        self.counts.iter().sum()
    }

    pub fn reset(&mut self) {
        self.counts.fill(0);
    }
}

// ─── CorticalColumnMoE ──────────────────────────────────────────────────────

/// Cortical Column Mixture-of-Experts with voting-based routing.
///
/// Architecture: n_experts=8, top_k=2, d_model=768, d_expert=384
///
/// Routing mechanism:
/// 1. Each expert computes confidence = dot(timescale_vec, x).tanh().abs()
/// 2. Top-k experts are selected by confidence
/// 3. Selected experts run forward pass: h = relu(W·x + bias)
/// 4. Expert hidden outputs are projected back to d_model
/// 5. Final output = weighted average using softmax(top-k confidences)
#[derive(Clone, Debug)]
pub struct CorticalColumnMoE {
    pub n_experts: usize,
    pub top_k: usize,
    pub d_model: usize,
    pub d_expert: usize,
    pub experts: Vec<ColumnExpert>,
    /// Output projection per expert: d_model × d_expert (row-major)
    pub output_projections: Vec<Vec<f32>>,
    pub load_tracker: LoadTracker,
}

impl CorticalColumnMoE {
    pub fn new(
        n_experts: usize,
        top_k: usize,
        d_model: usize,
        d_expert: usize,
    ) -> Result<Self, MoeError> {
        if top_k > n_experts {
            return Err(MoeError::InvalidConfig(format!(
                "top_k ({top_k}) must be <= n_experts ({n_experts})"
            )));
        }
        if top_k == 0 || n_experts == 0 || d_model == 0 || d_expert == 0 {
            return Err(MoeError::InvalidConfig(
                "all dimensions must be > 0".to_string(),
            ));
        }

        let experts: Vec<ColumnExpert> = (0..n_experts)
            .map(|i| ColumnExpert::new(d_model, d_expert, (i as u64 + 1) * 10000))
            .collect();

        let inv_sqrt = 1.0 / (d_expert as f32).sqrt();
        let output_projections: Vec<Vec<f32>> = (0..n_experts)
            .map(|i| {
                let seed_base = (i as u64 + 1) * 20000;
                (0..d_model * d_expert)
                    .map(|j| {
                        let v = lehmer_rng(seed_base + j as u64 + 1) as f32 / u32::MAX as f32 * 2.0
                            - 1.0;
                        v * inv_sqrt
                    })
                    .collect()
            })
            .collect();

        Ok(Self {
            n_experts,
            top_k,
            d_model,
            d_expert,
            experts,
            output_projections,
            load_tracker: LoadTracker::new(n_experts),
        })
    }

    /// Default configuration: n_experts=8, top_k=2, d_model=768, d_expert=384
    pub fn default_config() -> Result<Self, MoeError> {
        Self::new(8, 2, 768, 384)
    }

    /// Forward pass through the MoE.
    ///
    /// 1. Compute confidence scores for all experts
    /// 2. Select top-k experts by confidence
    /// 3. Run forward on selected experts
    /// 4. Project expert outputs back to d_model
    /// 5. Weighted average using softmax(top-k confidences)
    pub fn forward(&mut self, x: &[f32]) -> Result<Vec<f32>, MoeError> {
        if x.len() != self.d_model {
            return Err(MoeError::DimensionMismatch {
                expected: self.d_model,
                got: x.len(),
            });
        }

        // 1. Compute all expert confidences
        let confidences: Vec<f32> = self.experts.iter().map(|e| e.confidence(x)).collect();

        // 2. Select top-k experts
        let top_k_indices = self.select_top_k(&confidences);

        // 3. Record for load balancing
        self.load_tracker.record(&top_k_indices);

        // 4. Get top-k confidence scores and compute softmax weights
        let top_k_scores: Vec<f32> = top_k_indices.iter().map(|&i| confidences[i]).collect();
        let weights = softmax_topk(&top_k_scores);

        // 5. Forward through selected experts and weighted average
        let mut output = vec![0.0f32; self.d_model];
        for (rank, &expert_idx) in top_k_indices.iter().enumerate() {
            let expert_hidden = self.experts[expert_idx].forward(x);
            let expert_output = mat_vec_mul(
                &self.output_projections[expert_idx],
                &expert_hidden,
                self.d_model,
                self.d_expert,
            );
            let w = weights[rank];
            for (o, &e) in output.iter_mut().zip(expert_output.iter()) {
                *o += w * e;
            }
        }

        Ok(output)
    }

    /// Forward pass that also returns the selected expert indices and weights.
    pub fn forward_with_info(
        &mut self,
        x: &[f32],
    ) -> Result<(Vec<f32>, Vec<usize>, Vec<f32>), MoeError> {
        if x.len() != self.d_model {
            return Err(MoeError::DimensionMismatch {
                expected: self.d_model,
                got: x.len(),
            });
        }

        let confidences: Vec<f32> = self.experts.iter().map(|e| e.confidence(x)).collect();
        let top_k_indices = self.select_top_k(&confidences);
        self.load_tracker.record(&top_k_indices);

        let top_k_scores: Vec<f32> = top_k_indices.iter().map(|&i| confidences[i]).collect();
        let weights = softmax_topk(&top_k_scores);

        let mut output = vec![0.0f32; self.d_model];
        for (rank, &expert_idx) in top_k_indices.iter().enumerate() {
            let expert_hidden = self.experts[expert_idx].forward(x);
            let expert_output = mat_vec_mul(
                &self.output_projections[expert_idx],
                &expert_hidden,
                self.d_model,
                self.d_expert,
            );
            let w = weights[rank];
            for (o, &e) in output.iter_mut().zip(expert_output.iter()) {
                *o += w * e;
            }
        }

        Ok((output, top_k_indices, weights))
    }

    /// Select top-k expert indices by confidence score (descending).
    fn select_top_k(&self, confidences: &[f32]) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = confidences
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(self.top_k).map(|(i, _)| *i).collect()
    }

    /// Get individual expert outputs (projected to d_model) for verification.
    pub fn expert_outputs(&self, x: &[f32]) -> Vec<Vec<f32>> {
        self.experts
            .iter()
            .enumerate()
            .map(|(i, expert)| {
                let hidden = expert.forward(x);
                mat_vec_mul(
                    &self.output_projections[i],
                    &hidden,
                    self.d_model,
                    self.d_expert,
                )
            })
            .collect()
    }

    /// Get the load balance coefficient of variation.
    pub fn load_balance_cv(&self) -> f32 {
        self.load_tracker.coefficient_of_variation()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a pseudo-random input vector using xorshift.
    fn make_input(seed: u64, dim: usize) -> Vec<f32> {
        let mut s = seed.wrapping_add(1);
        (0..dim)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    // ─── Column routing tests ─────────────────────────────────────────────

    #[test]
    fn column_routing_forward_shape() {
        let mut moe = CorticalColumnMoE::new(8, 2, 768, 384).unwrap();
        let x = make_input(42, 768);
        let out = moe.forward(&x).unwrap();
        assert_eq!(out.len(), 768, "output dim must match d_model");
    }

    #[test]
    fn column_routing_top_k_selected() {
        let mut moe = CorticalColumnMoE::new(8, 2, 64, 32).unwrap();
        let x = make_input(99, 64);
        let (_, indices, weights) = moe.forward_with_info(&x).unwrap();
        assert_eq!(indices.len(), 2, "exactly k=2 experts must be selected");
        assert_eq!(weights.len(), 2, "exactly k=2 weights");
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn column_routing_different_inputs() {
        let mut moe = CorticalColumnMoE::new(8, 2, 64, 32).unwrap();
        let mut different_count = 0;
        for seed in 0..20u64 {
            let xa = make_input(seed * 1000, 64);
            let xb = make_input(seed * 1000 + 500, 64);
            let (_, ia, _) = moe.forward_with_info(&xa).unwrap();
            let (_, ib, _) = moe.forward_with_info(&xb).unwrap();
            if ia != ib {
                different_count += 1;
            }
        }
        assert!(
            different_count > 5,
            "at least 25% of input pairs should route differently, got {different_count}/20"
        );
    }

    #[test]
    fn column_routing_dimension_mismatch() {
        let mut moe = CorticalColumnMoE::new(8, 2, 64, 32).unwrap();
        let x_wrong = vec![0.1f32; 32];
        let result = moe.forward(&x_wrong);
        assert!(result.is_err(), "should error on dimension mismatch");
    }

    #[test]
    fn column_routing_top_k_validation() {
        let result = CorticalColumnMoE::new(4, 5, 64, 32);
        assert!(result.is_err(), "top_k > n_experts should fail");
    }

    #[test]
    fn column_routing_output_finite() {
        let mut moe = CorticalColumnMoE::new(8, 2, 128, 64).unwrap();
        for seed in 0..50u64 {
            let x = make_input(seed * 777, 128);
            let out = moe.forward(&x).unwrap();
            assert!(
                out.iter().all(|v| v.is_finite()),
                "all outputs must be finite (seed={seed})"
            );
        }
    }

    #[test]
    fn column_routing_default_config() {
        let moe = CorticalColumnMoE::default_config().unwrap();
        assert_eq!(moe.n_experts, 8);
        assert_eq!(moe.top_k, 2);
        assert_eq!(moe.d_model, 768);
        assert_eq!(moe.d_expert, 384);
    }

    // ─── Load balance tests ──────────────────────────────────────────────

    #[test]
    fn load_balance_cv_below_threshold() {
        let mut moe = CorticalColumnMoE::new(8, 2, 64, 32).unwrap();
        for seed in 0..1000u64 {
            let x = make_input(seed * 31337, 64);
            let _ = moe.forward(&x).unwrap();
        }
        let cv = moe.load_balance_cv();
        println!("load_balance_cv: {cv:.4}");
        println!("counts: {:?}", moe.load_tracker.counts);
        assert!(
            cv < 0.3,
            "coefficient of variation must be < 0.3, got {cv:.4}"
        );
    }

    #[test]
    fn load_tracker_counts_experts() {
        let mut tracker = LoadTracker::new(4);
        tracker.record(&[0, 1]);
        tracker.record(&[1, 2]);
        tracker.record(&[0, 3]);
        assert_eq!(tracker.counts, vec![2, 2, 1, 1]);
        assert_eq!(tracker.total_activations(), 6);
    }

    #[test]
    fn load_tracker_cv_uniform() {
        let mut tracker = LoadTracker::new(4);
        for _ in 0..10 {
            tracker.record(&[0, 1, 2, 3]);
        }
        let cv = tracker.coefficient_of_variation();
        assert!(cv < 1e-6, "perfectly uniform should have CV ~ 0, got {cv}");
    }

    #[test]
    fn load_tracker_reset() {
        let mut tracker = LoadTracker::new(4);
        tracker.record(&[0, 1]);
        tracker.reset();
        assert_eq!(tracker.counts, vec![0, 0, 0, 0]);
    }

    // ─── Voting confidence tests ─────────────────────────────────────────

    #[test]
    fn voting_confidence_nonzero() {
        let moe = CorticalColumnMoE::new(8, 2, 64, 32).unwrap();
        let x = make_input(42, 64);
        let confidences: Vec<f32> = moe.experts.iter().map(|e| e.confidence(&x)).collect();
        let all_nonzero = confidences.iter().all(|&c| c > 0.0);
        assert!(
            all_nonzero,
            "non-zero input must produce non-zero confidences: {confidences:?}"
        );
    }

    #[test]
    fn voting_confidence_bounded() {
        let moe = CorticalColumnMoE::new(8, 2, 64, 32).unwrap();
        for seed in 0..100u64 {
            let x = make_input(seed * 99, 64);
            for expert in &moe.experts {
                let c = expert.confidence(&x);
                assert!(c >= 0.0 && c <= 1.0, "confidence must be in [0,1], got {c}");
            }
        }
    }

    #[test]
    fn voting_output_is_weighted_average() {
        let mut moe = CorticalColumnMoE::new(8, 2, 64, 32).unwrap();
        let x = make_input(42, 64);
        let (output, indices, weights) = moe.forward_with_info(&x).unwrap();

        // Manually compute weighted average of selected expert outputs
        let all_expert_outputs = moe.expert_outputs(&x);
        let mut expected = vec![0.0f32; 64];
        for (rank, &idx) in indices.iter().enumerate() {
            let w = weights[rank];
            for (e, &eo) in expected.iter_mut().zip(all_expert_outputs[idx].iter()) {
                *e += w * eo;
            }
        }

        let max_diff = output
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "output must be weighted average of experts, max diff = {max_diff}"
        );
    }

    #[test]
    fn voting_weights_sum_to_one() {
        let mut moe = CorticalColumnMoE::new(8, 2, 64, 32).unwrap();
        for seed in 0..50u64 {
            let x = make_input(seed * 111, 64);
            let (_, _, weights) = moe.forward_with_info(&x).unwrap();
            let sum: f32 = weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "weights must sum to 1.0, got {sum} (seed={seed})"
            );
        }
    }

    // ─── Expert unit tests ──────────────────────────────────────────────

    #[test]
    fn expert_forward_shape() {
        let expert = ColumnExpert::new(64, 32, 12345);
        let x = make_input(1, 64);
        let h = expert.forward(&x);
        assert_eq!(h.len(), 32, "expert output must have d_out dimensions");
    }

    #[test]
    fn expert_relu_nonnegative() {
        let expert = ColumnExpert::new(64, 32, 12345);
        let x = make_input(1, 64);
        let h = expert.forward(&x);
        assert!(
            h.iter().all(|&v| v >= 0.0),
            "relu output must be non-negative"
        );
    }

    // ─── Softmax helper tests ───────────────────────────────────────────

    #[test]
    fn softmax_topk_sums_to_one() {
        let scores = vec![1.0, 2.0, 3.0];
        let weights = softmax_topk(&scores);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1, got {sum}");
    }

    #[test]
    fn softmax_topk_ordering() {
        let scores = vec![1.0, 3.0, 2.0];
        let weights = softmax_topk(&scores);
        assert!(
            weights[1] > weights[2],
            "higher score should get higher weight"
        );
        assert!(
            weights[2] > weights[0],
            "middle score should get middle weight"
        );
    }

    // ─── Property tests (proptest) ──────────────────────────────────────

    proptest! {
        #[test]
        fn property_output_bounded(
            x in proptest::collection::vec(-1.0f32..1.0f32, 64),
        ) {
            let mut moe = CorticalColumnMoE::new(8, 2, 64, 32).unwrap();
            let output = moe.forward(&x).unwrap();
            let out_norm = vec_norm(&output);

            let expert_outputs = moe.expert_outputs(&x);
            let max_expert_norm = expert_outputs.iter()
                .map(|eo| vec_norm(eo))
                .fold(0.0f32, f32::max);

            // Convex combination: ||sum w_i x_i|| <= max ||x_i||
            prop_assert!(
                out_norm <= max_expert_norm + 1e-3,
                "output norm ({}) must be <= max expert norm ({}) for convex combination",
                out_norm, max_expert_norm
            );
        }

        #[test]
        fn property_output_is_weighted_average(
            x in proptest::collection::vec(-1.0f32..1.0f32, 64),
        ) {
            let mut moe = CorticalColumnMoE::new(8, 2, 64, 32).unwrap();
            let (output, indices, weights) = moe.forward_with_info(&x).unwrap();

            prop_assert!(weights.iter().all(|&w| w >= 0.0), "weights must be non-negative");
            let wsum: f32 = weights.iter().sum();
            prop_assert!((wsum - 1.0).abs() < 1e-5, "weights must sum to 1, got {}", wsum);

            let expert_outputs = moe.expert_outputs(&x);
            let mut expected = vec![0.0f32; 64];
            for (rank, &idx) in indices.iter().enumerate() {
                for (e, &eo) in expected.iter_mut().zip(expert_outputs[idx].iter()) {
                    *e += weights[rank] * eo;
                }
            }
            let max_diff = output.iter().zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            prop_assert!(
                max_diff < 1e-4,
                "output must equal weighted average, max diff = {}", max_diff
            );
        }
    }
}
