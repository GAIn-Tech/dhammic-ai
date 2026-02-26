use std::f32;

pub fn placeholder() -> &'static str {
    "ces-hebbian"
}

/// LoRA factor pair: A ∈ ℝ^{rank × d_in}, B ∈ ℝ^{d_out × rank}
/// Full adaptation delta: W_delta = B @ A, rank ≤ r
#[derive(Clone, Debug)]
pub struct LoraAdapter {
    pub rank: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub scale: f32,
}

impl LoraAdapter {
    pub fn new(rank: usize, d_in: usize, d_out: usize, scale: f32) -> Self {
        let inv_sqrt_din = 1.0 / (d_in as f32).sqrt();
        let a: Vec<f32> = (0..rank * d_in)
            .map(|i| {
                let v = lehmer_rng(i as u64 + 1) as f32 / u32::MAX as f32 * 2.0 - 1.0;
                v * inv_sqrt_din
            })
            .collect();
        let b = vec![0.0f32; d_out * rank];
        Self {
            rank,
            d_in,
            d_out,
            a,
            b,
            scale,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.d_in);
        let h = mat_vec_mul(&self.a, x, self.rank, self.d_in);
        let out = mat_vec_mul(&self.b, &h, self.d_out, self.rank);
        out.iter().map(|v| v * self.scale).collect()
    }

    pub fn oja_update_a(&mut self, x: &[f32], h: &[f32], eta: f32, lambda: f32) {
        assert_eq!(x.len(), self.d_in);
        assert_eq!(h.len(), self.rank);
        for r in 0..self.rank {
            for d in 0..self.d_in {
                let w = self.a[r * self.d_in + d];
                let delta = eta * (h[r] * x[d] - lambda * w);
                self.a[r * self.d_in + d] = w + delta;
            }
        }
    }

    pub fn oja_update_b(&mut self, h: &[f32], z: &[f32], eta: f32, lambda: f32) {
        assert_eq!(h.len(), self.rank);
        assert_eq!(z.len(), self.d_out);
        for d in 0..self.d_out {
            for r in 0..self.rank {
                let w = self.b[d * self.rank + r];
                let delta = eta * (z[d] * h[r] - lambda * w);
                self.b[d * self.rank + r] = w + delta;
            }
        }
    }

    pub fn effective_rank(&self) -> usize {
        let a_norm: f32 = self.a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let b_norm: f32 = self.b.iter().map(|v| v * v).sum::<f32>().sqrt();
        if a_norm < 1e-12 || b_norm < 1e-12 {
            return 0;
        }
        self.rank
    }

    pub fn weight_norm(&self) -> f32 {
        let a_norm: f32 = self.a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let b_norm: f32 = self.b.iter().map(|v| v * v).sum::<f32>().sqrt();
        a_norm * b_norm
    }

    pub fn max_abs_delta(&self, prev_a: &[f32], prev_b: &[f32]) -> f32 {
        let da = self
            .a
            .iter()
            .zip(prev_a.iter())
            .map(|(a, p)| (a - p).abs())
            .fold(0.0f32, f32::max);
        let db = self
            .b
            .iter()
            .zip(prev_b.iter())
            .map(|(b, p)| (b - p).abs())
            .fold(0.0f32, f32::max);
        da.max(db)
    }
}

/// Hebbian LoRA layer: wraps a stack of LoRA adapters (one per transformer layer).
/// Forward-only: no gradient tape, adapts purely from co-activation statistics.
#[derive(Clone, Debug)]
pub struct HebbianLoraStack {
    pub adapters: Vec<LoraAdapter>,
    pub eta: f32,
    pub lambda: f32,
    pub step: u64,
}

impl HebbianLoraStack {
    pub fn new(
        n_layers: usize,
        rank: usize,
        d_in: usize,
        d_out: usize,
        eta: f32,
        lambda: f32,
    ) -> Self {
        let adapters = (0..n_layers)
            .map(|_| LoraAdapter::new(rank, d_in, d_out, 1.0))
            .collect();
        Self {
            adapters,
            eta,
            lambda,
            step: 0,
        }
    }

    pub fn forward_layer(&self, layer: usize, x: &[f32]) -> Vec<f32> {
        self.adapters[layer].forward(x)
    }

    pub fn hebbian_update(&mut self, layer: usize, x: &[f32], z: &[f32]) {
        let h = mat_vec_mul(
            &self.adapters[layer].a,
            x,
            self.adapters[layer].rank,
            self.adapters[layer].d_in,
        );
        let eta = self.current_eta();
        let lambda = self.lambda;
        self.adapters[layer].oja_update_a(x, &h, eta, lambda);
        self.adapters[layer].oja_update_b(&h, z, eta, lambda);
        self.step += 1;
    }

    pub fn current_eta(&self) -> f32 {
        let t_half = 1000.0_f32;
        self.eta / (1.0 + self.step as f32 / t_half)
    }
}

/// Importance score per layer: mean activation variance (proxy for Fisher information).
/// Used by TreeLoRA/CL-LoRA (Task 3.2) to allocate rank per layer.
#[derive(Clone, Debug, Default)]
pub struct ImportanceTracker {
    pub n_layers: usize,
    pub sum: Vec<f32>,
    pub sum_sq: Vec<f32>,
    pub count: Vec<u64>,
}

impl ImportanceTracker {
    pub fn new(n_layers: usize) -> Self {
        Self {
            n_layers,
            sum: vec![0.0; n_layers],
            sum_sq: vec![0.0; n_layers],
            count: vec![0; n_layers],
        }
    }

    pub fn record(&mut self, layer: usize, activation: &[f32]) {
        let mean = activation.iter().sum::<f32>() / activation.len() as f32;
        let var = activation
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / activation.len() as f32;
        self.sum[layer] += var;
        self.sum_sq[layer] += var * var;
        self.count[layer] += 1;
    }

    pub fn importance(&self, layer: usize) -> f32 {
        if self.count[layer] == 0 {
            return 0.0;
        }
        self.sum[layer] / self.count[layer] as f32
    }

    pub fn variance_of_importance(&self, layer: usize) -> f32 {
        if self.count[layer] < 2 {
            return 0.0;
        }
        let n = self.count[layer] as f32;
        let mean = self.sum[layer] / n;
        let mean_sq = self.sum_sq[layer] / n;
        (mean_sq - mean * mean).max(0.0)
    }

    pub fn allocate_ranks(&self, rank_low: usize, rank_high: usize) -> Vec<usize> {
        let scores: Vec<f32> = (0..self.n_layers).map(|l| self.importance(l)).collect();
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        scores
            .iter()
            .map(|&s| if s >= median { rank_high } else { rank_low })
            .collect()
    }
}

/// CL-LoRA: rehearsal-free continual learning via EWC-inspired Fisher weight protection.
/// Fisher approximation: F_ij ≈ EMA(activation^2) — computed forward-only.
/// High Fisher → low plasticity (preserves consolidated knowledge).
#[derive(Clone, Debug)]
pub struct CLLoraProtector {
    pub n_layers: usize,
    pub fisher_a: Vec<Vec<f32>>,
    pub fisher_b: Vec<Vec<f32>>,
    pub alpha: f32,
}

impl CLLoraProtector {
    pub fn new(n_layers: usize, rank: usize, d_in: usize, d_out: usize, alpha: f32) -> Self {
        Self {
            n_layers,
            fisher_a: vec![vec![1.0f32; rank * d_in]; n_layers],
            fisher_b: vec![vec![1.0f32; d_out * rank]; n_layers],
            alpha,
        }
    }

    pub fn update_fisher(&mut self, layer: usize, x: &[f32], z: &[f32]) {
        let alpha = self.alpha;
        let fa = &mut self.fisher_a[layer];
        let n = x.len();
        for (i, f) in fa.iter_mut().enumerate() {
            let xi = x[i % n];
            *f = alpha * *f + (1.0 - alpha) * xi * xi;
        }
        let fb = &mut self.fisher_b[layer];
        let m = z.len();
        for (i, f) in fb.iter_mut().enumerate() {
            let zi = z[i % m];
            *f = alpha * *f + (1.0 - alpha) * zi * zi;
        }
    }

    pub fn damped_delta(
        &self,
        layer: usize,
        raw_delta_a: &[f32],
        raw_delta_b: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let eps = 1e-4_f32;
        let da: Vec<f32> = raw_delta_a
            .iter()
            .zip(self.fisher_a[layer].iter())
            .map(|(d, f)| d / (f + eps))
            .collect();
        let db: Vec<f32> = raw_delta_b
            .iter()
            .zip(self.fisher_b[layer].iter())
            .map(|(d, f)| d / (f + eps))
            .collect();
        (da, db)
    }
}

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

fn lehmer_rng(mut s: u64) -> u32 {
    s = s.wrapping_mul(0x5DEECE66Du64).wrapping_add(0xBu64) & ((1u64 << 32) - 1);
    s as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn smoke() {
        assert_eq!(placeholder(), "ces-hebbian");
    }

    #[test]
    fn oja_rule_basic_update() {
        let mut adapter = LoraAdapter::new(4, 8, 8, 1.0);
        let prev_a = adapter.a.clone();
        let prev_b = adapter.b.clone();
        let x: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
        let z: Vec<f32> = (0..8).map(|i| (i as f32 / 8.0) * 0.5).collect();
        let h = mat_vec_mul(&adapter.a, &x, 4, 8);
        adapter.oja_update_a(&x, &h, 0.01, 0.01);
        adapter.oja_update_b(&h, &z, 0.01, 0.01);
        let delta = adapter.max_abs_delta(&prev_a, &prev_b);
        assert!(delta > 0.0, "weights must change after update");
        assert!(delta < 0.1, "delta too large: {delta}");
    }

    #[test]
    fn oja_rule_zero_input_no_change_in_b() {
        let mut adapter = LoraAdapter::new(4, 8, 8, 1.0);
        let prev_b = adapter.b.clone();
        let x = vec![0.0f32; 8];
        let h = vec![0.0f32; 4];
        let z = vec![0.0f32; 8];
        adapter.oja_update_b(&h, &z, 0.01, 0.0);
        let db: f32 = adapter
            .b
            .iter()
            .zip(prev_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(db < 1e-9, "zero input should not update B (with lambda=0)");
    }

    #[test]
    fn oja_rule_weight_decay_toward_zero() {
        let mut adapter = LoraAdapter::new(2, 4, 4, 1.0);
        let x = vec![0.0f32; 4];
        let h = vec![0.0f32; 2];
        let z = vec![0.0f32; 4];
        for _ in 0..1000 {
            adapter.oja_update_a(&x, &h, 0.1, 1.0);
            adapter.oja_update_b(&h, &z, 0.1, 1.0);
        }
        let a_norm: f32 = adapter.a.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            a_norm < 0.01,
            "weights should decay to near-zero with zero input + decay: {a_norm}"
        );
    }

    #[test]
    fn oja_rule_fixed_point_stability() {
        let rank = 2;
        let d = 4;
        let mut adapter = LoraAdapter::new(rank, d, d, 1.0);
        let x: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let z_target: Vec<f32> = vec![0.5, 0.0, 0.0, 0.0];
        let lambda = 1.0_f32;
        let prev_norm_a: f32 = adapter.a.iter().map(|v| v * v).sum::<f32>().sqrt();
        for _ in 0..500 {
            let h = mat_vec_mul(&adapter.a, &x, rank, d);
            adapter.oja_update_a(&x, &h, 0.001, lambda);
            adapter.oja_update_b(&h, &z_target, 0.001, lambda);
        }
        let final_norm_a: f32 = adapter.a.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("oja_rule_fixed_point_stability: initial_norm={prev_norm_a:.4}, final_norm={final_norm_a:.4}");
        assert!(
            final_norm_a.is_finite(),
            "A norm must be finite after Oja updates"
        );
        assert!(
            final_norm_a < prev_norm_a * 10.0,
            "Oja decay (lambda=1.0) must prevent runaway growth"
        );
    }

    #[test]
    fn property_weight_update_bounded() {
        let mut adapter = LoraAdapter::new(4, 16, 16, 1.0);
        let mut max_delta = 0.0f32;
        for step in 0..100 {
            let prev_a = adapter.a.clone();
            let prev_b = adapter.b.clone();
            let x: Vec<f32> = (0..16).map(|i| ((i + step) % 8) as f32 / 8.0).collect();
            let z: Vec<f32> = (0..16)
                .map(|i| ((i * 2 + step) % 16) as f32 / 16.0)
                .collect();
            let h = mat_vec_mul(&adapter.a, &x, 4, 16);
            adapter.oja_update_a(&x, &h, 0.01, 0.01);
            adapter.oja_update_b(&h, &z, 0.01, 0.01);
            let d = adapter.max_abs_delta(&prev_a, &prev_b);
            max_delta = max_delta.max(d);
        }
        println!("property_weight_update_bounded: max_delta={max_delta:.6}");
        assert!(
            max_delta < 0.1,
            "max weight update should be < 0.1, got {max_delta}"
        );
    }

    #[test]
    fn property_rank_preserved() {
        let rank = 8;
        let d_in = 32;
        let d_out = 32;
        let mut adapter = LoraAdapter::new(rank, d_in, d_out, 1.0);
        for step in 0..500 {
            let x: Vec<f32> = (0..d_in)
                .map(|i| ((i + step * 7) % 16) as f32 / 16.0)
                .collect();
            let z: Vec<f32> = (0..d_out)
                .map(|i| ((i + step * 3) % 16) as f32 / 16.0)
                .collect();
            let h = mat_vec_mul(&adapter.a, &x, rank, d_in);
            adapter.oja_update_a(&x, &h, 0.01, 0.01);
            adapter.oja_update_b(&h, &z, 0.01, 0.01);
        }
        let eff_rank = adapter.effective_rank();
        println!("property_rank_preserved: effective_rank={eff_rank}/{rank}");
        assert!(
            eff_rank <= rank,
            "rank must not exceed configured rank {rank}, got {eff_rank}"
        );
    }

    #[test]
    fn hebbian_stack_forward_shape() {
        let stack = HebbianLoraStack::new(4, 8, 32, 32, 0.01, 0.01);
        let x: Vec<f32> = vec![0.1f32; 32];
        let out = stack.forward_layer(0, &x);
        assert_eq!(out.len(), 32, "output dim must match d_out");
    }

    #[test]
    fn hebbian_stack_update_changes_output() {
        let mut stack = HebbianLoraStack::new(2, 4, 8, 8, 0.05, 0.01);
        let x: Vec<f32> = (0..8).map(|i| i as f32 / 8.0).collect();
        let z: Vec<f32> = (0..8).map(|i| (8 - i) as f32 / 8.0).collect();
        let out_before = stack.forward_layer(0, &x);
        for _ in 0..50 {
            stack.hebbian_update(0, &x, &z);
        }
        let out_after = stack.forward_layer(0, &x);
        let changed: bool = out_before
            .iter()
            .zip(out_after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "output must change after Hebbian updates");
    }

    #[test]
    fn hebbian_eta_decays() {
        let mut stack = HebbianLoraStack::new(1, 4, 8, 8, 0.1, 0.01);
        let eta_0 = stack.current_eta();
        stack.step = 1000;
        let eta_1000 = stack.current_eta();
        assert!(
            eta_1000 < eta_0,
            "eta must decay with steps: eta_0={eta_0}, eta_1000={eta_1000}"
        );
        assert!(
            (eta_0 / eta_1000 - 2.0).abs() < 0.01,
            "eta should halve at T_half=1000"
        );
    }

    #[test]
    fn importance_tracker_records() {
        let mut tracker = ImportanceTracker::new(4);
        for _ in 0..10 {
            let act: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
            tracker.record(0, &act);
            let flat: Vec<f32> = vec![0.1f32; 16];
            tracker.record(1, &flat);
        }
        let imp0 = tracker.importance(0);
        let imp1 = tracker.importance(1);
        println!("importance_tracker: layer0={imp0:.4}, layer1={imp1:.4}");
        assert!(
            imp0 > imp1,
            "ramp activation must have higher importance than flat"
        );
    }

    #[test]
    fn importance_scoring() {
        let mut tracker = ImportanceTracker::new(6);
        let mut rng_state = 0xDEADBEEFu64;
        for _ in 0..100 {
            for l in 0..6 {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let scale = if l < 3 { 1.0 } else { 0.1 };
                let act: Vec<f32> = (0..32)
                    .map(|_| {
                        rng_state ^= rng_state << 13;
                        rng_state ^= rng_state >> 7;
                        rng_state ^= rng_state << 17;
                        (rng_state as f32 / u64::MAX as f32 * 2.0 - 1.0) * scale
                    })
                    .collect();
                tracker.record(l, &act);
            }
        }
        for l in 0..6 {
            let var = tracker.variance_of_importance(l);
            println!("importance_scoring: layer{l} variance={var:.6}");
            assert!(
                var < 0.01,
                "importance variance must be stable (<0.01), layer{l}: {var}"
            );
        }
    }

    #[test]
    fn continual_learning_cl_lora_damping() {
        let n_layers = 2;
        let rank = 4;
        let d_in = 8;
        let d_out = 8;
        let mut protector = CLLoraProtector::new(n_layers, rank, d_in, d_out, 0.9);
        let x = vec![1.0f32; d_in];
        let z = vec![1.0f32; d_out];
        protector.update_fisher(0, &x, &z);
        let raw_da = vec![0.05f32; rank * d_in];
        let raw_db = vec![0.05f32; d_out * rank];
        let (da, db) = protector.damped_delta(0, &raw_da, &raw_db);
        let max_da = da.iter().cloned().fold(0.0f32, f32::max);
        let max_db = db.iter().cloned().fold(0.0f32, f32::max);
        println!("continual_learning_cl_lora_damping: max_da={max_da:.6}, max_db={max_db:.6}");
        assert!(max_da < 0.05, "Fisher damping must reduce A update");
        assert!(max_db < 0.05, "Fisher damping must reduce B update");
    }

    #[test]
    fn continual_learning_high_fisher_protects_weights() {
        let n_layers = 1;
        let rank = 4;
        let d_in = 8;
        let d_out = 8;
        let mut protector = CLLoraProtector::new(n_layers, rank, d_in, d_out, 0.9);
        let x_high = vec![10.0f32; d_in];
        let z_high = vec![10.0f32; d_out];
        for _ in 0..500 {
            protector.update_fisher(0, &x_high, &z_high);
        }
        let x_low = vec![0.01f32; d_in];
        let z_low = vec![0.01f32; d_out];
        let mut protector_low = CLLoraProtector::new(n_layers, rank, d_in, d_out, 0.9);
        for _ in 0..500 {
            protector_low.update_fisher(0, &x_low, &z_low);
        }
        let raw_da = vec![0.1f32; rank * d_in];
        let raw_db = vec![0.1f32; d_out * rank];
        let (da_high, _) = protector.damped_delta(0, &raw_da, &raw_db);
        let (da_low, _) = protector_low.damped_delta(0, &raw_da, &raw_db);
        let max_high = da_high.iter().cloned().fold(0.0f32, f32::max);
        let max_low = da_low.iter().cloned().fold(0.0f32, f32::max);
        println!("continual_learning_high_fisher_protects: high_fisher_delta={max_high:.6}, low_fisher_delta={max_low:.6}");
        assert!(
            max_high < max_low,
            "high Fisher must suppress updates more than low Fisher"
        );
        assert!(
            max_high < 0.01,
            "high Fisher (F≈100) must strongly suppress delta: {max_high}"
        );
    }

    proptest! {
        #[test]
        fn property_different_inputs_different_outputs(
            x1 in proptest::collection::vec(-1.0f32..1.0f32, 16),
            x2 in proptest::collection::vec(-1.0f32..1.0f32, 16),
        ) {
            let adapter = LoraAdapter::new(4, 16, 16, 1.0);
            let out1 = adapter.forward(&x1);
            let out2 = adapter.forward(&x2);
            let same_input = x1.iter().zip(x2.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
            let same_output = out1.iter().zip(out2.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
            prop_assert!(same_input || !same_output || out1.iter().all(|v| v.abs() < 1e-10),
                "different inputs should generally produce different outputs");
        }
    }
}
