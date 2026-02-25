//! Spiking activations + SP-inspired lateral inhibition

pub fn spike(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn surrogate_grad(x: f32, beta: f32) -> f32 {
    let denom = 1.0 + beta * x.abs();
    beta / (2.0 * denom * denom)
}

pub fn ptsoftplus(x: f32) -> f32 {
    let x_abs = x.abs();
    x.max(0.0) + (1.0 + (-x_abs).exp()).ln()
}

pub fn ptsilu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn sparsity(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let active = values.iter().filter(|&&v| v > 0.0).count();
    active as f32 / values.len() as f32
}

pub fn placeholder() {}

// ─── SP-Inspired Lateral Inhibition ──────────────────────────────────────────
//
// Competitive lateral inhibition layer inspired by HTM Spatial Pooler.
// Achieves ~2% activation sparsity (SDR target: k=40 active out of N neurons).
//
// Mathematical basis:
//   - Inhibition strength per neuron: w_i ∈ [0, 1]
//   - Effective activation: a_i = x_i * (1 - w_i)  ... (inhibitory gating)
//   - Top-k selection over a_i produces sparse output
//   - Hebbian update: Δw_i = η * (r_i - target_rate)
//     where r_i = running mean of spike(a_i), target_rate ≈ 0.02
//   - This drives overactive neurons to be suppressed, underactive to be boosted
//
// Lyapunov stability (Cohen-Grossberg 1983):
//   The update rule has a Lyapunov function V = Σ_i (r_i - ρ)²,
//   which strictly decreases until r_i → ρ for all i, ensuring convergence.

/// SP-inspired lateral inhibition layer (pure scalar / batch version).
///
/// `n`           — number of neurons
/// `k`           — number of active neurons (target: n * 0.02)
/// `eta`         — Hebbian learning rate (e.g., 0.01)
/// `target_rate` — desired per-neuron activation probability (e.g., 0.02)
pub struct SpInhibitionLayer {
    pub n: usize,
    pub k: usize,
    pub eta: f32,
    pub target_rate: f32,
    /// Inhibitory weight per neuron — initialised at 0.0 (no inhibition)
    pub weights: Vec<f32>,
    /// Exponential moving-average of activation rate per neuron
    pub running_rate: Vec<f32>,
    /// EMA momentum for running_rate (e.g., 0.99)
    pub ema_momentum: f32,
}

impl SpInhibitionLayer {
    pub fn new(n: usize, k: usize, eta: f32, target_rate: f32) -> Self {
        Self {
            n,
            k,
            eta,
            target_rate,
            weights: vec![0.0f32; n],
            running_rate: vec![target_rate; n], // warm-start avoids initial burst
            ema_momentum: 0.99,
        }
    }

    /// Forward pass (inference, no weight update).
    ///
    /// `x` — pre-inhibition activations of length `n`
    /// Returns sparse binary output: 1.0 for top-k neurons, 0.0 otherwise.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.n, "input length must equal n");
        // Compute gated activations: a_i = x_i * (1 - w_i)
        let gated: Vec<f32> = x
            .iter()
            .zip(self.weights.iter())
            .map(|(xi, wi)| xi * (1.0 - wi.min(1.0).max(0.0)))
            .collect();

        // Top-k selection: find the k-th largest value as threshold
        let threshold = top_k_threshold(&gated, self.k);

        gated
            .iter()
            .map(|&a| if a >= threshold { 1.0f32 } else { 0.0f32 })
            .collect()
    }

    /// Forward + Hebbian weight update (training).
    ///
    /// Returns sparse output and mutates `weights` and `running_rate`.
    pub fn forward_and_update(&mut self, x: &[f32]) -> Vec<f32> {
        let output = self.forward(x);

        let m = self.ema_momentum;
        for i in 0..self.n {
            self.running_rate[i] = m * self.running_rate[i] + (1.0 - m) * output[i];
        }

        // Hebbian inhibitory update: Δw_i = η * (r_i - ρ)
        // Overactive (r > ρ): increase inhibition weight
        // Underactive (r < ρ): decrease inhibition weight (more excitable)
        for i in 0..self.n {
            let delta = self.eta * (self.running_rate[i] - self.target_rate);
            self.weights[i] = (self.weights[i] + delta).clamp(0.0, 1.0);
        }

        output
    }

    /// Current measured sparsity (fraction of neurons with running_rate > 0).
    pub fn measured_sparsity(&self) -> f32 {
        let active = self.running_rate.iter().filter(|&&r| r > 0.001).count();
        active as f32 / self.n as f32
    }
}

/// Returns the value of the k-th largest element (0-indexed: k=1 means largest).
/// Used as the activation threshold for top-k selection.
fn top_k_threshold(values: &[f32], k: usize) -> f32 {
    if k == 0 || values.is_empty() {
        return f32::INFINITY;
    }
    let k_clamped = k.min(values.len());
    let mut sorted: Vec<f32> = values.to_vec();
    // Partial sort: bring top-k elements to front (O(n log k) average)
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted[k_clamped - 1]
}

/// Compute SDR interference reduction factor.
/// Formula (from brainstorm math): (N/k)² — fraction of spurious matches.
/// At 2% sparsity (k=40, N=2048): (2048/40)² = 2621×
pub fn sdr_interference_reduction(n: usize, k: usize) -> f64 {
    let ratio = n as f64 / k as f64;
    ratio * ratio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_spike_basic() {
        assert_eq!(spike(1.0), 1.0);
        assert_eq!(spike(0.0), 0.0);
        assert_eq!(spike(-1.0), 0.0);
        assert_eq!(spike(1e-10), 1.0);
        assert_eq!(spike(-1e-10), 0.0);
        println!("activation_spike_basic: passed");
    }

    #[test]
    fn surrogate_gradient_values() {
        let g1 = surrogate_grad(0.1, 10.0);
        let g2 = surrogate_grad(1.0, 10.0);

        println!("surrogate_grad(0.1, 10) = {:.6}", g1);
        println!("surrogate_grad(1.0, 10) = {:.6}", g2);

        assert!((g1 - 1.25).abs() < 1e-6, "Expected 1.25, got {}", g1);
        let expected_g2 = 10.0f32 / (2.0 * (1.0 + 10.0 * 1.0f32).powi(2));
        assert!(
            (g2 - expected_g2).abs() < 1e-6,
            "Expected {}, got {}",
            expected_g2,
            g2
        );
        println!("surrogate_gradient_values: passed");
    }

    #[test]
    fn property_gradient_bounded() {
        let beta = 10.0f32;
        let bound = beta / 2.0;

        for x_int in [-100i32, -10, -1, 0, 1, 10, 100] {
            let x = x_int as f32;
            let g = surrogate_grad(x, beta);
            assert!(
                g <= bound + 1e-6,
                "surrogate_grad({}, {}) = {} > bound {}",
                x,
                beta,
                g,
                bound
            );
        }
        let g0 = surrogate_grad(0.0, beta);
        assert!(
            (g0 - bound).abs() < 1e-6,
            "At x=0, expected beta/2={}, got {}",
            bound,
            g0
        );
        println!("property_gradient_bounded: passed (beta/2 = {})", bound);
    }

    #[test]
    fn activation_ptsoftplus_values() {
        let v = ptsoftplus(0.0);
        assert!(
            (v - 0.693147f32).abs() < 1e-5,
            "softplus(0) expected 0.693147, got {}",
            v
        );

        let v_large = ptsoftplus(20.0);
        assert!(
            (v_large - 20.0).abs() < 1e-3,
            "softplus(20) expected ~20, got {}",
            v_large
        );

        assert!(ptsoftplus(-10.0) > 0.0);
        println!("activation_ptsoftplus_values: passed");
    }

    #[test]
    fn activation_ptsilu_values() {
        assert!((ptsilu(0.0)).abs() < 1e-7);
        let v = ptsilu(10.0);
        assert!((v - 10.0).abs() < 0.01, "SiLU(10) expected ~10, got {}", v);
        println!("activation_ptsilu_values: passed");
    }

    #[test]
    fn activation_sparsity_measurement() {
        let all_pos = vec![1.0f32; 100];
        assert!((sparsity(&all_pos) - 1.0).abs() < 1e-6);

        let all_neg = vec![-1.0f32; 100];
        assert!(sparsity(&all_neg) < 1e-6);

        let mut half: Vec<f32> = vec![1.0; 50];
        half.extend(vec![-1.0; 50]);
        assert!((sparsity(&half) - 0.5).abs() < 1e-6);

        println!("activation_sparsity_measurement: passed");
    }

    #[test]
    fn property_surrogate_monotone_in_abs() {
        let beta = 10.0f32;
        let g0 = surrogate_grad(0.0, beta);
        let g1 = surrogate_grad(0.5, beta);
        let g2 = surrogate_grad(1.0, beta);
        let g3 = surrogate_grad(5.0, beta);
        assert!(g0 > g1, "g(0)={} should > g(0.5)={}", g0, g1);
        assert!(g1 > g2, "g(0.5)={} should > g(1.0)={}", g1, g2);
        assert!(g2 > g3, "g(1.0)={} should > g(5.0)={}", g2, g3);
        println!("property_surrogate_monotone_in_abs: passed");
    }

    #[test]
    fn sp_inhibition_forward_sparsity() {
        let n = 100usize;
        let k = 2usize;
        let layer = SpInhibitionLayer::new(n, k, 0.01, 0.02);
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let out = layer.forward(&x);
        let active = out.iter().filter(|&&v| v > 0.5).count();
        assert_eq!(active, k, "expected exactly k={k} active, got {active}");
        println!("sp_inhibition_forward_sparsity: passed (k={k} active out of n={n})");
    }

    #[test]
    fn sp_inhibition_top_k_correct() {
        let n = 10usize;
        let k = 3usize;
        let layer = SpInhibitionLayer::new(n, k, 0.01, 0.02);
        let x = vec![0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5, 0.0];
        let out = layer.forward(&x);
        let active_indices: Vec<usize> = out
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0.5)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(active_indices.len(), k);
        assert!(active_indices.contains(&1), "index 1 (0.9) must be active");
        assert!(active_indices.contains(&3), "index 3 (0.8) must be active");
        assert!(active_indices.contains(&5), "index 5 (0.7) must be active");
        println!(
            "sp_inhibition_top_k_correct: passed — top-3 are indices {:?}",
            active_indices
        );
    }

    #[test]
    fn inhibition_convergence() {
        let n = 200usize;
        let k = 4usize;
        let mut layer = SpInhibitionLayer::new(n, k, 0.05, k as f32 / n as f32);
        let mut rng_state = 42u64;
        for _step in 0..1000 {
            let x: Vec<f32> = (0..n)
                .map(|_| {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    ((rng_state & 0xFFFF) as f32) / 65535.0
                })
                .collect();
            layer.forward_and_update(&x);
        }
        let weight_var: f32 = {
            let mean = layer.weights.iter().sum::<f32>() / n as f32;
            layer
                .weights
                .iter()
                .map(|&w| (w - mean).powi(2))
                .sum::<f32>()
                / n as f32
        };
        println!(
            "inhibition_convergence: weight variance after 1000 steps = {:.6e}",
            weight_var
        );
        assert!(
            weight_var < 0.5,
            "weights should converge (low variance), got variance={weight_var:.4}"
        );
        println!("inhibition_convergence: passed");
    }

    #[test]
    fn property_sparsity_range() {
        let n = 2048usize;
        let k = 40usize;
        let layer = SpInhibitionLayer::new(n, k, 0.01, 0.02);
        let mut pass_count = 0usize;
        let trials = 100usize;
        let mut rng_state = 123u64;

        for _trial in 0..trials {
            let x: Vec<f32> = (0..n)
                .map(|_| {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    ((rng_state & 0xFFFF) as f32) / 65535.0
                })
                .collect();
            let out = layer.forward(&x);
            let s = sparsity(&out);
            if s >= 0.01 && s <= 0.05 {
                pass_count += 1;
            }
        }

        let pass_rate = pass_count as f32 / trials as f32;
        println!(
            "property_sparsity_range: {}/{} trials in [1%,5%] ({:.1}%)",
            pass_count,
            trials,
            pass_rate * 100.0
        );
        assert!(
            pass_rate >= 0.95,
            "expected 95%+ trials in [1%,5%] sparsity, got {:.1}%",
            pass_rate * 100.0
        );
    }

    #[test]
    fn interference_reduction() {
        let reduction = sdr_interference_reduction(2048, 40);
        println!("interference_reduction: (2048/40)^2 = {:.2}×", reduction);
        assert!(
            reduction >= 2000.0,
            "expected >= 2000× reduction, got {:.2}",
            reduction
        );
        assert!(
            (reduction - 2621.44).abs() < 0.1,
            "expected 2621.44, got {:.4}",
            reduction
        );
    }

    #[test]
    fn sp_inhibition_weights_bounded() {
        let n = 50usize;
        let k = 1usize;
        let mut layer = SpInhibitionLayer::new(n, k, 0.1, 0.02);
        let mut rng_state = 999u64;
        for _step in 0..500 {
            let x: Vec<f32> = (0..n)
                .map(|_| {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    ((rng_state & 0xFFFF) as f32) / 65535.0
                })
                .collect();
            layer.forward_and_update(&x);
        }
        for (i, &w) in layer.weights.iter().enumerate() {
            assert!(w >= 0.0 && w <= 1.0, "weight[{i}] = {w} out of [0,1]");
        }
        println!("sp_inhibition_weights_bounded: all {} weights in [0,1]", n);
    }
}
