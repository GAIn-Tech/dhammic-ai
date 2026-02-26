//! End-to-end CES inference pipeline with DVFS power management
//!
//! Dynamic Voltage and Frequency Scaling (DVFS) for adaptive compute allocation
//! based on input complexity signals (surprise, sparsity).

use std::collections::VecDeque;

/// Complexity signal derived from input characteristics
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexitySignal {
    /// Surprise score: 0.0 (expected) to 1.0 (highly surprising)
    pub surprise: f32,
    /// Sparsity: 0.0 (dense) to 1.0 (sparse), fraction of active neurons
    pub sparsity: f32,
}

impl ComplexitySignal {
    pub fn new(surprise: f32, sparsity: f32) -> Self {
        Self {
            surprise: surprise.clamp(0.0, 1.0),
            sparsity: sparsity.clamp(0.0, 1.0),
        }
    }
}

/// Compute mode determines precision and expert allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComputeMode {
    /// Minimal compute: low precision, single expert
    Minimal,
    /// Reduced compute: medium precision, half experts
    Reduced,
    /// Full compute: high precision, all experts
    Full,
}

/// DVFS controller: maps complexity signals to compute modes
#[derive(Debug, Clone)]
pub struct DvfsController {
    /// High surprise threshold (above = Full)
    pub high_surprise_threshold: f32,
    /// Low surprise threshold (below = Minimal)
    pub low_surprise_threshold: f32,
    /// Low sparsity threshold (below = Full, dense activation)
    pub low_sparsity_threshold: f32,
    /// High sparsity threshold (above = Minimal, sparse activation)
    pub high_sparsity_threshold: f32,
}

impl Default for DvfsController {
    fn default() -> Self {
        Self {
            high_surprise_threshold: 0.7,
            low_surprise_threshold: 0.3,
            low_sparsity_threshold: 0.05,
            high_sparsity_threshold: 0.15,
        }
    }
}

impl DvfsController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Determine compute mode from complexity signal
    pub fn compute_mode(&self, signal: &ComplexitySignal) -> ComputeMode {
        // High complexity: high surprise OR very dense activation
        if signal.surprise > self.high_surprise_threshold
            || signal.sparsity < self.low_sparsity_threshold
        {
            return ComputeMode::Full;
        }

        // Low complexity: low surprise AND sparse activation
        if signal.surprise < self.low_surprise_threshold
            && signal.sparsity > self.high_sparsity_threshold
        {
            return ComputeMode::Minimal;
        }

        // Medium complexity
        ComputeMode::Reduced
    }

    /// Number of active experts for given mode
    pub fn n_active_experts(&self, mode: ComputeMode, n_total: usize) -> usize {
        match mode {
            ComputeMode::Full => n_total,
            ComputeMode::Reduced => (n_total / 2).max(1),
            ComputeMode::Minimal => 1,
        }
    }

    /// Precision bits for given mode
    pub fn precision_bits(&self, mode: ComputeMode) -> u8 {
        match mode {
            ComputeMode::Full => 32,
            ComputeMode::Reduced => 16,
            ComputeMode::Minimal => 8,
        }
    }
}

/// Decision log for DVFS controller (bounded to last 1000 entries)
#[derive(Debug, Clone)]
pub struct DvfsLog {
    entries: VecDeque<(ComplexitySignal, ComputeMode)>,
    max_size: usize,
}

impl Default for DvfsLog {
    fn default() -> Self {
        Self::new(1000)
    }
}

impl DvfsLog {
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    pub fn record(&mut self, signal: ComplexitySignal, mode: ComputeMode) {
        if self.entries.len() >= self.max_size {
            self.entries.pop_front();
        }
        self.entries.push_back((signal, mode));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &VecDeque<(ComplexitySignal, ComputeMode)> {
        &self.entries
    }

    /// Average compute mode over recent history
    pub fn avg_compute_level(&self) -> f32 {
        if self.entries.is_empty() {
            return 0.0;
        }

        let sum: usize = self
            .entries
            .iter()
            .map(|(_, mode)| match mode {
                ComputeMode::Minimal => 0,
                ComputeMode::Reduced => 1,
                ComputeMode::Full => 2,
            })
            .sum();

        sum as f32 / self.entries.len() as f32
    }
}

/// Linux cpufreq integration (cfg-gated)
#[cfg(target_os = "linux")]
pub fn read_cpufreq_khz() -> Option<u64> {
    use std::fs;

    let path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

#[cfg(target_os = "linux")]
pub fn set_cpufreq_governor(governor: &str) -> std::io::Result<()> {
    use std::fs;
    use std::io::ErrorKind;

    let path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor";

    // Attempt write, but don't fail if permission denied
    match fs::write(path, governor) {
        Ok(_) => {
            tracing::debug!("Set cpufreq governor to: {}", governor);
            Ok(())
        }
        Err(e) if e.kind() == ErrorKind::PermissionDenied => {
            tracing::debug!("Permission denied setting governor (expected without root)");
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Stub implementations for non-Linux platforms
#[cfg(not(target_os = "linux"))]
pub fn read_cpufreq_khz() -> Option<u64> {
    tracing::debug!("cpufreq not available on non-Linux platform");
    None
}

#[cfg(not(target_os = "linux"))]
pub fn set_cpufreq_governor(_governor: &str) -> std::io::Result<()> {
    tracing::debug!("cpufreq governor setting not available on non-Linux platform");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dvfs_high_surprise_gives_full_compute() {
        let controller = DvfsController::new();
        let signal = ComplexitySignal::new(0.9, 0.01);
        let mode = controller.compute_mode(&signal);
        assert_eq!(mode, ComputeMode::Full, "High surprise should trigger Full");
    }

    #[test]
    fn dvfs_low_surprise_gives_minimal_compute() {
        let controller = DvfsController::new();
        let signal = ComplexitySignal::new(0.1, 0.5);
        let mode = controller.compute_mode(&signal);
        assert_eq!(
            mode,
            ComputeMode::Minimal,
            "Low surprise + high sparsity should trigger Minimal"
        );
    }

    #[test]
    fn dvfs_medium_gives_reduced() {
        let controller = DvfsController::new();
        let signal = ComplexitySignal::new(0.5, 0.1);
        let mode = controller.compute_mode(&signal);
        assert_eq!(
            mode,
            ComputeMode::Reduced,
            "Medium complexity should trigger Reduced"
        );
    }

    #[test]
    fn dvfs_complexity_heuristic_monotone() {
        let controller = DvfsController::new();

        // Test monotonicity: higher surprise should give same or higher compute mode
        let signals = vec![
            ComplexitySignal::new(0.1, 0.5),
            ComplexitySignal::new(0.3, 0.5),
            ComplexitySignal::new(0.5, 0.5),
            ComplexitySignal::new(0.7, 0.5),
            ComplexitySignal::new(0.9, 0.5),
        ];

        let modes: Vec<ComputeMode> = signals.iter().map(|s| controller.compute_mode(s)).collect();

        for i in 1..modes.len() {
            assert!(
                modes[i] >= modes[i - 1],
                "Higher surprise should give same or higher compute mode: {:?} vs {:?}",
                modes[i - 1],
                modes[i]
            );
        }
    }

    #[test]
    fn dvfs_n_active_experts_full() {
        let controller = DvfsController::new();
        let n_total = 8;
        let n_active = controller.n_active_experts(ComputeMode::Full, n_total);
        assert_eq!(n_active, n_total, "Full mode should use all experts");
    }

    #[test]
    fn dvfs_n_active_experts_reduced() {
        let controller = DvfsController::new();
        let n_total = 8;
        let n_active = controller.n_active_experts(ComputeMode::Reduced, n_total);
        assert_eq!(
            n_active,
            n_total / 2,
            "Reduced mode should use half experts"
        );

        // Test minimum of 1
        let n_active_small = controller.n_active_experts(ComputeMode::Reduced, 1);
        assert_eq!(
            n_active_small, 1,
            "Reduced mode should use at least 1 expert"
        );
    }

    #[test]
    fn dvfs_n_active_experts_minimal() {
        let controller = DvfsController::new();
        let n_total = 8;
        let n_active = controller.n_active_experts(ComputeMode::Minimal, n_total);
        assert_eq!(n_active, 1, "Minimal mode should use 1 expert");
    }

    #[test]
    fn dvfs_stub_logs_without_panic() {
        let controller = DvfsController::new();
        let mut log = DvfsLog::new(1000);

        // Run 100 iterations with varying signals
        for i in 0..100 {
            let surprise = (i as f32 / 100.0).clamp(0.0, 1.0);
            let sparsity = ((100 - i) as f32 / 100.0).clamp(0.0, 1.0);
            let signal = ComplexitySignal::new(surprise, sparsity);
            let mode = controller.compute_mode(&signal);
            log.record(signal, mode);
        }

        assert_eq!(log.len(), 100, "Log should contain 100 entries");
    }

    #[test]
    fn dvfs_log_bounded() {
        let mut log = DvfsLog::new(1000);
        let controller = DvfsController::new();

        // Add 1500 entries
        for _ in 0..1500 {
            let signal = ComplexitySignal::new(0.5, 0.5);
            let mode = controller.compute_mode(&signal);
            log.record(signal, mode);
        }

        assert_eq!(
            log.len(),
            1000,
            "Log should be bounded to 1000 entries, got {}",
            log.len()
        );
    }

    #[test]
    fn dvfs_precision_bits_ordered() {
        let controller = DvfsController::new();

        let full_bits = controller.precision_bits(ComputeMode::Full);
        let reduced_bits = controller.precision_bits(ComputeMode::Reduced);
        let minimal_bits = controller.precision_bits(ComputeMode::Minimal);

        assert!(
            full_bits >= reduced_bits,
            "Full precision should be >= Reduced"
        );
        assert!(
            reduced_bits >= minimal_bits,
            "Reduced precision should be >= Minimal"
        );

        assert_eq!(full_bits, 32, "Full should be 32-bit");
        assert_eq!(reduced_bits, 16, "Reduced should be 16-bit");
        assert_eq!(minimal_bits, 8, "Minimal should be 8-bit");
    }

    #[test]
    fn dvfs_log_avg_compute_level() {
        let mut log = DvfsLog::new(1000);

        // All minimal
        for _ in 0..10 {
            log.record(ComplexitySignal::new(0.1, 0.5), ComputeMode::Minimal);
        }
        let avg = log.avg_compute_level();
        assert!(
            (avg - 0.0).abs() < 0.01,
            "All Minimal should average to 0.0"
        );

        // All full
        let mut log2 = DvfsLog::new(1000);
        for _ in 0..10 {
            log2.record(ComplexitySignal::new(0.9, 0.01), ComputeMode::Full);
        }
        let avg2 = log2.avg_compute_level();
        assert!((avg2 - 2.0).abs() < 0.01, "All Full should average to 2.0");

        // Mixed
        let mut log3 = DvfsLog::new(1000);
        log3.record(ComplexitySignal::new(0.1, 0.5), ComputeMode::Minimal);
        log3.record(ComplexitySignal::new(0.5, 0.1), ComputeMode::Reduced);
        log3.record(ComplexitySignal::new(0.9, 0.01), ComputeMode::Full);
        let avg3 = log3.avg_compute_level();
        assert!(
            (avg3 - 1.0).abs() < 0.01,
            "Mixed (0,1,2) should average to 1.0"
        );
    }

    #[test]
    fn dvfs_dense_activation_triggers_full() {
        let controller = DvfsController::new();
        // Low surprise but very dense (low sparsity) should still trigger Full
        let signal = ComplexitySignal::new(0.2, 0.01);
        let mode = controller.compute_mode(&signal);
        assert_eq!(
            mode,
            ComputeMode::Full,
            "Dense activation should trigger Full compute"
        );
    }

    #[test]
    fn dvfs_cpufreq_stub_no_panic() {
        // Test that cpufreq functions don't panic
        let freq = read_cpufreq_khz();
        println!("Current CPU frequency: {:?} kHz", freq);

        // Test governor setting - should succeed or fail gracefully
        // On systems without cpufreq, the file won't exist (NotFound error)
        // On systems with cpufreq but no permissions, we get PermissionDenied (handled as Ok)
        let result = set_cpufreq_governor("performance");

        // Either succeeds, or fails with NotFound (no cpufreq), both are acceptable
        if let Err(e) = result {
            assert_eq!(
                e.kind(),
                std::io::ErrorKind::NotFound,
                "Only NotFound error is acceptable (no cpufreq on system)"
            );
        }
    }

    #[test]
    fn dvfs_complexity_signal_clamping() {
        // Test that values are clamped to [0.0, 1.0]
        let signal1 = ComplexitySignal::new(-0.5, 1.5);
        assert_eq!(signal1.surprise, 0.0, "Negative surprise should clamp to 0");
        assert_eq!(signal1.sparsity, 1.0, "Sparsity > 1.0 should clamp to 1.0");

        let signal2 = ComplexitySignal::new(2.0, -1.0);
        assert_eq!(signal2.surprise, 1.0, "Surprise > 1.0 should clamp to 1.0");
        assert_eq!(signal2.sparsity, 0.0, "Negative sparsity should clamp to 0");
    }
}
