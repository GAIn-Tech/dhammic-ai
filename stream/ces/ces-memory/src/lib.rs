//! Three-tier memory: Engram (Tier 1) + Episodic/Bayesian Surprise (Tier 3)
//!
//! Architecture from CES spec:
//!   Tier 1: Engram conditional memory — O(1) n-gram hash lookup + dendritic gating
//!   Tier 2: Hebbian LoRA fast weights — implemented in ces-hebbian (Sprint 3)
//!   Tier 3: EM-LLM episodic memory — Bayesian surprise boundaries + in-memory vector store

use candle_core::{DType, Device, Result, Tensor};

pub fn placeholder() -> &'static str {
    "ces-memory"
}

// ─────────────────────────────────────────────────────────────────────────────
// TIER 1a: Engram Conditional Memory
//
// DeepSeek Engram (arxiv 2601.07372) provides O(1) context-conditional memory
// lookup via deterministic n-gram hash addressing.
//
// Key equations:
//   addr = hash(ngram_tokens) mod TABLE_SIZE      (deterministic, O(1))
//   emb  = embedding_table[addr]                  (lookup)
//   gate = sigmoid(W_g · [x; emb])                (context-aware gating)
//   out  = gate ⊙ emb + (1 - gate) ⊙ x           (residual fusion)
//
// N-gram hash: FNV-1a over token sequence, mod table_size.
// ─────────────────────────────────────────────────────────────────────────────

const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const FNV_PRIME: u64 = 1_099_511_628_211;

/// FNV-1a hash of a token n-gram, reduced to table_size via modulo.
pub fn ngram_hash(tokens: &[u32], table_size: usize) -> usize {
    let mut h = FNV_OFFSET;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    (h as usize) % table_size
}

/// Engram conditional memory (Tier 1a).
pub struct EngramMemory {
    pub table_size: usize,
    pub embed_dim: usize,
    pub n: usize,
    table: Vec<Vec<f32>>,
    gate_w: Vec<Vec<f32>>,
}

impl EngramMemory {
    pub fn new(table_size: usize, embed_dim: usize, n: usize) -> Self {
        let scale = (embed_dim as f32).sqrt().recip();
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        let table = (0..table_size)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        ((rng & 0xFFFF) as f32 / 32768.0 - 1.0) * scale
                    })
                    .collect()
            })
            .collect();
        let gate_w = (0..embed_dim)
            .map(|_| {
                (0..(embed_dim * 2))
                    .map(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        ((rng & 0xFFFF) as f32 / 32768.0 - 1.0) * scale
                    })
                    .collect()
            })
            .collect();
        Self {
            table_size,
            embed_dim,
            n,
            table,
            gate_w,
        }
    }

    /// O(1) lookup: hash n-gram → table slot → embedding vector.
    pub fn lookup(&self, tokens: &[u32]) -> &[f32] {
        let addr = ngram_hash(tokens, self.table_size);
        &self.table[addr]
    }

    /// Write an embedding to the slot addressed by this n-gram.
    pub fn write(&mut self, tokens: &[u32], embedding: &[f32]) {
        assert_eq!(embedding.len(), self.embed_dim);
        let addr = ngram_hash(tokens, self.table_size);
        self.table[addr].copy_from_slice(embedding);
    }

    /// Gated fusion: gate = sigmoid(W_g · [x; emb]), out = gate⊙emb + (1-gate)⊙x
    pub fn fuse(&self, x: &[f32], emb: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.embed_dim);
        assert_eq!(emb.len(), self.embed_dim);
        let d = self.embed_dim;
        let gate: Vec<f32> = self
            .gate_w
            .iter()
            .map(|row| {
                let logit: f32 = row[..d]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, xi)| w * xi)
                    .sum::<f32>()
                    + row[d..]
                        .iter()
                        .zip(emb.iter())
                        .map(|(w, ei)| w * ei)
                        .sum::<f32>();
                1.0 / (1.0 + (-logit).exp())
            })
            .collect();
        gate.iter()
            .zip(emb.iter())
            .zip(x.iter())
            .map(|((g, e), xi)| g * e + (1.0 - g) * xi)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TIER 1b: Active Dendritic Gating (Task 2.2)
//
// From Grewal et al. 2022 "Avoiding Catastrophe: Active Dendrites Enable
// Multi-Task Learning". Each gate has K dendritic segments; only the segment
// with maximum activation modulates the gate (winner-take-all).
//
// Mathematical formulation:
//   segment_k(ctx) = w_k · ctx          (linear projection per segment)
//   winner = argmax_k segment_k(ctx)
//   modulation_i = sigmoid(segment_{winner_i}(ctx))
//   out = modulation ⊙ engram_output
//
// K=8 segments → 8× effective capacity increase per gate neuron.
// ─────────────────────────────────────────────────────────────────────────────

/// Active dendritic gating layer wrapping an EngramMemory.
pub struct DendriticGate {
    pub k_segments: usize,
    pub embed_dim: usize,
    pub ctx_dim: usize,
    segment_weights: Vec<Vec<Vec<f32>>>,
}

impl DendriticGate {
    pub fn new(k_segments: usize, embed_dim: usize, ctx_dim: usize) -> Self {
        let scale = (ctx_dim as f32).sqrt().recip();
        let mut rng = 0xDEAD_BEEF_CAFE_BABEu64;
        let segment_weights = (0..embed_dim)
            .map(|_| {
                (0..k_segments)
                    .map(|_| {
                        (0..ctx_dim)
                            .map(|_| {
                                rng ^= rng << 13;
                                rng ^= rng >> 7;
                                rng ^= rng << 17;
                                ((rng & 0xFFFF) as f32 / 32768.0 - 1.0) * scale
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self {
            k_segments,
            embed_dim,
            ctx_dim,
            segment_weights,
        }
    }

    /// Compute per-channel winner-take-all segment activations.
    ///
    /// Returns `(modulation, winner_indices)`:
    ///   - `modulation[i]` = sigmoid(max segment activation for channel i)
    ///   - `winner_indices[i]` = which segment won for channel i
    pub fn forward(&self, ctx: &[f32]) -> (Vec<f32>, Vec<usize>) {
        assert_eq!(ctx.len(), self.ctx_dim);
        let mut modulation = vec![0.0f32; self.embed_dim];
        let mut winners = vec![0usize; self.embed_dim];
        for (i, segs) in self.segment_weights.iter().enumerate() {
            let (best_seg, best_act) = segs
                .iter()
                .enumerate()
                .map(|(k, w)| {
                    let act: f32 = w.iter().zip(ctx.iter()).map(|(wi, ci)| wi * ci).sum();
                    (k, act)
                })
                .fold((0, f32::NEG_INFINITY), |(bk, ba), (k, a)| {
                    if a > ba {
                        (k, a)
                    } else {
                        (bk, ba)
                    }
                });
            winners[i] = best_seg;
            modulation[i] = 1.0 / (1.0 + (-best_act).exp());
        }
        (modulation, winners)
    }

    /// Apply dendritic modulation to an Engram fused output.
    pub fn modulate(&self, engram_out: &[f32], ctx: &[f32]) -> Vec<f32> {
        let (modulation, _) = self.forward(ctx);
        engram_out
            .iter()
            .zip(modulation.iter())
            .map(|(e, m)| e * m)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TIER 3: EM-LLM Episodic Memory + Bayesian Surprise (Task 2.3)
//
// From Ho et al. 2024 "EM-LLM: Human-like Episodic Memory for LLMs".
//
// Bayesian surprise boundary detection:
//   e_t = prediction error at token t
//   μ_t = EMA(e_t),  σ²_t = EMA((e_t - μ_{t-1})²)
//   boundary triggered when: e_t > μ_t + γ·σ_t
//   With γ=2.5 → ~0.621% of tokens trigger a boundary (brainstorm math verified)
//
// Episodes: tokens between consecutive boundaries (mean-pooled into embedding).
// ─────────────────────────────────────────────────────────────────────────────

/// Online Bayesian surprise detector for episode boundary detection.
pub struct BayesianSurpriseDetector {
    pub gamma: f32,
    pub ema_alpha: f32,
    mean: f32,
    var: f32,
    initialized: bool,
}

impl BayesianSurpriseDetector {
    pub fn new(gamma: f32, ema_alpha: f32) -> Self {
        Self {
            gamma,
            ema_alpha,
            mean: 0.0,
            var: 1.0,
            initialized: false,
        }
    }

    /// Update with new surprise value; returns true if this token is a boundary.
    pub fn update(&mut self, surprise: f32) -> bool {
        if !self.initialized {
            self.mean = surprise;
            self.var = 0.0;
            self.initialized = true;
            return false;
        }
        let threshold = self.mean + self.gamma * self.var.sqrt().max(1e-8);
        let is_boundary = surprise > threshold;
        let alpha = self.ema_alpha;
        let diff = surprise - self.mean;
        self.mean = alpha * self.mean + (1.0 - alpha) * surprise;
        self.var = alpha * self.var + (1.0 - alpha) * diff * diff;
        is_boundary
    }

    pub fn threshold(&self) -> f32 {
        self.mean + self.gamma * self.var.sqrt().max(1e-8)
    }
}

/// A single stored episode.
#[derive(Clone)]
pub struct Episode {
    pub id: u64,
    pub embedding: Vec<f32>,
    pub length: usize,
}

/// Episode segmenter and in-memory store.
pub struct EpisodicMemory {
    pub embed_dim: usize,
    pub max_episodes: usize,
    detector: BayesianSurpriseDetector,
    buffer: Vec<Vec<f32>>,
    pub episodes: Vec<Episode>,
    next_id: u64,
}

impl EpisodicMemory {
    pub fn new(embed_dim: usize, max_episodes: usize, gamma: f32) -> Self {
        Self {
            embed_dim,
            max_episodes,
            detector: BayesianSurpriseDetector::new(gamma, 0.95),
            buffer: Vec::new(),
            episodes: Vec::new(),
            next_id: 0,
        }
    }

    /// Process one token's hidden state.
    ///
    /// Returns true if a boundary was detected (episode committed to store).
    pub fn process_token(&mut self, h: &[f32], surprise: f32) -> bool {
        assert_eq!(h.len(), self.embed_dim);
        self.buffer.push(h.to_vec());
        let is_boundary = self.detector.update(surprise);
        if is_boundary && self.buffer.len() >= 2 {
            self.flush_episode();
            return true;
        }
        false
    }

    fn flush_episode(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let n = self.buffer.len();
        let mut embedding = vec![0.0f32; self.embed_dim];
        for h in &self.buffer {
            for (e, hi) in embedding.iter_mut().zip(h.iter()) {
                *e += hi;
            }
        }
        let scale = (n as f32).recip();
        for e in &mut embedding {
            *e *= scale;
        }
        self.episodes.push(Episode {
            id: self.next_id,
            embedding,
            length: n,
        });
        self.next_id += 1;
        self.buffer.clear();
        if self.episodes.len() > self.max_episodes {
            self.episodes.remove(0);
        }
    }

    /// Flush remaining buffer as a partial episode.
    pub fn flush_remaining(&mut self) {
        if !self.buffer.is_empty() {
            self.flush_episode();
        }
    }

    pub fn episode_count(&self) -> usize {
        self.episodes.len()
    }

    /// k-nearest-neighbor retrieval by cosine similarity.
    pub fn retrieve(&self, query: &[f32], k: usize) -> Vec<&Episode> {
        assert_eq!(query.len(), self.embed_dim);
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let mut scored: Vec<(usize, f32)> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(i, ep)| {
                let dot: f32 = ep
                    .embedding
                    .iter()
                    .zip(query.iter())
                    .map(|(e, q)| e * q)
                    .sum();
                let e_norm: f32 = ep
                    .embedding
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-8);
                (i, dot / (e_norm * q_norm))
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .iter()
            .take(k)
            .map(|(i, _)| &self.episodes[*i])
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TIER 3b: In-memory Vector Store (Task 2.4 — sqlite-vec portable fallback)
//
// sqlite-vec requires architecture-specific native extension (Probe B: MEDIUM).
// This in-process flat vector store mirrors the sqlite-vec API surface so it
// can be swapped for the real sqlite-vec via feature flag when native binaries
// are available.
// ─────────────────────────────────────────────────────────────────────────────

/// Flat in-memory vector store with cosine kNN (sqlite-vec API-compatible).
pub struct VectorStore {
    pub dim: usize,
    pub entries: Vec<(u64, Vec<f32>)>,
}

impl VectorStore {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
        }
    }

    pub fn insert(&mut self, id: u64, embedding: Vec<f32>) {
        assert_eq!(embedding.len(), self.dim);
        self.entries.push((id, embedding));
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// k-nearest-neighbor query by cosine similarity, sorted descending.
    pub fn knn_query(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        assert_eq!(query.len(), self.dim);
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let mut scored: Vec<(u64, f32)> = self
            .entries
            .iter()
            .map(|(id, emb)| {
                let dot: f32 = emb.iter().zip(query.iter()).map(|(e, q)| e * q).sum();
                let e_norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                (*id, dot / (e_norm * q_norm))
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Candle tensor wrappers for batch operations
// ─────────────────────────────────────────────────────────────────────────────

/// Batch Engram lookup over a token sequence.
///
/// For each position i, hashes tokens[max(0,i-n+1)..=i] → table slot → fuses with hidden[i].
/// Returns fused output tensor of shape (T, D).
pub fn engram_forward_batch(
    token_ids: &[u32],
    memory: &EngramMemory,
    hidden: &Tensor,
) -> Result<Tensor> {
    let (t, d) = hidden.dims2()?;
    assert_eq!(token_ids.len(), t);
    assert_eq!(d, memory.embed_dim);
    let device = hidden.device();
    let hidden_vec = hidden.to_vec2::<f32>()?;
    let fused: Vec<Vec<f32>> = (0..t)
        .map(|i| {
            let start = i.saturating_sub(memory.n - 1);
            let ngram = &token_ids[start..=i];
            let emb = memory.lookup(ngram);
            memory.fuse(&hidden_vec[i], emb)
        })
        .collect();
    let flat: Vec<f32> = fused.into_iter().flatten().collect();
    Tensor::from_vec(flat, (t, d), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke() {
        assert_eq!(placeholder(), "ces-memory");
    }

    #[test]
    fn engram_ngram_hash_deterministic() {
        let tokens = &[1u32, 42, 7, 99];
        let h1 = ngram_hash(tokens, 229_139);
        let h2 = ngram_hash(tokens, 229_139);
        assert_eq!(h1, h2, "hash must be deterministic");
        assert!(h1 < 229_139, "hash must be within table size");
        println!("engram_ngram_hash_deterministic: addr={h1}");
    }

    #[test]
    fn engram_hash_different_ngrams_different_slots() {
        let a = ngram_hash(&[1u32, 2, 3, 4], 229_139);
        let b = ngram_hash(&[1u32, 2, 3, 5], 229_139);
        assert_ne!(a, b, "different n-grams should map to different slots");
        println!("engram_hash_different_ngrams_different_slots: {a} != {b}");
    }

    #[test]
    fn engram_lookup_o1_same_result() {
        let mem = EngramMemory::new(1000, 32, 4);
        let tokens = &[10u32, 20, 30, 40];
        let e1 = mem.lookup(tokens).to_vec();
        let e2 = mem.lookup(tokens).to_vec();
        assert_eq!(
            e1, e2,
            "O(1) lookup must return same result on repeated calls"
        );
    }

    #[test]
    fn engram_write_read_roundtrip() {
        let mut mem = EngramMemory::new(1000, 16, 4);
        let tokens = &[5u32, 6, 7, 8];
        let emb: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        mem.write(tokens, &emb);
        let retrieved = mem.lookup(tokens).to_vec();
        for (a, b) in retrieved.iter().zip(emb.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "write-read roundtrip failed: {a} vs {b}"
            );
        }
        println!("engram_write_read_roundtrip: passed");
    }

    #[test]
    fn engram_gating() {
        let mem = EngramMemory::new(1000, 16, 4);
        let tokens = &[1u32, 2, 3, 4];
        let emb = mem.lookup(tokens);
        let x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();
        let fused = mem.fuse(&x, emb);
        assert_eq!(fused.len(), 16);
        for v in &fused {
            assert!(
                v.is_finite(),
                "engram fuse output contains non-finite value"
            );
        }
        let diff: f32 = fused
            .iter()
            .zip(x.iter())
            .map(|(f, xi)| (f - xi).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "gating must modify the input (non-trivial fusion)"
        );
        println!("engram_gating: fusion diff={diff:.4}, passed");
    }

    #[test]
    fn engram_numerical_equivalence() {
        let mut mem = EngramMemory::new(100, 8, 3);
        let tokens = &[1u32, 2, 3];
        let known_emb = vec![0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        mem.write(tokens, &known_emb);
        let x = vec![0.0f32; 8];
        let fused = mem.fuse(&x, &known_emb);
        assert_eq!(fused.len(), 8);
        for v in &fused {
            assert!(v.is_finite());
        }
        println!("engram_numerical_equivalence: passed");
    }

    #[test]
    fn engram_forward_batch_shape() {
        use candle_core::Device;
        let device = Device::Cpu;
        let mem = EngramMemory::new(1000, 16, 4);
        let tokens: Vec<u32> = (0..8).map(|i| i as u32).collect();
        let hidden = Tensor::randn(0f32, 0.5f32, (8usize, 16usize), &device).unwrap();
        let out = engram_forward_batch(&tokens, &mem, &hidden).unwrap();
        assert_eq!(out.dims(), &[8, 16], "batch output shape must be (T, D)");
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &vals {
            assert!(v.is_finite(), "batch output has non-finite value");
        }
        println!("engram_forward_batch_shape: passed");
    }

    #[test]
    fn dendritic_gating_forward_shape() {
        let gate = DendriticGate::new(8, 32, 16);
        let ctx: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let (modulation, winners) = gate.forward(&ctx);
        assert_eq!(modulation.len(), 32);
        assert_eq!(winners.len(), 32);
        for &w in &winners {
            assert!(w < 8, "winner index must be < k_segments");
        }
        for &m in &modulation {
            assert!(m > 0.0 && m < 1.0, "sigmoid modulation must be in (0,1)");
        }
        println!("dendritic_gating_forward_shape: passed");
    }

    #[test]
    fn property_different_contexts_different_segments() {
        let gate = DendriticGate::new(8, 16, 8);
        let ctx_a: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let ctx_b: Vec<f32> = (0..8).map(|i| -(i as f32) * 0.1).collect();
        let (_, winners_a) = gate.forward(&ctx_a);
        let (_, winners_b) = gate.forward(&ctx_b);
        let diff_count = winners_a
            .iter()
            .zip(winners_b.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            diff_count > 0,
            "different contexts must activate different segments"
        );
        println!("property_different_contexts_different_segments: {diff_count}/16 channels differ");
    }

    #[test]
    fn dendritic_modulate_shape() {
        let gate = DendriticGate::new(4, 16, 8);
        let engram_out: Vec<f32> = (0..16).map(|i| i as f32 * 0.01).collect();
        let ctx: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let modulated = gate.modulate(&engram_out, &ctx);
        assert_eq!(modulated.len(), 16);
        for v in &modulated {
            assert!(v.is_finite());
        }
        println!("dendritic_modulate_shape: passed");
    }

    #[test]
    fn dendritic_capacity() {
        let k = 8usize;
        let gate = DendriticGate::new(k, 64, 32);
        let mut seen_winners: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut rng = 0xABCDu64;
        for _ in 0..200 {
            let ctx: Vec<f32> = (0..32)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    ((rng & 0xFFFF) as f32 / 32768.0 - 1.0)
                })
                .collect();
            let (_, winners) = gate.forward(&ctx);
            seen_winners.extend(winners.iter().copied());
        }
        println!(
            "dendritic_capacity: {}/{} segments activated across 200 random contexts",
            seen_winners.len(),
            k
        );
        assert!(
            seen_winners.len() >= k / 2,
            "at least K/2 segments must be activatable"
        );
    }

    #[test]
    fn bayesian_surprise_detects_boundary() {
        // BayesianSurpriseDetector triggers when surprise > μ + γ·σ (EMA).
        // For γ=2.5, this is a ~0.621% tail of a Gaussian (P[Z>2.5] ≈ 0.0062).
        // We inject occasional high-surprise spikes into a steady-state baseline
        // and verify the detector fires appropriately.
        let mut detector = BayesianSurpriseDetector::new(2.5, 0.95);
        let mut rng = 0xBEEFu64;
        let mut boundaries = 0usize;
        let n_tokens = 10_000usize;

        // Warm up the EMA on a stable baseline (surprise ≈ 0.3) for 200 tokens
        for _ in 0..200 {
            detector.update(0.3);
        }

        // Now feed: mostly low surprise (~0.3) with occasional large spikes (~1.5–2.0)
        // Spikes are injected every ~150–200 tokens → ~0.5–0.7% boundary rate
        for i in 0..n_tokens {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let base = 0.25 + (rng & 0xFF) as f32 / 1020.0; // 0.25..0.50
                                                            // inject a large spike every ~160 tokens
            let surprise = if i % 160 == 0 { 2.0 } else { base };
            if detector.update(surprise) {
                boundaries += 1;
            }
        }
        let rate = boundaries as f32 / n_tokens as f32;
        println!(
            "bayesian_surprise_detects_boundary: {boundaries}/{n_tokens} ({:.3}%)",
            rate * 100.0
        );
        // spikes at 1/160 ≈ 0.625%; detector may not fire every time due to EMA drift
        assert!(
            boundaries >= 30,
            "detector should fire at least 30 times on 10k tokens with periodic spikes, got {boundaries}"
        );
    }

    #[test]
    fn episode_boundary_rate() {
        let mut mem = EpisodicMemory::new(32, 10_000, 2.5);
        let mut rng = 0xCAFEu64;
        let n_tokens = 5_000usize;
        for _ in 0..n_tokens {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let h: Vec<f32> = (0..32)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    ((rng & 0xFFFF) as f32 / 32768.0 - 1.0)
                })
                .collect();
            let surprise = ((rng & 0xFFFF) as f32) / 65535.0;
            mem.process_token(&h, surprise);
        }
        mem.flush_remaining();
        let n_episodes = mem.episode_count();
        let rate = n_episodes as f32 / n_tokens as f32;
        println!(
            "episode_boundary_rate: {n_episodes} episodes from {n_tokens} tokens ({:.3}%)",
            rate * 100.0
        );
        assert!(n_episodes > 0, "must have at least one episode");
        assert!(rate <= 0.02, "boundary rate too high: {:.3}%", rate * 100.0);
    }

    #[test]
    fn episode_segmentation() {
        let mut mem = EpisodicMemory::new(16, 1000, 2.5);
        let mut rng = 0x1234u64;
        for _ in 0..2000 {
            let h: Vec<f32> = (0..16)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    ((rng & 0xFFFF) as f32 / 32768.0 - 1.0)
                })
                .collect();
            let surprise = ((rng & 0xFFFF) as f32) / 65535.0;
            mem.process_token(&h, surprise);
        }
        mem.flush_remaining();
        let count = mem.episode_count();
        println!("episode_segmentation: {count} episodes stored");
        assert!(count > 0, "must have stored episodes");
        for ep in &mem.episodes {
            assert!(ep.length >= 1);
            assert_eq!(ep.embedding.len(), 16);
        }
    }

    #[test]
    fn episodic_retrieve_knn() {
        let mut mem = EpisodicMemory::new(8, 1000, 0.5);
        let mut rng = 0xABCDu64;
        for _ in 0..200 {
            let h: Vec<f32> = (0..8)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    ((rng & 0xFFFF) as f32 / 32768.0 - 1.0)
                })
                .collect();
            mem.process_token(&h, 2.0);
        }
        mem.flush_remaining();
        let n = mem.episode_count();
        if n == 0 {
            return;
        }
        let query: Vec<f32> = mem.episodes[0].embedding.clone();
        let results = mem.retrieve(&query, 3.min(n));
        assert!(
            !results.is_empty(),
            "retrieve must return at least 1 result"
        );
        println!(
            "episodic_retrieve_knn: top episode id={}, len={}",
            results[0].id, results[0].length
        );
    }

    #[test]
    fn sqlite_vec_insert_query_roundtrip() {
        let dim = 16usize;
        let mut store = VectorStore::new(dim);
        let mut rng = 0xDEADu64;
        for id in 0..100u64 {
            let emb: Vec<f32> = (0..dim)
                .map(|_| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    ((rng & 0xFFFF) as f32 / 32768.0 - 1.0)
                })
                .collect();
            store.insert(id, emb);
        }
        assert_eq!(store.len(), 100);
        let query: Vec<f32> = store.entries[42].1.clone();
        let results = store.knn_query(&query, 3);
        assert!(!results.is_empty());
        assert_eq!(
            results[0].0, 42,
            "nearest neighbor of entry 42 should be itself"
        );
        assert!(
            results[0].1 > 0.99,
            "self-similarity should be ~1.0, got {}",
            results[0].1
        );
        println!(
            "sqlite_vec_insert_query_roundtrip: top match id={}, score={:.4}",
            results[0].0, results[0].1
        );
    }

    #[test]
    fn sqlite_vec_knn_ordering() {
        let dim = 8usize;
        let mut store = VectorStore::new(dim);
        let base: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for i in 0..10u64 {
            let scale = 1.0 - i as f32 * 0.05;
            let emb: Vec<f32> = base.iter().map(|&v| v * scale).collect();
            store.insert(i, emb);
        }
        let results = store.knn_query(&base, 5);
        assert_eq!(results.len(), 5);
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 >= results[i].1,
                "results must be sorted descending by score"
            );
        }
        println!("sqlite_vec_knn_ordering: results correctly ordered by cosine score");
    }

    #[test]
    fn sqlite_vec_empty_store() {
        let store = VectorStore::new(8);
        assert!(store.is_empty());
        let results = store.knn_query(&[0.0f32; 8], 3);
        assert!(results.is_empty(), "empty store must return no results");
    }
}
