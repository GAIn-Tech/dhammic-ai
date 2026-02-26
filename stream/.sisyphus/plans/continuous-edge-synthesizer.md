# Continuous Edge Synthesizer — Production Roadmap

## TL;DR

> **Quick Summary**: Multi-sprint roadmap to build the Continuous Edge Synthesizer (CES) — a biologically-inspired, mathematically-proven AI platform in Rust (candle) + Python (PyO3). Implements continuous stream-of-consciousness inference via spiking SSMs, three-tier hierarchical memory, Hebbian learning, and cortical column routing. Core thesis: 130M parameters with architectural innovation rivals 780M+ standard Transformers.
>
> **Deliverables**:
> - `ces` Rust workspace with 7 crates (ssm, spiking, memory, hebbian, moe, inference, bindings)
> - Working 130M-param CES model (trained on OpenWebText, benchmarked)
> - Python bindings via PyO3 + ONNX export
> - Android cross-compilation target (aarch64)
> - Convergence benchmark results with confidence intervals
> - 5 Lean 4 formal proofs of key architectural properties
>
> **Estimated Effort**: XL (~35-55 agent sessions across 7 sprints)
> **Parallel Execution**: YES — 2-3 waves per sprint
> **Critical Path**: S0 probes → S1 SSM port → S2 memory → S3 Hebbian → S4 MoE → S5 integration → S6 validation

---

## Context

### Original Request
User provided a 116-line theoretical blueprint (`Edge AI: Continuous Stream-of-Consciousness`) proposing a "Continuous Edge Synthesizer" combining SSMs, spiking neurons, episodic memory, and Hebbian learning. Requested a production roadmap using hermeneutic circle analysis, first-principles axiom decomposition, and mathematical rigor via theory2.

### Interview Summary
**Key Discussions** (from extensive brainstorming — see `docs/brainstorms/2026-02-25-continuous-edge-synthesizer-brainstorm.md`):
- Architecture locked: Dual-track SSM (Mamba-2 SSD primary + RWKV v6 secondary), three-tier memory (Engram + Hebbian LoRA + EM-LLM/sqlite-vec), cortical column MoE routing, distributed biological routing
- All 4 HTM-inspired transformations included (SP-inhibition, dendritic gating, multi-scale heads, column MoE), 3 excluded (Semantic Folding=proprietary, Full HTM=legacy, Neural SEM Router=non-biological centralization)
- Mathematical convergence thesis verified: 5.4×–6.75× architectural multiplier (130M → 702M–877M effective params)
- 14 mathematical properties formally verified via theory2 symbolic computation
- 19+ upstream GitHub repos mapped with specific integration plans
- TDD confirmed: RED-GREEN-REFACTOR, cargo test + pytest, property-based tests (quickcheck + hypothesis)
- Fictional technologies identified and removed: SpikySpace (doesn't exist), XAMBA (doesn't exist), CLONE (mischaracterized)

**Research Findings** (7 librarian agents, 29 sequential thinking steps, 14 theory2 computations):
- Mamba-2 is CUDA-only — must port SSD algorithm to Rust/candle
- RWKV is the only edge-deployable SSM (ONNX export, NPU proven)
- DeepSeek Engram (3.8K★, Apache 2.0) proven at 27B scale, <3% overhead offloading 100B params
- Hebbian LoRA has ZERO production implementations — this is novel R&D
- SDR-inspired structured sparsity resolves gradient incompatibility via surrogate gradients

### Metis Review
**Identified Gaps** (addressed):
- Parameter budget allocation: Sprint 0 architecture spec task (not pre-guessed)
- Training pipeline: Added to Sprint 0 (required for convergence gates)
- Baseline model: Default GPT-2 Large (774M) — disclose, user can override
- Benchmark suite: HellaSwag, ARC-Easy, WinoGrande, PIQA, WikiText-103 ppl — disclose
- "Rivals" threshold: ≥90% average benchmark score — disclose
- RESEARCH vs ENGINEERING task tagging: Mandatory for all tasks
- Incremental convergence gates: Every sprint boundary, not just Sprint 6
- Explicit fallbacks: All RESEARCH tasks have defined fallback if they fail
- Dual-track merge strategy: Learned gating (g(x)·mamba + (1-g(x))·rwkv) — disclose
- RWKV version: v6 — disclose
- Tokenizer: SentencePiece BPE 32K vocab — disclose
- Sprint 0 split: Phase A (setup) + Phase B (feasibility probes) — adopted
- 5.4× multiplier validity: Confirmed — math uses ONLY real components (SP-inhibition, Engram, multi-scale, Hebbian), none of the fictional SpikySpace/XAMBA

---

## Work Objectives

### Core Objective
Build and validate a 130M-parameter AI model demonstrating that architectural innovation (spiking SSMs + three-tier memory + Hebbian learning + cortical column routing + SP-inhibition sparsity) achieves performance equivalent to a 780M+ standard Transformer, proving the convergence thesis with mathematical rigor.

### Concrete Deliverables
- `ces-ssm`: Mamba-2 SSD + multi-scale temporal heads in Rust/candle
- `ces-spiking`: Spiking activations + SP-inspired lateral inhibition
- `ces-memory`: Engram + dendritic gating + EM-LLM + sqlite-vec
- `ces-hebbian`: Hebbian LoRA + TreeLoRA continual learning
- `ces-moe`: Cortical Column MoE routing + software DVFS
- `ces-inference`: End-to-end inference pipeline + dual-track merge
- `ces-bindings`: PyO3 Python bindings + ONNX export
- Trained 130M-param CES model checkpoint
- Convergence benchmark results JSON
- 5 Lean 4 formal proofs (or documented proof sketches)

### Definition of Done
- [ ] `cargo test --workspace` → 0 failures
- [ ] `pytest tests/` → 0 failures
- [ ] Convergence multiplier ≥ 5.0× (conservative floor)
- [ ] CES-130M average benchmark score ≥ 90% of GPT-2 Large (774M) on 5-benchmark suite
- [ ] PyO3 bindings: `import ces; model.infer("hello")` → non-empty output
- [ ] ONNX export: output within 1% of native Rust inference
- [ ] Android: `cargo build --target aarch64-linux-android` → produces .so
- [ ] All 5 Lean 4 proofs verified or documented as proof sketches

### Must Have
- Mamba-2 SSD in Rust with numerical equivalence to CUDA reference (max_abs_error < 1e-5 fp32)
- Three-tier memory: Engram (O(1) lookup) + Hebbian LoRA (fast weights) + EM-LLM/sqlite-vec (episodic)
- Spiking activations with ~2% sparsity via SP-inspired lateral inhibition
- Multi-scale temporal heads (K=4, heterogeneous Δt)
- Cortical Column MoE with voting-based consensus routing
- Incremental convergence gates at every sprint boundary
- TDD: RED-GREEN-REFACTOR for every task
- Property-based tests for all mathematical properties
- Explicit fallbacks for all RESEARCH-tagged tasks
- Fork/extend real GitHub repos for every component

### Must NOT Have (Guardrails)
- No simplified substitutes or watered-down algorithms (user mandate)
- No explore agents — librarian or category-based task() only (user ban)
- No human-intervention acceptance criteria — all verification is agent-executed
- No ONNX decomposition of custom ops — export as opaque custom operator nodes
- No full Android app — cross-compilation + .so + inference benchmark only in Sprint 5
- No platform-specific DVFS beyond Linux cpufreq — stubs for other platforms
- No dynamic temporal scale discovery — fixed K=4 heads at design time
- No infinite R&D loops — all RESEARCH tasks time-boxed with explicit fallback
- No silent algorithm simplification — agents must document if they can't implement full version
- No arbitrary exclusion of biologically-plausible concepts (user mandate)

---

## Defaults Applied (from Metis Gap Analysis — Override If Needed)

| Decision | Default | Rationale |
|----------|---------|-----------|
| Baseline model | GPT-2 Large (774M) | Closest to 780M target, widely benchmarked, public weights/scores |
| "Rivals" threshold | ≥90% average score | Concrete falsifiable threshold |
| Benchmark suite | HellaSwag, ARC-Easy, WinoGrande, PIQA, WikiText-103 ppl | Standard LM eval suite, SSM-friendly |
| RWKV version | v6 | Stable, well-documented, multiple reference impls |
| Tokenizer | SentencePiece BPE, 32K vocab | Standard for 130M-scale models |
| Training data | OpenWebText (deduplicated) | Open, standard, reproducible |
| Dual-track merge | Learned gating: `g(x)·mamba + (1-g(x))·rwkv` | Simple, proven, upgradeable to voting |
| Temporal scales (K=4) | Δt ∈ {0.01, 0.1, 1.0, 10.0} | From brainstorm math (1000× frequency range) |
| Android target | aarch64-linux-android, API 26+ | Standard modern Android |
| ONNX scope | SSM inference path only | Custom ops (spiking, dendritic) as custom nodes |

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan are verifiable WITHOUT any human action.
> Every criterion is executed by the agent using tools (Bash, cargo test, pytest, etc.).

### Test Decision
- **Infrastructure exists**: NO (new project — Sprint 0 sets it up)
- **Automated tests**: YES (TDD — RED-GREEN-REFACTOR)
- **Framework**: cargo test (Rust), pytest (Python), quickcheck/proptest (Rust property-based), hypothesis (Python property-based)

### Task Classification
Every task is tagged as **RESEARCH** or **ENGINEERING**:

| Type | Estimation | Fallback Required | Time-Box |
|------|-----------|-------------------|----------|
| **ENGINEERING** | Standard effort estimate | No (predictable outcome) | No |
| **RESEARCH** | Range estimate (optimistic/pessimistic) | YES (explicit fallback defined) | YES (max agent sessions before fallback) |

### Incremental Convergence Gates (CRITICAL)

These gates validate the convergence thesis **incrementally** at every sprint boundary. Failure triggers investigation or fallback — never silent continuation.

| Gate | Sprint | Threshold | What It Proves | Fallback |
|------|--------|-----------|----------------|----------|
| G1 | After Sprint 1 | multiplier ≥ 1.5× | SSM + spiking + sparsity contribute | RWKV-only architecture |
| G2 | After Sprint 2 | multiplier ≥ 3.0× | Memory system amplifies capacity | Debug memory integration |
| G3 | After Sprint 3 | multiplier ≥ 4.5× | Hebbian adaptation works | Standard LoRA + Hebbian LR schedule |
| G4 | After Sprint 4 | multiplier ≥ 5.0× | Full system compounding | Simplify MoE to standard top-k |
| G6 | Sprint 6 | multiplier ≥ 5.4× | Convergence thesis validated | Document partial success + analysis |

Gate measurement: Train CES model at gate, evaluate WikiText-103 perplexity, compare against same-architecture Transformer baseline at various scales. Output: `{"sprint": N, "multiplier": X.Y, "threshold": Z.Z, "pass": bool}`.

### Numerical Equivalence Standards (Mamba-2 Port)
- fp32: max absolute error ≤ 1e-5 vs CUDA reference
- fp16: max absolute error ≤ 1e-2 vs CUDA reference
- Full forward pass relative error: ≤ 0.1%
- Minimum 1000 random input test cases per layer type
- Golden vectors generated from CUDA reference BEFORE Rust port begins

---

## Execution Strategy

### Sprint Overview

| Sprint | Focus | Tasks | Effort (Sessions) | Risk | Gate |
|--------|-------|-------|--------------------|------|------|
| **0** | Foundation + Feasibility | 6 | 6-8 | Low | All probes green |
| **1** | Spiking SSM Core | 6 | 8-12 | **Very High** | G1: ≥1.5× |
| **2** | Three-Tier Memory | 5 | 6-8 | Medium | G2: ≥3.0× |
| **3** | Hebbian LoRA | 4 | 5-8 | **High** | G3: ≥4.5× |
| **4** | MoE + Efficiency | 4 | 4-6 | Medium | G4: ≥5.0× |
| **5** | Integration | 4 | 4-6 | Low | Cross-platform works |
| **6** | Validation | 3 | 5-8 | Medium | G6: ≥5.4× |
| | **TOTAL** | **32** | **38-56** | | |

### Parallel Waves (Per Sprint)

```
SPRINT 0:
  Wave 1: [0.1 workspace] ∥ [0.2 CI]
  Wave 2: [0.3 Lean4] ∥ [0.4 feasibility probes]
  Wave 3: [0.5 arch spec] → [0.6 training infra]

SPRINT 1:
  Wave 1: [1.1 golden vectors]
  Wave 2: [1.2 Mamba-2 port] ∥ [1.3 spiking layer]
  Wave 3: [1.4 SP-inhibition] ∥ [1.5 multi-scale heads]
  Wave 4: [1.6 convergence gate G1]

SPRINT 2:
  Wave 1: [2.1 Engram port] ∥ [2.3 EM-LLM] ∥ [2.4 sqlite-vec]
  Wave 2: [2.2 dendritic gating]
  Wave 3: [2.5 convergence gate G2]

SPRINT 3:
  Wave 1: [3.1 Hebbian LoRA core]
  Wave 2: [3.2 TreeLoRA patterns] ∥ [3.3 forgetting tests]
  Wave 3: [3.4 convergence gate G3]

SPRINT 4:
  Wave 1: [4.1 Column MoE] ∥ [4.2 DVFS]
  Wave 2: [4.3 dual-track merge]
  Wave 3: [4.4 convergence gate G4]

SPRINT 5:
  Wave 1: [5.1 PyO3] ∥ [5.2 ONNX] ∥ [5.3 Android]
  Wave 2: [5.4 integration tests]

SPRINT 6:
  Wave 1: [6.1 full training]
  Wave 2: [6.2 benchmarks] ∥ [6.3 Lean proofs + report]
```

### Dependency Matrix

| Task | Depends On | Blocks | Parallel With |
|------|-----------|--------|---------------|
| 0.1 | None | 0.3-0.6, all S1+ | 0.2 |
| 0.2 | None | — | 0.1 |
| 0.3 | 0.1 | 6.3 | 0.4 |
| 0.4 | 0.1 | 0.5 | 0.3 |
| 0.5 | 0.4 | 0.6 | — |
| 0.6 | 0.5 | 1.6, 2.5, 3.4, 4.4 | — |
| 1.1 | S0 | 1.2 | — |
| 1.2 | 1.1 | 1.4, 1.5 | 1.3 |
| 1.3 | S0 | 1.4 | 1.2 |
| 1.4 | 1.2, 1.3 | 1.6 | 1.5 |
| 1.5 | 1.2 | 1.6 | 1.4 |
| 1.6 | 1.4, 1.5 | S2 | — |
| 2.1 | S1 | 2.2 | 2.3, 2.4 |
| 2.2 | 2.1 | 2.5 | — |
| 2.3 | S1 | 2.5 | 2.1, 2.4 |
| 2.4 | S1 | 2.5 | 2.1, 2.3 |
| 2.5 | 2.2, 2.3, 2.4 | S3 | — |
| 3.1 | S2 | 3.2, 3.3 | — |
| 3.2 | 3.1 | 3.4 | 3.3 |
| 3.3 | 3.1 | 3.4 | 3.2 |
| 3.4 | 3.2, 3.3 | S4 | — |
| 4.1 | S3 | 4.3 | 4.2 |
| 4.2 | S3 | 4.4 | 4.1 |
| 4.3 | 4.1 | 4.4 | — |
| 4.4 | 4.3, 4.2 | S5 | — |
| 5.1 | S4 | 5.4 | 5.2, 5.3 |
| 5.2 | S4 | 5.4 | 5.1, 5.3 |
| 5.3 | S4 | 5.4 | 5.1, 5.2 |
| 5.4 | 5.1, 5.2, 5.3 | S6 | — |
| 6.1 | S5 | 6.2 | — |
| 6.2 | 6.1 | — | 6.3 |
| 6.3 | 6.1 | — | 6.2 |

---

## TODOs

> Every task tagged **[RESEARCH]** or **[ENGINEERING]**. RESEARCH tasks have time-boxes and fallbacks.
> Implementation + Test = ONE Task. Never separate.
> Agent effort in sessions (each ~30-60 min equivalent).

---

### Sprint 0: Foundation + Feasibility

> **Goal**: Workspace, CI, toolchain, feasibility probes, architecture spec, training infra.
> **MUST NOT**: Write any ML algorithm code. Only scaffolding, probes, and specification.
> **Effort**: 6-8 agent sessions.

---

- [x] 0.1. Rust Workspace Scaffolding **[ENGINEERING]**

  **What to do**:
  - Create `ces` Cargo workspace with 7 member crates: `ces-ssm`, `ces-spiking`, `ces-memory`, `ces-hebbian`, `ces-moe`, `ces-inference`, `ces-bindings`
  - Configure shared dependencies: `candle-core`, `candle-nn`, `candle-transformers` (pin exact version), `sqlite-vec` (via rusqlite FFI), `tokenizers` (HuggingFace), `serde`, `criterion` (benchmarks), `proptest` (property-based testing)
  - Create `Cargo.toml` workspace with profile optimizations (`[profile.release]` LTO, codegen-units=1)
  - Add `rust-toolchain.toml` pinning stable + aarch64 targets
  - Create placeholder `lib.rs` + `tests/` for each crate with a single passing smoke test
  - Add `.cargo/config.toml` with Android NDK cross-compilation settings (commented out, activated in Sprint 5)

  **Must NOT do**: Implement any ML logic. Only scaffolding.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`coding-standards`]

  **Parallelization**: Wave 1 — can run in parallel with 0.2. Blocks: 0.3-0.6 and all Sprint 1+.

  **References**:
  - `huggingface/candle` — Cargo.toml structure, dependency versions, feature flags
  - `tracel-ai/burn` — Alternative workspace structure reference
  - `PyO3/pyo3` — PyO3 integration in Cargo workspace (for ces-bindings crate)

  **Acceptance Criteria**:
  - [ ] `cargo build --workspace` → 0 errors
  - [ ] `cargo test --workspace` → 7 smoke tests pass (1 per crate)
  - [ ] `cargo clippy --workspace` → 0 warnings
  - [ ] Each crate has `src/lib.rs` + `tests/*.rs`

  **QA Scenario**:
  ```
  Scenario: Workspace builds and all smoke tests pass
    Tool: Bash
    Steps:
      1. cargo build --workspace 2>&1 | tail -1 → contains "Finished"
      2. cargo test --workspace -- --nocapture 2>&1 → "7 passed" or "test result: ok"
      3. cargo clippy --workspace 2>&1 | grep -c "warning" → 0
    Evidence: Build and test output captured
  ```

  **Effort**: 1 agent session
  **Commit**: YES — `chore(workspace): scaffold CES Rust workspace with 7 crates`

---

- [x] 0.2. CI/CD Pipeline **[ENGINEERING]**

  **What to do**:
  - Create `.github/workflows/ci.yml`: cargo build, cargo test, cargo clippy, cargo fmt --check
  - Add matrix: `{os: [ubuntu-latest], rust: [stable]}`
  - Add caching for cargo registry + target dir
  - Add pytest job (runs after cargo, uses PyO3 — activated in Sprint 5)
  - Add benchmark comparison job (criterion, activated after Sprint 1)

  **Must NOT do**: Add deployment workflows. CI only.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`coding-standards`]

  **Parallelization**: Wave 1 — parallel with 0.1. Blocks: nothing (CI is informational).

  **References**:
  - `huggingface/candle/.github/workflows/` — candle's CI patterns
  - GitHub Actions caching docs for Rust

  **Acceptance Criteria**:
  - [ ] `.github/workflows/ci.yml` exists and is valid YAML
  - [ ] Workflow would trigger on push to main and PRs

  **QA Scenario**:
  ```
  Scenario: CI config is valid
    Tool: Bash
    Steps:
      1. python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" → exits 0
      2. grep -c "cargo test" .github/workflows/ci.yml → ≥ 1
      3. grep -c "cargo clippy" .github/workflows/ci.yml → ≥ 1
    Evidence: Validation output captured
  ```

  **Effort**: 1 agent session
  **Commit**: YES — `ci: add GitHub Actions workflow for Rust workspace`

---

- [x] 0.3. Lean 4 Environment Setup **[ENGINEERING]**

  **What to do**:
  - Install `elan` (Lean 4 version manager) and `lake` (build tool)
  - Create `proofs/` directory with `lakefile.lean` and `lean-toolchain` file
  - Add `Mathlib4` dependency (for real analysis, linear algebra lemmas)
  - Create a trivial proof file (`proofs/CES/Basic.lean`) that imports Mathlib and proves `1 + 1 = 2`
  - Document proof stubs for the 5 required proofs (SP-inhibition convergence, Hebbian convergence, SDR preservation, column voting, convergence multiplier)

  **Must NOT do**: Attempt the actual proofs (those are Sprint 6).

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
  - **Skills**: []

  **Parallelization**: Wave 2 — after 0.1. Parallel with 0.4. Blocks: 6.3.

  **References**:
  - `leanprover/lean4` — installation docs
  - `leanprover-community/mathlib4` — lakefile dependency setup
  - theory2 `theory2_prove_lean` — tool notes (requires `lake` binary)

  **Acceptance Criteria**:
  - [ ] `lake build` in `proofs/` → exits 0
  - [ ] `proofs/CES/Basic.lean` compiles and proves `1 + 1 = 2`
  - [ ] 5 proof stub files exist with `sorry` placeholders

  **QA Scenario**:
  ```
  Scenario: Lean 4 environment compiles trivial proof
    Tool: Bash
    Steps:
      1. cd proofs && lake build 2>&1 | tail -5 → no errors
      2. grep -r "sorry" proofs/CES/ | wc -l → 5 (one per stub)
    Evidence: lake build output captured
  ```

  **Effort**: 1 agent session
  **Commit**: YES — `chore(proofs): setup Lean 4 environment with Mathlib4`

---

- [x] 0.4. Technical Feasibility Probes **[ENGINEERING]**

  **What to do**:
  - **Probe A — candle selective scan**: Implement a minimal 1D selective scan (10 lines) in candle to verify the required tensor ops exist. If custom kernel needed, document which ops are missing.
  - **Probe B — sqlite-vec FFI**: Create a minimal Rust program that opens a sqlite DB, loads the vec extension, inserts 100 random vectors, queries k-nearest-neighbors.
  - **Probe C — Mamba-2 code audit**: Read `state-spaces/mamba` repo, count lines of CUDA/Triton code in the SSD implementation, identify all external dependencies, document the pure-algorithmic core vs hardware-specific code. Output: feasibility report.
  - **Probe D — PyO3 + candle interop**: Build a trivial candle tensor operation exposed via PyO3, call from Python, verify tensor data roundtrips correctly.
  - **Probe E — RWKV v6 state shape**: Document RWKV v6 hidden state dimensionality and update mechanics. Needed for dual-track merge design.

  **Must NOT do**: Build production implementations. These are PROBES — minimal experiments to validate assumptions.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: [`coding-standards`]

  **Parallelization**: Wave 2 — after 0.1. Parallel with 0.3. Blocks: 0.5.

  **References**:
  - `state-spaces/mamba/mamba_ssm/ops/triton/` — Mamba-2 Triton kernels to audit
  - `huggingface/candle/candle-examples/` — candle usage patterns
  - `asg017/sqlite-vec` — FFI integration reference
  - `BlinkDL/RWKV-LM/RWKV-v6/` — RWKV v6 state mechanics

  **Acceptance Criteria**:
  - [ ] Probe A: Report exists documenting candle ops availability for selective scan (YES/NO per required op)
  - [ ] Probe B: `cargo test --package ces-memory -- probe_sqlite_vec` → PASS (insert + query works)
  - [ ] Probe C: Report exists with Mamba-2 SSD line count, dependency list, pure-algorithm extraction plan
  - [ ] Probe D: Python script calls Rust via PyO3, creates tensor, gets result back → values match
  - [ ] Probe E: Document exists with RWKV v6 state shape, update equations, merge compatibility analysis

  **QA Scenario**:
  ```
  Scenario: All 5 probes produce green results
    Tool: Bash
    Steps:
      1. ls probes/reports/ → 5 report files (probe_a.md through probe_e.md)
      2. cargo test --workspace -- probe_ → all probe tests pass
      3. python probes/test_pyo3.py → exits 0, output contains "PASS"
    Evidence: Probe reports and test output captured
  ```

  **Effort**: 2-3 agent sessions (can dispatch parallel sub-agents per probe)
  **Commit**: YES — `chore(probes): validate technical feasibility for all core dependencies`

---

- [x] 0.5. Architecture Specification Document **[ENGINEERING]**

  **What to do**:
  - Create `scripts/verify_param_budget.py` helper that parses the spec and verifies params sum to 130M ± 1M
  - Write `docs/architecture-spec.md` containing:
    - **Parameter budget**: Exact allocation of 130M params across components (embedding, SSM layers, RWKV layers, MoE experts, Engram, LoRA, dendritic gates, SP-inhibition). Must sum to 130M ± 1M.
    - **Layer config**: Number of layers, hidden dim, head count, head dim for each track
    - **Baseline model**: GPT-2 Large (774M) with published benchmark scores (document exact values)
    - **Benchmark suite**: 5 benchmarks with scoring methodology and thresholds
    - **Convergence gate thresholds**: Exact multiplier targets per sprint (from Verification Strategy)
    - **Training config**: Optimizer, LR schedule, batch size, sequence length, total tokens, warmup
    - **Tokenizer**: SentencePiece BPE 32K config
    - **Data**: OpenWebText download + preprocessing pipeline spec
  - This document becomes the SINGLE SOURCE OF TRUTH for all subsequent sprints.

  **Must NOT do**: Implement anything. This is a specification document.

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []
  - Reason: Parameter budget allocation requires careful mathematical reasoning across coupled constraints.

  **Parallelization**: Wave 3 — after 0.4 (needs probe results). Blocks: 0.6.

  **References**:
  - Brainstorm doc `docs/brainstorms/2026-02-25-continuous-edge-synthesizer-brainstorm.md` — mathematical studies, convergence thesis
  - `state-spaces/mamba` — Mamba-2 parameter counts at various scales
  - `BlinkDL/RWKV-LM` — RWKV v6 parameter allocation patterns
  - `deepseek-ai/Engram` — Engram scaling law (75-80% MoE + 20-25% Engram)
  - Probe C report — Mamba-2 SSD architecture details
  - Probe E report — RWKV v6 state shape

  **Acceptance Criteria**:
  - [ ] `docs/architecture-spec.md` exists with all sections listed above
  - [ ] Parameter budget sums to 130M ± 1M (verified by a check script)
  - [ ] GPT-2 Large benchmark scores documented with sources
  - [ ] All 5 convergence gate thresholds have concrete numeric values

  **QA Scenario**:
  ```
  Scenario: Architecture spec is complete and parameter budget is consistent
    Tool: Bash
    Steps:
      1. python scripts/verify_param_budget.py docs/architecture-spec.md → "TOTAL: 130M ± 1M, PASS"
      2. grep -c "GPT-2 Large" docs/architecture-spec.md → ≥ 5
      3. grep -c "Gate G" docs/architecture-spec.md → ≥ 5
    Evidence: Verification output captured
  ```

  **Effort**: 1-2 agent sessions
  **Commit**: YES — `docs: architecture specification with parameter budget and benchmark targets`

---

- [x] 0.6. Minimal Training Infrastructure **[ENGINEERING]**

  **What to do**:
  - Create `ces-train/` Python package (not a Rust crate — training uses PyTorch for flexibility, inference is Rust)
  - Implement: data loading (OpenWebText via HuggingFace datasets), tokenization (SentencePiece), batching, standard cross-entropy loss, AdamW optimizer, cosine LR schedule with warmup
  - Create a minimal training loop that can train ANY `nn.Module`-compatible model (this will be used with PyTorch reference models for gate checks)
  - Create `benchmarks/` directory with gate measurement scripts: `measure_perplexity.py`, `measure_multiplier.py`, `run_gate.py`
  - Training loop must support: checkpoint save/load, WandB logging (optional), gradient accumulation
  - NOTE: The Rust CES model will be exported to PyTorch-compatible format for training, OR training will use a Python reference implementation that mirrors the Rust architecture (decided per sprint based on what's faster).

  **Must NOT do**: Implement the CES model architecture (that's Sprint 1-4). Only the training scaffolding.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: [`backend-patterns`]

  **Parallelization**: Wave 3 — after 0.5 (needs arch spec for config). Blocks: 1.6, 2.5, 3.4, 4.4 (all convergence gates).

  **References**:
  - `state-spaces/mamba/train/` — Mamba-2 training setup patterns
  - `BlinkDL/RWKV-LM/RWKV-v6/train.py` — RWKV training loop reference
  - OpenWebText dataset on HuggingFace

  **Acceptance Criteria**:
  - [ ] `pip install -e ces-train/` → succeeds
  - [ ] `python -m ces_train.test_loop --steps 10 --model dummy` → completes without error
  - [ ] `python benchmarks/measure_perplexity.py --model dummy --data test` → outputs perplexity number
  - [ ] `python benchmarks/run_gate.py --sprint 0 --model dummy` → outputs JSON with `{"sprint": 0, "pass": true}`
  - [ ] Checkpoint save/load roundtrip: save → load → forward pass produces identical output

  **QA Scenario**:
  ```
  Scenario: Training loop runs 10 steps on dummy model
    Tool: Bash
    Steps:
      1. pip install -e ces-train/ → exits 0
      2. python -m ces_train.test_loop --steps 10 --model dummy → "Step 10/10 complete"
      3. ls checkpoints/ → checkpoint file exists
      4. python benchmarks/run_gate.py --sprint 0 --model dummy → valid JSON output
    Evidence: Training output and gate JSON captured
  ```

  **Effort**: 1-2 agent sessions
  **Commit**: YES — `feat(train): minimal training infrastructure with gate measurement`

---

### Sprint 0 Go/No-Go Gate

**BEFORE proceeding to Sprint 1, ALL must be true**:
- [ ] All 5 feasibility probes return GREEN
- [ ] Architecture spec complete with parameter budget
- [ ] Training infrastructure runs on dummy model
- [ ] CI pipeline configured
- [ ] If Probe A (candle selective scan) is RED: Document missing ops, estimate custom kernel effort. If >5 agent sessions for custom kernels: **PIVOT to candle-alternative (burn?) or PyTorch-only training with Rust inference port**.

---

### Sprint 1: Spiking SSM Core

> **Goal**: Port Mamba-2 SSD to Rust/candle, add spiking activations, SP-inhibition, multi-scale heads.
> **MUST NOT**: Implement memory, MoE, or adaptation. SSM core ONLY.
> **Risk**: VERY HIGH — novel Rust port of CUDA-only algorithm + spiking activations have zero precedent in SSMs.
> **Effort**: 8-12 agent sessions.

---

- [x] 1.1. Mamba-2 SSD Reference Extraction + Golden Vectors **[ENGINEERING]**

  **What to do**:
  - Clone `state-spaces/mamba`, isolate the SSD (Structured State Space Duality) algorithm
  - Extract the pure-algorithmic selective scan logic from Triton kernels into readable Python
  - Generate golden test vectors: 1000+ random inputs at various sequence lengths (128, 512, 2048), hidden dims matching arch spec
  - Save golden vectors as `.npz` files: input → expected output for every layer type
  - Document the SSD algorithm in pseudocode (no CUDA/Triton specifics)
  - Identify numerical precision requirements per operation

  **Must NOT do**: Write any Rust code yet. This task is REFERENCE ONLY.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Requires careful extraction of algorithm from heavily optimized CUDA/Triton code.

  **Parallelization**: Wave 1 — first task in Sprint 1. Blocks: 1.2.

  **References**:
  - `state-spaces/mamba/mamba_ssm/ops/triton/ssd_combined.py` — core SSD Triton kernel
  - `state-spaces/mamba/mamba_ssm/modules/mamba2.py` — Mamba-2 module (high-level)
  - Brainstorm: "SSD proves SSMs and attention are DUAL — same math, different compute paths"
  - Probe C report from Sprint 0 (Mamba-2 audit)

  **Acceptance Criteria**:
  - [ ] `ssd_reference/pseudocode.md` documents full SSD algorithm in math notation
  - [ ] `ssd_reference/extract.py` runs SSD forward pass on reference inputs
  - [ ] `ssd_reference/golden_vectors/` contains .npz files (≥1000 test cases)
  - [ ] `python ssd_reference/verify.py` → all golden vectors match Mamba-2 output (tolerance 1e-6)

  **QA Scenario**:
  ```
  Scenario: Golden vectors match Mamba-2 reference
    Tool: Bash
    Steps:
      1. python ssd_reference/extract.py --generate-golden --n-cases 1000 → exits 0
      2. ls ssd_reference/golden_vectors/*.npz | wc -l → ≥ 3 (one per seq_len)
      3. python ssd_reference/verify.py → "1000/1000 vectors match, max_error=X.Xe-Y"
    Evidence: Verification output captured
  ```

  **Effort**: 2 agent sessions
  **Commit**: YES — `feat(reference): extract Mamba-2 SSD algorithm with golden test vectors`

---

- [x] 1.2. Mamba-2 SSD Rust/candle Port **[RESEARCH]**

  **What to do**:
  - Implement the selective scan (SSD) algorithm in `ces-ssm` crate using candle tensor ops
  - Follow pseudocode from 1.1, NOT the CUDA/Triton kernels
  - Implement: discretization (Δ → A_bar, B_bar), parallel scan, output projection
  - Handle multi-head structure per architecture spec
  - Write numerical equivalence tests against golden vectors from 1.1
  - Property-based tests: associativity of scan operator, stability under random perturbation

  **Must NOT do**: Add spiking, sparsity, multi-scale, or MoE. Pure SSD only.

  **Time-box**: 4 agent sessions. **Fallback**: If candle lacks required ops after 4 sessions, use `burn` framework or implement critical ops as raw BLAS calls via `ndarray`.

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []
  - Reason: Novel port requiring deep understanding of both SSM math and Rust/candle API.

  **Parallelization**: Wave 2 — after 1.1. Parallel with 1.3. Blocks: 1.4, 1.5.

  **References**:
  - `ssd_reference/pseudocode.md` — algorithm specification (from 1.1)
  - `ssd_reference/golden_vectors/` — test vectors (from 1.1)
  - `huggingface/candle/candle-transformers/src/models/mamba.rs` — candle's Mamba-1 impl (reference for candle API patterns, NOT SSD)
  - `fla-org/flash-linear-attention` — DeltaNet/FWP production kernels as scan reference
  - Architecture spec `docs/architecture-spec.md` — hidden dim, head count, layer count

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-ssm -- numerical_equivalence` → 0 failures, 1000+ test cases
  - [ ] Max absolute error < 1e-5 (fp32) across all golden vectors
  - [ ] `cargo test --package ces-ssm -- property_` → proptest passes (100+ cases per property)
  - [ ] Forward pass on seq_len=2048: completes without OOM on 16GB RAM

  **QA Scenario**:
  ```
  Scenario: SSD port matches reference within tolerance
    Tool: Bash
    Steps:
      1. cargo test --package ces-ssm -- numerical_equivalence --nocapture 2>&1 → "test result: ok"
      2. cargo test --package ces-ssm -- numerical_equivalence --nocapture 2>&1 | grep "max_abs_error" → value < 1e-5
      3. cargo test --package ces-ssm -- property_ --nocapture → "test result: ok"
    Evidence: Test output captured with error statistics
  ```

  **Effort**: 3-4 agent sessions (RESEARCH — may need iteration)
  **Commit**: YES — `feat(ssm): Mamba-2 SSD selective scan in Rust/candle`

---

- [x] 1.3. Spiking Activation Layer **[RESEARCH]**

  **What to do**:
  - Implement `ces-spiking` crate with:
    - `PTsoftplus` activation: bit-shift approximation of softplus (from SpikeGPT patterns)
    - `PTSiLU` activation: bit-shift SiLU variant
    - Surrogate gradient functions: `1/(1 + β|x|)²` with configurable β
    - Forward: exact spiking (threshold + reset). Backward: surrogate gradient (straight-through estimator with smooth surrogate)
  - Property-based tests: surrogate gradient bounded by β/2, surrogate approaches true gradient as β→∞, activation sparsity measurable
  - Benchmark: spiking forward pass vs standard activations (FLOP count comparison)
  - Verify theory2 results: surrogate_grad(β=10, x=0.1) ≈ 1.250

  **Must NOT do**: Add lateral inhibition (that's 1.4) or integrate with SSM (that's 1.4-1.5).

  **Time-box**: 2 agent sessions. **Fallback**: If bit-shift approximation has >5% error vs exact softplus, use standard softplus with sparsity mask.

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []
  - Reason: Surrogate gradient implementation requires careful numerical handling.

  **Parallelization**: Wave 2 — parallel with 1.2 (independent). Blocks: 1.4.

  **References**:
  - `ridgerchu/SpikeGPT` — spiking activation patterns, surrogate gradient reference
  - Brainstorm: surrogate gradient formula `β/(2(1+β|x|)²)`, verified values
  - Dragon Hatchling paper (arxiv 2509.26507) — Hebbian + spiking neuron patterns
  - theory2 verified: surrogate_grad(β=10, x=0.1) = 1.250, surrogate_grad(β=10, x=1.0) = 0.04132

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-spiking -- activation_` → all pass
  - [ ] `cargo test --package ces-spiking -- surrogate_gradient_values` → matches theory2 verified values within 1e-6
  - [ ] `cargo test --package ces-spiking -- property_gradient_bounded` → proptest confirms bound ≤ β/2
  - [ ] `cargo bench --package ces-spiking -- activation_benchmark` → spiking FLOPS < 50% of standard (bit-shift advantage)

  **QA Scenario**:
  ```
  Scenario: Spiking activations match theory2 verified values
    Tool: Bash
    Steps:
      1. cargo test --package ces-spiking -- surrogate_gradient_values --nocapture → values within 1e-6 of theory2
      2. cargo test --package ces-spiking -- property_ → proptest ok (100+ cases)
      3. cargo bench --package ces-spiking → benchmark results captured
    Evidence: Test output with numerical values captured
  ```

  **Effort**: 1-2 agent sessions
  **Commit**: YES — `feat(spiking): spiking activations with surrogate gradients`

---

- [x] 1.4. SP-Inspired Lateral Inhibition **[RESEARCH]**

  **What to do**:
  - Implement in `ces-spiking` crate:
    - Competitive lateral inhibition layer (replaces naive Top-K)
    - Online Hebbian learning on inhibitory weights (SP-inspired: boost underactive neurons, suppress overactive)
    - Target: ~2% activation sparsity (SDR-inspired, from brainstorm math: 40/2048)
    - Lyapunov stability: inhibition weights converge to stable equilibrium (Cohen-Grossberg 1983)
  - Integrate with spiking activation from 1.3: spike → inhibit → sparse output
  - Property-based tests: sparsity level in [1%, 5%] across random inputs, inhibitory weights bounded
  - Verify brainstorm math: interference reduction = (N/k)² = 2621× for 2% sparsity

  **Must NOT do**: Full HTM Spatial Pooler. SP-INSPIRED lateral inhibition only (simpler, gradient-compatible).

  **Time-box**: 3 agent sessions. **Fallback**: If competitive inhibition doesn't converge, use deterministic Top-K with learnable temperature.

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []
  - Reason: Novel algorithm combining spiking neurons with competitive inhibition — no reference implementation exists.

  **Parallelization**: Wave 3 — after 1.2 and 1.3. Parallel with 1.5. Blocks: 1.6.

  **References**:
  - `htm-community/htm.core` — SP algorithm reference (C++17, patterns only — not porting the framework)
  - Numenta 2017: "The HTM Spatial Pooler" — online learning algorithm for sparse coding
  - Cohen-Grossberg 1983: Lyapunov convergence theorem for competitive networks
  - Brainstorm math: sparsity interference = 1/s² = (N/k)² = 2621×, combined isolation ≈ 83,872×
  - theory2 verified: `(2048/40)**2` = 2621.44

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-spiking -- sp_inhibition_` → all pass
  - [ ] `cargo test --package ces-spiking -- property_sparsity_range` → sparsity in [1%, 5%] for 95% of random inputs
  - [ ] `cargo test --package ces-spiking -- inhibition_convergence` → weights stabilize within 1000 steps (variance < 1e-4)
  - [ ] `cargo test --package ces-spiking -- interference_reduction` → measured reduction ≥ 2000× (conservative floor of 2621×)

  **QA Scenario**:
  ```
  Scenario: SP-inhibition achieves target sparsity and converges
    Tool: Bash
    Steps:
      1. cargo test --package ces-spiking -- sp_inhibition_ --nocapture → all pass
      2. cargo test --package ces-spiking -- property_sparsity_range --nocapture → "95%+ inputs in [1%, 5%]"
      3. cargo test --package ces-spiking -- inhibition_convergence --nocapture → "converged at step N"
    Evidence: Test output with sparsity statistics captured
  ```

  **Effort**: 2-3 agent sessions (RESEARCH)
  **Commit**: YES — `feat(spiking): SP-inspired lateral inhibition with ~2% sparsity`

---

- [x] 1.5. Multi-Scale Temporal Heads **[RESEARCH]**

  **What to do**:
  - Modify `ces-ssm` to support heterogeneous Δt per head group:
    - K=4 head groups with Δt ∈ {0.01, 0.1, 1.0, 10.0} (from brainstorm math)
    - Each group processes same input but with different temporal resolution
    - Fast heads (Δt=0.01): token-level features, τ≈100 steps
    - Slow heads (Δt=10): document-level features, near-instant decay
    - Output: concatenation of all head group outputs (merged via projection)
  - Verify brainstorm math: K=4, r=10 → 1000× frequency range coverage
  - Property-based tests: each head group's effective memory length scales with 1/Δt, stability for all Δt values

  **Must NOT do**: Dynamic Δt selection. Fixed values per architecture spec.

  **Time-box**: 2 agent sessions. **Fallback**: If heterogeneous Δt causes training instability, use learnable Δt initialized at these values.

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []
  - Reason: Modifying SSM core math for multi-scale operation.

  **Parallelization**: Wave 3 — after 1.2. Parallel with 1.4. Blocks: 1.6.

  **References**:
  - `ces-ssm` crate from task 1.2 — base SSD implementation to modify
  - HiPPO/S4 papers — multi-scale state space theory
  - Architecture spec — head group allocation
  - Brainstorm math: exp(-0.01)=0.990/step, exp(-10)=4.54×10⁻⁵/step, 1000× range

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-ssm -- multi_scale_` → all pass
  - [ ] `cargo test --package ces-ssm -- property_memory_length_scales` → fast heads remember longer, slow heads forget faster
  - [ ] `cargo test --package ces-ssm -- frequency_range_coverage` → measured range ≥ 500× (conservative floor of 1000×)
  - [ ] Numerical equivalence maintained: multi-scale output on uniform Δt matches single-scale output

  **QA Scenario**:
  ```
  Scenario: Multi-scale heads cover 1000× frequency range
    Tool: Bash
    Steps:
      1. cargo test --package ces-ssm -- multi_scale_ --nocapture → all pass
      2. cargo test --package ces-ssm -- frequency_range_coverage --nocapture → "range: N×" where N ≥ 500
    Evidence: Test output captured
  ```

  **Effort**: 1-2 agent sessions
  **Commit**: YES — `feat(ssm): multi-scale temporal heads with heterogeneous Δt`

---

- [x] 1.6. Sprint 1 Convergence Gate + RWKV Secondary Track **[ENGINEERING]**

  **What to do**:
  - Implement RWKV v6 inference in `ces-ssm` (can use existing ONNX model via `onnxruntime` crate, or minimal Rust port of WKV mechanism)
  - Implement learned gating merge: `g(x)·mamba_out + (1-g(x))·rwkv_out` where g is a small linear layer
  - Create a Python-compatible CES model definition (PyTorch) that mirrors the Rust SSM architecture (for training)
  - Train a 130M CES-SSM model (SSM core only, no memory/Hebbian/MoE) on OpenWebText subset
  - Run convergence gate G1: measure WikiText-103 perplexity, compare against same-size Transformer baseline
  - Output: `gate_results/sprint1.json` with `{"sprint": 1, "multiplier": X.Y, "threshold": 1.5, "pass": bool}`

  **Must NOT do**: Add memory, Hebbian, or MoE components. SSM-only validation.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: End-to-end training + evaluation pipeline with careful measurement.

  **Parallelization**: Wave 4 — after 1.4, 1.5. Blocks: Sprint 2.

  **References**:
  - `BlinkDL/RWKV-LM/RWKV-v6/` — RWKV v6 reference implementation
  - `RWKV/rwkv-onnx` — ONNX export for RWKV
  - `ces-train/` from task 0.6 — training infrastructure
  - Architecture spec — model config, training hyperparams
  - Probe E report — RWKV v6 state shape and merge compatibility

  **Acceptance Criteria**:
  - [ ] RWKV inference produces valid output: `cargo test --package ces-ssm -- rwkv_forward` → PASS
  - [ ] Dual-track merge tested: `cargo test --package ces-ssm -- dual_track_merge` → PASS
  - [ ] Training completes on OpenWebText subset (≥100M tokens)
  - [ ] `gate_results/sprint1.json` exists with valid measurements
  - [ ] **GATE G1: multiplier ≥ 1.5×** — if FAIL, trigger investigation before Sprint 2

  **QA Scenario**:
  ```
  Scenario: Sprint 1 convergence gate passes
    Tool: Bash
    Steps:
      1. python benchmarks/run_gate.py --sprint 1 → JSON output
      2. python -c "import json; d=json.load(open('gate_results/sprint1.json')); assert d['pass'], f'GATE FAILED: {d}'"
      3. Assert: multiplier value ≥ 1.5
    Evidence: gate_results/sprint1.json captured
  ```

  **Effort**: 2-3 agent sessions
  **Commit**: YES — `feat(ssm): RWKV secondary track + dual merge + Sprint 1 convergence gate`

---

### Sprint 1 Go/No-Go Gate

**GATE G1: Effective multiplier ≥ 1.5×**
- 130M SSM-only perplexity ≤ 195M Transformer perplexity on WikiText-103
- **IF PASS**: Proceed to Sprint 2 (memory system)
- **IF FAIL (multiplier < 1.5×)**: STOP. Investigate:
  1. Is spiking sparsity too aggressive? Test at 5%, 10%
  2. Is multi-scale Δt helping? Ablation study
  3. **Fallback**: RWKV-only architecture (already edge-proven)

---

### Sprint 2: Three-Tier Memory System

> **Goal**: Engram conditional memory, dendritic gating, EM-LLM episodic memory, sqlite-vec storage.
> **MUST NOT**: Add online learning. Memory is read/write but NOT self-modifying weights.
> **Risk**: Medium — Engram has reference impl, EM-LLM has reference impl, sqlite-vec is production.
> **Effort**: 6-8 agent sessions.

---

- [x] 2.1. Engram Conditional Memory Port **[ENGINEERING]**

  **What to do**:
  - Port DeepSeek Engram to Rust in `ces-memory` crate:
    - N-gram hash function (deterministic addressing)
    - Multi-head embedding table (conditional lookup)
    - Context-aware gating (Query/Key/Value projections)
    - Depthwise causal convolution
    - Residual connection
  - Follow `deepseek-ai/Engram` reference implementation exactly
  - Generate golden vectors from Python reference, test Rust port against them
  - Verify: O(1) lookup latency, <3% overhead on forward pass

  **Must NOT do**: Add dendritic gating (that's 2.2). Pure Engram as specified in the paper.

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []
  - Reason: Porting a specific paper implementation with numerical precision requirements.

  **Parallelization**: Wave 1 — parallel with 2.3, 2.4. Blocks: 2.2.

  **References**:
  - `deepseek-ai/Engram` — reference implementation (3.8K★, Apache 2.0)
  - arxiv 2601.07372 — Engram paper (architecture details, scaling law, gating mechanism)
  - Brainstorm: "O(1) hashed N-gram lookup, 27B proven, BBH +5.0, NIAH 84.2→97.0"
  - Architecture spec — Engram allocation (~25% of sparse params, from scaling law)

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-memory -- engram_numerical_equivalence` → matches Python reference (max_abs_error < 1e-5)
  - [ ] `cargo bench --package ces-memory -- engram_lookup_latency` → O(1) confirmed (constant regardless of table size)
  - [ ] `cargo test --package ces-memory -- engram_overhead` → forward pass overhead < 5% vs without Engram
  - [ ] `cargo test --package ces-memory -- engram_gating` → gating correctly modulates output

  **QA Scenario**:
  ```
  Scenario: Engram port matches DeepSeek reference
    Tool: Bash
    Steps:
      1. cargo test --package ces-memory -- engram_ --nocapture → all pass
      2. cargo bench --package ces-memory -- engram_lookup → "O(1): latency stable across table sizes"
    Evidence: Test output and benchmark results captured
  ```

  **Effort**: 2-3 agent sessions
  **Commit**: YES — `feat(memory): Engram conditional memory port from DeepSeek`

---

- [x] 2.2. Active Dendritic Gating **[RESEARCH]**

  **What to do**:
  - Implement active dendritic gating in `ces-memory` crate:
    - K dendritic segments per Engram gate (from Grewal et al. 2022)
    - Each segment receives different context input
    - Winner-take-all: segment with max activation modulates the gate
    - This TRANSFORMS Engram's existing Q/K/V gating into context-dependent multi-path gating
  - Property-based tests: different contexts activate different segments, WTA is sharp (max >> second-max)
  - Verify brainstorm math: K=8 segments → 8× capacity increase per gate

  **Must NOT do**: Online learning of dendritic weights (that's Sprint 3 territory). Fixed weights for now.

  **Time-box**: 2 agent sessions. **Fallback**: If dendritic gating doesn't improve over standard gating, keep standard Engram gating and mark dendrites as "no improvement observed."

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []

  **Parallelization**: Wave 2 — after 2.1 (needs Engram base). Blocks: 2.5.

  **References**:
  - Grewal et al. 2022: "Avoiding Catastrophe: Active Dendrites Enable Multi-Task Learning"
  - `numenta/nupic.torch` — Active dendrites PyTorch impl (patterns, not code)
  - `ces-memory` Engram impl from 2.1 — base to enhance
  - Brainstorm: "K dendritic segments per gate, winner-take-all context selection"

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-memory -- dendritic_gating_` → all pass
  - [ ] `cargo test --package ces-memory -- property_different_contexts_different_segments` → proptest passes
  - [ ] `cargo test --package ces-memory -- dendritic_capacity` → K=8 segments demonstrably increase effective capacity
  - [ ] Integration test: Engram + dendrites produces valid output

  **QA Scenario**:
  ```
  Scenario: Dendritic gating activates different segments for different contexts
    Tool: Bash
    Steps:
      1. cargo test --package ces-memory -- dendritic_gating_ --nocapture → all pass
      2. cargo test --package ces-memory -- property_different_contexts --nocapture → proptest ok
    Evidence: Test output captured
  ```

  **Effort**: 1-2 agent sessions (RESEARCH)
  **Commit**: YES — `feat(memory): active dendritic gating on Engram context`

---

- [x] 2.3. EM-LLM Episodic Memory + Bayesian Surprise **[ENGINEERING]**

  **What to do**:
  - Port EM-LLM episodic memory logic to Rust in `ces-memory` crate:
    - Bayesian surprise computation: online mean + variance of prediction error
    - Event boundary detection: trigger when surprise > μ + γσ (from brainstorm: γ=2.5 → 0.621% of tokens)
    - Episode segmentation: group tokens between boundaries into episodes
    - Episode embedding: compress episode into vector for storage
    - Retrieval: k-nearest-neighbor lookup of relevant past episodes
  - Use `ces-spiking` SP-sparse activations as the signal for surprise computation (the SLOW SSM head drives boundaries — from brainstorm emergent property #4)

  **Must NOT do**: Implement vector storage (that's 2.4 sqlite-vec task).

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Porting reference implementation with statistical computation.

  **Parallelization**: Wave 1 — parallel with 2.1, 2.4. Blocks: 2.5.

  **References**:
  - `em-llm/EM-LLM-model` (211★) — reference implementation
  - `KIC/bayesian_changepoint_detection` — additional surprise computation reference
  - Brainstorm math: γ=2.5 → 1−Φ(γ) = 0.621% → 1/161 tokens trigger episodic storage
  - Brainstorm emergent property: "Slow SSM head drives episodic boundaries (semantic, not noise)"

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-memory -- bayesian_surprise_` → all pass
  - [ ] `cargo test --package ces-memory -- episode_boundary_rate` → boundary rate ≈ 0.5%-1.0% of tokens (near theoretical 0.621%)
  - [ ] `cargo test --package ces-memory -- episode_segmentation` → episodes have reasonable lengths (10-500 tokens)
  - [ ] Property test: surprise threshold γ produces correct trigger rate (within 2× of theoretical)

  **QA Scenario**:
  ```
  Scenario: Bayesian surprise detects episode boundaries at correct rate
    Tool: Bash
    Steps:
      1. cargo test --package ces-memory -- bayesian_surprise_ --nocapture → all pass
      2. cargo test --package ces-memory -- episode_boundary_rate --nocapture → rate in [0.3%, 1.5%]
    Evidence: Test output with boundary rate statistics captured
  ```

  **Effort**: 2 agent sessions
  **Commit**: YES — `feat(memory): EM-LLM episodic memory with Bayesian surprise boundaries`

---

- [x] 2.4. sqlite-vec Integration **[ENGINEERING]**

  **What to do**:
  - Integrate sqlite-vec in `ces-memory` crate via `rusqlite` FFI:
    - Create episode vector table (episode_id, embedding BLOB, metadata JSON)
    - Insert episode embeddings from 2.3 into sqlite-vec
    - k-nearest-neighbor query for episode retrieval
    - WAL mode for concurrent read/write during continuous inference
    - BoundlessBPE compression for episode text storage (optional — standard BPE fallback)
  - Benchmark: insert latency, query latency at 1K/10K/100K episodes

  **Must NOT do**: Implement episode segmentation (that's 2.3) or Engram (that's 2.1).

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
  - **Skills**: [`backend-patterns`]
  - Reason: Standard database integration — well-understood engineering.

  **Parallelization**: Wave 1 — parallel with 2.1, 2.3. Blocks: 2.5.

  **References**:
  - `asg017/sqlite-vec` — sqlite-vec documentation and API
  - Probe B from Sprint 0 — sqlite-vec FFI validation
  - `kensho-technologies/boundlessbpe` — BoundlessBPE reference (optional compression)

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-memory -- sqlite_vec_` → all pass
  - [ ] `cargo test --package ces-memory -- sqlite_vec_insert_query_roundtrip` → insert 100 vectors, query nearest, correct results
  - [ ] `cargo bench --package ces-memory -- sqlite_vec_query_latency` → p99 < 10ms for k=10 on 100K vectors
  - [ ] WAL mode: concurrent read during write doesn't block

  **QA Scenario**:
  ```
  Scenario: sqlite-vec insert and k-NN query work correctly
    Tool: Bash
    Steps:
      1. cargo test --package ces-memory -- sqlite_vec_ --nocapture → all pass
      2. cargo bench --package ces-memory -- sqlite_vec_query → p99 latency logged
    Evidence: Test and benchmark output captured
  ```

  **Effort**: 1 agent session
  **Commit**: YES — `feat(memory): sqlite-vec integration for episodic vector storage`

---

- [x] 2.5. Sprint 2 Convergence Gate **[ENGINEERING]**

  **What to do**:
  - Integrate all memory components with SSM from Sprint 1:
    - Engram + dendritic gating receives SSM hidden states
    - EM-LLM stores/retrieves episodes during continuous inference
    - Memory output injected back into SSM via residual connection
  - Train CES model with memory (retrain or fine-tune from Sprint 1 checkpoint)
  - Run convergence gate G2: measure WikiText-103 perplexity, compare against Sprint 1 result
  - Memory ablation: measure perplexity WITH vs WITHOUT memory (must be statistically significant improvement, p<0.05)
  - Output: `gate_results/sprint2.json`

  **Must NOT do**: Add Hebbian learning or MoE.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**: Wave 3 — after 2.2, 2.3, 2.4. Blocks: Sprint 3.

  **References**:
  - Sprint 1 trained model checkpoint
  - `ces-train/` training infrastructure
  - Architecture spec — memory integration points

  **Acceptance Criteria**:
  - [ ] End-to-end forward pass with memory: `cargo test --package ces-inference -- memory_integration` → PASS
  - [ ] Training completes with memory active
  - [ ] Memory ablation: `python benchmarks/memory_ablation.py` → p < 0.05 improvement
  - [ ] `gate_results/sprint2.json` exists with valid measurements
  - [ ] **GATE G2: multiplier ≥ 3.0×** — if FAIL, investigate memory contribution

  **QA Scenario**:
  ```
  Scenario: Sprint 2 convergence gate passes with memory contribution
    Tool: Bash
    Steps:
      1. python benchmarks/run_gate.py --sprint 2 → JSON output
      2. python -c "import json; d=json.load(open('gate_results/sprint2.json')); assert d['pass']"
      3. python benchmarks/memory_ablation.py → "p-value: X.XXX, SIGNIFICANT"
    Evidence: gate_results/sprint2.json and ablation results captured
  ```

  **Effort**: 2 agent sessions
  **Commit**: YES — `feat(memory): three-tier memory integration + Sprint 2 convergence gate`

---

### Sprint 2 Go/No-Go Gate

**GATE G2: Effective multiplier ≥ 3.0×**
- 130M + memory perplexity ≤ 390M Transformer perplexity
- Memory ablation shows statistically significant improvement
- **IF PASS**: Proceed to Sprint 3 (Hebbian LoRA)
- **IF FAIL (multiplier < 3.0×)**: Debug memory integration. Check: Is Engram retrieving relevant content? Are episode boundaries sensible? Is dendritic gating selecting useful contexts?

---

### Sprint 3: Hebbian LoRA Adaptation Engine

> **Goal**: Forward-only Hebbian LoRA, TreeLoRA/CL-LoRA continual learning, catastrophic forgetting prevention.
> **MUST NOT**: Touch MoE routing. Hebbian LoRA is isolated to weight adaptation.
> **Risk**: HIGH — Hebbian LoRA has ZERO production implementations. This is novel R&D.
> **Effort**: 5-8 agent sessions.

---

- [x] 3.1. Hebbian LoRA Core **[RESEARCH]**

  **What to do**:
  - Implement `ces-hebbian` crate:
    - Oja's rule for rank-r LoRA factor updates: ΔW = η(xy^T - λW) applied to LoRA A and B matrices
    - Forward-only computation: no backprop, no activation caching — uses only current activations
    - SP-sparse activations (from ces-spiking) as input — reduces interference by ~2621× (brainstorm math)
    - Competitive learning variant: neurons that fire together wire together, neurons that don't compete
    - Learning rate scheduling: Hebbian LR decays as weights approach fixed point W* = xy^T/λ
  - Micro-model convergence test: train 1M-param model with Hebbian LoRA for 1000 steps, verify loss decreases
  - Property-based tests: weight updates bounded, fixed-point stability, rank preservation

  **Must NOT do**: Standard backprop LoRA. This MUST be forward-only Hebbian. No gradient-based adaptation.

  **Time-box**: 4 agent sessions. **Fallback**: If Hebbian LoRA doesn't converge after 4 sessions, implement standard LoRA with Hebbian-inspired learning rate schedule (LR proportional to activation correlation). Document what failed and why.

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []
  - Reason: Novel algorithm with no reference implementation. Requires deep mathematical understanding.

  **Parallelization**: Wave 1 — first task in Sprint 3. Blocks: 3.2, 3.3.

  **References**:
  - Brainstorm math: Hebbian fixed point W* = xy^T/λ (theory2 verified), LoRA params = 294,912 (12 layers × r=16 × 2 × d=768)
  - Dragon Hatchling (arxiv 2509.26507) — Hebbian plasticity + spiking neuron architecture
  - Hebbian Transformers (arxiv 2510.21908) — Hebbian + gradient-based plasticity patterns
  - `IDSIA/lmtool-fwp` — Original Fast Weight Programmer (Schmidhuber) — closest existing implementation pattern
  - `fla-org/flash-linear-attention` — DeltaNet as fast weight reference

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-hebbian -- oja_rule_` → all pass
  - [ ] `cargo test --package ces-hebbian -- property_weight_update_bounded` → max delta < 0.1 per step
  - [ ] `cargo test --package ces-hebbian -- property_fixed_point_stability` → converges to W* ≈ xy^T/λ
  - [ ] `cargo test --package ces-hebbian -- property_rank_preserved` → LoRA matrices maintain rank ≤ r throughout
  - [ ] Micro-model test: `python benchmarks/sprint3_hebbian_micro.py` → loss decreases over 1000 steps (≤5% non-monotonic)

  **QA Scenario**:
  ```
  Scenario: Hebbian LoRA converges on micro-model
    Tool: Bash
    Steps:
      1. cargo test --package ces-hebbian -- oja_rule_ --nocapture → all pass
      2. python benchmarks/sprint3_hebbian_micro.py → "Converged: loss decreased from X to Y over 1000 steps"
      3. Assert: final_loss < initial_loss * 0.8 (at least 20% improvement)
    Evidence: Test output and convergence plot captured
  ```

  **Effort**: 3-4 agent sessions (RESEARCH — may require multiple approaches)
  **Commit**: YES — `feat(hebbian): Hebbian LoRA with Oja's rule and competitive learning`

---

- [x] 3.2. TreeLoRA / CL-LoRA Continual Learning Patterns **[RESEARCH]**

  **What to do**:
  - Enhance `ces-hebbian` with continual learning patterns:
    - TreeLoRA pattern: layer-wise LoRA allocation — different layers get different rank based on importance
    - CL-LoRA pattern: rehearsal-free adaptation — new knowledge doesn't require replay of old knowledge
    - Combine with Hebbian: Hebbian drives weight updates, TreeLoRA/CL-LoRA structure prevents catastrophic forgetting
  - Implement importance scoring: which layers/weights are most critical for existing knowledge (Fisher information or activation variance)
  - Property tests: importance scores are stable, allocation changes smoothly

  **Must NOT do**: Gradient-based rehearsal or experience replay (we're forward-only).

  **Time-box**: 2 agent sessions. **Fallback**: Skip TreeLoRA/CL-LoRA structure, use uniform LoRA rank across layers.

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []

  **Parallelization**: Wave 2 — after 3.1. Parallel with 3.3. Blocks: 3.4.

  **References**:
  - TreeLoRA (arxiv 2506.10355) — layer-wise LoRA allocation for continual learning
  - CL-LoRA (arxiv 2505.24816) — rehearsal-free continual LoRA adaptation
  - `ces-hebbian` from 3.1 — base Hebbian LoRA to enhance

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-hebbian -- continual_learning_` → all pass
  - [ ] `cargo test --package ces-hebbian -- importance_scoring` → scores are stable over 100 forward passes (variance < 0.01)
  - [ ] Layer allocation test: high-importance layers get higher rank, low-importance get lower

  **QA Scenario**:
  ```
  Scenario: Continual learning patterns produce stable importance scores
    Tool: Bash
    Steps:
      1. cargo test --package ces-hebbian -- continual_learning_ --nocapture → all pass
      2. cargo test --package ces-hebbian -- importance_scoring --nocapture → variance reported < 0.01
    Evidence: Test output captured
  ```

  **Effort**: 1-2 agent sessions (RESEARCH)
  **Commit**: YES — `feat(hebbian): TreeLoRA/CL-LoRA continual learning patterns`

---

- [x] 3.3. Catastrophic Forgetting Tests **[ENGINEERING]**

  **What to do**:
  - Implement forgetting measurement suite in `benchmarks/`:
    - Train model on Task A (e.g., text completion on domain A)
    - Continuously adapt via Hebbian LoRA on Task B (different domain)
    - Measure Task A performance after N steps of Task B adaptation
    - Success: Task A performance ≥ 90% of original after Task B learning
  - Compare: Hebbian LoRA vs standard LoRA vs no adaptation (ablation)
  - Test with SP-sparse activations (interference reduction) vs dense activations (higher interference)

  **Must NOT do**: Implement adaptation algorithms — use what's built in 3.1/3.2.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**: Wave 2 — after 3.1. Parallel with 3.2. Blocks: 3.4.

  **References**:
  - Brainstorm math: SP-sparse interference reduction = 2621× vs dense, combined isolation ≈ 83,872×
  - Grewal et al. 2022 — Active dendrites for continual learning benchmarks
  - `ces-hebbian` from 3.1, `ces-spiking` from Sprint 1

  **Acceptance Criteria**:
  - [ ] `python benchmarks/sprint3_forgetting_test.py` → Task A performance ≥ 90% after Task B (500 steps)
  - [ ] Ablation: sparse activations show less forgetting than dense (quantified)
  - [ ] Results JSON: `benchmarks/forgetting_results.json` with per-step performance on both tasks

  **QA Scenario**:
  ```
  Scenario: Hebbian LoRA resists catastrophic forgetting
    Tool: Bash
    Steps:
      1. python benchmarks/sprint3_forgetting_test.py → "Task A retention: XX% (threshold: 90%)"
      2. python -c "import json; d=json.load(open('benchmarks/forgetting_results.json')); assert d['task_a_retention'] >= 0.9"
    Evidence: forgetting_results.json captured
  ```

  **Effort**: 1-2 agent sessions
  **Commit**: YES — `test(hebbian): catastrophic forgetting measurement suite`

---

- [x] 3.4. Sprint 3 Convergence Gate **[ENGINEERING]**

  **What to do**:
  - Integrate Hebbian LoRA with SSM + memory from Sprint 1-2
  - Train/fine-tune CES model with all components active (SSM + memory + Hebbian LoRA)
  - Run convergence gate G3: WikiText-103 perplexity comparison
  - Hebbian ablation: measure performance WITH vs WITHOUT Hebbian adaptation
  - Output: `gate_results/sprint3.json`
  - **IF GATE FAILS**: Implement fallback — replace Hebbian LoRA with standard LoRA using Hebbian-inspired LR schedule

  **Must NOT do**: Add MoE (Sprint 4).

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**: Wave 3 — after 3.2, 3.3. Blocks: Sprint 4.

  **References**:
  - Sprint 1-2 model checkpoint, `ces-train/`, architecture spec

  **Acceptance Criteria**:
  - [ ] End-to-end inference with Hebbian: `cargo test --package ces-inference -- hebbian_integration` → PASS
  - [ ] `gate_results/sprint3.json` exists with valid measurements
  - [ ] **GATE G3: multiplier ≥ 4.5×** — if FAIL, implement standard LoRA fallback
  - [ ] If fallback triggered: `gate_results/sprint3_fallback.json` shows multiplier with standard LoRA

  **QA Scenario**:
  ```
  Scenario: Sprint 3 convergence gate (with fallback path)
    Tool: Bash
    Steps:
      1. python benchmarks/run_gate.py --sprint 3 → JSON output
      2. python -c "import json; d=json.load(open('gate_results/sprint3.json')); print(f'multiplier={d[\"multiplier\"]}, pass={d[\"pass\"]}')"
      3. If pass=false: python benchmarks/run_gate.py --sprint 3 --fallback standard-lora → fallback JSON
    Evidence: Gate results captured (primary and fallback if needed)
  ```

  **Effort**: 2 agent sessions
  **Commit**: YES — `feat(hebbian): Hebbian LoRA integration + Sprint 3 convergence gate`

---

### Sprint 3 Go/No-Go Gate

**GATE G3: Effective multiplier ≥ 4.5×**
- 130M + all components perplexity ≤ 585M Transformer
- **IF PASS**: Proceed to Sprint 4 with Hebbian LoRA
- **IF FAIL but multiplier ≥ 4.0× with standard LoRA fallback**: Proceed with standard LoRA, document Hebbian findings
- **IF FAIL and multiplier < 4.0× even with fallback**: STOP. The convergence thesis may be too aggressive — recalibrate targets

---

### Sprint 4: Cortical Column MoE + Efficiency

> **Goal**: Voting-based MoE routing, software DVFS, dual-track merge finalization, full pipeline.
> **MUST NOT**: Change the inference pipeline fundamentals. MoE wraps existing components.
> **Risk**: Medium — MoE is research but has fallback (standard top-k routing).
> **Effort**: 4-6 agent sessions.

---

- [ ] 4.1. Cortical Column MoE Routing **[RESEARCH]**

  **What to do**:
  - Implement `ces-moe` crate:
    - Each MoE "expert" is a cortical column operating at a specific timescale
    - Voting-based consensus: experts "vote" on output, weighted by confidence (not softmax gate)
    - Column confidence = how well the expert's temporal scale matches the current input pattern
    - Start with top-k routing (proven) + Hebbian voting signal as auxiliary loss
    - Load balancing: coefficient of variation across experts < 0.3
  - Property tests: voting is differentiable, load is balanced, output is weighted average of expert outputs

  **Must NOT do**: Pure voting-based routing from scratch (too risky). Start with top-k + voting auxiliary.

  **Time-box**: 3 agent sessions. **Fallback**: Standard top-k MoE routing (remove voting entirely).

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []

  **Parallelization**: Wave 1 — parallel with 4.2. Blocks: 4.3.

  **References**:
  - `thousandbrainsproject/tbp.monty` (MIT) — cortical column voting patterns
  - Clay, Leadholm, Hawkins 2024 (arxiv 2412.18354) — Thousand Brains Project
  - Brainstorm: "Each MoE expert = column at different timescale, consensus voting replaces softmax gate"
  - Entmax 2019: "Adaptively Sparse Transformers" — differentiable sparsity reference for voting

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-moe -- column_routing_` → all pass
  - [ ] `cargo test --package ces-moe -- load_balance` → CV across experts < 0.3
  - [ ] `cargo test --package ces-moe -- voting_differentiable` → gradients flow through voting mechanism
  - [ ] `cargo test --package ces-moe -- property_output_is_weighted_average` → proptest passes

  **QA Scenario**:
  ```
  Scenario: MoE routing balances load and maintains differentiability
    Tool: Bash
    Steps:
      1. cargo test --package ces-moe -- column_routing_ --nocapture → all pass
      2. cargo test --package ces-moe -- load_balance --nocapture → "CV: X.XX (< 0.3)"
    Evidence: Test output captured
  ```

  **Effort**: 2-3 agent sessions (RESEARCH)
  **Commit**: YES — `feat(moe): cortical column MoE with voting-based routing`

---

- [ ] 4.2. Software DVFS Power Management **[ENGINEERING]**

  **What to do**:
  - Implement power management in `ces-inference` crate:
    - Software DVFS: adjust compute intensity based on input complexity
    - Simple complexity heuristic: token surprise (from memory system) + activation sparsity level
    - Low complexity → reduce precision / skip MoE experts / increase sparsity threshold
    - High complexity → full precision / all experts / standard sparsity
    - Linux cpufreq interface via sysfs (real DVFS on supported hardware)
    - Other platforms: no-op stub with same API (logs decisions, doesn't act)
  - Benchmark: measure power consumption on Linux with RAPL (if available)

  **Must NOT do**: Hardware-specific power management beyond Linux cpufreq. Platform-agnostic stubs.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
  - **Skills**: [`backend-patterns`]

  **Parallelization**: Wave 1 — parallel with 4.1. Blocks: 4.4.

  **References**:
  - Brainstorm: "Software DVFS + spiking sparsity + SP-inhibition + Engram offload"
  - Linux cpufreq sysfs documentation
  - `ces-spiking` — sparsity level as complexity signal
  - `ces-memory` — surprise as complexity signal

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-inference -- dvfs_` → all pass
  - [ ] `cargo test --package ces-inference -- dvfs_complexity_heuristic` → high-surprise inputs get full compute, low-surprise get reduced
  - [ ] Non-Linux: `cargo test --package ces-inference -- dvfs_stub` → stub logs decisions without acting

  **QA Scenario**:
  ```
  Scenario: DVFS adjusts compute based on input complexity
    Tool: Bash
    Steps:
      1. cargo test --package ces-inference -- dvfs_ --nocapture → all pass
      2. Output shows: "high complexity → full compute" and "low complexity → reduced compute"
    Evidence: Test output captured
  ```

  **Effort**: 1 agent session
  **Commit**: YES — `feat(inference): software DVFS power management`

---

- [ ] 4.3. Dual-Track SSM Merge Finalization **[RESEARCH]**

  **What to do**:
  - Finalize the Mamba-2 + RWKV dual-track merge in `ces-inference`:
    - Learned gating: `g(x)·mamba_out + (1-g(x))·rwkv_out`
    - g(x) is a 2-layer MLP with sigmoid output, trained end-to-end
    - Test state divergence: after 1000 steps, measure information overlap between Mamba-2 and RWKV states
    - If states diverge too much: add periodic state alignment (project RWKV state into Mamba-2 space or vice versa)
  - End-to-end inference pipeline: input → tokenize → dual-track SSM → MoE → memory → Hebbian → output

  **Must NOT do**: Change individual components. Integrate what exists.

  **Time-box**: 2 agent sessions. **Fallback**: If dual-track doesn't improve over Mamba-2 alone, make RWKV a cold-start fallback (used only when ONNX is needed) rather than always-on secondary.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**: Wave 2 — after 4.1 (needs MoE for full pipeline). Blocks: 4.4.

  **References**:
  - `ces-ssm` dual-track from 1.6
  - Probe E report — RWKV v6 state shape
  - Architecture spec — merge configuration

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-inference -- dual_track_full_pipeline` → PASS
  - [ ] `cargo test --package ces-inference -- state_divergence_measurement` → divergence metric logged
  - [ ] End-to-end: input text → tokenize → forward → logits → valid probability distribution

  **QA Scenario**:
  ```
  Scenario: Full inference pipeline produces valid output
    Tool: Bash
    Steps:
      1. cargo test --package ces-inference -- dual_track_full_pipeline --nocapture → PASS
      2. cargo test --package ces-inference -- e2e_inference --nocapture → logits are valid probabilities
    Evidence: Test output captured
  ```

  **Effort**: 1-2 agent sessions (RESEARCH)
  **Commit**: YES — `feat(inference): dual-track SSM merge + end-to-end pipeline`

---

- [ ] 4.4. Sprint 4 Convergence Gate **[ENGINEERING]**

  **What to do**:
  - Train full CES model with ALL components (SSM + memory + Hebbian + MoE + DVFS)
  - Run convergence gate G4: full system multiplier measurement
  - MoE ablation: WITH vs WITHOUT MoE routing
  - DVFS ablation: measure tokens/second and power with vs without DVFS
  - Output: `gate_results/sprint4.json`

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**: Wave 3 — after 4.3, 4.2. Blocks: Sprint 5.

  **Acceptance Criteria**:
  - [ ] Full system training completes
  - [ ] `gate_results/sprint4.json` with valid measurements
  - [ ] **GATE G4: multiplier ≥ 5.0×** — if FAIL, simplify MoE to standard top-k
  - [ ] DVFS ablation results in `gate_results/sprint4_dvfs.json`

  **QA Scenario**:
  ```
  Scenario: Sprint 4 full system convergence gate
    Tool: Bash
    Steps:
      1. python benchmarks/run_gate.py --sprint 4 → JSON output
      2. python -c "import json; d=json.load(open('gate_results/sprint4.json')); assert d['multiplier'] >= 5.0"
    Evidence: Gate results captured
  ```

  **Effort**: 2 agent sessions
  **Commit**: YES — `feat(system): full CES integration + Sprint 4 convergence gate`

---

### Sprint 4 Go/No-Go Gate

**GATE G4: Effective multiplier ≥ 5.0×**
- **IF PASS**: Full system works. Proceed to integration (Sprint 5).
- **IF FAIL but ≥ 4.5×**: Acceptable. Proceed with note that MoE contribution is marginal.
- **IF FAIL and < 4.5×**: MoE is hurting. Revert to non-MoE architecture, proceed to Sprint 5.

---

### Sprint 5: Integration & Cross-Platform

> **Goal**: PyO3 Python bindings, ONNX export, Android cross-compilation.
> **MUST NOT**: Add new functionality. Bindings and export of EXISTING code only.
> **Risk**: Low — standard engineering tasks with known approaches.
> **Effort**: 4-6 agent sessions.

---

- [ ] 5.1. PyO3 Python Bindings **[ENGINEERING]**

  **What to do**:
  - Implement `ces-bindings` crate with PyO3:
    - `CESModel` class: load(path) → model, infer(text) → output, generate(prompt, max_tokens) → text
    - `CESConfig` class: expose all architecture parameters
    - `CESMemory` class: query episodic memory, inspect Engram state
    - Tensor interop: numpy arrays ↔ candle tensors (zero-copy where possible)
    - Memory ownership: Rust owns GPU/tensor memory, Python gets views
  - Package as `ces` pip-installable wheel

  **Must NOT do**: Implement new functionality. Wrap existing Rust API only.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: [`backend-patterns`]

  **Parallelization**: Wave 1 — parallel with 5.2, 5.3. Blocks: 5.4.

  **References**:
  - `PyO3/pyo3` — PyO3 patterns and best practices
  - Probe D from Sprint 0 — PyO3 + candle interop validation
  - `huggingface/candle/candle-pyo3/` — candle's Python bindings reference

  **Acceptance Criteria**:
  - [ ] `pip install -e ces-bindings/` → succeeds
  - [ ] `python -c "import ces; m = ces.CESModel.load('test'); print(m.infer('hello'))"` → non-empty output
  - [ ] `python -c "import ces; import numpy as np; t = ces.tensor(np.zeros(10)); print(t.shape)"` → "(10,)"
  - [ ] Memory safety: no segfaults under adversarial Python usage (None inputs, double-free attempts)

  **QA Scenario**:
  ```
  Scenario: Python bindings expose full model API
    Tool: Bash
    Steps:
      1. pip install -e ces-bindings/ → exits 0
      2. python -c "import ces; m = ces.CESModel.load('test_model'); out = m.infer('Hello world'); assert len(out) > 0"
      3. python -c "import ces; m = ces.CESModel.load('test_model'); text = m.generate('Once upon', max_tokens=50); assert len(text) > 10"
    Evidence: Python output captured
  ```

  **Effort**: 2 agent sessions
  **Commit**: YES — `feat(bindings): PyO3 Python bindings for CES model`

---

- [ ] 5.2. ONNX Export **[ENGINEERING]**

  **What to do**:
  - Implement ONNX export for the SSM inference path:
    - Standard ops: matmul, activation, normalization → standard ONNX ops
    - Custom ops (spiking, dendritic, Hebbian, SP-inhibition) → custom ONNX operator nodes with domain "ces"
    - Export script: `python scripts/export_onnx.py --model checkpoint --output model.onnx`
    - Validate: run ONNX model via onnxruntime, compare output to native Rust (within 1%)
  - Document custom op specifications for downstream ONNX consumers

  **Must NOT do**: Decompose custom ops into standard ONNX primitives. Export as-is with custom domain.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**: Wave 1 — parallel with 5.1, 5.3. Blocks: 5.4.

  **References**:
  - `microsoft/onnxruntime` — ONNX custom operator registration
  - `RWKV/rwkv-onnx` — ONNX export patterns for SSMs
  - ONNX custom operator docs

  **Acceptance Criteria**:
  - [ ] `python scripts/export_onnx.py --model test_model --output test.onnx` → exits 0, .onnx file created
  - [ ] ONNX validation: `python -c "import onnxruntime; sess = onnxruntime.InferenceSession('test.onnx')"` → loads without error
  - [ ] Output comparison: max relative error < 1% between ONNX and native Rust output

  **QA Scenario**:
  ```
  Scenario: ONNX export produces valid and accurate model
    Tool: Bash
    Steps:
      1. python scripts/export_onnx.py --model test_model --output test.onnx → exits 0
      2. python scripts/validate_onnx.py --onnx test.onnx --native test_model → "max_rel_error: X.XX% (< 1%)"
    Evidence: Export and validation output captured
  ```

  **Effort**: 1-2 agent sessions
  **Commit**: YES — `feat(export): ONNX model export with custom operator support`

---

- [ ] 5.3. Android Cross-Compilation **[ENGINEERING]**

  **What to do**:
  - Configure Rust cross-compilation for `aarch64-linux-android`:
    - Activate NDK settings in `.cargo/config.toml` (commented out in Sprint 0)
    - Build `ces-inference` for Android target
    - Link sqlite-vec for ARM64 (from Probe B validation)
    - Create minimal inference benchmark binary that runs on Android
  - Test: build succeeds, .so file produced, can be loaded by a minimal Android test harness (or adb push + run)

  **Must NOT do**: Build Android app, JNI wrapper, or Kotlin/Java interface. .so binary + benchmark only.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**: Wave 1 — parallel with 5.1, 5.2. Blocks: 5.4.

  **References**:
  - `microsoft/onnxruntime` — Android build patterns (issue #26747)
  - NDK cross-compilation docs for Rust
  - Sprint 0 Probe B — sqlite-vec ARM validation

  **Acceptance Criteria**:
  - [ ] `cargo build --target aarch64-linux-android --package ces-inference` → exits 0
  - [ ] `ls target/aarch64-linux-android/release/*.so` → .so file exists
  - [ ] File size logged: `ls -lh target/aarch64-linux-android/release/*.so`

  **QA Scenario**:
  ```
  Scenario: Android cross-compilation produces valid binary
    Tool: Bash
    Steps:
      1. cargo build --target aarch64-linux-android --package ces-inference --release → exits 0
      2. file target/aarch64-linux-android/release/libces_inference.so → "ELF 64-bit LSB shared object, ARM aarch64"
    Evidence: Build output and file info captured
  ```

  **Effort**: 1 agent session
  **Commit**: YES — `feat(android): cross-compilation for aarch64-linux-android`

---

- [ ] 5.4. Integration Tests **[ENGINEERING]**

  **What to do**:
  - End-to-end integration tests across all interfaces:
    - Rust native: `cargo test --package ces-inference -- integration_` → full pipeline
    - Python: `pytest tests/integration/` → PyO3 bindings end-to-end
    - ONNX: `pytest tests/onnx/` → ONNX model inference matches native
    - Memory roundtrip: create episodes → query → retrieve → verify content
    - Continuous inference: run 1000 tokens through the system, verify no memory leaks, no crashes
  - Performance regression: compare tokens/sec against Sprint 4 baseline (must not regress > 5%)

  **Must NOT do**: Fix bugs discovered here by changing architecture. Only fix integration wiring.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**: Wave 2 — after 5.1, 5.2, 5.3. Blocks: Sprint 6.

  **References**:
  - All crate APIs from Sprint 1-4
  - `ces-bindings` from 5.1
  - ONNX model from 5.2

  **Acceptance Criteria**:
  - [ ] `cargo test --package ces-inference -- integration_` → all pass
  - [ ] `pytest tests/integration/ tests/onnx/` → all pass
  - [ ] Continuous inference: 1000 tokens without crash or memory growth > 10%
  - [ ] Performance: tokens/sec ≥ 95% of Sprint 4 baseline

  **QA Scenario**:
  ```
  Scenario: Full integration test suite passes
    Tool: Bash
    Steps:
      1. cargo test --workspace -- integration_ → all pass
      2. pytest tests/ -v → all pass
      3. python benchmarks/continuous_inference.py --tokens 1000 → "No crashes, memory stable"
    Evidence: Test output captured
  ```

  **Effort**: 1-2 agent sessions
  **Commit**: YES — `test(integration): end-to-end integration test suite`

---

### Sprint 5 Go/No-Go Gate

**All integration tests pass. Cross-platform targets build. Performance within 5% of Sprint 4.**
- **IF PASS**: Proceed to Sprint 6 (final validation).
- **IF FAIL**: Debug integration issues. Do NOT proceed to Sprint 6 with broken integration.

---

### Sprint 6: Validation & Convergence Thesis

> **Goal**: Full training run, comprehensive benchmark suite, Lean 4 proofs, convergence thesis report.
> **MUST NOT**: Fix bugs or add features. Benchmark and document ONLY.
> **Risk**: Medium — training compute + formal proofs may take significant agent sessions.
> **Effort**: 5-8 agent sessions.

---

- [ ] 6.1. Full Training Run **[ENGINEERING]**

  **What to do**:
  - Train the final 130M CES model on full OpenWebText (deduplicated):
    - Architecture: full CES with all components (SSM + spiking + memory + Hebbian + MoE)
    - Training config from architecture spec (optimizer, LR, schedule, batch size, total tokens)
    - All Sprint 1-4 gate results inform any hyperparameter adjustments
    - Checkpoint every N steps, log all metrics to WandB (or local JSON)
    - Final checkpoint: `checkpoints/ces-130m-final/`
  - Also train baseline: standard Transformer at 130M, 195M, 390M, 585M, 780M params (for scaling curve)
    - Use existing reference implementations (GPT-2 configs at various scales)
    - Same data, same tokenizer, same training tokens

  **Must NOT do**: Change the CES architecture. Train what was built.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Long-running training with monitoring and checkpoint management.

  **Parallelization**: Wave 1 — first task. Blocks: 6.2.

  **References**:
  - `ces-train/` infrastructure from Sprint 0
  - Architecture spec — full training configuration
  - All gate results from Sprints 1-4
  - OpenWebText dataset

  **Acceptance Criteria**:
  - [ ] CES-130M training completes on full dataset
  - [ ] Checkpoint exists: `checkpoints/ces-130m-final/`
  - [ ] Baseline models trained at 5 scales (130M, 195M, 390M, 585M, 780M)
  - [ ] Training loss curves show convergence (no divergence)
  - [ ] Training metrics JSON logged per checkpoint

  **QA Scenario**:
  ```
  Scenario: CES model and baselines complete training
    Tool: Bash
    Steps:
      1. ls checkpoints/ces-130m-final/ → model files exist
      2. ls checkpoints/baseline-*/  → 5 baseline checkpoint dirs exist
      3. python scripts/check_training.py → "All models converged, no divergence detected"
    Evidence: Training logs and checkpoint listings captured
  ```

  **Effort**: 2-3 agent sessions (mostly compute-bound, agent manages and monitors)
  **Commit**: YES — `feat(training): final CES-130M and baseline model training`

---

- [ ] 6.2. Benchmark Suite Execution **[ENGINEERING]**

  **What to do**:
  - Run the 5-benchmark suite on CES-130M and all 5 baselines:
    - HellaSwag (common sense reasoning)
    - ARC-Easy (science QA)
    - WinoGrande (coreference resolution)
    - PIQA (physical intuition)
    - WikiText-103 perplexity (language modeling)
  - Compute: per-benchmark scores, average score, confidence intervals (bootstrap, N=1000)
  - Compute: effective multiplier = scale at which Transformer matches CES-130M
  - Generate: scaling curve plot (benchmark score vs model size, CES point overlaid)
  - Generate: comprehensive results JSON: `results/convergence_benchmark.json`
  - Generate: comparison table for documentation

  **Must NOT do**: Cherry-pick benchmarks. Run ALL 5, report ALL results.

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**: Wave 2 — after 6.1. Parallel with 6.3.

  **References**:
  - Architecture spec — benchmark suite and scoring methodology
  - GPT-2 Large published scores — for cross-validation
  - lm-evaluation-harness (EleutherAI) — standard benchmark runner

  **Acceptance Criteria**:
  - [ ] `results/convergence_benchmark.json` exists with all 6 models × 5 benchmarks = 30 scores
  - [ ] Confidence intervals computed (bootstrap N=1000)
  - [ ] Effective multiplier calculated and logged
  - [ ] **GATE G6: multiplier ≥ 5.4×** (conservative thesis target)
  - [ ] Scaling curve plot generated: `results/scaling_curve.png`

  **QA Scenario**:
  ```
  Scenario: Full convergence benchmark validates thesis
    Tool: Bash
    Steps:
      1. python benchmarks/run_full_benchmark.py → exits 0
      2. python -c "import json; d=json.load(open('results/convergence_benchmark.json')); print(f'Multiplier: {d[\"effective_multiplier\"]}×, Pass: {d[\"thesis_validated\"]}')"
      3. ls results/scaling_curve.png → file exists
    Evidence: convergence_benchmark.json and scaling_curve.png captured
  ```

  **Effort**: 2 agent sessions
  **Commit**: YES — `feat(benchmark): full convergence thesis validation`

---

- [ ] 6.3. Lean 4 Formal Proofs + Convergence Report **[RESEARCH]**

  **What to do**:
  - Attempt the 5 Lean 4 formal proofs:
    1. SP-inhibition convergence via Lyapunov function (Cohen-Grossberg 1983)
    2. Hebbian convergence on coupled LoRA factors (rank-r case, Oja's rule)
    3. SDR properties preservation under surrogate gradient training
    4. Column voting consensus probability bound
    5. Convergence multiplier lower bound (worst-case analysis)
  - For each proof: attempt via `lake build`. If proof takes > 2 agent sessions, document as proof SKETCH (LaTeX) and mark "MANUAL-PROOF-NEEDED"
  - Write convergence thesis report: `docs/convergence-thesis-report.md`
    - Mathematical framework (4 axioms from brainstorm)
    - Experimental validation (benchmark results from 6.2)
    - Formal proofs (completed or sketched)
    - Comparison with published scaling laws
    - Limitations and future work

  **Must NOT do**: Spend more than 2 sessions per proof. Document and move on if stuck.

  **Time-box**: 3 agent sessions total for all 5 proofs. **Fallback**: LaTeX proof sketches with `sorry`-free Lean for whatever portion compiles.

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []
  - Reason: Formal verification requires deep mathematical reasoning.

  **Parallelization**: Wave 2 — parallel with 6.2 (independent). Blocks: nothing.

  **References**:
  - `proofs/` directory from Sprint 0 (Lean 4 environment + stubs)
  - Brainstorm mathematical studies — all 14 theory2-verified properties
  - Cohen-Grossberg 1983 — Lyapunov convergence reference
  - Brainstorm emergent properties — for report discussion
  - Benchmark results from 6.2 — for report

  **Acceptance Criteria**:
  - [ ] `lake build` in `proofs/` → exits 0 (even if some proofs use `sorry`)
  - [ ] Count of `sorry`-free proofs logged
  - [ ] `docs/convergence-thesis-report.md` exists with all sections
  - [ ] Report includes benchmark data tables and scaling curve
  - [ ] Proof sketches documented for any `sorry`-using proofs

  **QA Scenario**:
  ```
  Scenario: Lean proofs compile and report is complete
    Tool: Bash
    Steps:
      1. cd proofs && lake build 2>&1 → exits 0
      2. grep -r "sorry" proofs/CES/*.lean | wc -l → logged (0 = all proven, >0 = sketches needed)
      3. wc -l docs/convergence-thesis-report.md → > 200 lines (substantive report)
    Evidence: Proof compilation output and report captured
  ```

  **Effort**: 2-3 agent sessions (RESEARCH)
  **Commit**: YES — `docs: convergence thesis report with formal proofs`

---

### Sprint 6 Final Gate

**GATE G6: Convergence thesis validation**
- Effective multiplier ≥ 5.4× → **THESIS VALIDATED** (conservative target met)
- Effective multiplier ≥ 6.0× → **THESIS STRONGLY VALIDATED** (near optimistic target)
- Effective multiplier < 5.0× → **THESIS PARTIALLY VALIDATED** — document achievable multiplier, analyze which components underperformed
- Formal proofs: ≥ 3/5 fully verified in Lean 4 → **FORMAL RIGOR ACHIEVED**
- Formal proofs: < 3/5 → **PROOF SKETCHES DOCUMENTED** (acceptable, can be completed later)

---

## Commit Strategy

| Sprint | After Task | Message | Key Files |
|--------|------------|---------|-----------|
| 0 | 0.1 | `chore(workspace): scaffold CES Rust workspace with 7 crates` | Cargo.toml, crate dirs |
| 0 | 0.2 | `ci: add GitHub Actions workflow for Rust workspace` | .github/workflows/ |
| 0 | 0.3 | `chore(proofs): setup Lean 4 environment with Mathlib4` | proofs/ |
| 0 | 0.4 | `chore(probes): validate technical feasibility` | probes/ |
| 0 | 0.5 | `docs: architecture specification` | docs/architecture-spec.md |
| 0 | 0.6 | `feat(train): minimal training infrastructure` | ces-train/ |
| 1 | 1.1 | `feat(reference): Mamba-2 SSD golden test vectors` | ssd_reference/ |
| 1 | 1.2 | `feat(ssm): Mamba-2 SSD selective scan in Rust/candle` | ces-ssm/ |
| 1 | 1.3 | `feat(spiking): spiking activations with surrogate gradients` | ces-spiking/ |
| 1 | 1.4 | `feat(spiking): SP-inspired lateral inhibition` | ces-spiking/ |
| 1 | 1.5 | `feat(ssm): multi-scale temporal heads` | ces-ssm/ |
| 1 | 1.6 | `feat(ssm): RWKV + dual merge + Sprint 1 gate` | ces-ssm/, gate_results/ |
| 2 | 2.1 | `feat(memory): Engram conditional memory port` | ces-memory/ |
| 2 | 2.2 | `feat(memory): active dendritic gating` | ces-memory/ |
| 2 | 2.3 | `feat(memory): EM-LLM episodic + Bayesian surprise` | ces-memory/ |
| 2 | 2.4 | `feat(memory): sqlite-vec integration` | ces-memory/ |
| 2 | 2.5 | `feat(memory): three-tier integration + Sprint 2 gate` | ces-inference/, gate_results/ |
| 3 | 3.1 | `feat(hebbian): Hebbian LoRA with Oja's rule` | ces-hebbian/ |
| 3 | 3.2 | `feat(hebbian): TreeLoRA/CL-LoRA patterns` | ces-hebbian/ |
| 3 | 3.3 | `test(hebbian): catastrophic forgetting suite` | benchmarks/ |
| 3 | 3.4 | `feat(hebbian): integration + Sprint 3 gate` | ces-inference/, gate_results/ |
| 4 | 4.1 | `feat(moe): cortical column MoE routing` | ces-moe/ |
| 4 | 4.2 | `feat(inference): software DVFS` | ces-inference/ |
| 4 | 4.3 | `feat(inference): dual-track merge + full pipeline` | ces-inference/ |
| 4 | 4.4 | `feat(system): full integration + Sprint 4 gate` | gate_results/ |
| 5 | 5.1 | `feat(bindings): PyO3 Python bindings` | ces-bindings/ |
| 5 | 5.2 | `feat(export): ONNX model export` | scripts/ |
| 5 | 5.3 | `feat(android): aarch64 cross-compilation` | .cargo/ |
| 5 | 5.4 | `test(integration): end-to-end test suite` | tests/ |
| 6 | 6.1 | `feat(training): final CES-130M + baselines` | checkpoints/ |
| 6 | 6.2 | `feat(benchmark): convergence thesis validation` | results/ |
| 6 | 6.3 | `docs: convergence thesis report + formal proofs` | docs/, proofs/ |

---

## Success Criteria

### Verification Commands
```bash
# Build
cargo build --workspace                    # Expected: 0 errors
cargo clippy --workspace                   # Expected: 0 warnings
cargo test --workspace                     # Expected: 0 failures

# Python
pip install -e ces-bindings/ && pytest tests/  # Expected: 0 failures

# Convergence thesis
python benchmarks/run_full_benchmark.py    # Expected: multiplier ≥ 5.4×

# Cross-platform
cargo build --target aarch64-linux-android --package ces-inference  # Expected: .so produced

# ONNX
python scripts/export_onnx.py && python scripts/validate_onnx.py   # Expected: < 1% error

# Formal proofs
cd proofs && lake build                    # Expected: 0 errors (some sorry acceptable)
```

### Final Checklist
- [ ] All "Must Have" present (checked against list)
- [ ] All "Must NOT Have" absent (checked against guardrails)
- [ ] All cargo test pass (0 failures)
- [ ] All pytest pass (0 failures)
- [ ] Convergence multiplier ≥ 5.0× (conservative floor)
- [ ] CES-130M benchmarks within 90% of GPT-2 Large (774M)
- [ ] PyO3 bindings functional
- [ ] ONNX export validated
- [ ] Android cross-compilation successful
- [ ] Convergence thesis report complete
- [ ] ≥ 3/5 Lean 4 proofs verified (or proof sketches documented)
- [ ] All gate results JSON files exist and are consistent
- [ ] No silent algorithm simplifications (all RESEARCH tasks documented their outcome)
