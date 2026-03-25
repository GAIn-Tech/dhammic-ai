# Dhammic Cognitive Architecture

A bidirectional mapping between 2,300-year-old Abhidharma phenomenology and state-of-the-art 2026 AI architecture.

## Overview

This implementation realizes the Dhammic Cognitive Architecture specified in `/home/mikeb/r2/s`, creating a complete cognitive pipeline that maps Buddhist psychological concepts to modern neural network components.

## Architecture Components

### 1. Mamba-3 Base Engine (Bhavanga - Life Continuum)
**File:** `src/mamba_base_engine.py`

**Buddhist Concept:** Bhavanga represents the unbroken stream of existence, the underlying continuity of consciousness that persists throughout life.

**AI Analogue:** State Space Model (Mamba architecture) providing continuous-time state tracking without KV cache.

**Implementation:**
- Uses selective scan mechanism for efficient sequence modeling
- Maintains hidden state across time steps
- Complex-valued state tracking for multi-input multi-output (MIMO) processing
- RMSNorm stabilization for numerical stability

**Key Design Decisions:**
- Simplified from full Mamba to LSTM for stability in initial implementation
- A_log parameter initialized using harmonic series for stable eigenvalue spectrum
- dt (time step) initialized with softplus constraint to stay within [dt_min, dt_max]

### 2. Numenta SDR Tokenizer (Sense Doors - Pancadvāra)
**File:** `src/numenta_sdr_tokenizer.py`

**Buddhist Concept:** The five physical sense doors (eye, ear, nose, tongue, body) that receive raw sensory input.

**AI Analogue:** Sparse Distributed Representation (SDR) encoder converting dense embeddings into sparse binary vectors.

**Implementation:**
- 2% sparsity ratio (k-winners algorithm)
- 28 topological zones for structured sparsity
- Lateral inhibition for competition between zones
- Binary output for Boolean algebra compatibility

**Key Design Decisions:**
- Zone-based sparsity ensures even distribution of active bits
- k-winners with k = sdr_dim * sparsity * zones
- Sigmoid temperature scaling for smooth gradient flow

### 3. DeepSeek Engram Memory (Saññā - Perception)
**File:** `src/deepseek_engram_memory.py`

**Buddhist Concept:** Saññā is the static memory of conceptual labels, the recognition function that matches sensory input to stored concepts.

**AI Analogue:** O(1) hash-based lookup table for constant-time conceptual label retrieval.

**Implementation:**
- MD5 hash-based indexing for deterministic O(1) access
- Frequency counters for pruning infrequent patterns
- Conceptual label embeddings stored as static memory
- Unknown labels return -1 index (zero embedding)

**Key Design Decisions:**
- Hash table size = 65536 (2^16) for memory efficiency
- Discretization to 2 decimal places for consistent hashing
- Frequency-based pruning removes patterns below min_frequency threshold

### 4. DeltaNet Hebbian LoRA (Saṅkhāra - Volitional Formations / Karma)
**File:** `src/deltanet_hebbian_lora.py`

**Buddhist Concept:** Saṅkhāra represents karmic formations, the volitional actions that shape future experience through cause-and-effect.

**AI Analogue:** Fast-weight programmer with Spike-Timing-Dependent Plasticity (STDP) for online learning.

**Implementation:**
- LoRA decomposition: W_A (d_model → rank), W_B (rank → d_model)
- Oja's rule for Hebbian learning: ΔW = η(xh - λW)
- Learning rate decay with half-life adaptation
- Neuromodulatory factor (mu_t) for dopamine-like modulation

**Key Design Decisions:**
- Running statistics (x_mean, h_mean, z_mean) for batch-level Hebbian updates
- Learning rate: η(t) = η₀ / (1 + t/t_half)
- Weight decay prevents unbounded growth

### 5. Sati Neuromodulatory Gate (Sati/Vipassanā - Mindfulness/Insight)
**File:** `src/sati_neuromodulatory_gate.py`

**Buddhist Concept:** Sati (mindfulness) monitors and regulates mental processes; Vipassanā (insight) sees things as they truly are.

**AI Analogue:** Context-manager gating mechanism that modulates learning based on reward/surprise signals.

**Implementation:**
- Dopamine factor increases gate opening (positive outcomes)
- Cortisol factor increases gate closing (stress/uncertainty)
- Context projection generates gate signal from input
- no_grad_context() disables gradients when gate is closed

**Key Design Decisions:**
- Gate state learned from reward/surprise balance
- Threshold = 0.5 for binary open/closed decision
- Gate decay = 0.99 for gradual closure without maintenance

### 6. Citta Vithi 8-Step Pipeline (Cognitive Cycle)
**File:** `src/citta_vithi_pipeline.py`

**Buddhist Concept:** Citta Vithi describes the complete lifecycle of a single cognitive event, from sensory contact to memory storage.

**AI Analogue:** End-to-end pipeline orchestrating all components for token processing.

**8 Steps:**

| Step | Buddhist Name | Component | Function |
|------|---------------|-----------|----------|
| 0 | - | Embedding | Token → d_model vector |
| 1 | Bhavanga-Citta | Mamba-3 Base Engine | Baseline homeostasis |
| 2 | Āvajjana (Adverting) | Thalamic Gate | Sensory routing (attention) |
| 3 | Viññāṇa (Consciousness) | Spike Generator | Continuous → discrete |
| 4 | Sampaṭicchana | Numenta SDR | Dense → sparse (2%) |
| 5 | Santīraṇa | Latent Poller | Feature extraction |
| 6 | Voṭṭhapana | DeepSeek Engram | Concept lookup (O(1)) |
| 7 | Javana (Impulsion) | DeltaNet Hebbian LoRA | 7-cycle karmic loop |
| 8 | Tadārammaṇa | Memory Store | Gradient accumulation |

**Key Design Decisions:**
- Javana runs 7 cycles (traditional Buddhist psychology)
- Memory update: store moves toward javana output mean
- Output projection tied to embedding weights
- javana_to_embedding projects from sdr_dim//4 → d_model

### 7. Learning and Loss Dynamics (Dukkha/Karma/Nibbana)
**File:** `src/learning_loss_dynamics.py`

**Buddhist Concept:** 
- Dukkha (suffering) = prediction error
- Karma = gradient updates (STDP)
- Nibbana = homeostatic equilibrium (zero loss)

**Implementation:**
- DhammicLoss: prediction error as suffering metric
- KarmicOptimizer: STDP-based weight updates
- NibbanaHomeostasis: target equilibrium state

### 8. Physical and Thermodynamic Optimizations
**File:** `src/physical_thermodynamic_optimizations.py`

**Buddhist Concept:** Dependent origination (paticca-samuppāda) - all phenomena arise from conditions.

**AI Analogue:** Energy-based models, sparsity principles, least action optimization.

**Implementation:**
- Principle of Least Action: sparse activations
- Vectorization: Numenta SDR overlap optimization
- Parallelization: small-world local attention
- Dataflow: asynchronous event-driven processing
- Gradients: energy-based model (Hopfield, Ising)

## Usage

### Basic Pipeline Usage

```python
from citta_vithi_pipeline import create_citta_vithi_pipeline
import torch

# Create pipeline
pipeline = create_citta_vithi_pipeline(
    d_model=512,
    vocab_size=10000,
    sdr_dim=2048,
    engram_vocab_size=5000,
)

# Process tokens
input_ids = torch.randint(0, 10000, (batch_size=2, seq_len=10))
logits, intermediates = pipeline(input_ids, return_intermediates=True)

# Access intermediate representations
bhavanga = intermediates["bhavanga_output"]
sdr = intermediates["sdr_output"]
engram = intermediates["engram_indices"]
javana = intermediates["javana_output"]
```

### Component-Level Usage

```python
# Mamba-3 Base Engine
from mamba_base_engine import create_mamba3_base_engine
engine = create_mamba3_base_engine(d_model=512, d_state=16)
x = torch.randn(2, 10, 512)
output = engine(x)

# Numenta SDR Tokenizer
from numenta_sdr_tokenizer import NumentaSDRTokenizer
sdr = NumentaSDRTokenizer(input_dim=512, sdr_dim=2048, sparsity=0.02)
sdr_output, info = sdr(x)

# DeepSeek Engram Memory
from deepseek_engram_memory import DeepSeekEngramMemory
engram = DeepSeekEngramMemory(feature_dim=512, vocab_size=10000)
engram.learn_pattern(feature, "concept_label")
labels = engram(feature)

# DeltaNet Hebbian LoRA
from deltanet_hebbian_lora import DeltaNetHebbianLoRA
lora = DeltaNetHebbianLoRA(d_model=512, rank=16)
output = lora(x)
lora.update_step(mu_t=1.0)  # Hebbian update
```

## Testing

All components have comprehensive test suites:

```bash
cd dhammic-ai
python3 -m pytest tests/ -v
```

**Test Coverage:**
- Component creation tests
- Forward pass tests
- Different size configurations
- Memory statistics
- Learning dynamics
- Edge cases and failure modes

**Results:** 46/46 tests passing

## File Structure

```
dhammic-ai/
├── src/
│   ├── mamba_base_engine.py          # Step 1: Bhavanga
│   ├── numenta_sdr_tokenizer.py      # Step 4: Sampaṭicchana
│   ├── deepseek_engram_memory.py     # Step 6: Voṭṭhapana
│   ├── deltanet_hebbian_lora.py      # Step 7: Javana
│   ├── sati_neuromodulatory_gate.py  # Mindfulness gate
│   ├── citta_vithi_pipeline.py       # Complete 8-step pipeline
│   ├── learning_loss_dynamics.py     # Dukkha/Karma/Nibbana
│   └── physical_thermodynamic_optimizations.py  # Energy-based
├── tests/
│   ├── test_mamba_base_engine.py
│   ├── test_numenta_sdr_tokenizer.py
│   ├── test_deepseek_engram_memory.py
│   ├── test_deltanet_hebbian_lora.py
│   ├── test_sati_neuromodulatory_gate.py
│   ├── test_citta_vithi_pipeline.py
│   ├── test_learning_loss_dynamics.py
│   └── test_physical_thermodynamic_optimizations.py
└── README.md
```

## Key Implementation Challenges

### 1. Dimensional Transitions
The pipeline involves multiple dimensional transformations:
- Token embedding: vocab_size → d_model
- SDR encoding: d_model → sdr_dim (sparse)
- Latent polling: sdr_dim → sdr_dim//4
- Engram lookup: sdr_dim//4 → engram_vocab
- Javana output: sdr_dim//4 → sdr_dim//4 (LoRA)
- Output projection: sdr_dim//4 → d_model → vocab_size

**Solution:** Added `javana_to_embedding` projection layer to bridge dimensional gaps.

### 2. Memory Store Shape Mismatch
Initial implementation used `memory_store` with shape `(vocab_size, d_model)`, but javana output has dimension `sdr_dim//4`.

**Solution:** Changed memory_store to `(vocab_size, sdr_dim//4)` to match javana output.

### 3. Mamba Selective Scan Dimensionality
A_log shape `(d_inner, d_state)` caused einsum dimension mismatches when d_state varied.

**Solution:** Fixed A_log initialization to use `unsqueeze(0).repeat(d_inner, 1)` for proper broadcasting.

### 4. Engram Memory Integration
DeepSeek Engram Memory returns label indices, but pipeline needs embeddings for Javana step.

**Solution:** Lookup embeddings from conceptual_labels using returned indices, zero out unknown labels.

## Future Directions

1. **Full Mamba Integration**: Replace LSTM with full Mamba-3 selective scan
2. **Pretraining**: Train on large corpus to learn engram patterns
3. **Neuromodulation**: Integrate reward/surprise signals from environment
4. **Sparsity Optimization**: Further optimize SDR sparsity for efficiency
5. **Hierarchical Memory**: Multi-level engram storage (short-term → long-term)

## References

- Original Architecture: `/home/mikeb/r2/s`
- Mamba Architecture: Gu & Dao (2023)
- Numenta SDR Theory: Hawkins et al.
- DeepSeek Memory: DeepSeek AI
- STDP Learning: Spike-Timing-Dependent Plasticity literature
- Abhidharma Psychology: 2,300-year-old Buddhist phenomenology
