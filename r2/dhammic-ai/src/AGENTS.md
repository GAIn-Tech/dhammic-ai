<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-25 | Updated: 2026-03-25 -->

# src

## Purpose
Eight neural modules implementing the Dhammic Cognitive Architecture. Each module maps a Buddhist Abhidharma concept to a modern AI component. Together they form the Citta Vithi 8-step cognitive pipeline.

## Key Files

| File | Buddhist Concept | AI Component | Description |
|------|-----------------|--------------|-------------|
| `mamba_base_engine.py` | Bhavanga (Life-Continuum) | Mamba-3 SSM | Continuous-time state space model with MIMO complex-valued state tracking. LSTM fallback for stability. Config: `Mamba3Config` dataclass. |
| `numenta_sdr_tokenizer.py` | Pancadvara (Sense Doors) | Sparse Distributed Representation | 2% sparsity encoder with 28 topological zones, k-winners lateral inhibition. Dense → sparse binary vectors. |
| `deepseek_engram_memory.py` | Sanna (Perception) | O(1) Hash Lookup | MD5-based hash table (65536 slots) for constant-time conceptual label retrieval. Frequency-based pruning. |
| `deltanet_hebbian_lora.py` | Sankhara (Karma/Ego) | Hebbian LoRA + STDP | Fast-weight programmer with Oja's rule, spike-timing-dependent plasticity, and neuromodulatory gating. |
| `sati_neuromodulatory_gate.py` | Sati (Mindfulness) | Dopamine/Cortisol Gate | Context manager that modulates learning rate to 0, enabling weight decay. Threshold-based open/closed. |
| `citta_vithi_pipeline.py` | Citta Vithi (Cognitive Cycle) | End-to-End Pipeline | Orchestrates all 8 steps: embedding → mamba → gate → spike → SDR → poll → engram → javana(x7) → memory. |
| `learning_loss_dynamics.py` | Dukkha/Karma/Nibbana | Loss + Optimizer + Homeostasis | `DhammicLossFunction` (prediction error), `KarmicOptimizer` (STDP updates), `NibbanaHomeostasis` (equilibrium target). |
| `physical_thermodynamic_optimizations.py` | Paticca-samuppada (Dependent Origination) | Energy-Based Optimizations | Sparse activations, SDR overlap vectorization, small-world attention, async event-driven processing, Hopfield/Ising gradients. |

## For AI Agents

### Working In This Directory
- **No `__init__.py`** -- modules are imported directly by name from the `dhammic-ai/` working directory
- Every module has a factory function (`create_*`) as the standard instantiation pattern
- All modules are `nn.Module` subclasses with `device`/`dtype` kwargs
- `_amp_disabled = True` flag indicates modules that disable mixed precision internally
- The pipeline (`citta_vithi_pipeline.py`) imports and wraps all other modules

### Dimensional Flow (Critical)
```
vocab_size → d_model → d_model → d_model → sdr_dim (sparse) → sdr_dim//4 → engram_vocab → sdr_dim//4 → d_model → vocab_size
```

Key transitions:
- `d_model` (768 default) is the base hidden dimension
- `sdr_dim` (2048 default) is the sparse representation width
- `sdr_dim//4` (512) is the latent polling output and Javana working dimension
- The `javana_to_embedding` projection bridges `sdr_dim//4` back to `d_model`

### Modification Guidelines
- When changing dimensions in one module, trace the full pipeline flow to ensure compatibility
- Engram memory returns indices (not embeddings) -- the pipeline looks up embeddings from `conceptual_labels`
- Javana runs exactly 7 cycles (Buddhist tradition) -- this is intentional, not a magic number
- `memory_store` shape is `(vocab_size, sdr_dim//4)` to match javana output (not d_model)

### Testing Requirements
- Each module has a corresponding test file in `../tests/`
- Run: `cd dhammic-ai && python3 -m pytest tests/test_<module_name>.py -v`
- Test various configurations (small/medium/large dimensions)

## Dependencies

### Internal
- `citta_vithi_pipeline.py` depends on all other modules in this directory
- `learning_loss_dynamics.py` is standalone (loss/optimizer utilities)
- `physical_thermodynamic_optimizations.py` is standalone (optimization utilities)

### External
- `torch`, `torch.nn`, `torch.nn.functional` -- all modules
- `numpy` -- SDR tokenizer, engram memory
- `hashlib` -- engram memory (MD5 hashing)
- `pickle` -- engram memory (serialization)

<!-- MANUAL: -->
