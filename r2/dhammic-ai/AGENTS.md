<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-25 | Updated: 2026-03-25 -->

# dhammic-ai

## Purpose
Complete implementation of the Dhammic Cognitive Architecture. Contains 8 neural modules that map Buddhist Abhidharma concepts to modern AI components, orchestrated through the Citta Vithi 8-step pipeline. Each module is a standalone `nn.Module` with its own test suite.

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Comprehensive architecture docs with component descriptions, usage examples, and design decisions |
| `benchmark_sdr.py` | SDR tokenizer performance benchmark (100 iterations, batch 32x128x512) |
| `artifacts_benchmark_profile.json` | Benchmark profiling results (JSON) |
| `artifacts_tests_profile_top.txt` | Top profiler hotspots from test suite execution |
| `.coverage` | pytest-cov coverage data |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | 8 source modules implementing the cognitive pipeline (see `src/AGENTS.md`) |
| `tests/` | Test suites for all components, 46+ tests (see `tests/AGENTS.md`) |
| `docs/` | Documentation directory (currently empty, reserved for future use) |
| `artifacts_profiler_traces/` | Chrome-compatible JSON profiler traces per component (generated output) |

## For AI Agents

### Working In This Directory
- **Read `README.md` first** -- it documents every component's Buddhist concept, AI analogue, implementation details, and design decisions
- The architecture spec is at `../s` (YAML) -- consult it for the ontological mapping
- All source is in `src/` with no `__init__.py` -- imports use direct module names
- The 8-step pipeline (`src/citta_vithi_pipeline.py`) orchestrates all other components

### Architecture Overview

```
Token → Embedding → Mamba-3 SSM → Thalamic Gate → Spike Generator → SDR Tokenizer
  → Latent Poller → Engram Lookup → Javana LoRA (×7 cycles) → Memory Store → Logits
```

| Step | Buddhist Name | Module | Dimension |
|------|---------------|--------|-----------|
| 0 | - | Embedding | vocab → d_model |
| 1 | Bhavanga | `mamba_base_engine` | d_model → d_model |
| 2 | Avajjana | Thalamic Gate (inline) | d_model → d_model |
| 3 | Vinnana | Spike Generator (inline) | d_model → d_model |
| 4 | Sampaticchana | `numenta_sdr_tokenizer` | d_model → sdr_dim (2% sparse) |
| 5 | Santiirana | Latent Poller (inline) | sdr_dim → sdr_dim//4 |
| 6 | Votthapana | `deepseek_engram_memory` | sdr_dim//4 → engram_vocab |
| 7 | Javana | `deltanet_hebbian_lora` | sdr_dim//4 → sdr_dim//4 (×7) |
| 8 | Tadarammana | Memory Store | gradient accumulation |

### Testing Requirements
```bash
cd dhammic-ai
python3 -m pytest tests/ -v          # All tests
python3 -m pytest tests/ -v --amp-test  # Include AMP tests
```

### Common Patterns
- Factory functions: `create_mamba3_base_engine()`, `create_numenta_sdr_tokenizer()`, etc.
- All modules accept `device` and `dtype` kwargs for placement control
- `_amp_disabled = True` on modules that don't support mixed precision
- Dimensional transitions are the main integration challenge -- see README "Key Implementation Challenges"
- Javana step runs exactly 7 cycles (traditional Buddhist psychology)

## Dependencies

### Internal
- `../s` -- Architecture specification (YAML ontology)

### External
- `torch` -- All neural components
- `numpy` -- SDR operations, hash computation
- `hashlib` -- MD5 hashing for engram memory O(1) lookup
- `pickle` -- Engram memory serialization

<!-- MANUAL: -->
