<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-25 | Updated: 2026-03-25 -->

# tests

## Purpose
Comprehensive test suites for all 8 Dhammic Cognitive Architecture modules. 46+ tests covering creation, forward pass, dimensional variations, memory statistics, learning dynamics, and edge cases.

## Key Files

| File | Tests | Description |
|------|-------|-------------|
| `conftest.py` | - | Pytest configuration: adds `--amp-test` CLI option for AMP-specific tests |
| `test_mamba_base_engine.py` | ~6 | Mamba-3 SSM: creation, forward pass, state tracking, different d_state values |
| `test_numenta_sdr_tokenizer.py` | ~6 | SDR encoder: sparsity verification, zone distribution, k-winners, different sdr_dim |
| `test_deepseek_engram_memory.py` | ~6 | Engram memory: hash lookup, learn/retrieve cycle, frequency pruning, unknown labels |
| `test_deltanet_hebbian_lora.py` | ~6 | Hebbian LoRA: forward pass, STDP updates, learning rate decay, weight normalization |
| `test_sati_neuromodulatory_gate.py` | ~6 | Neuromodulatory gate: open/closed states, dopamine/cortisol modulation, gate decay |
| `test_citta_vithi_pipeline.py` | ~6 | Full pipeline: 8-step forward pass, intermediate outputs, dimensional consistency |
| `test_learning_loss_dynamics.py` | ~5 | Loss/optimizer: DhammicLoss computation, KarmicOptimizer updates, homeostasis target |
| `test_physical_thermodynamic_optimizations.py` | ~5 | Optimizations: sparsity application, overlap vectorization, energy-based gradients |

## For AI Agents

### Working In This Directory
- Run all tests: `cd dhammic-ai && python3 -m pytest tests/ -v`
- Run single module: `python3 -m pytest tests/test_mamba_base_engine.py -v`
- AMP tests: `python3 -m pytest tests/ -v --amp-test`
- Tests import directly from `src/` (no package structure)

### Testing Patterns
- Each test file mirrors its corresponding source module in `src/`
- Tests use `torch.manual_seed()` for reproducibility
- Common test structure: creation test → forward pass test → edge case tests → config variation tests
- All tests run on CPU by default (no GPU required)

### Adding New Tests
- Follow naming convention: `test_<module_name>.py`
- Include at minimum: creation, forward pass, and dimensional variation tests
- Use the `--amp-test` flag for any mixed-precision-specific tests

## Dependencies

### Internal
- All modules in `../src/` -- imported directly by module name

### External
- `pytest` -- test framework
- `torch` -- tensor creation and assertions

<!-- MANUAL: -->
