<!-- Generated: 2026-03-25 | Updated: 2026-03-25 -->

# r2

## Purpose
Research repository for the **Dhammic Cognitive Architecture** -- a bidirectional mapping between 2,300-year-old Abhidharma Buddhist phenomenology and state-of-the-art 2026 AI architecture. Implements a complete 8-step cognitive pipeline (Citta Vithi) using Mamba SSM, Numenta SDR, DeepSeek Engram memory, DeltaNet Hebbian LoRA, and neuromodulatory gating.

## Key Files

| File | Description |
|------|-------------|
| `s` | Architecture specification (YAML) defining the full Dhammic Cognitive Architecture ontology, component mappings, and 8-step pipeline |
| `benchmark_sdr.py` | SDR tokenizer benchmark script (latency measurement) |
| `test_overlap.py` | Quick SDR bitwise overlap computation test |
| `.tmp_model_repl_summary.txt` | Temporary model REPL session summary (ephemeral) |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `dhammic-ai/` | Main implementation: all source modules, tests, benchmarks, and profiler artifacts (see `dhammic-ai/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- The architecture spec lives in `s` (YAML format) -- read it first to understand the Buddhist-to-AI mapping before modifying any component
- Root-level scripts (`benchmark_sdr.py`, `test_overlap.py`) are quick utilities; the real code is in `dhammic-ai/src/`
- Python dependencies: `torch`, `numpy` -- use `uv` for package management (NOT pip)
- All source imports assume `dhammic-ai/` as the working directory

### Testing Requirements
- Run tests from inside `dhammic-ai/`: `cd dhammic-ai && python3 -m pytest tests/ -v`
- 46+ tests across 8 test files
- Use `--amp-test` flag for AMP-specific tests (custom pytest option in `conftest.py`)

### Common Patterns
- Every neural component maps to a Buddhist psychological concept (documented in docstrings)
- Components use `_amp_disabled = True` flag to control mixed precision behavior
- Factory functions (`create_*`) are the standard way to instantiate components
- All modules are pure PyTorch `nn.Module` subclasses

## Dependencies

### External
- `torch` (PyTorch) -- core neural network framework
- `numpy` -- numerical operations
- `hashlib` -- MD5 hashing for engram memory

<!-- MANUAL: -->
