"""Chunked-offload scaling sweep for the dhammic-ai pipeline.

This is the *chunked* counterpart to ``scaling_sweep.py``. It runs the
SAME architecture and the SAME (seq_len, batch) grid, but routes the
forward through ``ChunkedOffloadRunner`` so the backbone activations
are staged on pinned-CPU instead of resident on the GPU.

------------------------------------------------------------
Strategy
------------------------------------------------------------

The pipeline is:

    token_ids ── embedding ──► (B, T, D) ── backbone[0..N] ──► (B, T, D)
                                                              │
                                                              └► final_norm ──► hidden
                                                                               │
                                                                               head ──► loss

We chose strategy (b) from the task brief: one ``runner.forward()`` call
wraps the *entire* sequential backbone (plus ``final_norm``) as a single
``layer_fn``. The embedding stays eager because token IDs are tiny, and
the head stays eager because it already implements an internal-chunked
fused kernel; staging its (B, T, D) input back to CPU and feeding it
chunk-by-chunk would require a second runner wrap which is out of scope
for *this* validation.

(a) would wrap each backbone layer individually, which is finer-grained
but the SSM block carries cross-chunk state via its ``cache`` dict —
plumbing that state across runner-managed chunks is itself a research
project. The runtime's existing unit test was written against a dummy
MLP for exactly this reason. (b) is enough to prove the activations-
offload claim (the user's stated goal) without re-engineering the SSM
cache.

------------------------------------------------------------
Outputs
------------------------------------------------------------
    runs/scaling_sweep_chunked.csv
    runs/scaling_sweep_chunked.md

The md file contains an apples-to-apples comparison table against
``runs/scaling_sweep.csv`` (the eager baseline) and the two key
assertions the validation hinges on:

    1. chunked@8192 / chunked@256  ≤ 1.5×   (flat VRAM)
    2. chunked < eager at seq ≥ 1024        (offload wins at scale)

If either assertion fails, the script prints the raw numbers and exits
without shimming or hiding the failure.
"""

from __future__ import annotations

import csv
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from citta_vithi_pipeline import CittaVithiPipeline  # noqa: E402
from kernels import FusedLMHeadXentModule  # noqa: E402
from runtime.chunked_offload import ChunkedOffloadRunner  # noqa: E402


# Architecture — same knobs as scripts/scaling_sweep.py (apples-to-apples).
D_MODEL = 64
N_LAYERS = 2
D_STATE = 8
EXPAND = 2
N_HEADS = 4
CHUNK_SIZE = 256
VOCAB_SIZE = 1024
SDR_DIM = 256
SDR_K = 10
ENG_COLS = 64
ENG_CELLS = 4
ENG_K = 4

DEVICE = "cuda"


# --------------------------------------------------------------------------- #
# Composite "backbone+norm" module — what we hand to the runner as layer_fn.
# --------------------------------------------------------------------------- #
class BackboneAndNorm(nn.Module):
    """Wraps ``pipeline.backbone`` + ``pipeline.final_norm`` as a single
    ``layer_fn`` for ``ChunkedOffloadRunner``.

    Input shape:  (B, chunk, D)
    Output shape: (B, chunk, D)
    """

    def __init__(self, pipeline: CittaVithiPipeline) -> None:
        super().__init__()
        self.backbone = pipeline.backbone
        self.final_norm = pipeline.final_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The runner calls us outside of any autocast context, so re-enter
        # bf16 autocast here to stay apples-to-apples with the eager sweep
        # (which wraps the whole forward in bf16 autocast).
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for layer in self.backbone:
                x = layer(x)
            return self.final_norm(x)


def build_model() -> tuple[CittaVithiPipeline, BackboneAndNorm, FusedLMHeadXentModule]:
    torch.manual_seed(42)
    model = CittaVithiPipeline(
        d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
        mamba_expand=EXPAND, n_heads=N_HEADS, chunk_size=CHUNK_SIZE,
        vocab_size=VOCAB_SIZE, sdr_dim=SDR_DIM, sdr_k_active=SDR_K,
        engram_n_columns=ENG_COLS, engram_cells_per_col=ENG_CELLS,
        engram_k_active=ENG_K,
    ).to(DEVICE)
    body = BackboneAndNorm(model)  # shares parameters with model
    head = FusedLMHeadXentModule(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE,
        weight=model.embedding.dense_embed.weight,
    ).to(DEVICE)
    return model, body, head


def _bytes_to_mb(n: int) -> float:
    return n / (1024.0 * 1024.0)


def run_one_chunked(
    model: CittaVithiPipeline,
    body: BackboneAndNorm,
    head: FusedLMHeadXentModule,
    runner: ChunkedOffloadRunner,
    batch: int,
    seq: int,
    warmup: int = 1,
    iters: int = 3,
) -> dict | None:
    """Run a single warmed-up, timed forward+backward through the runner."""
    assert seq % CHUNK_SIZE == 0, f"seq {seq} not divisible by chunk {CHUNK_SIZE}"

    try:
        # -------- Warm-up (untimed, primes allocator + ring buffer) --------
        for _ in range(warmup):
            x = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE)
            y = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE)

            # 1. Embedding stays eager (GPU).
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb_gpu = model.embedding(x)

            # 2. Move embedded activations to pinned CPU; runner streams them.
            emb_cpu = torch.empty(
                emb_gpu.shape, dtype=emb_gpu.dtype,
                device="cpu", pin_memory=True,
            )
            emb_cpu.copy_(emb_gpu.detach())
            del emb_gpu

            # 3. Chunked backbone+norm.
            hidden_cpu = runner.forward(body, emb_cpu, CHUNK_SIZE)

            # 4. Head: pull hidden back to GPU for the fused-xent kernel.
            hidden_gpu = hidden_cpu.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = head(hidden_gpu, y)

            loss.backward()
            model.zero_grad(set_to_none=True)
            del hidden_gpu, hidden_cpu, emb_cpu, loss, x, y

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # -------- Timed iterations --------
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            x = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE)
            y = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                emb_gpu = model.embedding(x)
            emb_cpu = torch.empty(
                emb_gpu.shape, dtype=emb_gpu.dtype,
                device="cpu", pin_memory=True,
            )
            emb_cpu.copy_(emb_gpu.detach())
            del emb_gpu

            hidden_cpu = runner.forward(body, emb_cpu, CHUNK_SIZE)
            hidden_gpu = hidden_cpu.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = head(hidden_gpu, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed_s = (time.perf_counter() - t0) / iters

        peak_mb = _bytes_to_mb(torch.cuda.max_memory_allocated())
        tokens = batch * seq
        tok_per_s = tokens / elapsed_s

        return {
            "batch": batch,
            "seq_len": seq,
            "elapsed_ms": elapsed_s * 1000.0,
            "peak_vram_mb": peak_mb,
            "tokens": tokens,
            "tok_per_sec": tok_per_s,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return None
    except Exception as e:  # noqa: BLE001 — surface the error verbatim
        import traceback
        print(f"  ERROR at batch={batch} seq={seq}: {type(e).__name__}: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return None


def _load_eager_baseline() -> dict[tuple[int, int], float]:
    """Read the eager sweep csv and return {(batch, seq_len): peak_vram_mb}.

    Multiple rows for the same (batch, seq_len) → keep the *last* one (matches
    how scaling_sweep.py writes the file).
    """
    csv_path = ROOT / "runs" / "scaling_sweep.csv"
    if not csv_path.exists():
        return {}
    out: dict[tuple[int, int], float] = {}
    with csv_path.open() as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            if row.get("status") != "OK":
                continue
            try:
                key = (int(row["batch"]), int(row["seq_len"]))
                out[key] = float(row["peak_vram_mb"])
            except (ValueError, KeyError):
                continue
    return out


def main() -> int:
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    csv_path = runs_dir / "scaling_sweep_chunked.csv"
    md_path = runs_dir / "scaling_sweep_chunked.md"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f"VRAM free / total: {_bytes_to_mb(free_b):.0f} / "
          f"{_bytes_to_mb(total_b):.0f} MB")
    print(f"Architecture: d={D_MODEL} l={N_LAYERS} ds={D_STATE} "
          f"exp={EXPAND} heads={N_HEADS} V={VOCAB_SIZE} chunk={CHUNK_SIZE}")
    print("Strategy: (b) — single runner.forward() wrapping the whole "
          "backbone + final_norm.")

    SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192]
    BATCHES = [1, 2, 4]

    model, body, head = build_model()
    runner = ChunkedOffloadRunner()
    results: list[dict[str, Any]] = []

    print("\n=== Sweep 1 (chunked): seq_len at batch=2 ===")
    for seq in SEQ_LENS:
        rec = run_one_chunked(model, body, head, runner, batch=2, seq=seq)
        if rec is None:
            print(f"  seq={seq}: OOM / ERROR")
            results.append({
                "batch": 2, "seq_len": seq,
                "elapsed_ms": float("nan"),
                "peak_vram_mb": float("nan"),
                "tokens": 2 * seq,
                "tok_per_sec": float("nan"),
                "status": "OOM",
            })
            continue
        rec["status"] = "OK"
        print(f"  seq={seq}: {rec['peak_vram_mb']:7.1f} MB | "
              f"{rec['elapsed_ms']:8.1f} ms | {rec['tok_per_sec']:8.0f} tok/s")
        results.append(rec)

    print("\n=== Sweep 2 (chunked): batch at seq=2048 ===")
    for batch in BATCHES:
        rec = run_one_chunked(model, body, head, runner, batch=batch, seq=2048)
        if rec is None:
            print(f"  batch={batch}: OOM / ERROR")
            results.append({
                "batch": batch, "seq_len": 2048,
                "elapsed_ms": float("nan"),
                "peak_vram_mb": float("nan"),
                "tokens": batch * 2048,
                "tok_per_sec": float("nan"),
                "status": "OOM",
            })
            continue
        rec["status"] = "OK"
        print(f"  batch={batch}: {rec['peak_vram_mb']:7.1f} MB | "
              f"{rec['elapsed_ms']:8.1f} ms | {rec['tok_per_sec']:8.0f} tok/s")
        results.append(rec)

    # Write CSV (same columns as the eager sweep).
    with csv_path.open("w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=["batch", "seq_len", "elapsed_ms",
                        "peak_vram_mb", "tokens", "tok_per_sec", "status"],
        )
        wr.writeheader()
        for r in results:
            wr.writerow(r)

    # -------- Markdown report (with eager-vs-chunked comparison) --------
    eager = _load_eager_baseline()

    lines: list[str] = []
    lines.append("# Scaling Sweep — Chunked Offload")
    lines.append("")
    lines.append("Counterpart to `scaling_sweep.md`. Same architecture and "
                 "the same (seq_len, batch) grid, but the backbone + "
                 "`final_norm` are run through `ChunkedOffloadRunner` so "
                 "their activations are staged on pinned-CPU between chunks.")
    lines.append("")
    lines.append(f"- GPU: `{torch.cuda.get_device_name(0)}`")
    lines.append(f"- VRAM total / free at start: "
                 f"{_bytes_to_mb(total_b):.0f} / {_bytes_to_mb(free_b):.0f} MB")
    lines.append(f"- Architecture: d={D_MODEL} l={N_LAYERS} ds={D_STATE} "
                 f"exp={EXPAND} heads={N_HEADS} V={VOCAB_SIZE} "
                 f"chunk={CHUNK_SIZE}")
    lines.append("- Wrap strategy: (b) — single `runner.forward()` over the "
                 "whole backbone + `final_norm`. The embedding is eager "
                 "(token IDs are tiny) and the head is eager (its fused "
                 "kernel already chunks vocab internally).")
    lines.append("")
    lines.append("## Sweep 1: seq_len at batch=2 — chunked vs eager")
    lines.append("")
    lines.append("| seq_len | eager VRAM (MB) | chunked VRAM (MB) | "
                 "ratio eager/chunked | chunked/chunked@256 | chunked wall (ms) | "
                 "chunked tok/s |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    chunked_b2: dict[int, dict] = {
        r["seq_len"]: r for r in results
        if r["batch"] == 2 and r["status"] == "OK"
    }
    base_chunked = chunked_b2.get(256, {}).get("peak_vram_mb")

    for seq in SEQ_LENS:
        eager_v = eager.get((2, seq))
        ch = chunked_b2.get(seq)
        if ch is None or eager_v is None:
            eager_str = f"{eager_v:.1f}" if eager_v is not None else "n/a"
            ch_str = f"{ch['peak_vram_mb']:.1f}" if ch is not None else "n/a"
            lines.append(f"| {seq} | {eager_str} | {ch_str} "
                         f"| n/a | n/a | n/a | n/a |")
            continue
        ratio_e_c = eager_v / ch["peak_vram_mb"]
        ratio_c_base = (ch["peak_vram_mb"] / base_chunked
                        if base_chunked else float("nan"))
        lines.append(f"| {seq} | {eager_v:.1f} | {ch['peak_vram_mb']:.1f} | "
                     f"{ratio_e_c:.2f}x | {ratio_c_base:.2f}x | "
                     f"{ch['elapsed_ms']:.1f} | {ch['tok_per_sec']:.0f} |")

    lines.append("")
    lines.append("## Sweep 2: batch at seq=2048 — chunked vs eager")
    lines.append("")
    lines.append("| batch | eager VRAM (MB) | chunked VRAM (MB) | "
                 "ratio eager/chunked | chunked wall (ms) | chunked tok/s |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    chunked_s2048: dict[int, dict] = {
        r["batch"]: r for r in results
        if r["seq_len"] == 2048 and r["status"] == "OK"
    }
    for batch in BATCHES:
        eager_v = eager.get((batch, 2048))
        ch = chunked_s2048.get(batch)
        if ch is None or eager_v is None:
            continue
        ratio_e_c = eager_v / ch["peak_vram_mb"]
        lines.append(f"| {batch} | {eager_v:.1f} | {ch['peak_vram_mb']:.1f} | "
                     f"{ratio_e_c:.2f}x | {ch['elapsed_ms']:.1f} | "
                     f"{ch['tok_per_sec']:.0f} |")

    # ---- Key assertions ----
    lines.append("")
    lines.append("## Key assertions")
    lines.append("")

    flat_pass: bool | None = None
    wins_at_scale: bool | None = None

    if base_chunked is not None and 8192 in chunked_b2:
        top = chunked_b2[8192]["peak_vram_mb"]
        ratio = top / base_chunked
        flat_pass = ratio <= 1.5
        verdict = "PASS" if flat_pass else "FAIL"
        lines.append(f"- **Flat-VRAM:** chunked@8192 / chunked@256 = "
                     f"**{ratio:.2f}x** ({top:.1f} MB / {base_chunked:.1f} MB) "
                     f"— target ≤ 1.5x — **{verdict}**.")
    else:
        lines.append("- **Flat-VRAM:** could not be evaluated "
                     "(chunked sweep did not complete both 256 and 8192).")

    # chunked-wins-at-scale assertion (any of seq>=1024)
    wins_rows = []
    for seq in (1024, 2048, 4096, 8192):
        eager_v = eager.get((2, seq))
        ch = chunked_b2.get(seq)
        if eager_v is None or ch is None:
            continue
        wins_rows.append((seq, eager_v, ch["peak_vram_mb"],
                          ch["peak_vram_mb"] < eager_v))
    if wins_rows:
        wins_at_scale = all(w[3] for w in wins_rows)
        verdict = "PASS" if wins_at_scale else "FAIL"
        lines.append(f"- **chunked < eager at seq ≥ 1024:** **{verdict}**.")
        for seq, ev, cv, w in wins_rows:
            symbol = "✓" if w else "✗"
            lines.append(f"  - seq={seq}: eager={ev:.1f} MB vs "
                         f"chunked={cv:.1f} MB {symbol}")

    if flat_pass is not None and wins_at_scale is not None:
        lines.append("")
        if flat_pass and wins_at_scale:
            lines.append("**Flat-VRAM claim PROVEN** for the pipeline-level "
                         "chunked-offload wrap (strategy b).")
        else:
            lines.append("**Flat-VRAM claim NOT yet proven**. See the raw "
                         "numbers above. Probable suspects:")
            lines.append("- the runner's ring-buffer GPU input chunks "
                         "(3-deep, size B×chunk×D) — these scale with B and "
                         "D, NOT T, so should be flat across the sweep;")
            lines.append("- the head's eager (B, T, D) input — this is the "
                         "biggest known seq-proportional residual; chunked "
                         "head-loss would be a future optimization;")
            lines.append("- transient torch / cublas workspace allocations.")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
