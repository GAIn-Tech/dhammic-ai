"""Scaling sweep for the fused dhammic-ai pipeline.

Runs a single forward+backward step at varying ``seq_len`` and
``batch_size`` for a tiny but architecture-faithful CittaVithiPipeline,
recording peak VRAM (MB), wall time (ms), and throughput (tok/s).

Writes:
  runs/scaling_sweep.csv
  runs/scaling_sweep.md

RTX 3060 has ~1.4 GB free VRAM at peak; the script bails on any seq_len
that OOMs and reports the largest feasible value.
"""

from __future__ import annotations

import csv
import gc
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from citta_vithi_pipeline import CittaVithiPipeline  # noqa: E402
from kernels import FusedLMHeadXentModule  # noqa: E402


# Architecture knobs held fixed (small but representative of the
# winning gen 9 config: d=128, l=4 — we shrink for the 1.4 GB budget).
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


def build_model() -> tuple[CittaVithiPipeline, FusedLMHeadXentModule]:
    torch.manual_seed(42)
    model = CittaVithiPipeline(
        d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
        mamba_expand=EXPAND, n_heads=N_HEADS, chunk_size=CHUNK_SIZE,
        vocab_size=VOCAB_SIZE, sdr_dim=SDR_DIM, sdr_k_active=SDR_K,
        engram_n_columns=ENG_COLS, engram_cells_per_col=ENG_CELLS,
        engram_k_active=ENG_K,
    ).to(DEVICE)
    head = FusedLMHeadXentModule(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE,
        weight=model.embedding.dense_embed.weight,
    ).to(DEVICE)
    return model, head


def _bytes_to_mb(n: int) -> float:
    return n / (1024.0 * 1024.0)


def run_one(model, head, batch: int, seq: int, warmup: int = 1,
            iters: int = 3) -> dict | None:
    assert seq % CHUNK_SIZE == 0, f"seq {seq} not divisible by chunk {CHUNK_SIZE}"

    try:
        # Warm-up
        for _ in range(warmup):
            x = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE)
            y = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                hidden = model(x)
                loss = head(hidden, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            del hidden, loss, x, y

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Timed iterations
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            x = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE)
            y = torch.randint(0, VOCAB_SIZE, (batch, seq), device=DEVICE)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                hidden = model(x)
                loss = head(hidden, y)
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


def main():
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    csv_path = runs_dir / "scaling_sweep.csv"
    md_path = runs_dir / "scaling_sweep.md"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f"VRAM free / total: {_bytes_to_mb(free_b):.0f} / {_bytes_to_mb(total_b):.0f} MB")
    print(f"Architecture: d={D_MODEL} l={N_LAYERS} ds={D_STATE} exp={EXPAND} "
          f"heads={N_HEADS} V={VOCAB_SIZE} chunk={CHUNK_SIZE}")

    # Sweep 1: seq_len at fixed batch=2
    SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192]
    BATCHES = [1, 2, 4]

    model, head = build_model()
    results: list[dict] = []

    print("\n=== Sweep 1: seq_len at batch=2 ===")
    for seq in SEQ_LENS:
        rec = run_one(model, head, batch=2, seq=seq)
        if rec is None:
            print(f"  seq={seq}: OOM")
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
              f"{rec['elapsed_ms']:7.1f} ms | {rec['tok_per_sec']:8.0f} tok/s")
        results.append(rec)

    print("\n=== Sweep 2: batch at seq=2048 ===")
    for batch in BATCHES:
        rec = run_one(model, head, batch=batch, seq=2048)
        if rec is None:
            print(f"  batch={batch}: OOM")
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
              f"{rec['elapsed_ms']:7.1f} ms | {rec['tok_per_sec']:8.0f} tok/s")
        results.append(rec)

    # Write CSV
    with csv_path.open("w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=["batch", "seq_len", "elapsed_ms",
                        "peak_vram_mb", "tokens", "tok_per_sec", "status"],
        )
        wr.writeheader()
        for r in results:
            wr.writerow(r)

    # Markdown summary with key assertions
    seq_runs = [r for r in results if r["batch"] == 2 and r["status"] == "OK"]
    batch_runs = [r for r in results if r["seq_len"] == 2048 and r["status"] == "OK"]

    seq_runs_sorted = sorted(seq_runs, key=lambda r: r["seq_len"])
    vram_256 = next((r["peak_vram_mb"] for r in seq_runs_sorted if r["seq_len"] == 256), None)
    vram_largest = seq_runs_sorted[-1]["peak_vram_mb"] if seq_runs_sorted else None
    largest_seq = seq_runs_sorted[-1]["seq_len"] if seq_runs_sorted else None
    ratio = (vram_largest / vram_256) if (vram_256 and vram_largest) else None

    lines: list[str] = []
    lines.append("# Scaling Sweep — Fused Dhammic-AI Pipeline")
    lines.append("")
    lines.append(f"- GPU: `{torch.cuda.get_device_name(0)}`")
    lines.append(f"- VRAM total / free at start: "
                 f"{_bytes_to_mb(total_b):.0f} / {_bytes_to_mb(free_b):.0f} MB")
    lines.append(f"- Architecture: d={D_MODEL} l={N_LAYERS} ds={D_STATE} "
                 f"exp={EXPAND} heads={N_HEADS} V={VOCAB_SIZE} chunk={CHUNK_SIZE}")
    lines.append("")
    lines.append("## Sweep 1: seq_len at batch=2")
    lines.append("")
    lines.append("| seq_len | peak VRAM (MB) | wall (ms) | tok/s | status |")
    lines.append("|---:|---:|---:|---:|:---|")
    for r in sorted([rr for rr in results if rr["batch"] == 2],
                    key=lambda rr: rr["seq_len"]):
        lines.append(f"| {r['seq_len']} | {r['peak_vram_mb']:.1f} | "
                     f"{r['elapsed_ms']:.1f} | {r['tok_per_sec']:.0f} | "
                     f"{r['status']} |")
    lines.append("")
    lines.append("## Sweep 2: batch at seq_len=2048")
    lines.append("")
    lines.append("| batch | peak VRAM (MB) | wall (ms) | tok/s | status |")
    lines.append("|---:|---:|---:|---:|:---|")
    for r in sorted([rr for rr in results if rr["seq_len"] == 2048],
                    key=lambda rr: rr["batch"]):
        lines.append(f"| {r['batch']} | {r['peak_vram_mb']:.1f} | "
                     f"{r['elapsed_ms']:.1f} | {r['tok_per_sec']:.0f} | "
                     f"{r['status']} |")
    lines.append("")
    lines.append("## Key assertions")
    lines.append("")
    if vram_256 is not None and vram_largest is not None:
        lines.append(f"- VRAM ratio (seq={largest_seq} / seq=256): "
                     f"**{ratio:.2f}x** "
                     f"({vram_largest:.1f} MB / {vram_256:.1f} MB)")
        if largest_seq == 8192:
            lines.append(f"- Target: ratio ≤ 1.5x for chunked-offload to be "
                         f"\"working\" — observed {ratio:.2f}x.")
        else:
            lines.append(f"- 8192 OOM'd; largest feasible seq = {largest_seq}.")
    if batch_runs:
        b1 = next((r for r in batch_runs if r["batch"] == 1), None)
        b2 = next((r for r in batch_runs if r["batch"] == 2), None)
        b4 = next((r for r in batch_runs if r["batch"] == 4), None)
        if b1 and b2:
            lines.append(f"- VRAM batch=2 / batch=1 ratio: "
                         f"{b2['peak_vram_mb'] / b1['peak_vram_mb']:.2f}x "
                         f"(target ≈ 2x — roughly linear)")
        if b2 and b4:
            lines.append(f"- VRAM batch=4 / batch=2 ratio: "
                         f"{b4['peak_vram_mb'] / b2['peak_vram_mb']:.2f}x "
                         f"(target ≈ 2x — roughly linear)")
    if len(seq_runs_sorted) >= 2:
        tps_short = seq_runs_sorted[0]["tok_per_sec"]
        tps_long = seq_runs_sorted[-1]["tok_per_sec"]
        lines.append(f"- Throughput seq={largest_seq} vs seq=256: "
                     f"{tps_long:.0f} vs {tps_short:.0f} tok/s "
                     f"(target: roughly flat as seq grows)")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
