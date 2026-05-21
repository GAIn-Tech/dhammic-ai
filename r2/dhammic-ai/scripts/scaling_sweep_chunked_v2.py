"""Chunked-offload scaling sweep — v2, with chunk-streaming LM head.

This is the V2 counterpart to ``scaling_sweep_chunked.py``.  Same
architecture, same (seq_len, batch) grid, same chunked-offload backbone
wrapping — but the LM head is now ``ChunkedLMHeadXent`` instead of
``FusedLMHeadXentModule``.  The head's input no longer needs to be a full
``(B, T, D)`` tensor pulled to GPU; instead we slice it chunk-by-chunk and
feed each slice through the fused kernel, with manual per-chunk backward
keeping the kernel's transient ``d_weight`` and ``d_hidden_flat`` working
tensors flat across ``n_chunks``.

The V3 (original chunked) sweep failed the flat-VRAM target with a ratio of
``chunked@8192 / chunked@256 = 3.54x`` (target ≤ 1.5x); the residual growth
came from the head materializing ``(B, T, D)`` hidden on GPU and the
kernel's per-row ``d_hidden_flat (M=B*T, D)`` fp32 accumulator.  This V2
sweep tests whether the chunk-streaming head closes that gap.

Outputs:
    runs/scaling_sweep_chunked_v2.csv
    runs/scaling_sweep_chunked_v2.md
"""

from __future__ import annotations

import csv
import gc
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
from runtime.chunked_offload import ChunkedOffloadRunner  # noqa: E402
from runtime.chunked_lmhead import ChunkedLMHeadXent  # noqa: E402


# Same architecture knobs as scripts/scaling_sweep_chunked.py for apples-to-apples.
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


class BackboneAndNorm(nn.Module):
    """Wraps ``pipeline.backbone`` + ``pipeline.final_norm`` as a single
    ``layer_fn`` for ``ChunkedOffloadRunner``."""

    def __init__(self, pipeline: CittaVithiPipeline) -> None:
        super().__init__()
        self.backbone = pipeline.backbone
        self.final_norm = pipeline.final_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for layer in self.backbone:
                x = layer(x)
            return self.final_norm(x)


def build_model() -> tuple[CittaVithiPipeline, BackboneAndNorm, ChunkedLMHeadXent]:
    torch.manual_seed(42)
    model = CittaVithiPipeline(
        d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
        mamba_expand=EXPAND, n_heads=N_HEADS, chunk_size=CHUNK_SIZE,
        vocab_size=VOCAB_SIZE, sdr_dim=SDR_DIM, sdr_k_active=SDR_K,
        engram_n_columns=ENG_COLS, engram_cells_per_col=ENG_CELLS,
        engram_k_active=ENG_K,
    ).to(DEVICE)
    body = BackboneAndNorm(model)
    head = ChunkedLMHeadXent(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE,
        weight=model.embedding.dense_embed.weight,
        chunk_size=CHUNK_SIZE,
    ).to(DEVICE)
    return model, body, head


def _bytes_to_mb(n: int) -> float:
    return n / (1024.0 * 1024.0)


def run_one_chunked_v2(
    model: CittaVithiPipeline,
    body: BackboneAndNorm,
    head: ChunkedLMHeadXent,
    runner: ChunkedOffloadRunner,
    batch: int,
    seq: int,
    warmup: int = 1,
    iters: int = 3,
) -> dict | None:
    """Warmed-up, timed forward+backward through runner + chunked head."""
    assert seq % CHUNK_SIZE == 0, f"seq {seq} not divisible by chunk {CHUNK_SIZE}"

    try:
        for _ in range(warmup):
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
            # Chunk-streaming head: keep hidden on CPU and feed it directly;
            # the head's autograd Function moves to-device internally per
            # chunk and never materializes the full hidden on GPU.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = head(hidden_cpu, y)

            loss.backward()
            model.zero_grad(set_to_none=True)
            del hidden_cpu, emb_cpu, loss, x, y

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

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
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = head(hidden_cpu, y)
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
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"  ERROR at batch={batch} seq={seq}: {type(e).__name__}: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return None


def _load_csv(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    if not path.exists():
        return {}
    out: dict[tuple[int, int], dict[str, float]] = {}
    with path.open() as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            if row.get("status") != "OK":
                continue
            try:
                key = (int(row["batch"]), int(row["seq_len"]))
                out[key] = {
                    "peak_vram_mb": float(row["peak_vram_mb"]),
                    "elapsed_ms": float(row["elapsed_ms"]),
                    "tok_per_sec": float(row["tok_per_sec"]),
                }
            except (ValueError, KeyError):
                continue
    return out


def main() -> int:
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    csv_path = runs_dir / "scaling_sweep_chunked_v2.csv"
    md_path = runs_dir / "scaling_sweep_chunked_v2.md"

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f"VRAM free / total: {_bytes_to_mb(free_b):.0f} / "
          f"{_bytes_to_mb(total_b):.0f} MB")
    print(f"Architecture: d={D_MODEL} l={N_LAYERS} ds={D_STATE} "
          f"exp={EXPAND} heads={N_HEADS} V={VOCAB_SIZE} chunk={CHUNK_SIZE}")
    print("Strategy: chunked-offload backbone + ChunkedLMHeadXent.")

    SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192]
    BATCHES = [1, 2, 4]

    model, body, head = build_model()
    runner = ChunkedOffloadRunner()
    results: list[dict[str, Any]] = []

    print("\n=== Sweep 1 (chunked v2): seq_len at batch=2 ===")
    for seq in SEQ_LENS:
        rec = run_one_chunked_v2(model, body, head, runner, batch=2, seq=seq)
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

    print("\n=== Sweep 2 (chunked v2): batch at seq=2048 ===")
    for batch in BATCHES:
        rec = run_one_chunked_v2(model, body, head, runner, batch=batch, seq=2048)
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

    with csv_path.open("w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=["batch", "seq_len", "elapsed_ms",
                        "peak_vram_mb", "tokens", "tok_per_sec", "status"],
        )
        wr.writeheader()
        for r in results:
            wr.writerow(r)

    # Markdown report — three-way comparison: eager, chunked v1, chunked v2.
    eager = _load_csv(runs_dir / "scaling_sweep.csv")
    chunked_v1 = _load_csv(runs_dir / "scaling_sweep_chunked.csv")

    lines: list[str] = []
    lines.append("# Scaling Sweep — Chunked Offload v2 (chunk-streaming LM head)")
    lines.append("")
    lines.append(
        "V2 of the chunked-offload sweep. Same architecture and (seq_len, "
        "batch) grid as `scaling_sweep_chunked.md`, but the LM head is "
        "`ChunkedLMHeadXent` so the head's input is streamed chunk-by-chunk "
        "and the kernel's per-chunk `d_weight` / `d_hidden_flat` transients "
        "are freed before the next chunk runs."
    )
    lines.append("")
    lines.append(f"- GPU: `{torch.cuda.get_device_name(0)}`")
    lines.append(f"- VRAM total / free at start: "
                 f"{_bytes_to_mb(total_b):.0f} / {_bytes_to_mb(free_b):.0f} MB")
    lines.append(f"- Architecture: d={D_MODEL} l={N_LAYERS} ds={D_STATE} "
                 f"exp={EXPAND} heads={N_HEADS} V={VOCAB_SIZE} "
                 f"chunk={CHUNK_SIZE}")
    lines.append("")
    lines.append("## Sweep 1: seq_len at batch=2 — eager vs chunked v1 vs chunked v2")
    lines.append("")
    lines.append(
        "| seq_len | eager VRAM (MB) | chunked v1 (MB) | chunked v2 (MB) | "
        "v2/v2@256 | v2 wall (ms) | v2 tok/s |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    chunked_v2_b2 = {
        r["seq_len"]: r for r in results
        if r["batch"] == 2 and r["status"] == "OK"
    }
    base_v2 = chunked_v2_b2.get(256, {}).get("peak_vram_mb")

    for seq in SEQ_LENS:
        eager_v = eager.get((2, seq), {}).get("peak_vram_mb")
        v1 = chunked_v1.get((2, seq), {}).get("peak_vram_mb")
        v2 = chunked_v2_b2.get(seq)
        cells = [
            f"{eager_v:.1f}" if eager_v is not None else "n/a",
            f"{v1:.1f}" if v1 is not None else "n/a",
            f"{v2['peak_vram_mb']:.1f}" if v2 is not None else "n/a",
            (f"{v2['peak_vram_mb'] / base_v2:.2f}x"
             if v2 is not None and base_v2 else "n/a"),
            f"{v2['elapsed_ms']:.1f}" if v2 is not None else "n/a",
            f"{v2['tok_per_sec']:.0f}" if v2 is not None else "n/a",
        ]
        lines.append(f"| {seq} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Sweep 2: batch at seq=2048 — eager vs chunked v1 vs chunked v2")
    lines.append("")
    lines.append(
        "| batch | eager VRAM (MB) | chunked v1 (MB) | chunked v2 (MB) | "
        "v2 wall (ms) | v2 tok/s |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")
    chunked_v2_s2048 = {
        r["batch"]: r for r in results
        if r["seq_len"] == 2048 and r["status"] == "OK"
    }
    for batch in BATCHES:
        eager_v = eager.get((batch, 2048), {}).get("peak_vram_mb")
        v1 = chunked_v1.get((batch, 2048), {}).get("peak_vram_mb")
        v2 = chunked_v2_s2048.get(batch)
        cells = [
            f"{eager_v:.1f}" if eager_v is not None else "n/a",
            f"{v1:.1f}" if v1 is not None else "n/a",
            f"{v2['peak_vram_mb']:.1f}" if v2 is not None else "n/a",
            f"{v2['elapsed_ms']:.1f}" if v2 is not None else "n/a",
            f"{v2['tok_per_sec']:.0f}" if v2 is not None else "n/a",
        ]
        lines.append(f"| {batch} | " + " | ".join(cells) + " |")

    # Key assertions
    lines.append("")
    lines.append("## Key assertions")
    lines.append("")

    flat_pass: bool | None = None

    if base_v2 is not None and 8192 in chunked_v2_b2:
        top = chunked_v2_b2[8192]["peak_vram_mb"]
        ratio = top / base_v2
        flat_pass = ratio <= 1.5
        verdict = "PASS" if flat_pass else "FAIL"
        lines.append(
            f"- **Flat-VRAM (v2):** chunked-v2@8192 / chunked-v2@256 = "
            f"**{ratio:.2f}x** ({top:.1f} MB / {base_v2:.1f} MB) "
            f"— target ≤ 1.5x — **{verdict}**."
        )
        # Side-by-side with V1.
        v1_base = chunked_v1.get((2, 256), {}).get("peak_vram_mb")
        v1_top = chunked_v1.get((2, 8192), {}).get("peak_vram_mb")
        if v1_base is not None and v1_top is not None:
            lines.append(
                f"- For comparison: v1 ratio was "
                f"**{v1_top / v1_base:.2f}x** "
                f"({v1_top:.1f} MB / {v1_base:.1f} MB) — "
                f"v2 improves the seq-scale residual by "
                f"**{(v1_top / v1_base) / (top / base_v2):.2f}x**."
            )
    else:
        lines.append("- **Flat-VRAM (v2):** could not be evaluated "
                     "(sweep did not complete both 256 and 8192).")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {csv_path} and {md_path}")

    if flat_pass is False:
        print("\n[NOTE] Flat-VRAM target NOT met — see md report for raw numbers.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
