"""VRAM offload audit for the fused dhammic-ai pipeline.

Runs one forward+backward pass at a representative configuration and
captures a peak-memory snapshot. Classifies allocations and emits a
Markdown report listing top-N entries plus optimisation opportunities.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from citta_vithi_pipeline import CittaVithiPipeline  # noqa: E402
from kernels import FusedLMHeadXentModule  # noqa: E402

D_MODEL = 64
N_LAYERS = 2
D_STATE = 8
EXPAND = 2
N_HEADS = 4
CHUNK_SIZE = 256
VOCAB_SIZE = 1024
SDR_DIM = 256
ENG_COLS = 64
ENG_CELLS = 4

BATCH = 2
SEQ = 2048


def _bytes_to_mb(n: int) -> float:
    return n / (1024.0 * 1024.0)


def _classify(name: str) -> str:
    """Coarse classification by parameter name."""
    lname = name.lower()
    if "embed" in lname or "to_sdr" in lname or "from_sdr" in lname or "cell_embed" in lname:
        return "W:emb"
    if "weight" in lname and "norm" not in lname:
        return "W:linear"
    if "bias" in lname or "norm" in lname:
        return "W:norm/bias"
    if "a_log" in lname or "dt_bias" in lname or "_bias" in lname:
        return "W:ssm-param"
    return "W:other"


def main():
    runs_dir = ROOT / "runs"
    audit_dir = ROOT / "docs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_path = audit_dir / "offload_audit.md"

    torch.manual_seed(0)
    model = CittaVithiPipeline(
        d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
        mamba_expand=EXPAND, n_heads=N_HEADS, chunk_size=CHUNK_SIZE,
        vocab_size=VOCAB_SIZE, sdr_dim=SDR_DIM, sdr_k_active=10,
        engram_n_columns=ENG_COLS, engram_cells_per_col=ENG_CELLS,
        engram_k_active=4,
    ).cuda()
    head = FusedLMHeadXentModule(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE,
        weight=model.embedding.dense_embed.weight,
    ).cuda()

    # Warmup
    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ), device="cuda")
    y = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = model(x)
        loss = head(h, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    del h, loss, x, y
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Run with snapshot collection
    torch.cuda.memory._record_memory_history(max_entries=10000)
    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ), device="cuda")
    y = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = model(x)
        loss = head(h, y)
    loss.backward()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)

    free_b, total_b = torch.cuda.mem_get_info(0)

    # ---- Parameter breakdown (W class) -------------------------------------
    weight_bytes = defaultdict(int)
    weight_rows: list[tuple[str, int]] = []
    for name, p in list(model.named_parameters()) + list(
        ("head." + n, q) for n, q in head.named_parameters()
        if not any(q is m for m in model.parameters())
    ):
        nb = p.numel() * p.element_size()
        cls = _classify(name)
        weight_bytes[cls] += nb
        weight_rows.append((name, nb))
    total_weight = sum(weight_bytes.values())
    weight_rows.sort(key=lambda r: -r[1])

    # ---- Allocations from the snapshot -------------------------------------
    # snapshot['segments'][i]['blocks'] is a list of {'state', 'size', ...}.
    # state == 'active_allocated' means a live tensor.
    allocs: list[tuple[int, str]] = []
    if isinstance(snapshot, dict):
        for seg in snapshot.get("segments", []):
            for blk in seg.get("blocks", []):
                if blk.get("state") not in {"active_allocated"}:
                    continue
                sz = int(blk.get("size", 0))
                # Frames: most recent first; pick the first non-torch one.
                fr_list = blk.get("frames") or []
                top = None
                for fr in fr_list:
                    name = fr.get("filename", "") + ":" + str(fr.get("line", ""))
                    if "/torch/" in name and "/dhammic" not in name:
                        continue
                    top = f"{fr.get('name', '?')} @ {name}"
                    break
                if top is None and fr_list:
                    fr = fr_list[0]
                    top = f"{fr.get('name', '?')} @ {fr.get('filename', '?')}:{fr.get('line', '?')}"
                allocs.append((sz, top or "<no frame>"))
    allocs.sort(key=lambda r: -r[0])
    top_allocs = allocs[: max(20, 10)]

    # ---- Write report ------------------------------------------------------
    lines: list[str] = []
    lines.append("# Offload Audit — Fused Dhammic-AI Pipeline")
    lines.append("")
    lines.append(f"- GPU: `{torch.cuda.get_device_name(0)}`")
    lines.append(f"- Probe config: batch={BATCH}, seq_len={SEQ}, "
                 f"d_model={D_MODEL}, n_layers={N_LAYERS}, "
                 f"vocab={VOCAB_SIZE}, chunk_size={CHUNK_SIZE}")
    lines.append(f"- Peak VRAM during fwd+bwd: **{_bytes_to_mb(peak_bytes):.2f} MB**")
    lines.append(f"- Free / total VRAM at audit start: "
                 f"{_bytes_to_mb(free_b):.0f} / {_bytes_to_mb(total_b):.0f} MB")
    lines.append("")

    # Parameter table
    lines.append("## Parameter (weight) memory by class")
    lines.append("")
    lines.append("| class | bytes | MB | % of weights |")
    lines.append("|:---|---:|---:|---:|")
    for cls in sorted(weight_bytes, key=lambda c: -weight_bytes[c]):
        nb = weight_bytes[cls]
        lines.append(f"| {cls} | {nb} | {_bytes_to_mb(nb):.3f} | "
                     f"{100 * nb / max(total_weight, 1):.1f}% |")
    lines.append(f"| **TOTAL** | {total_weight} | "
                 f"{_bytes_to_mb(total_weight):.3f} | 100.0% |")
    lines.append("")

    # Top parameter entries
    lines.append("## Top 15 parameter tensors by size")
    lines.append("")
    lines.append("| name | bytes | MB |")
    lines.append("|:---|---:|---:|")
    for name, nb in weight_rows[:15]:
        lines.append(f"| `{name}` | {nb} | {_bytes_to_mb(nb):.3f} |")
    lines.append("")

    # Top live allocations
    lines.append(f"## Top {min(20, len(top_allocs))} live allocations (> 1 MB)")
    lines.append("")
    lines.append("| MB | top frame |")
    lines.append("|---:|:---|")
    shown = 0
    for sz, frame in top_allocs:
        if _bytes_to_mb(sz) < 1.0:
            continue
        lines.append(f"| {_bytes_to_mb(sz):.2f} | `{frame}` |")
        shown += 1
        if shown >= 20:
            break
    if shown == 0:
        lines.append("| (none > 1 MB at probe size) | |")
    lines.append("")

    # Heuristics for optimization opportunities
    lines.append("## Optimisation opportunities")
    lines.append("")
    opps: list[str] = []
    sdr_b = next((nb for name, nb in weight_rows if "to_sdr" in name), 0)
    sdr_b += next((nb for name, nb in weight_rows if "from_sdr" in name), 0)
    if sdr_b > 0.10 * total_weight:
        opps.append(
            f"- SDR projections account for {_bytes_to_mb(sdr_b):.2f} MB "
            f"({100 * sdr_b / total_weight:.1f}% of weights). At larger "
            f"vocab the dense_embed dwarfs this; at small vocab consider "
            f"shrinking ``sdr_dim`` further."
        )
    cell_b = next((nb for name, nb in weight_rows if "cell_embed" in name), 0)
    if cell_b > 0.20 * total_weight:
        opps.append(
            f"- HTMEngram cell_embed is {_bytes_to_mb(cell_b):.2f} MB. The "
            f"K8 fused kernel only loads a small (K × C) slice per token; "
            f"the table itself is fine to stay GPU-resident but is an obvious "
            f"offload candidate for very-large-vocab regimes."
        )
    if peak_bytes < 200 * 1024 * 1024:
        opps.append(
            f"- Peak ({_bytes_to_mb(peak_bytes):.1f} MB) is well under the "
            f"1.4 GB budget for the chosen probe shape; the budget is "
            f"already comfortable at this config. The chunked-offload "
            f"runtime is the right tool for longer-seq scaling rather than "
            f"this probe."
        )
    if not opps:
        opps.append("- Nothing obviously over-provisioned at the probe size.")
    for o in opps:
        lines.append(o)
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    print(f"Peak VRAM: {_bytes_to_mb(peak_bytes):.2f} MB")
    print(f"Total weights: {_bytes_to_mb(total_weight):.2f} MB")
    print(f"Top alloc count > 1 MB: {shown}")


if __name__ == "__main__":
    main()
