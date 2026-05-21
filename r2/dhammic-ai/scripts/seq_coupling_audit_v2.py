"""Diff-based offload audit, V2 — Python-stack-aware.

Improvements over v1:
- Warmup pass at each shape to absorb Triton autotune compilation buffers
  (these confounded v1's seq=2048 first-call peak of 300 MB).
- torch.cuda.memory._record_memory_history(stacks="python", context="all")
  to capture real Python source frames per allocation.
- Per-allocation diff: any tensor that *grows linearly with T* is the leak.
- Reports the top callers (Python file:line) responsible for seq-coupling.
"""

from __future__ import annotations

import gc
import os
import sys
from collections import defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from citta_vithi_pipeline import CittaVithiPipeline
from runtime.chunked_offload import ChunkedOffloadRunner
from runtime.chunked_lmhead import ChunkedLMHeadXent


def build_model(d_model=64, n_layers=2, vocab=1024):
    m = CittaVithiPipeline(
        d_model=d_model, n_layers=n_layers, d_state=8, mamba_expand=2,
        n_heads=4, chunk_size=256, vocab_size=vocab,
        sdr_dim=256, sdr_k_active=10,
        engram_n_columns=64, engram_cells_per_col=4, engram_k_active=4,
    ).to("cuda").to(torch.bfloat16)
    return m


class BB(torch.nn.Module):
    """Backbone+norm wrapped so the ChunkedOffloadRunner can call it as one layer_fn."""
    def __init__(self, m):
        super().__init__()
        self.layers = m.backbone
        self.norm = m.final_norm

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


def one_step(seq_len, batch=2, chunk_size=256, d_model=64, vocab=1024):
    model = build_model(d_model=d_model, vocab=vocab)
    runner = ChunkedOffloadRunner()
    head = ChunkedLMHeadXent(
        d_model=d_model, vocab_size=vocab,
        weight=model.embedding.dense_embed.weight,
        chunk_size=chunk_size,
    ).to("cuda").to(torch.bfloat16)

    x = torch.randint(0, vocab, (batch, seq_len), device="cuda")
    y = torch.randint(0, vocab, (batch, seq_len), device="cuda")

    emb = model.embedding(x)
    bb = BB(model).to("cuda").to(torch.bfloat16)
    h_full = runner.forward(bb.forward, emb, chunk_size)
    loss = head(h_full, y)
    loss.backward()

    del model, runner, head, x, y, emb, h_full, loss, bb
    gc.collect(); torch.cuda.empty_cache()


def measure_at(seq_len):
    """Warmup once, then measure peak + snapshot."""
    # warmup
    one_step(seq_len)
    gc.collect(); torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(
        max_entries=100000, stacks="python", context="all"
    )

    one_step(seq_len)

    peak = torch.cuda.max_memory_allocated() / 1e6
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    return peak, snap


def best_python_frame(frames):
    """Find the most useful Python frame from a snapshot frame list.
    Prefers frames inside our repo, falls back to PyTorch internals."""
    if not frames:
        return "<no frame>"
    repo_marker = "/home/mikeb/r2/dhammic-ai/"
    best = None
    for fr in frames:
        if isinstance(fr, dict):
            name = fr.get("name", "")
            filename = fr.get("filename", "")
        else:
            name = str(fr)
            filename = ""
        # prefer repo-local Python frames
        if filename and repo_marker in filename:
            short = filename.replace(repo_marker, "")
            line = fr.get("line", "?")
            return f"{short}:{line}:{name}"
        if best is None and "site-packages" not in filename:
            best = (filename, fr.get("line", "?"), name)
    if best:
        return f"{best[0]}:{best[1]}:{best[2]}"
    return f"{frames[0].get('name', '<?>') if isinstance(frames[0], dict) else frames[0]}"


def summarize(snap):
    """Bucket active allocations by (rounded MB, python frame)."""
    buckets = defaultdict(lambda: {"count": 0, "bytes": 0})
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            size = blk.get("size", 0)
            if size < 64 * 1024:
                continue
            frame = best_python_frame(blk.get("frames", []))
            key = (round(size / 1e6, 3), frame)
            buckets[key]["count"] += 1
            buckets[key]["bytes"] += size
    return buckets


def main():
    seq_lens = [2048, 4096, 8192]
    snapshots = {}
    for T in seq_lens:
        peak, snap = measure_at(T)
        snapshots[T] = (peak, summarize(snap))
        print(f"seq={T:>5d}  peak={peak:7.2f} MB  buckets={len(snapshots[T][1])}")

    # Build all-keys diff table
    all_keys = set()
    for _, b in snapshots.values():
        all_keys.update(b.keys())

    # For each bucket, compute total bytes at each T
    rows = []
    for k in all_keys:
        size_mb, frame = k
        counts = [snapshots[T][1].get(k, {"count": 0, "bytes": 0})["count"] for T in seq_lens]
        totals_mb = [size_mb * c for c in counts]
        rows.append((size_mb, frame, counts, totals_mb))

    # Sort by total at 8k
    rows.sort(key=lambda r: -r[3][-1])

    print("\n# Top allocations (sorted by total MB at seq=8192)\n")
    print(f"{'size_MB':>8} | {'@2k':>4} | {'@4k':>4} | {'@8k':>4} | "
          f"{'tot@8k_MB':>10} | {'growth':>7} | frame")
    print("-" * 140)
    for size_mb, frame, counts, totals in rows[:30]:
        c2k, c4k, c8k = counts
        t8k = totals[-1]
        t2k = totals[0]
        if t2k > 0:
            growth = f"{t8k/t2k:.2f}x"
        elif t8k > 0:
            growth = "new"
        else:
            growth = "—"
        print(f"{size_mb:8.3f} | {c2k:4d} | {c4k:4d} | {c8k:4d} | "
              f"{t8k:10.2f} | {growth:>7} | {frame[:80]}")

    # Identify seq-coupled rows: growth ratio ≈ 4× (since T doubled twice)
    print("\n# Likely seq-coupled allocations (growth ≥ 1.8x and total > 1 MB at 8k)\n")
    leaks = []
    for size_mb, frame, counts, totals in rows:
        c2k, c4k, c8k = counts
        t2k, t4k, t8k = totals
        if t8k < 1.0:
            continue
        if t2k > 0 and (t8k / t2k) >= 1.8:
            leaks.append((t8k - t2k, size_mb, frame, counts, totals))
        elif t2k == 0 and t8k >= 1.0:
            leaks.append((t8k, size_mb, frame, counts, totals))

    leaks.sort(key=lambda r: -r[0])
    for delta, size_mb, frame, counts, totals in leaks[:10]:
        c2k, c4k, c8k = counts
        t2k, t4k, t8k = totals
        print(f"  + {delta:6.2f} MB grew | {size_mb:.3f} MB × counts ({c2k}, {c4k}, {c8k}) "
              f"| @8k={t8k:.2f} MB | frame: {frame[:90]}")

    # Peak summary
    print("\n# Peak summary\n")
    for T in seq_lens:
        peak, _ = snapshots[T]
        print(f"  seq={T:>5d}: peak={peak:7.2f} MB")
    p2, p4, p8 = (snapshots[T][0] for T in seq_lens)
    print(f"\n  growth peak@8k/peak@2k = {p8/p2:.2f}x")
    print(f"  growth peak@8k/peak@4k = {p8/p4:.2f}x")
    if p4 > 0:
        k_per_t = (p8 - p4) / (8192 - 4096) * 1e6  # bytes per token (B included)
        constant = p4 * 1e6 - 4096 * k_per_t
        print(f"  fit: peak(T) ≈ {constant/1e6:.2f} MB + {k_per_t:.2f} bytes/token")

    # Write markdown
    md_path = os.path.join(ROOT, "docs", "audit", "seq_coupling_diff_v2.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write("# Seq-coupling audit V2 — Python-stack frames\n\n")
        f.write("Each allocation observed at peak fwd+bwd after a warmup pass to absorb\n")
        f.write("Triton autotune compile buffers. Frames resolved to actual Python source.\n\n")
        f.write("## Peak by seq_len\n\n")
        for T in seq_lens:
            f.write(f"- seq={T}: peak={snapshots[T][0]:.2f} MB\n")
        if p4 > 0:
            f.write(f"\n**Linear fit**: peak(T) ≈ {constant/1e6:.2f} MB + {k_per_t:.2f} bytes/token\n\n")
        f.write("## Top 30 allocations at seq=8192\n\n")
        f.write("| size_MB | @2k | @4k | @8k | total@8k_MB | growth | frame |\n")
        f.write("|---:|---:|---:|---:|---:|---:|:---|\n")
        for size_mb, frame, counts, totals in rows[:30]:
            c2k, c4k, c8k = counts
            t8k = totals[-1]
            t2k = totals[0]
            if t2k > 0:
                growth = f"{t8k/t2k:.2f}x"
            elif t8k > 0:
                growth = "new"
            else:
                growth = "—"
            f.write(f"| {size_mb:.3f} | {c2k} | {c4k} | {c8k} | {t8k:.2f} | {growth} | `{frame}` |\n")
        f.write("\n## Identified seq-coupled allocations\n\n")
        if not leaks:
            f.write("None found above 1 MB threshold.\n")
        else:
            f.write("| growth (MB) | size each | counts (2k,4k,8k) | frame |\n")
            f.write("|---:|---:|:---|:---|\n")
            for delta, size_mb, frame, counts, totals in leaks[:10]:
                c2k, c4k, c8k = counts
                f.write(f"| +{delta:.2f} | {size_mb:.3f} | ({c2k},{c4k},{c8k}) | `{frame}` |\n")

    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
