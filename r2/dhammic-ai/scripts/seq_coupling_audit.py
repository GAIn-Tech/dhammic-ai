"""Diff-based offload audit. Run fwd+bwd at three seq_lens, identify every
allocation that grows linearly with T -- those are the residual seq-coupled
tensors keeping us from strict flat VRAM."""

from __future__ import annotations

import gc
import os
import sys
from collections import defaultdict

import torch

# Make repo modules importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from citta_vithi_pipeline import CittaVithiPipeline
from runtime.chunked_offload import ChunkedOffloadRunner
from runtime.chunked_lmhead import ChunkedLMHeadXent


def build_model(d_model=64, n_layers=2, vocab=1024, device="cuda"):
    m = CittaVithiPipeline(
        d_model=d_model, n_layers=n_layers, d_state=8, mamba_expand=2,
        n_heads=4, chunk_size=256, vocab_size=vocab,
        sdr_dim=256, sdr_k_active=10,
        engram_n_columns=64, engram_cells_per_col=4, engram_k_active=4,
    ).to(device).to(torch.bfloat16)
    return m


def step_at(seq_len, batch=2, chunk_size=256, d_model=64, vocab=1024):
    """One full fwd+bwd at the given seq, return peak-allocation snapshot."""
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.memory._record_memory_history(max_entries=100000)

    model = build_model(d_model=d_model, vocab=vocab)
    runner = ChunkedOffloadRunner()
    head = ChunkedLMHeadXent(
        d_model=d_model, vocab_size=vocab,
        weight=model.embedding.dense_embed.weight,
        chunk_size=chunk_size,
    ).to("cuda").to(torch.bfloat16)

    x = torch.randint(0, vocab, (batch, seq_len), device="cuda")
    y = torch.randint(0, vocab, (batch, seq_len), device="cuda")

    # mimic the chunked sweep wiring: embedding eager, backbone via runner, head chunked
    emb = model.embedding(x)

    # build a "BackboneAndNorm" the same way the v2 sweep does
    class BB(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.layers = m.backbone
            self.norm = m.final_norm
        def forward(self, h):
            for layer in self.layers:
                h = layer(h)
            return self.norm(h)

    bb = BB(model).to("cuda").to(torch.bfloat16)

    h_full = runner.forward(bb.forward, emb, chunk_size)
    loss = head(h_full, y)
    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)

    # Bucket live allocations by (size_class, top-frame). Use rounded MB so
    # near-equal sizes group together.
    buckets = defaultdict(int)
    for seg in snapshot.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            size = blk.get("size", 0)
            if size < 64 * 1024:  # skip <64KB
                continue
            mb = size / 1e6
            frames = blk.get("frames") or [{"name": "<no frame>"}]
            top = frames[0].get("name", "<?>") if isinstance(frames[0], dict) else str(frames[0])
            # truncate long frame names
            top = top[:80]
            buckets[(round(mb, 2), top)] += 1

    del model, runner, head, x, y, emb, h_full, loss, bb
    gc.collect(); torch.cuda.empty_cache()

    return peak_mb, buckets


def main():
    seq_lens = [2048, 4096, 8192]
    snapshots = {}
    for T in seq_lens:
        peak_mb, buckets = step_at(T)
        snapshots[T] = (peak_mb, buckets)
        print(f"seq_len={T:>5d}  peak={peak_mb:7.2f} MB  unique_bucket_keys={len(buckets)}")

    # Build a wide table: for each (size, frame) bucket, what was its count at each T?
    all_keys = set()
    for _, b in snapshots.values():
        all_keys.update(b.keys())

    rows = []
    for k in all_keys:
        size_mb, frame = k
        counts = [snapshots[T][1].get(k, 0) for T in seq_lens]
        # total bytes contributed at each T
        totals_mb = [size_mb * c for c in counts]
        rows.append((size_mb, frame, counts, totals_mb))

    # Sort by total-MB-at-largest-T descending so the worst T-scalers float to the top
    rows.sort(key=lambda r: -r[3][-1])

    print("\n# Per-allocation diff across seq_lens\n")
    print(f"{'size_MB':>8} | {'count@2k':>9} | {'count@4k':>9} | {'count@8k':>9} | "
          f"{'total@8k_MB':>12} | {'growth(8k/2k)':>14} | frame")
    print("-" * 130)
    for size_mb, frame, counts, totals in rows[:40]:
        c2k, c4k, c8k = counts
        t8k = totals[-1]
        t2k = totals[0]
        growth = (t8k / t2k) if t2k > 0 else float("inf") if t8k > 0 else 0
        print(f"{size_mb:8.3f} | {c2k:9d} | {c4k:9d} | {c8k:9d} | "
              f"{t8k:12.2f} | {growth:14.2f} | {frame}")

    # Summary: top T-scalers
    print("\n# Summary\n")
    for T in seq_lens:
        peak, _ = snapshots[T]
        print(f"  peak@seq={T:>5d}: {peak:7.2f} MB")
    p2, p4, p8 = (snapshots[T][0] for T in seq_lens)
    print(f"\n  growth ratio peak@8k / peak@2k = {p8/p2:.2f}x   "
          f"(target for strict-flat: ≤ ~1.2x; current chunked-v2 ratio in the sweep: 3.43x at @8k/@256, but at the @8k/@2k slice that's {p8/p2:.2f}x)")

    # Write the rows to a markdown file
    md_path = os.path.join(ROOT, "docs", "audit", "seq_coupling_diff.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write("# Seq-coupling diff audit\n\n")
        f.write("Each allocation observed at peak fwd+bwd, grouped by (size, top frame).\n")
        f.write("If `total@8k / total@2k ~ 4x`, that allocation is seq-coupled.\n")
        f.write("If the ratio is ~1x, that allocation is constant-size in T.\n\n")
        f.write(f"| size_MB | count@2k | count@4k | count@8k | total@8k_MB | growth(8k/2k) | frame |\n")
        f.write(f"|---:|---:|---:|---:|---:|---:|:---|\n")
        for size_mb, frame, counts, totals in rows[:40]:
            c2k, c4k, c8k = counts
            t8k = totals[-1]
            t2k = totals[0]
            growth = (t8k / t2k) if t2k > 0 else float("inf") if t8k > 0 else 0
            f.write(f"| {size_mb:.3f} | {c2k} | {c4k} | {c8k} | {t8k:.2f} | {growth:.2f} | `{frame}` |\n")
        f.write(f"\n\n## Peak\n\n")
        for T in seq_lens:
            f.write(f"- seq={T}: peak={snapshots[T][0]:.2f} MB\n")
        f.write(f"\nPeak growth 8k/2k = {p8/p2:.2f}x.\n")

    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
