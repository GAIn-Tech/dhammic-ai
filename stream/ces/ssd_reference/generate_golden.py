#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from extract import make_case, save_case
else:
    from ssd_reference.extract import make_case, save_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SSD golden vectors")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "golden_vectors",
    )
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--chunk-len", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for seq_len in args.seq_lens:
        for i in range(args.n_cases):
            seed = 42 + total
            case = make_case(
                seed=seed,
                batch_size=args.batch_size,
                seq_len=seq_len,
                n_heads=args.n_heads,
                d_state=args.d_state,
                chunk_len=args.chunk_len,
            )
            filename = f"ssd_B{args.batch_size}_T{seq_len}_H{args.n_heads}_N{args.d_state}_seed{seed}.npz"
            save_case(args.output_dir / filename, case)
            total += 1

    print(f"generated {total} golden vectors in {args.output_dir}")


if __name__ == "__main__":
    main()
