#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))
    from extract import ssd_forward_reference
else:
    from ssd_reference.extract import ssd_forward_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify SSD golden vectors")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "golden_vectors",
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    files = sorted(args.input_dir.glob("*.npz"))

    if not files:
        raise SystemExit(f"no vectors found in {args.input_dir}")

    matched = 0
    max_error = 0.0

    for path in files:
        data = np.load(path)
        y = ssd_forward_reference(
            x=torch.from_numpy(data["x"]).float(),
            a=torch.from_numpy(data["A"]).float(),
            b_proj=torch.from_numpy(data["B"]).float(),
            c_proj=torch.from_numpy(data["C"]).float(),
            dt=torch.from_numpy(data["dt"]).float(),
            d_skip=torch.from_numpy(data["D"]).float(),
            dt_bias=torch.from_numpy(data["dt_bias"]).float()
            if "dt_bias" in data.files
            else None,
        )
        expected = torch.from_numpy(data["expected_y"]).float()
        err = torch.max(torch.abs(y - expected)).item()
        max_error = max(max_error, err)
        if err <= args.tolerance:
            matched += 1

    total = len(files)
    if matched != total:
        raise SystemExit(f"{matched}/{total} vectors match, max_error={max_error:.3e}")

    print(f"{matched}/{total} vectors match, max_error={max_error:.3e}")


if __name__ == "__main__":
    main()
