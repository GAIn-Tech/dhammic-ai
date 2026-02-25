#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def ssd_forward_reference(
    x: torch.Tensor,
    a: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    dt: torch.Tensor,
    d_skip: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    chunk_len: int = 64,
) -> torch.Tensor:
    del chunk_len

    if x.dtype != torch.float32:
        x = x.float()
    if a.dtype != torch.float32:
        a = a.float()
    if b_proj.dtype != torch.float32:
        b_proj = b_proj.float()
    if c_proj.dtype != torch.float32:
        c_proj = c_proj.float()
    if dt.dtype != torch.float32:
        dt = dt.float()
    if d_skip.dtype != torch.float32:
        d_skip = d_skip.float()
    if dt_bias is not None and dt_bias.dtype != torch.float32:
        dt_bias = dt_bias.float()

    batch_size, seq_len, n_heads = x.shape
    d_state = b_proj.shape[-1]

    if dt_bias is not None:
        dt = dt + dt_bias.view(1, 1, n_heads)

    dt_disc = F.softplus(dt)
    dA = torch.exp(dt_disc * a.view(1, 1, n_heads))
    dB = dt_disc.unsqueeze(-1) * b_proj

    state = torch.zeros(
        batch_size, n_heads, d_state, device=x.device, dtype=torch.float32
    )
    outputs = []
    for t in range(seq_len):
        state = dA[:, t, :].unsqueeze(-1) * state + dB[:, t, :, :] * x[
            :, t, :
        ].unsqueeze(-1)
        y_t = (c_proj[:, t, :, :] * state).sum(dim=-1)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)
    y = y + d_skip.view(1, 1, n_heads) * x
    return y


def make_case(
    *,
    seed: int,
    batch_size: int,
    seq_len: int,
    n_heads: int,
    d_state: int,
    chunk_len: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.randn(batch_size, seq_len, n_heads, dtype=torch.float32)
    a = -torch.rand(n_heads, dtype=torch.float32) - 0.05
    b_proj = torch.randn(batch_size, seq_len, n_heads, d_state, dtype=torch.float32)
    c_proj = torch.randn(batch_size, seq_len, n_heads, d_state, dtype=torch.float32)
    dt = torch.randn(batch_size, seq_len, n_heads, dtype=torch.float32)
    d_skip = torch.randn(n_heads, dtype=torch.float32)
    dt_bias = torch.randn(n_heads, dtype=torch.float32) * 0.1

    expected_y = ssd_forward_reference(
        x=x,
        a=a,
        b_proj=b_proj,
        c_proj=c_proj,
        dt=dt,
        d_skip=d_skip,
        dt_bias=dt_bias,
        chunk_len=chunk_len,
    )

    metadata = {
        "B": batch_size,
        "T": seq_len,
        "H": n_heads,
        "N": d_state,
        "seed": seed,
        "chunk_len": chunk_len,
        "dtype": "float32",
    }

    return {
        "x": x.numpy(),
        "A": a.numpy(),
        "B": b_proj.numpy(),
        "C": c_proj.numpy(),
        "dt": dt.numpy(),
        "D": d_skip.numpy(),
        "dt_bias": dt_bias.numpy(),
        "expected_y": expected_y.detach().cpu().numpy(),
        "metadata": np.array(json.dumps(metadata)),
    }


def save_case(path: Path, case: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **case)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mamba-2 SSD pure-PyTorch reference tooling"
    )
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("make-case", help="Generate a single deterministic .npz case")
    gen.add_argument("--output", type=Path, required=True, help="Output .npz path")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--batch-size", type=int, default=2)
    gen.add_argument("--seq-len", type=int, default=128)
    gen.add_argument("--n-heads", type=int, default=4)
    gen.add_argument("--d-state", type=int, default=16)
    gen.add_argument("--chunk-len", type=int, default=64)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "make-case":
        case = make_case(
            seed=args.seed,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            n_heads=args.n_heads,
            d_state=args.d_state,
            chunk_len=args.chunk_len,
        )
        save_case(args.output, case)
        print(f"saved {args.output}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
