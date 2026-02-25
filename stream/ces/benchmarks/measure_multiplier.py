#!/usr/bin/env python3
import argparse
import sys


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _get_model(model_name: str):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from ces_train.dummy_model import DummyModel

    if model_name == "dummy":
        return DummyModel(vocab_size=32768, hidden=64, seq_len=128)
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dummy")
    parser.add_argument(
        "--baseline-params",
        type=float,
        default=774e6,
        help="Baseline model param count (default: GPT-2 Large 774M)",
    )
    args = parser.parse_args()

    model = _get_model(args.model)
    actual = _count_params(model)
    multiplier = 1.0
    effective = actual * multiplier

    print(f"Actual params:    {actual:,} ({actual / 1e6:.2f}M)")
    print(
        f"Multiplier:       {multiplier:.2f}x  (placeholder — Sprint 6 computes real value)"
    )
    print(f"Effective params: {effective:,.0f} ({effective / 1e6:.2f}M)")
    print(f"Baseline (GPT-2 Large): {args.baseline_params / 1e6:.0f}M")
    print(f"Equivalent fraction:    {100 * effective / args.baseline_params:.1f}%")


if __name__ == "__main__":
    main()
