from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW

from ces_train.checkpoint import load_checkpoint, save_checkpoint
from ces_train.config import TrainConfig
from ces_train.dummy_model import DummyModel
from ces_train.loop import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--model", type=str, default="dummy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model != "dummy":
        raise ValueError("Only --model dummy is supported in Task 0.6")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        config = TrainConfig(
            seq_len=128,
            vocab_size=32768,
            batch_size=4,
            grad_accum_steps=1,
            max_steps=args.steps,
            warmup_steps=max(1, min(10, args.steps)),
            save_every=args.steps,
            log_every=max(1, args.steps),
            checkpoint_dir=str(checkpoint_dir),
            dataset_name="synthetic",
            tokenizer_path=None,
            dtype="fp32",
        )

        model = DummyModel(
            vocab_size=config.vocab_size, hidden=64, seq_len=config.seq_len
        )
        train(model, config)
    print(f"Step {args.steps}/{args.steps} complete")

    persistent_dir = Path("/home/mikeb/stream/ces/checkpoints")
    persistent_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = persistent_dir / f"step_{args.steps:07d}.pt"
    save_checkpoint(model, None, args.steps, 0.0, ckpt_path)

    fresh_model = DummyModel(
        vocab_size=config.vocab_size, hidden=64, seq_len=config.seq_len
    )
    metadata = load_checkpoint(fresh_model, None, ckpt_path)
    if metadata["step"] != args.steps:
        raise RuntimeError("Invalid checkpoint metadata")

    sample_input = torch.randint(0, 32768, (2, 128), dtype=torch.long)
    model.eval()
    fresh_model.eval()
    with torch.no_grad():
        before = model.cpu()(sample_input)
        after = fresh_model(sample_input)

    if not torch.allclose(before, after, atol=1e-5, rtol=1e-3):
        raise RuntimeError("Checkpoint roundtrip verification failed")


if __name__ == "__main__":
    main()
