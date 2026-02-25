#!/usr/bin/env python3
import argparse
import math
import sys

import torch
import torch.nn.functional as F


def _get_model(model_name: str):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from ces_train.dummy_model import DummyModel

    if model_name == "dummy":
        return DummyModel(vocab_size=32768, hidden=64, seq_len=128)
    raise ValueError(f"Unknown model: {model_name}")


def _synthetic_batches(vocab_size: int, seq_len: int, n_batches: int, device):
    for _ in range(n_batches):
        yield torch.randint(0, vocab_size, (4, seq_len), device=device)


def measure(model, data_source: str, n_batches: int = 25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    vocab_size = model.embed.num_embeddings
    seq_len = 128

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in _synthetic_batches(vocab_size, seq_len, n_batches, device):
            logits = model(batch)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss)
            total_tokens += shift_labels.numel()

    ppl = math.exp(total_loss / total_tokens)
    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dummy")
    parser.add_argument("--data", default="test")
    args = parser.parse_args()

    model = _get_model(args.model)
    ppl = measure(model, args.data)
    print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
