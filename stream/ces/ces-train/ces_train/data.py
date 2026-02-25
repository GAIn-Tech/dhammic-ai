from __future__ import annotations

from typing import Iterator, Optional

import sentencepiece as spm
import torch
from datasets import load_dataset

from ces_train.config import TrainConfig


class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        self.processor = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(text, out_type=int)


def synthetic_batch_iterator(
    config: TrainConfig, device: torch.device
) -> Iterator[torch.Tensor]:
    while True:
        yield torch.randint(
            low=0,
            high=config.vocab_size,
            size=(config.batch_size, config.seq_len),
            device=device,
            dtype=torch.long,
        )


def openwebtext_batch_iterator(
    config: TrainConfig,
    device: torch.device,
    tokenizer: SentencePieceTokenizer,
) -> Iterator[torch.Tensor]:
    dataset = load_dataset(config.dataset_name, split="train", streaming=True)
    token_buffer: list[int] = []
    needed = config.batch_size * config.seq_len
    for row in dataset:
        text = row.get("text") if isinstance(row, dict) else None
        if not text:
            continue
        token_buffer.extend(tokenizer.encode(text))
        while len(token_buffer) >= needed:
            chunk = token_buffer[:needed]
            del token_buffer[:needed]
            batch = torch.tensor(chunk, dtype=torch.long, device=device).view(
                config.batch_size,
                config.seq_len,
            )
            yield batch


def get_batch_iterator(
    config: TrainConfig,
    device: torch.device,
    force_synthetic: bool = False,
) -> tuple[Iterator[torch.Tensor], bool]:
    if force_synthetic:
        return synthetic_batch_iterator(config, device), True
    if not config.tokenizer_path:
        return synthetic_batch_iterator(config, device), True
    try:
        tokenizer = SentencePieceTokenizer(config.tokenizer_path)
        return openwebtext_batch_iterator(config, device, tokenizer), False
    except Exception:
        return synthetic_batch_iterator(config, device), True
