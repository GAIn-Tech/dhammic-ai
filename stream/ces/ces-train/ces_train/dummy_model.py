import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 32768, hidden: int = 64, seq_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, hidden)
        self.linear = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        logits = self.linear(x)
        return logits
