from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    hidden_dim: int = 512
    seq_len: int = 1024
    vocab_size: int = 32768

    batch_size: int = 8
    grad_accum_steps: int = 8
    max_steps: int = 100_000
    warmup_steps: int = 2000
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    dataset_name: str = "Skylion007/openwebtext"
    tokenizer_path: Optional[str] = None

    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000

    use_wandb: bool = False
    log_every: int = 10

    dtype: str = "bf16"
