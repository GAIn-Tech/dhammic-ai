from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from ces_train.checkpoint import save_checkpoint
from ces_train.config import TrainConfig
from ces_train.data import get_batch_iterator


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(config: TrainConfig) -> torch.dtype:
    if config.dtype == "fp16":
        return torch.float16
    if config.dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def _lr_for_step(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.max_lr * float(step + 1) / float(max(1, config.warmup_steps))
    progress = (step - config.warmup_steps) / float(
        max(1, config.max_steps - config.warmup_steps)
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + cosine * (config.max_lr - config.min_lr)


def train(model: nn.Module, config: TrainConfig) -> None:
    device = _resolve_device()
    autocast_dtype = _resolve_dtype(config)
    autocast_enabled = device.type == "cuda" and config.dtype in {"fp16", "bf16"}

    model = model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    batch_iter, synthetic = get_batch_iterator(config, device)
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=(device.type == "cuda" and config.dtype == "fp16"),
    )
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for step in range(config.max_steps):
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for _ in range(config.grad_accum_steps):
            input_ids = next(batch_iter)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                scaled_loss = loss / config.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            running_loss += float(loss.detach().item())

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        lr = _lr_for_step(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        mean_loss = running_loss / config.grad_accum_steps

        if ((step + 1) % config.log_every) == 0 or step == 0:
            source = "synthetic" if synthetic else "openwebtext"
            print(f"step={step + 1} loss={mean_loss:.4f} lr={lr:.8f} data={source}")

        if ((step + 1) % config.save_every) == 0 or (step + 1) == config.max_steps:
            checkpoint_path = checkpoint_dir / f"step_{step + 1:07d}.pt"
            save_checkpoint(model, optimizer, step + 1, mean_loss, checkpoint_path)
