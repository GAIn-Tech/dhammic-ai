from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(model, optimizer, step, loss, path):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "loss": float(loss),
    }
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path_obj)


def load_checkpoint(model, optimizer, path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return {
        "step": int(checkpoint.get("step", 0)),
        "loss": float(checkpoint.get("loss", 0.0)),
    }
