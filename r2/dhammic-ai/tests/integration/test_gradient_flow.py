"""Per-parameter gradient-flow audit for the fused dhammic-ai pipeline.

Spins up a tiny CittaVithiPipeline + FusedLMHeadXentModule, runs one
forward+backward step under bf16 autocast, and asserts that EVERY
parameter has a finite, non-zero gradient. A zero-norm grad means the
parameter is detached from the graph (e.g. wrong dtype, wrong cast,
broken autograd Function) — a structural integration bug.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from citta_vithi_pipeline import CittaVithiPipeline  # noqa: E402
from kernels import FusedLMHeadXentModule  # noqa: E402


def _build_tiny_model() -> tuple[CittaVithiPipeline, FusedLMHeadXentModule]:
    model = CittaVithiPipeline(
        d_model=64, n_layers=2, d_state=8, mamba_expand=2,
        n_heads=4, chunk_size=256, vocab_size=512,
        sdr_dim=256, sdr_k_active=10,
        engram_n_columns=64, engram_cells_per_col=4, engram_k_active=4,
    ).cuda()
    head = FusedLMHeadXentModule(
        d_model=64, vocab_size=512,
        weight=model.embedding.dense_embed.weight,
    ).cuda()
    return model, head


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_every_param_receives_finite_nonzero_grad():
    torch.manual_seed(0)
    model, head = _build_tiny_model()
    model.train()

    # NOTE: HebbianLoRA initialises ``W_B = zeros`` (standard LoRA init).
    # That makes the LoRA branch a no-op on the first forward, so
    # ``W_A``, ``gate.weight`` and ``gate.bias`` would all have zero
    # grads on step 1. We take ONE warm-up step + a tiny optimiser
    # update so ``W_B`` becomes non-zero, then check grads on step 2 —
    # this is the only point at which a true detached-from-graph bug
    # is distinguishable from the legitimate zero-init quirk.
    opt = torch.optim.SGD(
        list(model.parameters()) + [
            p for p in head.parameters()
            if not any(p is m for m in model.parameters())
        ],
        lr=1e-2,
    )

    for _ in range(2):
        x = torch.randint(0, 512, (2, 256), device="cuda")
        y = torch.randint(0, 512, (2, 256), device="cuda")
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(x)
            loss = head(hidden, y)
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
        loss.backward()
        # final iteration: keep grads, don't step.
        if _ == 0:
            opt.step()

    # Walk parameters. We dedupe by `id(p)` because the LM head shares
    # its weight with the input embedding (tied weight) — the test
    # otherwise reports the same Parameter twice.
    seen: set[int] = set()
    rows: list[tuple[str, float, tuple[int, ...]]] = []
    zero_norm: list[str] = []
    none_grad: list[str] = []
    non_finite: list[str] = []

    for name, p in list(model.named_parameters()) + list(
        ("head." + n, q) for n, q in head.named_parameters()
    ):
        if id(p) in seen:
            continue
        seen.add(id(p))
        if p.grad is None:
            none_grad.append(name)
            continue
        if not torch.isfinite(p.grad).all():
            non_finite.append(name)
            continue
        gn = p.grad.detach().abs().sum().item()
        if gn == 0.0:
            zero_norm.append(name)
        rows.append((name, gn, tuple(p.shape)))

    # Pretty table
    print("\n=== gradient flow audit ===")
    print(f"{'param':<60s} {'grad_norm':>14s}  shape")
    for name, gn, shape in rows:
        print(f"{name:<60s} {gn:>14.6e}  {shape}")
    print(f"total: {len(rows)} params with grad ({len(zero_norm)} zero, "
          f"{len(none_grad)} none, {len(non_finite)} non-finite)")

    # Hard assertions
    assert not none_grad, f"params with grad=None: {none_grad}"
    assert not non_finite, f"params with non-finite grads: {non_finite}"
    assert not zero_norm, (
        f"params with zero-norm grad (detached from autograd graph): "
        f"{zero_norm}"
    )


if __name__ == "__main__":
    test_every_param_receives_finite_nonzero_grad()
    print("PASSED")
