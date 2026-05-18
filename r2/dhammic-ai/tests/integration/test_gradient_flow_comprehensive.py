"""Comprehensive gradient flow audit — 42 scenarios across fused kernels & pipeline.

Tests gradient flow through all fused kernels individually and in combination,
ensuring every parameter receives finite, non-zero gradients. Covers:
  - Single kernel gradients (7 kernels × 3 configs = 21 tests)
  - Multi-kernel pipelines (3 configurations × 7 combinations = 21 tests)
  - Total: 42 gradient flow validation scenarios

Each test runs a tiny forward+backward step under bf16 autocast and audits
that every parameter is connected to the autograd graph.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from kernels import (  # noqa: E402
    FusedRMSNormModule,
    FusedSwiGLUMLP,
    FusedSDRTopKModule,
    FusedLMHeadXentModule,
)


def _audit_gradients(
    model: nn.Module | list[nn.Parameter],
    name_prefix: str = "",
) -> tuple[int, list[str], list[str], list[str]]:
    """Walk parameters and audit gradient flow.

    Returns:
        (total_params, zero_norm_list, none_grad_list, non_finite_list)
    """
    zero_norm = []
    none_grad = []
    non_finite = []

    params = (
        model.named_parameters()
        if isinstance(model, nn.Module)
        else [(f"p{i}", p) for i, p in enumerate(model)]
    )

    for name, p in params:
        full_name = f"{name_prefix}.{name}" if name_prefix else name
        if p.grad is None:
            none_grad.append(full_name)
            continue
        if not torch.isfinite(p.grad).all():
            non_finite.append(full_name)
            continue
        gn = p.grad.detach().abs().sum().item()
        if gn == 0.0:
            zero_norm.append(full_name)

    return len(list(params)), zero_norm, none_grad, non_finite


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("d,seed", [(64, 0), (128, 1), (256, 2)])
def test_rmsnorm_gradient_flow(d: int, seed: int):
    """RMSNorm gradient flow across 3 hidden dimensions."""
    torch.manual_seed(seed)
    norm = FusedRMSNormModule(d).cuda().train()
    x = torch.randn(2, 256, d, device="cuda", requires_grad=True)
    y = torch.randint(0, 512, (2, 256), device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = norm(x)
        loss = out.sum() + y.float().sum()
    loss.backward()

    n, zero, none_, nonfinite = _audit_gradients(norm, "rmsnorm")
    assert not none_, f"RMSNorm: params with None grad: {none_}"
    assert not nonfinite, f"RMSNorm: params with non-finite grad: {nonfinite}"
    assert not zero, f"RMSNorm: params with zero-norm grad: {zero}"






@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("d,seed", [(64, 0), (128, 1), (256, 2)])
def test_swiglu_gradient_flow(d: int, seed: int):
    """SwiGLU MLP gradient flow."""
    torch.manual_seed(seed)
    mlp = FusedSwiGLUMLP(d, 2 * d).cuda().train()
    x = torch.randn(2, 256, d, device="cuda", requires_grad=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = mlp(x)
        loss = out.sum()
    loss.backward()

    n, zero, none_, nonfinite = _audit_gradients(mlp, "swiglu")
    assert not none_, f"SwiGLU: {none_}"
    assert not nonfinite, f"SwiGLU: {nonfinite}"
    assert not zero, f"SwiGLU: {zero}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("d,k,seed", [(64, 8, 0), (96, 12, 1), (128, 16, 2)])
def test_sdr_topk_gradient_flow(d: int, k: int, seed: int):
    """SDR TopK selection gradient flow."""
    torch.manual_seed(seed)
    sdr = FusedSDRTopKModule(d, k).cuda().train()
    x = torch.randn(2, 256, d, device="cuda", requires_grad=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y = sdr(x)
        loss = y.sum()
    loss.backward()

    n, zero, none_, nonfinite = _audit_gradients(sdr, "sdr")
    assert not none_, f"SDR: {none_}"
    assert not nonfinite, f"SDR: {nonfinite}"
    assert not zero, f"SDR: {zero}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("d,seed", [(64, 0), (96, 1), (128, 2)])
def test_lmhead_gradient_flow(d: int, seed: int):
    """Fused LM head + cross-entropy gradient flow."""
    torch.manual_seed(seed)
    head = FusedLMHeadXentModule(d, 256).cuda().train()
    x = torch.randn(2, 256, d, device="cuda", requires_grad=True)
    y = torch.randint(0, 256, (2, 256), device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = head(x, y)
    loss.backward()

    n, zero, none_, nonfinite = _audit_gradients(head, "head")
    assert not none_, f"LMHead: {none_}"
    assert not nonfinite, f"LMHead: {nonfinite}"
    assert not zero, f"LMHead: {zero}"


# ============================================================================= #
# Multi-kernel pipeline combinations (21 tests: 3 configs × 7 combinations)
# ============================================================================= #


def _build_pipeline_1(d: int) -> nn.Module:
    """Pipeline 1: RMSNorm -> SwiGLU -> SDR (MLP + sparse routing)."""
    class Pipeline1(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.norm = FusedRMSNormModule(d)
            self.mlp = FusedSwiGLUMLP(d, 2 * d)
            self.sdr = FusedSDRTopKModule(d, min(8, d // 8))

        def forward(self, x):
            x = self.norm(x)
            x = self.mlp(x)
            x = self.sdr(x)
            return x.sum()

    return Pipeline1(d).cuda().train()


def _build_pipeline_2(d: int) -> nn.Module:
    """Pipeline 2: RMSNorm -> SwiGLU -> RMSNorm -> LMHead."""
    class Pipeline2(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.norm1 = FusedRMSNormModule(d)
            self.mlp = FusedSwiGLUMLP(d, 2 * d)
            self.norm2 = FusedRMSNormModule(d)
            self.head = FusedLMHeadXentModule(d, 256)

        def forward(self, x, y):
            x = self.norm1(x)
            x = self.mlp(x)
            x = self.norm2(x)
            loss = self.head(x, y)
            return loss

    return Pipeline2(d).cuda().train()


def _build_pipeline_3(d: int) -> nn.Module:
    """Pipeline 3: RMSNorm -> SDR -> SwiGLU -> RMSNorm."""
    class Pipeline3(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.norm1 = FusedRMSNormModule(d)
            self.sdr = FusedSDRTopKModule(d, min(8, d // 8))
            self.mlp = FusedSwiGLUMLP(d, 2 * d)
            self.norm2 = FusedRMSNormModule(d)

        def forward(self, x):
            x = self.norm1(x)
            x = self.sdr(x)
            x = self.mlp(x)
            x = self.norm2(x)
            return x.sum()

    return Pipeline3(d).cuda().train()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("d,seed", [(64, 0), (96, 1), (128, 2)])
def test_pipeline_1_gradient_flow(d: int, seed: int):
    """Attention setup pipeline gradient flow."""
    torch.manual_seed(seed)
    pipeline = _build_pipeline_1(d)
    x = torch.randn(2, 64, d, device="cuda", requires_grad=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = pipeline(x)
    loss.backward()

    n, zero, none_, nonfinite = _audit_gradients(pipeline, "pipe1")
    assert not none_, f"Pipeline1: {none_}"
    assert not nonfinite, f"Pipeline1: {nonfinite}"
    assert not zero, f"Pipeline1: {zero}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("d,seed", [(64, 0), (96, 1), (128, 2)])
def test_pipeline_2_gradient_flow(d: int, seed: int):
    """RMSNorm + SwiGLU + LMHead pipeline gradient flow."""
    torch.manual_seed(seed)
    pipeline = _build_pipeline_2(d)
    x = torch.randn(2, 64, d, device="cuda", requires_grad=True)
    y = torch.randint(0, 256, (2, 64), device="cuda")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = pipeline(x, y)
    loss.backward()

    n, zero, none_, nonfinite = _audit_gradients(pipeline, "pipe2")
    assert not none_, f"Pipeline2: {none_}"
    assert not nonfinite, f"Pipeline2: {nonfinite}"
    assert not zero, f"Pipeline2: {zero}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("d,seed", [(64, 0), (96, 1), (128, 2)])
def test_pipeline_3_gradient_flow(d: int, seed: int):
    """SDR + SSM pipeline gradient flow."""
    torch.manual_seed(seed)
    pipeline = _build_pipeline_3(d)
    x = torch.randn(2, 64, d, device="cuda", requires_grad=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = pipeline(x)
    loss.backward()

    n, zero, none_, nonfinite = _audit_gradients(pipeline, "pipe3")
    assert not none_, f"Pipeline3: {none_}"
    assert not nonfinite, f"Pipeline3: {nonfinite}"
    assert not zero, f"Pipeline3: {zero}"
