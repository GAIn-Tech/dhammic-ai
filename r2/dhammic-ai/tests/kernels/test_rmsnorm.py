"""K1 — fused_rmsnorm parity & gradient tests against eager F.rms_norm."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load src/kernels/rmsnorm.py directly to avoid name collision with
# the tests/kernels package (both directories ship an __init__.py).
_RMSNORM_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "kernels", "rmsnorm.py")
)
_spec = importlib.util.spec_from_file_location("dhammic_kernels_rmsnorm", _RMSNORM_PATH)
_rmsnorm_mod = importlib.util.module_from_spec(_spec)
sys.modules["dhammic_kernels_rmsnorm"] = _rmsnorm_mod
_spec.loader.exec_module(_rmsnorm_mod)  # type: ignore[union-attr]

FusedRMSNorm = _rmsnorm_mod.FusedRMSNorm
FusedRMSNormModule = _rmsnorm_mod.FusedRMSNormModule
fused_rmsnorm = _rmsnorm_mod.fused_rmsnorm


CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="Triton kernel requires CUDA")

EPS = 1e-6


def _eager_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    # Reference matches the eager path in mamba_base_engine.RMSNorm:
    # x * rsqrt(mean(x**2, -1, keepdim) + eps) * weight, with weight broadcast.
    return F.rms_norm(x, (x.shape[-1],), weight.to(x.dtype), eps)


# --------------------------------------------------------------------------- #
# Forward parity — fp32
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(2, 256, 128), (4, 512, 256), (1, 1024, 96)])
def test_forward_parity_fp32(shape):
    torch.manual_seed(0)
    D = shape[-1]
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    w = torch.randn(D, device="cuda", dtype=torch.float32) * 0.5 + 1.0

    y_fused = fused_rmsnorm(x, w, EPS)
    y_eager = _eager_rmsnorm(x, w, EPS)

    err = (y_fused - y_eager).abs().max().item()
    assert err < 1e-5, f"shape {shape}: max_abs_err {err} >= 1e-5"


# --------------------------------------------------------------------------- #
# Forward parity — bf16
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(2, 256, 128), (4, 512, 256), (1, 1024, 96)])
def test_forward_parity_bf16(shape):
    torch.manual_seed(0)
    D = shape[-1]
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(D, device="cuda", dtype=torch.float32) * 0.5 + 1.0).to(torch.bfloat16)

    y_fused = fused_rmsnorm(x, w, EPS)
    y_eager = _eager_rmsnorm(x, w, EPS)

    assert y_fused.dtype == torch.bfloat16, f"expected bf16 output, got {y_fused.dtype}"
    err = (y_fused.float() - y_eager.float()).abs().max().item()
    assert err < 1e-2, f"shape {shape}: bf16 max_abs_err {err} >= 1e-2"


# --------------------------------------------------------------------------- #
# Backward parity — fp32
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(2, 256, 128), (4, 512, 256), (1, 1024, 96)])
def test_backward_parity(shape):
    torch.manual_seed(1)
    D = shape[-1]

    x = torch.randn(*shape, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(D, device="cuda", dtype=torch.float32, requires_grad=True) * 0.5 + 1.0
    w = w.detach().clone().requires_grad_(True)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    # Fused
    y = fused_rmsnorm(x, w, EPS)
    g = torch.randn_like(y)
    y.backward(g)

    # Eager reference
    y_ref = _eager_rmsnorm(x_ref, w_ref, EPS)
    y_ref.backward(g)

    dx_err = (x.grad - x_ref.grad).abs().max().item()
    dw_err = (w.grad - w_ref.grad).abs().max().item()

    assert dx_err < 1e-3, f"shape {shape}: dx max_abs_err {dx_err} >= 1e-3"
    assert dw_err < 1e-3, f"shape {shape}: dw max_abs_err {dw_err} >= 1e-3"


# --------------------------------------------------------------------------- #
# Shapes — incl. 4-D (B, T, H, D)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    [
        (2, 256, 128),
        (4, 512, 256),
        (1, 1024, 96),
        (2, 64, 4, 32),    # (B, T, H, D)
        (1, 128, 8, 16),
    ],
)
def test_shapes(shape):
    torch.manual_seed(2)
    D = shape[-1]
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    w = torch.randn(D, device="cuda", dtype=torch.float32)

    y = fused_rmsnorm(x, w, EPS)
    assert y.shape == x.shape, f"output shape {tuple(y.shape)} != input {tuple(x.shape)}"

    y_ref = _eager_rmsnorm(x, w, EPS)
    err = (y - y_ref).abs().max().item()
    assert err < 1e-5, f"shape {shape}: max_abs_err {err} >= 1e-5"


# --------------------------------------------------------------------------- #
# Module swap — instantiate FusedRMSNormModule, copy weights from nn.RMSNorm
# --------------------------------------------------------------------------- #
def test_module_swap():
    torch.manual_seed(3)
    B, T, D = 2, 128, 96

    eager = nn.RMSNorm(D, eps=EPS).cuda().to(torch.float32)
    fused = FusedRMSNormModule(D, eps=EPS).cuda().to(torch.float32)

    # Copy weights from nn.RMSNorm to fused module
    with torch.no_grad():
        fused.weight.copy_(eager.weight)

    x = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
    y_eager = eager(x)
    y_fused = fused(x)

    err = (y_fused - y_eager).abs().max().item()
    assert err < 1e-5, f"module swap max_abs_err {err} >= 1e-5"

    # Same for bf16
    eager_bf = nn.RMSNorm(D, eps=EPS).cuda().to(torch.bfloat16)
    fused_bf = FusedRMSNormModule(D, eps=EPS).cuda().to(torch.bfloat16)
    with torch.no_grad():
        fused_bf.weight.copy_(eager_bf.weight)

    xb = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
    yb_e = eager_bf(xb)
    yb_f = fused_bf(xb)

    err_bf = (yb_e.float() - yb_f.float()).abs().max().item()
    assert err_bf < 1e-2, f"module swap bf16 max_abs_err {err_bf} >= 1e-2"


# --------------------------------------------------------------------------- #
# Autotune selection probe — exposes the picked config for the report.
# --------------------------------------------------------------------------- #
def test_block_config_visible(capsys):
    """Smoke test confirming the picked BLOCK_SIZE / num_warps for the report."""
    torch.manual_seed(0)
    for D in [96, 128, 256, 512]:
        x = torch.randn(2, 64, D, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(D, device="cuda", dtype=torch.bfloat16)
        _ = fused_rmsnorm(x, w, EPS)
        bs = _rmsnorm_mod._next_pow2(D)
        nw = _rmsnorm_mod._pick_num_warps(bs)
        print(f"D={D}: BLOCK_SIZE={bs}, num_warps={nw}")
        assert bs >= D
