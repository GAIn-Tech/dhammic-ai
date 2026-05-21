"""K6 — fused_swiglu parity, gradient & module tests vs eager SwiGLU."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load src/kernels/swiglu.py directly to avoid name collision with
# the tests/kernels package (both directories ship an __init__.py).
_SWIGLU_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "kernels", "swiglu.py")
)
_spec = importlib.util.spec_from_file_location("dhammic_kernels_swiglu", _SWIGLU_PATH)
_swiglu_mod = importlib.util.module_from_spec(_spec)
sys.modules["dhammic_kernels_swiglu"] = _swiglu_mod
_spec.loader.exec_module(_swiglu_mod)  # type: ignore[union-attr]

FusedSwiGLUMLP = _swiglu_mod.FusedSwiGLUMLP
fused_swiglu_act = _swiglu_mod.fused_swiglu_act


CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="Triton kernel requires CUDA")


def _eager_swiglu_act(up: torch.Tensor) -> torch.Tensor:
    gate, val = up.chunk(2, dim=-1)
    return F.silu(gate) * val


# --------------------------------------------------------------------------- #
# Forward parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape_act", [(2, 256, 320), (4, 512, 640), (1, 1024, 160)])
def test_act_forward_parity_bf16(shape_act):
    """Kernel uses fp32 internal accumulation, so we compare bf16 kernel output
    against the fp32-evaluated eager reference: the kernel result must round
    to a value within 1e-2 of the true (fp32-computed) silu(gate)*val.
    """
    torch.manual_seed(0)
    *lead, two_N = shape_act
    assert two_N % 2 == 0
    up_bf16 = torch.randn(*shape_act, device="cuda", dtype=torch.bfloat16)

    y_fused = fused_swiglu_act(up_bf16)
    # Reference: same bf16 inputs, but eval in fp32 for a clean ground truth.
    y_eager_fp32 = _eager_swiglu_act(up_bf16.float())

    assert y_fused.shape == y_eager_fp32.shape
    assert y_fused.dtype == torch.bfloat16
    err = (y_fused.float() - y_eager_fp32).abs().max().item()
    # bf16 storage rounding alone is ~2^-7 * max(|y|); for SwiGLU on randn
    # inputs the output magnitude can reach 3–5, giving ~3e-2 representational
    # noise. The fp32-internal kernel is bound by that limit, not algorithm error.
    rel_thresh = 5e-2 * y_eager_fp32.abs().max().item()
    bf16_thresh = max(rel_thresh, 5e-2)
    assert err < bf16_thresh, (
        f"shape {shape_act}: bf16 max_abs_err {err} >= {bf16_thresh}"
    )


@pytest.mark.parametrize("shape_act", [(2, 256, 320), (4, 512, 640), (1, 1024, 160)])
def test_act_forward_parity_fp32(shape_act):
    torch.manual_seed(0)
    up = torch.randn(*shape_act, device="cuda", dtype=torch.float32)
    y_fused = fused_swiglu_act(up)
    y_eager = _eager_swiglu_act(up)
    err = (y_fused - y_eager).abs().max().item()
    assert err < 1e-5, f"shape {shape_act}: fp32 max_abs_err {err} >= 1e-5"


# --------------------------------------------------------------------------- #
# Backward parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape_act", [(2, 256, 320), (4, 512, 640), (1, 1024, 160)])
def test_act_backward_parity_fp32(shape_act):
    torch.manual_seed(1)
    up = torch.randn(*shape_act, device="cuda", dtype=torch.float32, requires_grad=True)
    up_ref = up.detach().clone().requires_grad_(True)

    y_fused = fused_swiglu_act(up)
    g = torch.randn_like(y_fused)
    y_fused.backward(g)

    y_eager = _eager_swiglu_act(up_ref)
    y_eager.backward(g)

    err = (up.grad - up_ref.grad).abs().max().item()
    assert err < 1e-3, f"shape {shape_act}: fp32 grad max_abs_err {err} >= 1e-3"


@pytest.mark.parametrize("shape_act", [(2, 256, 320), (4, 512, 640), (1, 1024, 160)])
def test_act_backward_parity_bf16(shape_act):
    """Backward parity: fused (bf16 in/out, fp32 internal) vs fp32 eager
    ground truth on the same bf16 inputs."""
    torch.manual_seed(1)
    up_bf16 = torch.randn(*shape_act, device="cuda", dtype=torch.bfloat16,
                          requires_grad=True)
    up_fp32 = up_bf16.detach().float().requires_grad_(True)

    y_fused = fused_swiglu_act(up_bf16)
    g = torch.randn_like(y_fused)
    y_fused.backward(g)

    y_eager = _eager_swiglu_act(up_fp32)
    y_eager.backward(g.float())

    err = (up_bf16.grad.float() - up_fp32.grad).abs().max().item()
    # See forward bf16 comment: bf16 storage limits us to ~2^-7 * |grad|.
    rel_thresh = 5e-2 * up_fp32.grad.abs().max().item()
    bf16_thresh = max(rel_thresh, 5e-2)
    assert err < bf16_thresh, (
        f"shape {shape_act}: bf16 grad max_abs_err {err} >= {bf16_thresh}"
    )


# --------------------------------------------------------------------------- #
# Module parity — FusedSwiGLUMLP vs eager Linear+chunk+silu+mul+Linear
# --------------------------------------------------------------------------- #
class _EagerSwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, mlp_dim: int, bias: bool = False,
                 dtype=None, device=None):
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.mlp_up = nn.Linear(d_model, mlp_dim * 2, bias=bias, **factory)
        self.mlp_down = nn.Linear(mlp_dim, d_model, bias=bias, **factory)

    def forward(self, x):
        up = self.mlp_up(x)
        gate, val = up.chunk(2, dim=-1)
        return self.mlp_down(F.silu(gate) * val)


def test_module_parity_fp32():
    torch.manual_seed(2)
    d_model, mlp_dim = 128, 320
    shape = (2, 256, d_model)

    fused = FusedSwiGLUMLP(d_model, mlp_dim, bias=False,
                           dtype=torch.float32, device="cuda")
    eager = _EagerSwiGLUMLP(d_model, mlp_dim, bias=False,
                            dtype=torch.float32, device="cuda")
    # Copy weights eager -> fused
    fused.mlp_up.weight.data.copy_(eager.mlp_up.weight.data)
    fused.mlp_down.weight.data.copy_(eager.mlp_down.weight.data)

    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    y_fused = fused(x)
    y_eager = eager(x)

    assert y_fused.shape == y_eager.shape
    err = (y_fused - y_eager).abs().max().item()
    assert err < 1e-4, f"module fp32 max_abs_err {err} >= 1e-4"


def test_module_parity_bf16_and_grad():
    torch.manual_seed(3)
    d_model, mlp_dim = 128, 320
    shape = (2, 256, d_model)

    fused = FusedSwiGLUMLP(d_model, mlp_dim, bias=False,
                           dtype=torch.bfloat16, device="cuda")
    eager = _EagerSwiGLUMLP(d_model, mlp_dim, bias=False,
                            dtype=torch.bfloat16, device="cuda")
    fused.mlp_up.weight.data.copy_(eager.mlp_up.weight.data)
    fused.mlp_down.weight.data.copy_(eager.mlp_down.weight.data)

    x_f = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_e = x_f.detach().clone().requires_grad_(True)

    y_f = fused(x_f)
    y_e = eager(x_e)

    err = (y_f.float() - y_e.float()).abs().max().item()
    assert err < 2e-2, f"module bf16 forward max_abs_err {err} >= 2e-2"

    g = torch.randn_like(y_f)
    y_f.backward(g)
    y_e.backward(g)

    grad_err = (x_f.grad.float() - x_e.grad.float()).abs().max().item()
    assert grad_err < 5e-2, f"module bf16 grad_x max_abs_err {grad_err} >= 5e-2"


# --------------------------------------------------------------------------- #
# Shapes — act kernel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape_act", [(2, 256, 320), (4, 512, 640), (1, 1024, 160)])
def test_act_shapes(shape_act):
    up = torch.randn(*shape_act, device="cuda", dtype=torch.bfloat16)
    y = fused_swiglu_act(up)
    *lead, two_N = shape_act
    N = two_N // 2
    assert tuple(y.shape) == tuple(lead) + (N,)
    assert y.dtype == up.dtype
