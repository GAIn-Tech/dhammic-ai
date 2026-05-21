"""Parity tests for FusedRoPE Triton kernel vs eager reference.

Covers both broadcast angles (B,T,1,D/2) — legacy API — and per-head
angles (B,T,H,D/2) — production caller in Mamba3Block.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch
from torch import Tensor


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_rope_mod = _load_module("_rope_kernel_under_test", os.path.join(_SRC, "kernels", "rope.py"))
fused_rope = _rope_mod.fused_rope


def apply_rope(x: Tensor, angles: Tensor) -> Tensor:
    """Eager reference RoPE used as the parity oracle.

    Accepts angles broadcast to ``(B,T,1,D/2)`` or full per-head
    ``(B,T,H,D/2)``. Pure PyTorch — relies on broadcasting.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    x_rot_even = cos_a * x1 - sin_a * x2
    x_rot_odd = sin_a * x1 + cos_a * x2
    return torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused_rope requires CUDA"
)


# Realistic shapes from the plan: (B, T, H, d_state)
SHAPES = [
    (2, 256, 8, 16),
    (4, 512, 8, 32),
    (1, 1024, 4, 8),
]


def _make_inputs(B, T, H, D, dtype, device="cuda", seed=0, ang_scale=2.0,
                 per_head: bool = False):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(B, T, H, D, generator=g, device=device, dtype=dtype)
    aH = H if per_head else 1
    angles = torch.randn(
        B, T, aH, D // 2, generator=g, device=device, dtype=dtype
    ) * ang_scale
    return x, angles


# --------------------------------------------------------------------------- #
# Forward parity — broadcast angles (legacy API)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"BTHD_{s}")
def test_forward_parity_fp32(shape):
    B, T, H, D = shape
    x, angles = _make_inputs(B, T, H, D, torch.float32)

    y_ref = apply_rope(x, angles)
    y = fused_rope(x, angles)

    max_abs = (y - y_ref).abs().max().item()
    assert max_abs < 1e-5, f"fp32 max_abs_err={max_abs} for shape {shape}"


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"BTHD_{s}")
def test_forward_parity_bf16(shape):
    B, T, H, D = shape
    g = torch.Generator(device="cuda").manual_seed(0)
    x_fp = torch.randn(B, T, H, D, generator=g, device="cuda", dtype=torch.float32) * 0.5
    angles_fp = torch.randn(B, T, 1, D // 2, generator=g, device="cuda", dtype=torch.float32) * 1.0
    x_bf = x_fp.bfloat16()
    angles_bf = angles_fp.bfloat16()
    x_fp = x_bf.float()
    angles_fp = angles_bf.float()

    y_oracle = apply_rope(x_fp, angles_fp)
    y_eager_bf = apply_rope(x_bf, angles_bf)
    y_fused_bf = fused_rope(x_bf, angles_bf)

    err_fused = (y_fused_bf.float() - y_oracle).abs().max().item()
    err_eager = (y_eager_bf.float() - y_oracle).abs().max().item()

    assert err_fused < 1e-2, (
        f"bf16-vs-fp32 max_abs_err={err_fused} for shape {shape} "
        f"(eager bf16 err = {err_eager})"
    )
    assert err_fused <= err_eager + 1e-3, (
        f"fused bf16 ({err_fused}) worse than eager bf16 ({err_eager}) "
        f"for shape {shape}"
    )


# --------------------------------------------------------------------------- #
# Backward parity — broadcast angles
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"BTHD_{s}")
def test_backward_parity_dx(shape):
    B, T, H, D = shape
    x, angles = _make_inputs(B, T, H, D, torch.float32, seed=42)

    x_ref = x.detach().clone().requires_grad_(True)
    a_ref = angles.detach().clone().requires_grad_(True)
    x_k = x.detach().clone().requires_grad_(True)
    a_k = angles.detach().clone().requires_grad_(True)

    g = torch.randn_like(x)

    y_ref = apply_rope(x_ref, a_ref)
    y_ref.backward(g)

    y_k = fused_rope(x_k, a_k)
    y_k.backward(g)

    max_abs = (x_k.grad - x_ref.grad).abs().max().item()
    assert max_abs < 1e-3, f"dx max_abs_err={max_abs} for shape {shape}"


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"BTHD_{s}")
def test_backward_parity_dangles(shape):
    B, T, H, D = shape
    x, angles = _make_inputs(B, T, H, D, torch.float32, seed=7)

    x_ref = x.detach().clone().requires_grad_(True)
    a_ref = angles.detach().clone().requires_grad_(True)
    x_k = x.detach().clone().requires_grad_(True)
    a_k = angles.detach().clone().requires_grad_(True)

    g = torch.randn_like(x)

    y_ref = apply_rope(x_ref, a_ref)
    y_ref.backward(g)

    y_k = fused_rope(x_k, a_k)
    y_k.backward(g)

    max_abs = (a_k.grad - a_ref.grad).abs().max().item()
    assert max_abs < 1e-3, f"dangles max_abs_err={max_abs} for shape {shape}"


# --------------------------------------------------------------------------- #
# Shape coverage (forward only — confirms strides and edge sizes)
# --------------------------------------------------------------------------- #

def test_shapes_all():
    for shape in SHAPES:
        B, T, H, D = shape
        x, angles = _make_inputs(B, T, H, D, torch.float32)
        y = fused_rope(x, angles)
        y_ref = apply_rope(x, angles)
        assert y.shape == x.shape, f"shape mismatch for {shape}"
        assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5), \
            f"forward mismatch for {shape}"


# --------------------------------------------------------------------------- #
# Per-head angles (production API): angles is (B, T, H, D/2)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"BTHD_{s}")
def test_forward_parity_fp32_per_head(shape):
    B, T, H, D = shape
    x, angles = _make_inputs(B, T, H, D, torch.float32, per_head=True)
    assert angles.shape == (B, T, H, D // 2)

    y_ref = apply_rope(x, angles)
    y = fused_rope(x, angles)

    max_abs = (y - y_ref).abs().max().item()
    assert max_abs < 1e-5, f"fp32 per-head max_abs_err={max_abs} for shape {shape}"


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"BTHD_{s}")
def test_forward_parity_bf16_per_head(shape):
    B, T, H, D = shape
    g = torch.Generator(device="cuda").manual_seed(0)
    x_fp = torch.randn(B, T, H, D, generator=g, device="cuda", dtype=torch.float32) * 0.5
    angles_fp = torch.randn(B, T, H, D // 2, generator=g, device="cuda", dtype=torch.float32) * 1.0
    x_bf = x_fp.bfloat16()
    angles_bf = angles_fp.bfloat16()
    x_fp = x_bf.float()
    angles_fp = angles_bf.float()

    y_oracle = apply_rope(x_fp, angles_fp)
    y_eager_bf = apply_rope(x_bf, angles_bf)
    y_fused_bf = fused_rope(x_bf, angles_bf)

    err_fused = (y_fused_bf.float() - y_oracle).abs().max().item()
    err_eager = (y_eager_bf.float() - y_oracle).abs().max().item()

    assert err_fused < 1e-2, (
        f"bf16 per-head err={err_fused} for shape {shape} (eager={err_eager})"
    )
    assert err_fused <= err_eager + 1e-3, (
        f"fused bf16 per-head ({err_fused}) worse than eager bf16 "
        f"({err_eager}) for shape {shape}"
    )


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"BTHD_{s}")
def test_backward_parity_dx_per_head(shape):
    B, T, H, D = shape
    x, angles = _make_inputs(B, T, H, D, torch.float32, seed=42, per_head=True)

    x_ref = x.detach().clone().requires_grad_(True)
    a_ref = angles.detach().clone().requires_grad_(True)
    x_k = x.detach().clone().requires_grad_(True)
    a_k = angles.detach().clone().requires_grad_(True)

    g = torch.randn_like(x)

    y_ref = apply_rope(x_ref, a_ref)
    y_ref.backward(g)

    y_k = fused_rope(x_k, a_k)
    y_k.backward(g)

    max_abs = (x_k.grad - x_ref.grad).abs().max().item()
    assert max_abs < 1e-3, f"dx per-head max_abs_err={max_abs} for shape {shape}"


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"BTHD_{s}")
def test_backward_parity_dangles_per_head(shape):
    B, T, H, D = shape
    x, angles = _make_inputs(B, T, H, D, torch.float32, seed=7, per_head=True)

    x_ref = x.detach().clone().requires_grad_(True)
    a_ref = angles.detach().clone().requires_grad_(True)
    x_k = x.detach().clone().requires_grad_(True)
    a_k = angles.detach().clone().requires_grad_(True)

    g = torch.randn_like(x)

    y_ref = apply_rope(x_ref, a_ref)
    y_ref.backward(g)

    y_k = fused_rope(x_k, a_k)
    y_k.backward(g)

    assert a_k.grad.shape == a_ref.grad.shape == angles.shape
    max_abs = (a_k.grad - a_ref.grad).abs().max().item()
    # Per-head has no H reduction → tighter tolerance than broadcast.
    assert max_abs < 1e-3, f"dangles per-head max_abs_err={max_abs} for shape {shape}"


def test_shapes_all_per_head():
    for shape in SHAPES:
        B, T, H, D = shape
        x, angles = _make_inputs(B, T, H, D, torch.float32, per_head=True)
        y = fused_rope(x, angles)
        y_ref = apply_rope(x, angles)
        assert y.shape == x.shape, f"shape mismatch for {shape}"
        assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5), \
            f"per-head forward mismatch for {shape}"


# --------------------------------------------------------------------------- #
# Module smoke: cumsum-of-angles upstream + apply downstream, eager vs fused.
# Mirrors the actual call site in Mamba3Block.forward.
# --------------------------------------------------------------------------- #

class _TinyRoPEModule(torch.nn.Module):
    """Reproduces the RoPE block at miniature scale, per-head angles."""

    def __init__(self, theta_dim: int, n_heads: int, use_fused: bool):
        super().__init__()
        self.theta_dim = theta_dim
        self.n_heads = n_heads
        self.use_fused = use_fused
        self.theta = torch.nn.Parameter(torch.randn(theta_dim) * 0.1)
        self.B_bias = torch.nn.Parameter(torch.ones(n_heads, 2 * theta_dim))

    def forward(self, x_proj: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        # x_proj: (B, T, n_heads, 2*theta_dim) — pre-RoPE B/C-like tensor
        # dt: (B, T, n_heads) — discretization step (positive)
        # Real Mamba3Block path: per-head angles, no mean-pool.
        raw_angles = dt.unsqueeze(-1) * self.theta.unsqueeze(0).unsqueeze(
            0
        ).unsqueeze(0)  # (B,T,n_heads,theta_dim)
        angles = (-raw_angles).cumsum(dim=1)  # (B,T,n_heads,theta_dim)
        B_h = x_proj + self.B_bias
        if self.use_fused:
            return fused_rope(B_h, angles)
        return apply_rope(B_h, angles)


def test_module_smoke():
    torch.manual_seed(0)
    B, T, n_heads, theta_dim = 2, 64, 4, 8
    D = 2 * theta_dim
    device = "cuda"
    dtype = torch.float32

    x_proj = torch.randn(B, T, n_heads, D, device=device, dtype=dtype)
    dt = torch.rand(B, T, n_heads, device=device, dtype=dtype) * 0.1 + 0.01

    mod_eager = _TinyRoPEModule(theta_dim, n_heads, use_fused=False).to(device)
    mod_fused = _TinyRoPEModule(theta_dim, n_heads, use_fused=True).to(device)
    mod_fused.load_state_dict(mod_eager.state_dict())

    y_eager = mod_eager(x_proj, dt)
    y_fused = mod_fused(x_proj, dt)

    max_abs = (y_eager - y_fused).abs().max().item()
    assert max_abs < 1e-4, f"module smoke max_abs_err={max_abs}"

    # Backward parity through the module
    g = torch.randn_like(y_eager)
    y_eager.backward(g)
    y_fused.backward(g)

    theta_diff = (mod_eager.theta.grad - mod_fused.theta.grad).abs().max().item()
    bias_diff = (mod_eager.B_bias.grad - mod_fused.B_bias.grad).abs().max().item()
    assert theta_diff < 1e-3, f"theta.grad mismatch: {theta_diff}"
    assert bias_diff < 1e-3, f"B_bias.grad mismatch: {bias_diff}"
