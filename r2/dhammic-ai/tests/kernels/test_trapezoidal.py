"""K4 — fused_trapezoidal parity & gradient tests against eager reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.kernels.trapezoidal import fused_trapezoidal


CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="Triton kernel requires CUDA")


# Realistic shapes from dhammic-ai (B, T, H). n_heads typically 8 or 16.
SHAPES = [(2, 256, 8), (4, 1024, 8), (1, 2048, 16)]


def _eager(
    dt_raw: torch.Tensor,
    lam_raw: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference path: mirrors mamba_base_engine.Mamba3Block.forward."""
    dt = F.softplus(dt_raw + dt_bias)
    lam = torch.sigmoid(lam_raw)
    A = -torch.exp(A_log)
    dA = dt * A
    alpha = torch.exp(dA)
    beta = (1.0 - lam) * dt * alpha
    gamma = lam * dt
    return dt, dA, beta, gamma


def _make_inputs(B: int, T: int, H: int, dtype: torch.dtype, seed: int = 0):
    torch.manual_seed(seed)
    dt_raw = torch.randn(B, T, H, device="cuda", dtype=dtype) * 0.5
    lam_raw = torch.randn(B, T, H, device="cuda", dtype=dtype) * 0.5
    # dt_bias: small positive ~ uniform(0.001, 0.1) like _init_ssm_params
    dt_bias = (torch.rand(H, device="cuda", dtype=dtype) * 0.099 + 0.001)
    # A_log: uniform(-4, -1) like _init_ssm_params
    A_log = (torch.rand(H, device="cuda", dtype=dtype) * 3.0 - 4.0)
    return dt_raw, lam_raw, dt_bias, A_log


# --------------------------------------------------------------------------- #
# Forward parity — fp32
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_parity_fp32(shape):
    B, T, H = shape
    dt_raw, lam_raw, dt_bias, A_log = _make_inputs(B, T, H, torch.float32)

    dt_f, dA_f, beta_f, gamma_f = fused_trapezoidal(dt_raw, lam_raw, dt_bias, A_log)
    dt_e, dA_e, beta_e, gamma_e = _eager(dt_raw, lam_raw, dt_bias, A_log)

    for name, f, e in [("dt", dt_f, dt_e), ("dA", dA_f, dA_e),
                       ("beta", beta_f, beta_e), ("gamma", gamma_f, gamma_e)]:
        err = (f - e).abs().max().item()
        assert err < 1e-5, f"shape {shape} {name}: max_abs_err {err} >= 1e-5"


# --------------------------------------------------------------------------- #
# Forward parity — bf16
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_parity_bf16(shape):
    B, T, H = shape
    dt_raw, lam_raw, dt_bias, A_log = _make_inputs(B, T, H, torch.bfloat16)

    dt_f, dA_f, beta_f, gamma_f = fused_trapezoidal(dt_raw, lam_raw, dt_bias, A_log)
    # Eager: do the math in fp32 (matches autocast), then cast the final outputs
    # to bf16. That's what the eager pipeline writes to memory in production,
    # so it's the apples-to-apples reference for a bf16-out fused kernel.
    dt_e32, dA_e32, beta_e32, gamma_e32 = _eager(
        dt_raw.float(), lam_raw.float(), dt_bias.float(), A_log.float()
    )
    dt_e = dt_e32.to(torch.bfloat16)
    dA_e = dA_e32.to(torch.bfloat16)
    beta_e = beta_e32.to(torch.bfloat16)
    gamma_e = gamma_e32.to(torch.bfloat16)

    for name, f, e in [("dt", dt_f, dt_e), ("dA", dA_f, dA_e),
                       ("beta", beta_f, beta_e), ("gamma", gamma_f, gamma_e)]:
        assert f.dtype == torch.bfloat16, f"{name}: expected bf16, got {f.dtype}"
        err = (f.float() - e.float()).abs().max().item()
        assert err < 5e-3, f"shape {shape} {name}: bf16 max_abs_err {err} >= 5e-3"


# --------------------------------------------------------------------------- #
# Backward parity — all 4 inputs, fp32
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", SHAPES)
def test_backward_all_grads(shape):
    B, T, H = shape
    dt_raw, lam_raw, dt_bias, A_log = _make_inputs(B, T, H, torch.float32)

    # --- Eager reference: build the same graph with autograd, get grads ---
    dt_raw_e = dt_raw.detach().clone().requires_grad_(True)
    lam_raw_e = lam_raw.detach().clone().requires_grad_(True)
    dt_bias_e = dt_bias.detach().clone().requires_grad_(True)
    A_log_e = A_log.detach().clone().requires_grad_(True)
    dt_e, dA_e, beta_e, gamma_e = _eager(dt_raw_e, lam_raw_e, dt_bias_e, A_log_e)

    # Use distinct random upstream gradients per output to test all paths.
    torch.manual_seed(7)
    g_dt    = torch.randn_like(dt_e)
    g_dA    = torch.randn_like(dA_e)
    g_beta  = torch.randn_like(beta_e)
    g_gamma = torch.randn_like(gamma_e)

    loss_e = (dt_e * g_dt).sum() + (dA_e * g_dA).sum() \
             + (beta_e * g_beta).sum() + (gamma_e * g_gamma).sum()
    loss_e.backward()

    # --- Fused path: same inputs, same upstream grads ---
    dt_raw_f = dt_raw.detach().clone().requires_grad_(True)
    lam_raw_f = lam_raw.detach().clone().requires_grad_(True)
    dt_bias_f = dt_bias.detach().clone().requires_grad_(True)
    A_log_f = A_log.detach().clone().requires_grad_(True)
    dt_f, dA_f, beta_f, gamma_f = fused_trapezoidal(
        dt_raw_f, lam_raw_f, dt_bias_f, A_log_f
    )
    loss_f = (dt_f * g_dt).sum() + (dA_f * g_dA).sum() \
             + (beta_f * g_beta).sum() + (gamma_f * g_gamma).sum()
    loss_f.backward()

    tol = 1e-3
    for name, gf, ge in [
        ("dt_raw",  dt_raw_f.grad,  dt_raw_e.grad),
        ("lam_raw", lam_raw_f.grad, lam_raw_e.grad),
        ("dt_bias", dt_bias_f.grad, dt_bias_e.grad),
        ("A_log",   A_log_f.grad,   A_log_e.grad),
    ]:
        err = (gf - ge).abs().max().item()
        denom = ge.abs().max().item() + 1e-12
        rel = err / denom
        assert err < tol or rel < tol, (
            f"shape {shape} grad[{name}]: max_abs_err {err}, rel {rel} >= {tol}"
        )


# --------------------------------------------------------------------------- #
# Shape sanity — covers small / medium / large
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", SHAPES)
def test_shapes(shape):
    B, T, H = shape
    dt_raw, lam_raw, dt_bias, A_log = _make_inputs(B, T, H, torch.bfloat16)
    dt, dA, beta, gamma = fused_trapezoidal(dt_raw, lam_raw, dt_bias, A_log)
    assert dt.shape == (B, T, H)
    assert dA.shape == (B, T, H)
    assert beta.shape == (B, T, H)
    assert gamma.shape == (B, T, H)
    for t in (dt, dA, beta, gamma):
        assert t.dtype == torch.bfloat16
        assert t.is_contiguous()
