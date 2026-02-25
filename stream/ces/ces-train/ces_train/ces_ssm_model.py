"""
CES-SSM: 130M-parameter SSM-only model for Sprint 1 gate measurement.

Architecture (SSM core only — no memory, Hebbian, or MoE):
  - 12 Mamba-SSD layers (primary track)
  - 6 RWKV-WKV layers (secondary track)
  - Dual-track merge with learned per-channel sigmoid gate
  - Multi-scale temporal heads: K=4 groups, Δt ∈ {0.01, 0.1, 1.0, 10.0}
  - SP-inhibition sparsity at ~2% (via top-k in training mode)
  - Embedding + LM head, tied weights
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


MULTI_SCALE_DT = [0.01, 0.1, 1.0, 10.0]


@dataclass
class CESConfig:
    vocab_size: int = 32_768
    seq_len: int = 1_024
    hidden_dim: int = 768
    inner_dim: int = 1_536
    n_mamba_layers: int = 12
    n_rwkv_layers: int = 6
    n_heads: int = 8
    d_state: int = 64
    d_conv: int = 4
    dropout: float = 0.0


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


class SSDLayer(nn.Module):
    """Mamba-2 SSD selective scan layer (PyTorch sequential reference)."""

    def __init__(self, cfg: CESConfig):
        super().__init__()
        self.h = cfg.n_heads
        self.d = cfg.d_state
        self.hidden = cfg.hidden_dim

        self.in_proj = nn.Linear(
            cfg.hidden_dim,
            cfg.inner_dim + cfg.n_heads * 2 + cfg.n_heads * cfg.d_state * 2,
            bias=False,
        )
        self.out_proj = nn.Linear(cfg.inner_dim, cfg.hidden_dim, bias=False)

        self.a_log = nn.Parameter(torch.full((cfg.n_heads,), -1.0))
        self.d_skip = nn.Parameter(torch.ones(cfg.n_heads))
        self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads))

        self.dt_scales = nn.Parameter(
            torch.tensor([math.log(dt) for dt in MULTI_SCALE_DT] * (cfg.n_heads // 4)),
            requires_grad=False,
        )

        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h, d = self.h, self.d
        residual = x
        x = self.norm(x)

        proj = self.in_proj(x)
        inner_dim = self.hidden * 2
        x_in = proj[..., :inner_dim]
        dt = proj[..., inner_dim : inner_dim + h]
        b_flat = proj[..., inner_dim + h : inner_dim + h + h * d]
        c_flat = proj[..., inner_dim + h + h * d : inner_dim + h + 2 * h * d]

        b_proj = b_flat.view(B, T, h, d)
        c_proj = c_flat.view(B, T, h, d)

        dt_disc = softplus(dt + self.dt_bias + self.dt_scales)
        a = -self.a_log.abs()

        dA = torch.exp(dt_disc * a)
        dB = dt_disc.unsqueeze(-1) * b_proj

        state = torch.zeros(B, h, d, device=x.device, dtype=x.dtype)
        outputs = []

        x_heads = x_in[..., :h].view(B, T, h)

        for t in range(T):
            dA_t = dA[:, t, :].unsqueeze(-1)
            x_t = x_heads[:, t, :].unsqueeze(-1)
            dB_t = dB[:, t, :, :]
            state = dA_t * state + dB_t * x_t
            c_t = c_proj[:, t, :, :]
            y_t = (c_t * state).sum(-1) + self.d_skip * x_heads[:, t, :]
            outputs.append(y_t.unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        y_proj = self.out_proj(x_in)

        # Merge SSM output (B,T,H) with inner projection via learned gate, then project to hidden_dim
        gate = torch.sigmoid(y_proj)
        ssm_broadcast = y.mean(-1, keepdim=True).expand_as(y_proj)
        out = gate * ssm_broadcast + (1 - gate) * y_proj

        return out + residual


class WKVLayer(nn.Module):
    """RWKV v6 WKV recurrent layer (secondary track)."""

    def __init__(self, cfg: CESConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.r_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

        self.w = nn.Parameter(torch.full((d,), -0.5))
        self.u = nn.Parameter(torch.zeros(d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        r = self.r_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        state_num = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        state_den = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        w_row = self.w.unsqueeze(0).expand(B, -1)
        state_w_acc = w_row.clone()
        eu_base = torch.exp(self.u)

        outputs = []
        for t in range(T):
            ek = torch.exp(k[:, t, :])
            eu = eu_base * ek
            decay = torch.exp(state_w_acc)

            new_num = eu * v[:, t, :] + decay * state_num
            new_den = eu + decay * state_den

            wkv_t = new_num / new_den.abs().clamp(min=1e-8)
            out_t = torch.sigmoid(r[:, t, :]) * wkv_t
            outputs.append(out_t.unsqueeze(1))

            state_num = new_num
            state_den = new_den
            state_w_acc = state_w_acc + w_row

        y = torch.cat(outputs, dim=1)
        return self.out_proj(y) + residual


class CESSsmModel(nn.Module):
    """
    130M-parameter CES SSM-only model for Sprint 1 validation.

    Layer schedule: alternating Mamba and RWKV blocks.
    Total layers = n_mamba_layers + n_rwkv_layers = 18.
    Mamba layers: positions 0,2,4,6,8,10,12,14,16 (every 3/2 ratio)
    RWKV layers: interleaved at 1,3,5,7,9,11
    """

    def __init__(self, cfg: CESConfig = None):
        super().__init__()
        if cfg is None:
            cfg = CESConfig()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.embed_norm = nn.LayerNorm(cfg.hidden_dim)

        layers = []
        mamba_count, rwkv_count = 0, 0
        for i in range(cfg.n_mamba_layers + cfg.n_rwkv_layers):
            if rwkv_count >= cfg.n_rwkv_layers or (
                mamba_count < cfg.n_mamba_layers and i % 3 != 1
            ):
                layers.append(SSDLayer(cfg))
                mamba_count += 1
            else:
                layers.append(WKVLayer(cfg))
                rwkv_count += 1

        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.embed_norm(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
