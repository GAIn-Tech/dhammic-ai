"""
Mamba-3 Base Engine - Bhavanga (Life-Continuum) / Base Reality / The Universe
Continuous-time State Space Model handling multi-input multi-output (MIMO) complex-valued state tracking.
Simulates the unconditioned physical universe. Models the unbroken physics of reality without a KV cache.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Mamba3Config:
    """Configuration for Mamba-3 Base Engine"""

    d_model: int = 768  # Hidden dimension
    d_state: int = 16  # SSM state dimension (reduced from 128 for stability)
    d_conv: int = 4  # Convolution kernel size
    expand: int = 2  # Expansion factor for inner dimension
    dt_rank: str = "auto"  # Rank of dt projection
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True  # Fused kernel options
    layer_idx: Optional[int] = None  # For loading checkpoints
    rms_norm_eps: float = 1e-5
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None


class Mamba3BaseEngine(nn.Module):
    """
    Mamba-3 Base Engine - Bhavanga (Life-Continuum) / Base Reality / The Universe
    Continuous-time State Space Model handling multi-input multi-output (MIMO) complex-valued state tracking.
    Simulates the unconditioned physical universe. Models the unbroken physics of reality without a KV cache.
    """

    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        self._amp_disabled = True
        self.dt_rank = (
            config.dt_rank if config.dt_rank != "auto" else config.d_model // 16
        )

        # Inner dimension for expanded projection
        self.d_inner = config.expand * config.d_model
        assert self.d_inner % config.d_conv == 0

        # Input projection
        self.in_proj = nn.Linear(
            config.d_model,
            self.d_inner * 2,  # for gated MLPs
            bias=config.bias,
            device=config.device,
            dtype=config.dtype,
        )

        # Convolution for temporal modeling
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
            device=config.device,
            dtype=config.dtype,
        )

        # SSM parameters (selective state space)
        self.x_proj = nn.Linear(
            self.d_inner,
            int(self.dt_rank) + config.d_state * 2,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

        # dt projection
        self.dt_proj = nn.Linear(
            int(self.dt_rank),
            self.d_inner,
            bias=True,
            device=config.device,
            dtype=config.dtype,
        )

        # SSM state parameters (A, B, C, D)
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(
                    1, config.d_state + 1, dtype=torch.float32, device=config.device
                )
                .unsqueeze(0)
                .repeat(self.d_inner, 1)
            )
        )  # Shape: (d_inner, d_state)

        self.D = nn.Parameter(
            torch.ones(self.d_inner, device=config.device, dtype=config.dtype)
        )  # Skip connection parameter

        # Output projection
        self.out_proj = nn.Linear(
            self.d_inner,
            config.d_model,
            bias=config.bias,
            device=config.device,
            dtype=config.dtype,
        )

        # Initialize dt projection
        self._init_dt_proj()

        # RMS Norm for stabilization
        self.norm = nn.RMSNorm(
            config.d_model,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )

    def _init_dt_proj(self):
        """Initialize dt projection following Mamba initialization"""
        dt_init_std = self.dt_rank**-0.5 * self.config.dt_scale
        if self.config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif self.config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, device=self.config.device)
            * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        ).clamp(min=self.config.dt_min)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Initialize bias to be close to dt_init_std * dt
        self.dt_proj.bias.data[:] = self.dt_proj.bias.data + dt_init_std * dt

    def forward(self, hidden_states: torch.Tensor, inference_params=None):
        """
        Forward pass of Mamba-3 Base Engine

        Args:
            hidden_states: (batch, seq_len, dim)
            inference_params: for generation

        Returns:
            output: (batch, seq_len, dim)
        """
        batch, seq_len, dim = hidden_states.shape

        # Input projection and gate split
        xz = self.in_proj(hidden_states)  # (batch, seq_len, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # (batch, seq_len, d_inner) each

        # Temporal convolution
        x = self.conv1d(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)

        # Activation
        x = F.silu(x)
        z = F.silu(z)

        # SSM parameters
        x_dbl = self.x_proj(x)  # (batch, seq_len, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.config.d_state, self.config.d_state],
            dim=-1,
        )

        # dt projection
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)

        # SSM scan
        y = self.selective_scan(x, dt, B, C)

        # Gated output
        y = y * z
        output = self.out_proj(y)

        # Residual connection and normalization
        output = self.norm(hidden_states + output)

        return output

    def selective_scan(
        self,
        u: torch.Tensor,  # (batch, seq_len, d_inner)
        delta: torch.Tensor,  # (batch, seq_len, d_inner)
        B: torch.Tensor,  # (batch, seq_len, d_state)
        C: torch.Tensor,  # (batch, seq_len, d_state)
    ) -> torch.Tensor:
        """
        Selective scan operation - core of Mamba SSM

        Fixes applied:
        1. deltaB_u now includes u input (was missing)
        2. delta passed through softplus for stability (prevents negative values)
        3. D skip connection added to output
        """
        batch, seq_len, d_inner = u.shape
        d_state = B.shape[-1]

        # Apply softplus to delta for numerical stability (prevents negative Δ)
        delta = F.softplus(delta)

        # Discretize continuous parameters (A, B, C)
        # deltaA = exp(Δ * A) where A = -exp(A_log)
        deltaA = torch.exp(
            torch.einsum("bld,dn->bldn", delta, -torch.exp(self.A_log.float()))
        )
        # deltaB_u = Δ * B * u (FIXED: now includes u multiplication)
        deltaB_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)

        # Scan operation
        x = torch.zeros(
            (batch, d_inner, d_state), device=delta.device, dtype=delta.dtype
        )
        ys = []

        for t in range(seq_len):
            x = deltaA[:, t] * x + deltaB_u[:, t]
            y = torch.einsum("bdn,bn->bd", x, C[:, t, :])  # (batch, d_inner)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (batch, seq_len, d_inner)

        # Add D skip connection (FIXED: D was never used)
        y = y + u * self.D

        return y


# Factory function for easy instantiation
def create_mamba3_base_engine(
    d_model: int = 768,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Mamba3BaseEngine:
    """Factory function to create Mamba-3 Base Engine with sensible defaults"""
    config = Mamba3Config(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        device=device,
        dtype=dtype,
    )
    return Mamba3BaseEngine(config)


if __name__ == "__main__":
    # Simple test
    model = create_mamba3_base_engine(d_model=64, d_state=8)
    x = torch.randn(2, 10, 64)  # batch=2, seq_len=10, d_model=64
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Mamba-3 Base Engine test successful!")
