"""
DeltaNet Hebbian LoRA - Saṅkhāra (Volitional Formations / Karma) / The Ego
Fast-Weight Programmer maintaining a dynamic 2D memory matrix updated in the forward pass.
Filters pristine reality through localized biases; generates 'Karma' via local Spike-Timing-Dependent Plasticity (STDP).
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class DeltaNetHebbianLoRA(nn.Module):
    """
    DeltaNet Hebbian LoRA - Saṅkhāra (Volitional Formations / Karma) / The Ego
    Fast-Weight Programmer maintaining a dynamic 2D memory matrix updated in the forward pass.
    Filters pristine reality through localized biases; generates 'Karma' via local Spike-Timing-Dependent Plasticity (STDP).

    Implements hybrid learning rule:
    - Oja's rule for weight normalization (stability)
    - STDP timing window for potentiation/depression (biological plausibility)
    - Eligibility traces with neuromodulatory gating (reward-based modulation)
    """

    def __init__(
        self,
        d_model: int,  # Model dimension
        rank: int = 16,  # LoRA rank
        eta: float = 1e-3,  # Learning rate for Hebbian updates
        lambda_decay: float = 0.01,  # Decay factor for weights
        t_half: float = 1000.0,  # Half-life for learning rate adaptation
        stdp_window: float = 20.0,  # STDP timing window in ms (tau_+ and tau_-)
        a_plus: float = 1.0,  # Potentiation amplitude
        a_minus: float = 1.2,  # Depression amplitude (slightly stronger)
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.eta_0 = float(eta)
        self.lambda_decay = float(lambda_decay)
        self.t_half = float(t_half)
        self.stdp_window = float(stdp_window)
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self._step = 0
        self._amp_disabled = True

        # LoRA matrices for fast weights
        self.W_A = nn.Parameter(
            torch.randn(d_model, rank, device=device, dtype=dtype)
            * (1.0 / math.sqrt(float(max(1, d_model))))
        )
        self.W_B = nn.Parameter(torch.zeros(rank, d_model, device=device, dtype=dtype))

        # Running traces for STDP (exponential moving average of pre/post activity)
        self.register_buffer(
            "pre_trace", torch.zeros(d_model, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "post_trace", torch.zeros(rank, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "pre_trace_decay", torch.tensor(0.9, device=device)
        )  # tau_- equivalent
        self.register_buffer(
            "post_trace_decay", torch.tensor(0.9, device=device)
        )  # tau_+ equivalent

        # Eligibility trace for neuromodulatory gating
        self.register_buffer(
            "eligibility_trace",
            torch.zeros(d_model, rank, dtype=torch.float32, device=device),
        )
        self.register_buffer("eligibility_decay", torch.tensor(0.95, device=device))

        # Running mean for Oja's rule (kept for stability)
        self.register_buffer(
            "_x_mean", torch.zeros(d_model, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "_h_mean", torch.zeros(rank, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "_z_mean", torch.zeros(d_model, dtype=torch.float32, device=device)
        )

    def current_eta(self) -> float:
        """Get current effective learning rate with half-life decay"""
        return self.eta_0 / (1.0 + float(self._step) / self.t_half)

    def forward(
        self, x: torch.Tensor, spike_times: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of DeltaNet Hebbian LoRA

        Args:
            x: Input tensor (batch, seq_len, d_model)
            spike_times: Optional timing info for STDP (batch, seq_len, d_model) with relative spike times

        Returns:
            output: x + LoRA(x) with Hebbian updated weights
        """
        # LoRA computation: x @ W_A @ W_B
        h = x @ self.W_A  # (batch, seq_len, rank)
        z = h @ self.W_B  # (batch, seq_len, d_model)
        lora_out = z  # Scale is implicitly 1.0, can be added as parameter

        with torch.no_grad():
            # Update running means for Oja's rule (stability)
            self._x_mean = x.detach().float().mean(dim=(0, 1))
            self._h_mean = h.detach().float().mean(dim=(0, 1))
            self._z_mean = z.detach().float().mean(dim=(0, 1))

            # Update pre/post traces for STDP (exponential moving average)
            pre_activity = x.detach().float().mean(dim=(0, 1))  # (d_model,)
            post_activity = h.detach().float().mean(dim=(0, 1))  # (rank,)

            self.pre_trace = self.pre_trace * self.pre_trace_decay + pre_activity * (
                1 - self.pre_trace_decay
            )
            self.post_trace = (
                self.post_trace * self.post_trace_decay
                + post_activity * (1 - self.post_trace_decay)
            )

            # Update eligibility trace (pre-synaptic activity stored for later modulation)
            self.eligibility_trace = (
                self.eligibility_trace * self.eligibility_decay
                + torch.outer(pre_activity, post_activity)
                * (1 - self.eligibility_decay)
            )

        # Add LoRA output to residual connection
        output = x + lora_out

        return output

    @torch.no_grad()
    def hebbian_update(
        self, eta: float, lambda_decay: float, reward_signal: float = 1.0
    ) -> float:
        """
        Perform hybrid Hebbian update: Oja's rule + STDP timing window + eligibility trace gating

        Args:
            eta: Effective learning rate
            lambda_decay: Weight decay factor
            reward_signal: Neuromodulatory signal (dopamine-like) for eligibility trace gating

        Returns:
            Total delta magnitude for monitoring
        """
        # === W_A Update: STDP + Oja hybrid ===
        w_a = self.W_A.data.float()
        x_mean = self._x_mean.to(device=self.W_A.device, dtype=torch.float32)
        h_mean = self._h_mean.to(device=self.W_A.device, dtype=torch.float32)
        pre_trace = self.pre_trace.to(device=self.W_A.device, dtype=torch.float32)
        post_trace = self.post_trace.to(device=self.W_A.device, dtype=torch.float32)
        eligibility = self.eligibility_trace.to(
            device=self.W_A.device, dtype=torch.float32
        )

        # Oja's rule component (normalization for stability)
        x_norm_sq = float((x_mean * x_mean).sum().item())
        eta_norm_a = float(eta) / max(x_norm_sq + 1e-8, 1.0)
        oja_delta_a = eta_norm_a * (
            x_mean.unsqueeze(-1) * h_mean.unsqueeze(0) - float(lambda_decay) * w_a
        )

        # STDP component: potentiation (pre-before-post) vs depression (post-before-pre)
        # outer(pre_trace, post_trace) gives (d_model, rank) matching W_A shape
        stdp_potentiation = self.a_plus * torch.outer(pre_trace, post_trace)
        stdp_depression = self.a_minus * torch.outer(self.pre_trace * 0.5, post_trace)
        stdp_delta_a = stdp_potentiation - stdp_depression

        # Eligibility trace gating by reward signal (neuromodulatory)
        gated_delta_a = oja_delta_a + reward_signal * stdp_delta_a * eligibility

        # Clamp for stability
        delta_a = torch.clamp(gated_delta_a, -1.0, 1.0)
        self.W_A.data.add_(delta_a.to(dtype=self.W_A.dtype))

        # === W_B Update: similar hybrid rule ===
        w_b = self.W_B.data.float()
        h_mean_b = self._h_mean.to(device=self.W_B.device, dtype=torch.float32)
        z_mean = self._z_mean.to(device=self.W_B.device, dtype=torch.float32)

        # Oja's rule component
        h_norm_sq = float((h_mean_b * h_mean_b).sum().item())
        eta_norm_b = float(eta) / max(h_norm_sq + 1e-8, 1.0)
        oja_delta_b = eta_norm_b * (
            h_mean_b.unsqueeze(-1) * z_mean.unsqueeze(0) - float(lambda_decay) * w_b
        )

        # STDP component (simpler: no separate trace for z, use running mean)
        stdp_delta_b = (
            self.a_plus * (h_mean_b.unsqueeze(-1) * z_mean.unsqueeze(0))
            - self.a_minus * float(lambda_decay) * w_b
        )

        # Clamp for stability
        delta_b = torch.clamp(oja_delta_b + stdp_delta_b, -1.0, 1.0)
        self.W_B.data.add_(delta_b.to(dtype=self.W_B.dtype))

        # Return total change for monitoring
        return float(delta_a.abs().sum().item() + delta_b.abs().sum().item())

    def update_step(
        self,
        post_layer_outputs: Optional[torch.Tensor] = None,
        mu_t: float = 1.0,
        reward_signal: float = 1.0,
    ) -> float:
        """
        Update the Hebbian LoRA layer with post-synaptic activity and STDP timing

        Args:
            post_layer_outputs: Optional post-layer activity for more accurate update
            mu_t: Neuromodulatory factor (e.g., dopamine/acetylcholine levels) - gates eligibility trace
            reward_signal: Reward prediction error for eligibility trace gating (Dukkha/Karma term)

        Returns:
            Total delta magnitude from Hebbian update
        """
        if mu_t <= 0.0:
            return 0.0

        # Effective learning rate modulated by neuromodulators
        eta_eff = self.current_eta() * float(mu_t)

        # Use provided post-layer outputs or estimate from our own output
        if post_layer_outputs is not None:
            z_mean = post_layer_outputs.detach().float().mean(dim=(0, 1))
            self._z_mean.copy_(z_mean)
        else:
            z_mean = self._z_mean

        # Perform hybrid Hebbian update (Oja + STDP + eligibility gating)
        return self.hebbian_update(
            eta_eff, self.lambda_decay, reward_signal=float(reward_signal)
        )


def create_deltanet_hebbian_lora(
    d_model: int = 768,
    rank: int = 16,
    eta: float = 1e-3,
    lambda_decay: float = 0.01,
    t_half: float = 1000.0,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> DeltaNetHebbianLoRA:
    """Factory function to create DeltaNet Hebbian LoRA with sensible defaults"""
    return DeltaNetHebbianLoRA(
        d_model=d_model,
        rank=rank,
        eta=eta,
        lambda_decay=lambda_decay,
        t_half=t_half,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    # Simple test
    lora = create_deltanet_hebbian_lora(d_model=64, rank=8)
    x = torch.randn(2, 10, 64)  # batch=2, seq_len=10, d_model=64
    output = lora(x)

    # Test Hebbian update
    delta_magnitude = lora.update_step(mu_t=1.0)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hebbian update magnitude: {delta_magnitude:.6f}")
    print("DeltaNet Hebbian LoRA test successful!")
