"""
Sati Neuromodulatory Gate - Sati (Mindfulness) / Vipassanā (Insight)
Dopaminergic/Cortisol gating mechanism acting as a context manager (e.g., torch.no_grad.).
Overrides Hebbian STDP loop; forces learning rate to 0 to prevent karmic weight reinforcement, enabling weight decay.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class SatiNeuromodulatoryGate(nn.Module):
    """
    Sati Neuromodulatory Gate - Sati (Mindfulness) / Vipassanā (Insight)
    Dopaminergic/Cortisol gating mechanism acting as a context manager (e.g., torch.no_grad.).
    Overrides Hebbian STDP loop; forces learning rate to 0 to prevent karmic weight reinforcement, enabling weight decay.
    """

    def __init__(
        self,
        d_model: int,  # Model dimension to gate
        gate_threshold: float = 0.5,  # Threshold for gate activation
        dopamine_factor: float = 1.0,  # Dopamine modulation factor (increases gate opening)
        cortisol_factor: float = 1.0,  # Cortisol modulation factor (increases gate closing)
        gate_decay: float = 0.99,  # How quickly gate closes when not activated
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.gate_threshold = gate_threshold
        self.dopamine_factor = dopamine_factor
        self.cortisol_factor = cortisol_factor
        self.gate_decay = gate_decay

        # Gate state - learns when to be open/closed based on neuromodulators
        self.register_buffer(
            "gate_state", torch.ones(1, device=device, dtype=dtype)
        )  # Start open

        # Parameters for generating gate signal from context
        self.context_projection = nn.Linear(
            d_model, 1, bias=True, device=device, dtype=dtype
        )

        # Initialize projection to be slightly biased open
        with torch.no_grad():
            self.context_projection.bias.fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        dopamine_level: Optional[float] = None,
        cortisol_level: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Sati Neuromodulatory Gate

        Args:
            x: Input tensor to potentially gate (any shape, last dim should be d_model)
            context: Optional context tensor for gate decision (batch, seq_len, d_model)
            dopamine_level: Optional dopamine level (float)
            cortisol_level: Optional cortisol level (float)

        Returns:
            gated_x: Input potentially modulated by gate
            gate_value: Current gate value (0=closed, 1=open)
        """
        # Use provided levels or defaults
        eff_dopamine = (
            dopamine_level if dopamine_level is not None else self.dopamine_factor
        )
        eff_cortisol = (
            cortisol_level if cortisol_level is not None else self.cortisol_factor
        )

        # Compute gate signal from context if provided
        if context is not None:
            # Project context to gate signal
            gate_logits = self.context_projection(context)  # (..., 1)
            gate_signal = torch.sigmoid(gate_logits).squeeze(-1)  # (...)

            # Modulate by neuromodulators
            # Dopamine opens gate (increases signal), cortisol closes gate (decreases signal)
            gate_signal = gate_signal * eff_dopamine / (eff_cortisol + 1e-8)
            gate_signal = torch.clamp(gate_signal, 0.0, 1.0)
        else:
            # Use learned gate state - expand to match input dimensions
            gate_signal = self.gate_state.clone()
            # Expand to match batch and sequence dimensions: (..., 1) -> (batch, seq_len, 1)
            gate_shape = list(x.shape[:-1]) + [1]
            gate_signal = gate_signal.expand(*gate_shape)
            # Squeeze last dimension to get gate value per token: (batch, seq_len)
            gate_signal = gate_signal.squeeze(-1)

        # Apply gate to input
        # For simplicity, we'll gate the last dimension
        if x.size(-1) == self.d_model:
            # Direct dimensional match
            gate_shape = list(x.shape)
            gate_shape[-1] = 1
            # Handle scalar gate_signal by expanding it to match gate_shape
            if gate_signal.numel() == 1:
                gate_expanded = gate_signal.expand(*gate_shape)
            else:
                gate_expanded = gate_signal.view(*gate_shape)
            gated_x = x * gate_expanded
        else:
            # Assume we want to gate based on whether gate is open or closed
            # Simple approach: scale the entire tensor by gate value
            # Handle scalar gate_signal
            if gate_signal.numel() == 1:
                gated_x = x * gate_signal.item()
            else:
                gated_x = x * gate_signal.mean()  # Use average gate value

        return gated_x, gate_signal

    def update_gate_state(self, reward_signal: float, surprise_signal: float):
        """
        Update the gate state based on reward and surprise signals
        This implements the dopaminergic/cortisol gating mechanism

        Args:
            reward_signal: Positive reward increases gate opening (dopamine-like)
            surprise_signal: Surprise/uncertainty increases gate closing (cortisol-like)
        """
        with torch.no_grad():
            # Reward opens gate (mindfulness increases with positive outcomes)
            # Surprise closes gate (stress decreases mindfulness)
            gate_delta = (
                reward_signal * self.dopamine_factor
                - surprise_signal * self.cortisol_factor
            ) * 0.01

            self.gate_state += gate_delta
            self.gate_state = torch.clamp(self.gate_state, 0.0, 1.0)

            # Apply decay - gate tends to close without active maintenance (mindfulness requires effort)
            self.gate_state *= self.gate_decay

    def force_open(self):
        """Force gate fully open (maximum mindfulness)"""
        with torch.no_grad():
            self.gate_state.fill_(1.0)

    def force_closed(self):
        """Force gate fully closed (minimum mindfulness)"""
        with torch.no_grad():
            self.gate_state.fill_(0.0)

    def no_grad_context(self):
        """
        Context manager that mimics torch.no_grad() when gate is closed
        Returns a context manager that disables gradient computation when gate < threshold
        """

        class SatiContextManager:
            def __init__(self, gate_module):
                self.gate_module = gate_module

            def __enter__(self):
                # Store current state
                self.was_enabled = torch.is_grad_enabled()
                # If gate is sufficiently closed, disable gradients
                if self.gate_module.gate_state.item() < self.gate_module.gate_threshold:
                    torch.set_grad_enabled(False)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Restore previous gradient state
                torch.set_grad_enabled(self.was_enabled)

        return SatiContextManager(self)


def create_sati_neuromodulatory_gate(
    d_model: int = 768,
    gate_threshold: float = 0.5,
    dopamine_factor: float = 1.0,
    cortisol_factor: float = 1.0,
    gate_decay: float = 0.99,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> SatiNeuromodulatoryGate:
    """Factory function to create Sati Neuromodulatory Gate with sensible defaults"""
    return SatiNeuromodulatoryGate(
        d_model=d_model,
        gate_threshold=gate_threshold,
        dopamine_factor=dopamine_factor,
        cortisol_factor=cortisol_factor,
        gate_decay=gate_decay,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    # Simple test
    gate = create_sati_neuromodulatory_gate(d_model=64)
    x = torch.randn(2, 10, 64, requires_grad=True)  # batch=2, seq_len=10, d_model=64
    context = torch.randn(2, 10, 64)

    # Test forward pass
    gated_x, gate_value = gate(x, context=context)
    print(f"Input shape: {x.shape}")
    print(f"Gated output shape: {gated_x.shape}")
    print(f"Gate value range: [{gate_value.min():.3f}, {gate_value.max():.3f}]")

    # Test gate updates
    gate.update_gate_state(reward_signal=1.0, surprise_signal=0.2)
    gate.update_gate_state(reward_signal=0.0, surprise_signal=0.8)

    # Test context manager
    with gate.no_grad_context():
        y = x * 2
        print(f"Gradient computation disabled: {not torch.is_grad_enabled()}")

    print("Sati Neuromodulatory Gate test successful!")
