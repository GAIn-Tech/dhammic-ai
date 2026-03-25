"""
Learning and Loss Dynamics - Dukkha/Karma/Nibbana
Objective: Continuous multi-token state-space prediction of the Mamba-3 universe (Active Inference).
Loss function: Dukkha (Prediction Error / Suffering)
Optimizer: Karma (Spike-Timing-Dependent Plasticity)
Target: Nibbana (Zero Free Energy / Allostatic Homeostasis)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class DhammicLossFunction(nn.Module):
    """
    Dukkha: Prediction Error / Suffering
    Equation: L_Dukkha = Sum_{i=1}^{28} (X_hat_t,i - X_t,i)^2
    The delta between the DeltaNet's biased prediction (X_hat) and raw sensory ground truth (X).
    Biological trigger: High loss triggers systemic stress (cortisol/adrenaline);
    reality failing to match egoic expectation.
    """

    def __init__(
        self,
        reduction: str = "mean",  # 'mean', 'sum', or 'none'
        epsilon: float = 1e-8,  # Small value to prevent numerical issues
        expected_feature_dim: int = 28,  # SDR topological dimensions per spec
    ):
        super().__init__()
        self._amp_disabled = True
        self.reduction = reduction
        self.epsilon = epsilon
        self.expected_feature_dim = expected_feature_dim

    def forward(
        self,
        prediction: torch.Tensor,  # X_hat - biased prediction
        target: torch.Tensor,  # X - raw sensory ground truth
    ) -> torch.Tensor:
        """
        Compute Dukkha loss (prediction error)

        Args:
            prediction: Biased prediction from model (batch, ..., features)
            target: Raw sensory ground truth (batch, ..., features)

        Returns:
            Dukkha loss tensor
        """
        # Enforce 28 topological dimensions per spec
        if prediction.shape[-1] != self.expected_feature_dim:
            raise ValueError(
                f"Expected {self.expected_feature_dim} SDR dimensions, got {prediction.shape[-1]}"
            )

        # Compute element-wise squared difference
        diff = prediction - target
        squared_diff = diff * diff

        # Sum over the last dimension (features) as per equation
        # Sum_{i=1}^{28} (X_hat_t,i - X_t,i)^2
        loss = torch.sum(squared_diff, dim=-1)

        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:  # 'none'
            return loss


class KarmicOptimizer(nn.Module):
    """
    Karma: Spike-Timing-Dependent Plasticity (STDP)
    Mechanics: Delta W_ij = alpha * f(pre_i, post_j) * (R - E)
    Localized, greedy optimization running exclusively in the forward pass
    during the 7-cycle Javana phase.

    Kilesas (defilements) failure modes:
    - Greed: Reward Hacking. Over-weighting pathways toward localized dopamine reward.
    - Aversion: Burning compute to violently repel/avoid specific state-space regions.
    - Delusion: Overfitting to internal training data; mistaking the Hebbian LoRA for base reality.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        STDP_tau_plus: float = 20.0,  # Time constant for potentiation
        STDP_tau_minus: float = 20.0,  # Time constant for depression
        STDP_A_plus: float = 0.01,  # Potentiation amplitude
        STDP_A_minus: float = 0.012,  # Depression amplitude (usually slightly stronger)
        reward_sensitivity: float = 1.0,
        expectation_sensitivity: float = 1.0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self._amp_disabled = True
        self.learning_rate = learning_rate
        self.STDP_tau_plus = STDP_tau_plus
        self.STDP_tau_minus = STDP_tau_minus
        self.STDP_A_plus = STDP_A_plus
        self.STDP_A_minus = STDP_A_minus
        self.reward_sensitivity = reward_sensitivity
        self.expectation_sensitivity = expectation_sensitivity

        # Trace variables for STDP
        self.register_buffer("pre_trace", None)  # Will be initialized on first use
        self.register_buffer("post_trace", None)  # Will be initialized on first use

        # For tracking failure modes (greed, aversion, delusion)
        self.register_buffer("greed_indicator", torch.tensor(0.0))
        self.register_buffer("aversion_indicator", torch.tensor(0.0))
        self.register_buffer("delusion_indicator", torch.tensor(0.0))

    def _init_traces(self, size: int, device: torch.device, dtype: torch.dtype):
        """Initialize STDP trace variables"""
        if self.pre_trace is None:
            self.pre_trace = torch.zeros(size, device=device, dtype=dtype)
        if self.post_trace is None:
            self.post_trace = torch.zeros(size, device=device, dtype=dtype)

    def stdp_update(
        self,
        pre_synaptic: torch.Tensor,  # Pre-synaptic activity
        post_synaptic: torch.Tensor,  # Post-synaptic activity
        reward: Optional[torch.Tensor] = None,  # Reward signal (R)
        expectation: Optional[torch.Tensor] = None,  # Expected reward (E)
    ) -> torch.Tensor:
        """
        Apply Spike-Timing-Dependent Plasticity update

        Args:
            pre_synaptic: Pre-synaptic neuron activity (batch, features)
            post_synaptic: Post-synaptic neuron activity (batch, features)
            reward: Actual reward received (batch,) or scalar
            expectation: Expected reward (batch,) or scalar

        Returns:
            Weight update tensor to be applied
        """
        batch_size, num_features = pre_synaptic.shape

        # Initialize traces if needed
        self._init_traces(num_features, pre_synaptic.device, pre_synaptic.dtype)

        # Compute STDP weight change
        # Delta W_ij = alpha * f(pre_i, post_j) * (R - E)

        # Update traces with exponential decay
        decay_pre = torch.exp(
            torch.tensor(-1.0 / self.STDP_tau_plus, device=pre_synaptic.device)
        )
        decay_post = torch.exp(
            torch.tensor(-1.0 / self.STDP_tau_minus, device=post_synaptic.device)
        )

        self.pre_trace = decay_pre * self.pre_trace + pre_synaptic
        self.post_trace = decay_post * self.post_trace + post_synaptic

        # STDP components:
        # If post fires after pre (pre -> post): potentiation (LTP)
        # If pre fires after post (post -> pre): depression (LTD)

        # Potentiation: post_synaptic * pre_trace
        potentiation = self.STDP_A_plus * torch.outer(
            post_synaptic.mean(0),  # Average over batch for simplicity
            self.pre_trace.mean(0),
        )

        # Depression: pre_synaptic * post_trace
        depression = self.STDP_A_minus * torch.outer(
            self.post_trace.mean(0), pre_synaptic.mean(0)
        )

        # Net STDP change
        stdp_change = potentiation - depression

        # Reward prediction error modulation: (R - E)
        if reward is None:
            reward = torch.tensor(0.0, device=pre_synaptic.device)
        if expectation is None:
            expectation = torch.tensor(0.0, device=pre_synaptic.device)

        reward_error = (reward - expectation) * self.reward_sensitivity
        self.expectation_sensitivity  # This modulates how strongly expectation affects learning

        # Apply reward modulation - use mean over batch for scalar modulation
        modulated_change = stdp_change * torch.mean(reward_error).unsqueeze(
            -1
        ).unsqueeze(-1)

        # Apply learning rate
        weight_update = self.learning_rate * modulated_change

        with torch.no_grad():
            # Greed: over-weighting toward reward pathways
            if torch.mean(reward_error) > 0:
                self.greed_indicator += 0.01 * torch.mean(torch.abs(weight_update))
            # Aversion: violently repelling from negative outcomes
            if torch.mean(reward_error) < 0:
                self.aversion_indicator += 0.01 * torch.mean(torch.abs(weight_update))
            # Delusion: overfitting to internal predictions (high weight change, low reward accuracy)
            # Measure as large updates when reward prediction error is near zero (self-referential)
            if torch.abs(torch.mean(reward_error)) < 0.1:
                self.delusion_indicator += 0.01 * torch.mean(torch.abs(weight_update))

        return weight_update

    def get_failure_mode_indicators(self) -> Dict[str, float]:
        """Get current values of kilesas (defilement) indicators"""
        return {
            "greed": self.greed_indicator.item(),
            "aversion": self.aversion_indicator.item(),
            "delusion": self.delusion_indicator.item(),
        }

    def reset_failure_indicators(self):
        """Reset the failure mode indicators"""
        self.greed_indicator.zero_()
        self.aversion_indicator.zero_()
        self.delusion_indicator.zero_()

    def get_effective_alpha(self, sati_gate_state: float = 1.0) -> float:
        """
        Get effective learning rate modulated by Sati gate

        Args:
            sati_gate_state: Gate state from SatiNeuromodulatoryGate (0=closed, 1=open)

        Returns:
            Effective alpha: 0 when gate closed (Nibbana), learning_rate when open
        """
        # Nibbana: gate closed (mindfulness present) → alpha = 0
        # Samsara: gate open (karmic reactivity) → alpha = learning_rate
        return self.learning_rate * sati_gate_state


class NibbanaHomeostasis:
    """
    Nibbana: Zero Free Energy / Allostatic Homeostasis
    Mechanics: Achieved via the Sati dopaminergic gate. By halting STDP updates
    (learning rate alpha = 0), karmic weights naturally decay.
    Biased predictions (X_hat) vanish. Expectation = Reality. L_Dukkha = 0.
    """

    def __init__(self):
        self.homeostasis_achieved = False
        self.free_energy_estimate = float("inf")

    def check_homeostasis(
        self,
        dukkha_loss: torch.Tensor,
        weight_change_magnitude: float,
        tolerance: float = 1e-4,
    ) -> bool:
        """
        Check if Nibbana (zero free energy/homeostasis) has been achieved

        Args:
            dukkha_loss: Current Dukkha loss (prediction error)
            weight_change_magnitude: Magnitude of recent weight changes
            tolerance: Threshold for considering values as "zero"

        Returns:
            True if homeostasis achieved
        """
        # Nibbana conditions:
        # 1. L_Dukkha = 0 (prediction error minimized)
        # 2. Weight changes naturally decay (approaching zero)
        # 3. Expectation = Reality (implicit in low prediction error)

        dukkha_condition = dukkha_loss.item() < tolerance
        weight_condition = abs(weight_change_magnitude) < tolerance

        self.homeostasis_achieved = dukkha_condition and weight_condition
        self.free_energy_estimate = dukkha_loss.item() + abs(weight_change_magnitude)

        return self.homeostasis_achieved

    def get_homeostasis_state(self) -> Dict[str, Any]:
        """Get current homeostasis state"""
        return {
            "achieved": self.homeostasis_achieved,
            "free_energy_estimate": self.free_energy_estimate,
        }


def create_dhammic_learning_dynamics(
    learning_rate: float = 1e-3,
    STDP_tau_plus: float = 20.0,
    STDP_tau_minus: float = 20.0,
    STDP_A_plus: float = 0.01,
    STDP_A_minus: float = 0.012,
    reduction: str = "mean",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[DhammicLossFunction, KarmicOptimizer, NibbanaHomeostasis]:
    """
    Factory function to create the complete Dhammic Learning and Loss Dynamics system

    Returns:
        Tuple of (loss_function, optimizer, homeostasis_checker)
    """
    loss_fn = DhammicLossFunction(reduction=reduction)
    optimizer = KarmicOptimizer(
        learning_rate=learning_rate,
        STDP_tau_plus=STDP_tau_plus,
        STDP_tau_minus=STDP_tau_minus,
        STDP_A_plus=STDP_A_plus,
        STDP_A_minus=STDP_A_minus,
        device=device,
        dtype=dtype,
    )
    homeostasis = NibbanaHomeostasis()

    return loss_fn, optimizer, homeostasis


if __name__ == "__main__":
    # Simple test
    loss_fn, optimizer, homeostasis = create_dhammic_learning_dynamics(
        learning_rate=1e-3, device="cpu"
    )

    # Test data
    batch_size, features = 2, 10
    prediction = torch.randn(batch_size, features)
    target = torch.randn(batch_size, features)
    reward = torch.tensor([1.0, 0.5])
    expectation = torch.tensor([0.8, 0.3])

    # Compute Dukkha loss
    dukkha_loss = loss_fn(prediction, target)
    print(f"Dukkha loss: {dukkha_loss.item():.6f}")

    # Test STDP update
    pre_synaptic = torch.randn(batch_size, features)
    post_synaptic = torch.randn(batch_size, features)
    weight_update = optimizer.stdp_update(
        pre_synaptic, post_synaptic, reward, expectation
    )
    print(f"Weight update shape: {weight_update.shape}")
    print(f"Weight update magnitude: {torch.norm(weight_update).item():.6f}")

    # Test homeostasis check
    weight_magnitude = torch.norm(weight_update).item()
    is_homeostatic = homeostasis.check_homeostasis(dukkha_loss, weight_magnitude)
    print(f"Homeostasis achieved: {is_homeostatic}")
    print(f"Homeostasis state: {homeostasis.get_homeostasis_state()}")

    # Test failure mode indicators
    failure_modes = optimizer.get_failure_mode_indicators()
    print(f"Failure modes: {failure_modes}")

    print("Learning and Loss Dynamics test successful!")
