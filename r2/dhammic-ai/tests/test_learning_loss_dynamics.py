"""
Tests for Learning and Loss Dynamics component
"""

import torch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from learning_loss_dynamics import (
    DhammicLossFunction,
    KarmicOptimizer,
    NibbanaHomeostasis,
    create_dhammic_learning_dynamics,
)


def test_dhammic_loss_function_creation():
    """Test that Dhammic Loss Function can be created"""
    loss_fn = DhammicLossFunction()
    assert isinstance(loss_fn, DhammicLossFunction)
    print("✓ Dhammic Loss Function creation test passed")


def test_dhammic_loss_function_forward():
    """Test forward pass of Dhammic Loss Function"""
    loss_fn = DhammicLossFunction(reduction="mean", expected_feature_dim=28)
    prediction = torch.randn(2, 28)
    target = torch.randn(2, 28)
    loss = loss_fn(prediction, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar output
    assert loss.item() >= 0  # Loss should be non-negative
    print("✓ Dhammic Loss Function forward pass test passed")


def test_dhammic_loss_function_reductions():
    """Test different reduction types"""
    loss_fn_mean = DhammicLossFunction(reduction="mean", expected_feature_dim=28)
    loss_fn_sum = DhammicLossFunction(reduction="sum", expected_feature_dim=28)
    loss_fn_none = DhammicLossFunction(reduction="none", expected_feature_dim=28)

    prediction = torch.randn(2, 28)
    target = torch.randn(2, 28)

    loss_mean = loss_fn_mean(prediction, target)
    loss_sum = loss_fn_sum(prediction, target)
    loss_none = loss_fn_none(prediction, target)

    assert loss_mean.dim() == 0
    assert loss_sum.dim() == 0
    assert loss_none.shape == (2,)  # Should preserve batch dimension
    assert torch.isclose(
        loss_sum, loss_mean * 2.0
    )  # Sum should be 2x mean for batch size 2
    print("✓ Dhammic Loss Function reductions test passed")


def test_karmic_optimizer_creation():
    """Test that Karmic Optimizer can be created"""
    optimizer = KarmicOptimizer()
    assert isinstance(optimizer, KarmicOptimizer)
    assert hasattr(optimizer, "learning_rate")
    assert hasattr(optimizer, "STDP_A_plus")
    assert hasattr(optimizer, "STDP_A_minus")
    print("✓ Karmic Optimizer creation test passed")


def test_karmic_optimizer_stdp_update():
    """Test STDP update in Karmic Optimizer"""
    optimizer = KarmicOptimizer(learning_rate=1e-3)
    pre_synaptic = torch.randn(2, 5)
    post_synaptic = torch.randn(2, 5)
    reward = torch.tensor([1.0, 0.5])
    expectation = torch.tensor([0.8, 0.3])

    # Initial state
    initial_trace = (
        optimizer.post_trace.clone() if optimizer.post_trace is not None else None
    )

    # Apply STDP update
    weight_update = optimizer.stdp_update(
        pre_synaptic, post_synaptic, reward, expectation
    )

    assert weight_update.shape == (5, 5)  # Outer product of 5-dim vectors
    assert isinstance(weight_update, torch.Tensor)
    # Check that traces were updated
    assert optimizer.post_trace is not None
    print("✓ Karmic Optimizer STDP update test passed")


def test_karmic_optimizer_failure_modes():
    """Test failure mode indicators in Karmic Optimizer"""
    optimizer = KarmicOptimizer()
    initial_greed = optimizer.greed_indicator.item()
    initial_aversion = optimizer.aversion_indicator.item()

    # Test with positive reward (should increase greed indicator)
    pre_synaptic = torch.ones(1, 3)
    post_synaptic = torch.ones(1, 3)
    reward = torch.tensor([2.0])  # High positive reward
    expectation = torch.tensor([0.0])  # Low expectation

    _ = optimizer.stdp_update(pre_synaptic, post_synaptic, reward, expectation)
    updated_greed = optimizer.greed_indicator.item()

    # Greed indicator should have increased
    assert updated_greed >= initial_greed
    print("✓ Karmic Optimizer failure modes test passed")


def test_nibbana_homeostasis():
    """Test Nibbana Homeostasis checker"""
    homeostasis = NibbanaHomeostasis()

    # Test far from homeostasis
    is_homeostatic = homeostasis.check_homeostasis(
        dukkha_loss=torch.tensor(1.0), weight_change_magnitude=0.5
    )
    assert not is_homeostatic

    # Test near homeostasis
    is_homeostatic = homeostasis.check_homeostasis(
        dukkha_loss=torch.tensor(0.00001), weight_change_magnitude=0.00001
    )
    assert is_homeostatic

    state = homeostasis.get_homeostasis_state()
    assert "achieved" in state
    assert "free_energy_estimate" in state
    print("✓ Nibbana Homeostasis test passed")


def test_create_dhammic_learning_dynamics():
    """Test factory function"""
    loss_fn, optimizer, homeostasis = create_dhammic_learning_dynamics(
        learning_rate=2e-3, device="cpu"
    )

    assert isinstance(loss_fn, DhammicLossFunction)
    assert isinstance(optimizer, KarmicOptimizer)
    assert isinstance(homeostasis, NibbanaHomeostasis)

    # Test that they work together with 28 SDR dimensions per spec
    prediction = torch.randn(1, 28)
    target = torch.randn(1, 28)
    loss = loss_fn(prediction, target)
    assert isinstance(loss, torch.Tensor)
    print("✓ Create Dhammic Learning Dynamics test passed")


if __name__ == "__main__":
    test_dhammic_loss_function_creation()
    test_dhammic_loss_function_forward()
    test_dhammic_loss_function_reductions()
    test_karmic_optimizer_creation()
    test_karmic_optimizer_stdp_update()
    test_karmic_optimizer_failure_modes()
    test_nibbana_homeostasis()
    test_create_dhammic_learning_dynamics()
    print("All Learning and Loss Dynamics tests passed!")
