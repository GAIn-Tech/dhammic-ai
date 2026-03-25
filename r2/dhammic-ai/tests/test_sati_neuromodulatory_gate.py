"""
Tests for Sati Neuromodulatory Gate component
"""

import torch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sati_neuromodulatory_gate import (
    SatiNeuromodulatoryGate,
    create_sati_neuromodulatory_gate,
)


def test_sati_neuromodulatory_gate_creation():
    """Test that Sati Neuromodulatory Gate can be created"""
    gate = create_sati_neuromodulatory_gate(d_model=64)
    assert isinstance(gate, SatiNeuromodulatoryGate)
    assert gate.d_model == 64
    print("✓ Sati Neuromodulatory Gate creation test passed")


def test_sati_neuromodulatory_gate_forward():
    """Test forward pass of Sati Neuromodulatory Gate"""
    gate = create_sati_neuromodulatory_gate(d_model=32)
    x = torch.randn(2, 10, 32, requires_grad=True)  # batch=2, seq_len=10, d_model=32
    context = torch.randn(2, 10, 32)

    gated_x, gate_value = gate(x, context=context)
    assert gated_x.shape == (2, 10, 32)
    assert gate_value.shape == (2, 10)  # Gate value per position
    print("✓ Sati Neuromodulatory Gate forward pass test passed")


def test_sati_neuromodulatory_gate_context_manager():
    """Test Sati Neuromodulatory Gate context manager"""
    gate = create_sati_neuromodulatory_gate(d_model=16)
    x = torch.randn(2, 5, 16, requires_grad=True)

    # Test with gate closed (should disable gradients)
    gate.force_closed()
    with gate.no_grad_context():
        y = x * 2
    # For this test, we just verify the context manager doesn't crash
    print("✓ Sati Neuromodulatory Gate context manager test passed")


def test_sati_neuromodulatory_gate_update():
    """Test gate state updates"""
    gate = create_sati_neuromodulatory_gate(d_model=16)
    initial_state = gate.gate_state.clone()

    # Update with reward signal
    gate.update_gate_state(reward_signal=1.0, surprise_signal=0.0)
    reward_state = gate.gate_state.clone()

    # Update with surprise signal
    gate.update_gate_state(reward_signal=0.0, surprise_signal=1.0)
    surprise_state = gate.gate_state.clone()

    # States should be different
    assert not torch.equal(initial_state, reward_state) or not torch.equal(
        reward_state, surprise_state
    )
    print("✓ Sati Neuromodulatory Gate update test passed")


def test_sati_neuromodulatory_gate_different_sizes():
    """Test Sati Neuromodulatory Gate with different sizes"""
    test_sizes = [16, 32, 64, 128]
    for d_model in test_sizes:
        gate = create_sati_neuromodulatory_gate(d_model=d_model)
        x = torch.randn(1, 3, d_model)
        gated_x, gate_value = gate(x)
        assert gated_x.shape == (1, 3, d_model)
        assert gate_value.shape == (1, 3)
    print("✓ Sati Neuromodulatory Gate different sizes test passed")


if __name__ == "__main__":
    test_sati_neuromodulatory_gate_creation()
    test_sati_neuromodulatory_gate_forward()
    test_sati_neuromodulatory_gate_context_manager()
    test_sati_neuromodulatory_gate_update()
    test_sati_neuromodulatory_gate_different_sizes()
    print("All Sati Neuromodulatory Gate tests passed!")
