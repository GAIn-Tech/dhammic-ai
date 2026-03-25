"""
Tests for DeltaNet Hebbian LoRA component
"""

import torch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deltanet_hebbian_lora import DeltaNetHebbianLoRA, create_deltanet_hebbian_lora


def test_deltanet_hebbian_lora_creation():
    """Test that DeltaNet Hebbian LoRA can be created"""
    lora = create_deltanet_hebbian_lora(d_model=64, rank=16)
    assert isinstance(lora, DeltaNetHebbianLoRA)
    assert lora.d_model == 64
    assert lora.rank == 16
    print("✓ DeltaNet Hebbian LoRA creation test passed")


def test_deltanet_hebbian_lora_forward():
    """Test forward pass of DeltaNet Hebbian LoRA"""
    lora = create_deltanet_hebbian_lora(d_model=32, rank=8)
    x = torch.randn(2, 10, 32)  # batch=2, seq_len=10, d_model=32
    output = lora(x)
    assert output.shape == (2, 10, 32)
    print("✓ DeltaNet Hebbian LoRA forward pass test passed")


def test_deltanet_hebbian_lora_update():
    """Test Hebbian update of DeltaNet Hebbian LoRA"""
    lora = create_deltanet_hebbian_lora(d_model=16, rank=4)
    x = torch.randn(2, 5, 16)
    _ = lora(x)  # Forward pass to set internal states

    # Test update step
    delta_magnitude = lora.update_step(mu_t=1.0)
    assert isinstance(delta_magnitude, float)
    assert delta_magnitude >= 0.0
    print("✓ DeltaNet Hebbian LoRA update test passed")


def test_deltanet_hebbian_lora_different_sizes():
    """Test DeltaNet Hebbian LoRA with different sizes"""
    test_configs = [(16, 4), (32, 8), (64, 16)]
    for d_model, rank in test_configs:
        lora = create_deltanet_hebbian_lora(d_model=d_model, rank=rank)
        x = torch.randn(1, 3, d_model)
        output = lora(x)
        assert output.shape == (1, 3, d_model)
    print("✓ DeltaNet Hebbian LoRA different sizes test passed")


if __name__ == "__main__":
    test_deltanet_hebbian_lora_creation()
    test_deltanet_hebbian_lora_forward()
    test_deltanet_hebbian_lora_update()
    test_deltanet_hebbian_lora_different_sizes()
    print("All DeltaNet Hebbian LoRA tests passed!")
