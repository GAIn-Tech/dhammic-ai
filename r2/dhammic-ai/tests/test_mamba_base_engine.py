"""
Tests for Mamba-3 Base Engine component
"""

import torch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_base_engine import Mamba3BaseEngine, create_mamba3_base_engine


def test_mamba_base_engine_creation():
    """Test that Mamba-3 Base Engine can be created"""
    engine = create_mamba3_base_engine(d_model=64, d_state=16)
    assert isinstance(engine, Mamba3BaseEngine)
    assert engine.config.d_model == 64
    assert engine.config.d_state == 16
    print("✓ Mamba-3 Base Engine creation test passed")


def test_mamba_base_engine_forward():
    """Test forward pass of Mamba-3 Base Engine"""
    engine = create_mamba3_base_engine(d_model=32, d_state=8)
    x = torch.randn(2, 10, 32)  # batch=2, seq_len=10, d_model=32
    output = engine(x)
    assert output.shape == (2, 10, 32)
    print("✓ Mamba-3 Base Engine forward pass test passed")


def test_mamba_base_engine_different_sizes():
    """Test Mamba-3 Base Engine with different sizes"""
    for d_model in [16, 64, 128]:
        for d_state in [4, 16, 32]:
            engine = create_mamba3_base_engine(d_model=d_model, d_state=d_state)
            x = torch.randn(1, 5, d_model)
            output = engine(x)
            assert output.shape == (1, 5, d_model)
    print("✓ Mamba-3 Base Engine different sizes test passed")


if __name__ == "__main__":
    test_mamba_base_engine_creation()
    test_mamba_base_engine_forward()
    test_mamba_base_engine_different_sizes()
    print("All Mamba-3 Base Engine tests passed!")
