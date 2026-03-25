"""
Tests for Numenta SDR Tokenizer component
"""

import torch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from numenta_sdr_tokenizer import NumentaSDRTokenizer, create_numenta_sdr_tokenizer


def test_numenta_sdr_tokenizer_creation():
    """Test that Numenta SDR Tokenizer can be created"""
    tokenizer = create_numenta_sdr_tokenizer(input_dim=64, sdr_dim=224)
    assert isinstance(tokenizer, NumentaSDRTokenizer)
    assert tokenizer.input_dim == 64
    assert tokenizer.sdr_dim == 224
    print("✓ Numenta SDR Tokenizer creation test passed")


def test_numenta_sdr_tokenizer_forward():
    """Test forward pass of Numenta SDR Tokenizer"""
    tokenizer = create_numenta_sdr_tokenizer(
        input_dim=32, sdr_dim=112, sdr_sparsity=0.02
    )
    x = torch.randn(2, 10, 32)  # batch=2, seq_len=10, input_dim=32
    sdr_output, info = tokenizer(x)
    assert sdr_output.shape == (2, 10, 112)
    assert "active_ratio" in info
    assert 0 <= info["active_ratio"] <= 1
    print("✓ Numenta SDR Tokenizer forward pass test passed")


def test_numenta_sdr_tokenizer_sparsity():
    """Test that Numenta SDR Tokenizer enforces sparsity"""
    tokenizer = create_numenta_sdr_tokenizer(
        input_dim=16, sdr_dim=56, sdr_sparsity=0.01
    )  # 1% sparsity
    x = torch.ones(2, 5, 16)  # All ones input
    sdr_output, info = tokenizer(x)
    # Should be close to target sparsity (allowing for variation)
    assert info["active_ratio"] < 0.05  # Should be much less than 5% active
    print("✓ Numenta SDR Tokenizer sparsity test passed")


def test_numenta_sdr_tokenizer_different_sizes():
    """Test Numenta SDR Tokenizer with different sizes"""
    test_configs = [(16, 56, 0.01), (32, 112, 0.02), (64, 224, 0.03)]
    for input_dim, sdr_dim, sparsity in test_configs:
        tokenizer = create_numenta_sdr_tokenizer(
            input_dim=input_dim, sdr_dim=sdr_dim, sdr_sparsity=sparsity
        )
        x = torch.randn(1, 3, input_dim)
        sdr_output, info = tokenizer(x)
        assert sdr_output.shape == (1, 3, sdr_dim)
    print("✓ Numenta SDR Tokenizer different sizes test passed")


if __name__ == "__main__":
    test_numenta_sdr_tokenizer_creation()
    test_numenta_sdr_tokenizer_forward()
    test_numenta_sdr_tokenizer_sparsity()
    test_numenta_sdr_tokenizer_different_sizes()
    print("All Numenta SDR Tokenizer tests passed!")
