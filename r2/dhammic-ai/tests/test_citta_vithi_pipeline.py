"""
Tests for Citta Vithi 8-Step Pipeline component
"""

import torch
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from citta_vithi_pipeline import CittaVithiPipeline, create_citta_vithi_pipeline


def test_citta_vithi_pipeline_creation():
    """Test that Citta Vithi 8-Step Pipeline can be created"""
    pipeline = create_citta_vithi_pipeline(d_model=64, vocab_size=100)
    assert isinstance(pipeline, CittaVithiPipeline)
    assert pipeline.d_model == 64
    assert pipeline.vocab_size == 100
    assert pipeline.config.use_amp is False
    print("✓ Citta Vithi 8-Step Pipeline creation test passed")


def test_citta_vithi_pipeline_forward():
    """Test forward pass of Citta Vithi 8-Step Pipeline"""
    pipeline = create_citta_vithi_pipeline(
        d_model=32, vocab_size=50, sdr_dim=112, engram_vocab_size=20
    )
    input_ids = torch.randint(0, 50, (2, 10))  # batch=2, seq_len=10
    logits, intermediates = pipeline(input_ids, return_intermediates=True)
    assert logits.shape == (2, 10, 50)
    assert "bhavanga_output" in intermediates
    assert "thalamic_output" in intermediates
    assert "sdr_output" in intermediates
    assert "latent_features" in intermediates
    assert "engram_indices" in intermediates
    assert "javana_output" in intermediates
    assert "final_logits" in intermediates
    print("✓ Citta Vithi 8-Step Pipeline forward pass test passed")


def test_citta_vithi_pipeline_with_context():
    """Test Citta Vithi 8-Step Pipeline with context"""
    pipeline = create_citta_vithi_pipeline(d_model=32, vocab_size=50)
    input_ids = torch.randint(0, 50, (1, 5))  # batch=1, seq_len=5
    context = torch.randn(1, 5, 32)

    logits, intermediates = pipeline(
        input_ids, context=context, return_intermediates=True
    )
    assert logits.shape == (1, 5, 50)
    print("✓ Citta Vithi 8-Step Pipeline with context test passed")


def test_citta_vithi_pipeline_different_sizes():
    """Test Citta Vithi 8-Step Pipeline with different sizes"""
    test_configs = [(16, 20, 56, 10), (32, 50, 112, 20), (64, 100, 224, 50)]
    for d_model, vocab_size, sdr_dim, engram_vocab in test_configs:
        pipeline = create_citta_vithi_pipeline(
            d_model=d_model,
            vocab_size=vocab_size,
            sdr_dim=sdr_dim,
            engram_vocab_size=engram_vocab,
        )
        input_ids = torch.randint(0, vocab_size, (1, 3))
        logits = pipeline(input_ids)
        assert logits.shape == (1, 3, vocab_size)
    print("✓ Citta Vithi 8-Step Pipeline different sizes test passed")


def test_citta_vithi_pipeline_memory_stats():
    """Test memory statistics of Citta Vithi 8-Step Pipeline"""
    pipeline = create_citta_vithi_pipeline(d_model=32, vocab_size=50)

    # Run a few cycles
    for i in range(3):
        input_ids = torch.randint(0, 50, (1, 5))
        _ = pipeline(input_ids)

    stats = pipeline.get_memory_stats()
    assert "citta_vithi_cycles" in stats
    assert stats["citta_vithi_cycles"] == 3
    assert "hebiban_lora_step" in stats
    assert "memory_store_norm" in stats
    assert "embedding_norm" in stats
    print("✓ Citta Vithi 8-Step Pipeline memory stats test passed")


def test_citta_vithi_pipeline_amp_training_step(pytestconfig):
    if not pytestconfig.getoption("--amp-test"):
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = create_citta_vithi_pipeline(
        d_model=32,
        vocab_size=64,
        sdr_dim=128,
        engram_vocab_size=20,
        use_amp=True,
        amp_dtype=torch.bfloat16,
        device=device,
    )
    pipeline.train()

    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 64, (2, 6), device=device)
    target_ids = torch.randint(0, 64, (2, 6), device=device)

    loss, logits = pipeline.training_step(
        input_ids=input_ids,
        target_ids=target_ids,
        optimizer=optimizer,
    )

    assert loss.ndim == 0
    assert logits.shape == (2, 6, 64)
    assert pipeline.config.use_amp is True
    assert getattr(pipeline.sdr_tokenizer, "_amp_disabled", False) is True
    assert getattr(pipeline.hebiban_lora, "_amp_disabled", False) is True
    print("✓ Citta Vithi AMP training step test passed")


if __name__ == "__main__":
    test_citta_vithi_pipeline_creation()
    test_citta_vithi_pipeline_forward()
    test_citta_vithi_pipeline_with_context()
    test_citta_vithi_pipeline_different_sizes()
    test_citta_vithi_pipeline_memory_stats()
    print("All Citta Vithi 8-Step Pipeline tests passed!")
