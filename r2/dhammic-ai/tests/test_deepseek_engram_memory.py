"""
Tests for DeepSeek Engram Memory component
"""

import torch

from src.deepseek_engram_memory import (
    DeepSeekEngramMemory,
    create_deepseek_engram_memory,
)


def test_deepseek_engram_memory_creation():
    """Test that DeepSeek Engram Memory can be created"""
    engram = create_deepseek_engram_memory(feature_dim=64, vocab_size=100)
    assert isinstance(engram, DeepSeekEngramMemory)
    assert engram.feature_dim == 64
    assert engram.vocab_size == 100
    print("✓ DeepSeek Engram Memory creation test passed")


def test_deepseek_engram_memory_learning():
    """Test learning and retrieval in DeepSeek Engram Memory"""
    engram = create_deepseek_engram_memory(
        feature_dim=32, vocab_size=50, hash_table_size=1024
    )

    # Learn some patterns
    features1 = torch.randn(32)
    features2 = torch.randn(32)

    engram.learn_pattern(features1, "concept_A")
    engram.learn_pattern(features2, "concept_B")

    # Test retrieval
    result, info = engram(features1, return_label=False)
    assert result is not None
    assert "found_ratio" in info
    print("✓ DeepSeek Engram Memory learning test passed")


def test_deepseek_engram_memory_batch():
    """Test batch learning in DeepSeek Engram Memory"""
    engram = create_deepseek_engram_memory(feature_dim=16, vocab_size=20)

    # Batch learning
    features_batch = torch.randn(4, 16)
    labels_batch = torch.tensor([1, 2, 3, 4])

    engram.batch_learn(features_batch, labels_batch)

    # Test retrieval
    test_features = features_batch[0]
    result, info = engram(test_features, return_label=False)
    assert result is not None
    print("✓ DeepSeek Engram Memory batch learning test passed")


def test_deepseek_engram_memory_different_sizes():
    """Test DeepSeek Engram Memory with different sizes"""
    test_configs = [(16, 50, 512), (32, 100, 1024), (64, 200, 2048)]
    for feature_dim, vocab_size, hash_size in test_configs:
        engram = create_deepseek_engram_memory(
            feature_dim=feature_dim, vocab_size=vocab_size, hash_table_size=hash_size
        )
        x = torch.randn(feature_dim)
        result, info = engram(x, return_label=False)
        assert result is not None
    print("✓ DeepSeek Engram Memory different sizes test passed")


def test_hash_parity_and_legacy_fallback_path():
    torch.manual_seed(7)
    engram = create_deepseek_engram_memory(
        feature_dim=32, vocab_size=64, hash_table_size=4096, use_vectorized_hash=True
    )

    batch = torch.randn(3, 5, 32)

    legacy_flat = []
    flat_batch = batch.reshape(-1, 32)
    for i in range(flat_batch.shape[0]):
        legacy_flat.append(engram._hash_features_legacy(flat_batch[i]))
    legacy_hashes = torch.tensor(legacy_flat, dtype=torch.long).reshape(3, 5)

    vectorized_hashes = engram._hash_features_vectorized(batch)

    engram.use_vectorized_hash = False
    fallback_hashes = engram._hash_batch(batch)

    assert torch.equal(fallback_hashes.cpu(), legacy_hashes)
    assert vectorized_hashes.shape == legacy_hashes.shape
    assert (vectorized_hashes >= 0).all()
    assert (vectorized_hashes < engram.hash_table_size).all()

    vectorized_hashes_2 = engram._hash_features_vectorized(batch)
    assert torch.equal(vectorized_hashes, vectorized_hashes_2)


def test_vectorized_collision_rate_vs_legacy():
    torch.manual_seed(123)
    engram = create_deepseek_engram_memory(
        feature_dim=64, vocab_size=128, hash_table_size=8192, use_vectorized_hash=True
    )

    def collision_count(indices: torch.Tensor) -> int:
        flat = indices.reshape(-1)
        unique = torch.unique(flat).numel()
        return int(flat.numel() - unique)

    edge_zeros = torch.zeros(128, 64)
    edge_ones = torch.ones(128, 64)
    random_data = torch.randn(2048, 64)

    datasets = [edge_zeros, edge_ones, random_data]

    for data in datasets:
        legacy_list = [
            engram._hash_features_legacy(data[i]) for i in range(data.shape[0])
        ]
        legacy = torch.tensor(legacy_list, dtype=torch.long)
        vectorized = engram._hash_features_vectorized(data)

        legacy_collisions = collision_count(legacy)
        vectorized_collisions = collision_count(vectorized)

        if torch.all(data == 0) or torch.all(data == 1):
            assert vectorized_collisions == legacy_collisions
        else:
            assert vectorized_collisions <= legacy_collisions + max(
                8, data.shape[0] // 50
            )


if __name__ == "__main__":
    test_deepseek_engram_memory_creation()
    test_deepseek_engram_memory_learning()
    test_deepseek_engram_memory_batch()
    test_deepseek_engram_memory_different_sizes()
    test_hash_parity_and_legacy_fallback_path()
    test_vectorized_collision_rate_vs_legacy()
    print("All DeepSeek Engram Memory tests passed!")
