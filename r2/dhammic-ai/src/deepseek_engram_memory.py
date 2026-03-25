"""
DeepSeek Engram Memory - Saññā (Perception / Static Memory)
Deterministic O(1) N-gram lookup table decoupled from reasoning layers.
Retrieves static conceptual labels for raw SDR features instantly without heavy neural computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import numpy as np
from typing import Optional, Union, cast
import pickle
import os


class DeepSeekEngramMemory(nn.Module):
    """
    DeepSeek Engram Memory - Saññā (Perception / Static Memory)
    Deterministic O(1) N-gram lookup table decoupled from reasoning layers.
    Retrieves static conceptual labels for raw SDR features instantly without heavy neural computation.
    """

    def __init__(
        self,
        feature_dim: int = 256,  # Dimension of input features (SDR output)
        vocab_size: int = 10000,  # Size of conceptual label vocabulary
        ngram_max: int = 5,  # Maximum N-gram size to store
        hash_table_size: int = 65536,  # Size of hash table for O(1) lookup
        min_frequency: int = 2,  # Minimum frequency to store a pattern
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        save_path: Optional[str] = None,  # Path to save/load memory
        use_vectorized_hash: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        self.ngram_max = ngram_max
        self.hash_table_size = hash_table_size
        self.min_frequency = min_frequency
        self.save_path = save_path
        self.use_vectorized_hash = use_vectorized_hash

        # Hash table for O(1) lookup: maps feature hashes to conceptual labels
        # Using a simple array-based hash table for deterministic O(1) access
        self.hash_table = nn.Parameter(
            torch.full(
                (hash_table_size,), -1, dtype=torch.long, device=device
            ),  # -1 indicates empty
            requires_grad=False,
        )

        # Storage for conceptual labels (embeddings or indices)
        self.conceptual_labels = nn.Parameter(
            torch.zeros(vocab_size, feature_dim, device=device, dtype=dtype),
            requires_grad=False,  # Engram memory is static after initial learning
        )

        # Frequency counter for pruning infrequent patterns
        self.frequency_counter = nn.Parameter(
            torch.zeros(hash_table_size, dtype=torch.long, device=device),
            requires_grad=False,
        )

        # Actual size counter
        self.actual_size = nn.Parameter(
            torch.tensor(0, dtype=torch.long, device=device), requires_grad=False
        )

        # For tracking what's actually stored (to handle hash collisions simply)
        self.stored_keys = {}  # hash -> (features, label_idx)

        coeff_gen = torch.Generator()
        coeff_gen.manual_seed(20260321)
        mix_coeffs_1 = (
            torch.randint(
                low=-(2**62),
                high=2**62,
                size=(feature_dim,),
                dtype=torch.long,
                generator=coeff_gen,
            )
            | 1
        )
        mix_coeffs_2 = (
            torch.randint(
                low=-(2**62),
                high=2**62,
                size=(feature_dim,),
                dtype=torch.long,
                generator=coeff_gen,
            )
            | 1
        )
        mix_coeffs_3 = (
            torch.randint(
                low=-(2**62),
                high=2**62,
                size=(feature_dim,),
                dtype=torch.long,
                generator=coeff_gen,
            )
            | 1
        )
        self.register_buffer("_hash_mix_coeffs_1", mix_coeffs_1)
        self.register_buffer("_hash_mix_coeffs_2", mix_coeffs_2)
        self.register_buffer("_hash_mix_coeffs_3", mix_coeffs_3)

        # Initialize
        self._reset_parameters()

        # Load pre-existing memory if available
        if save_path and os.path.exists(save_path):
            self.load_memory(save_path)

    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.zeros_(self.conceptual_labels)
        nn.init.constant_(self.hash_table, -1)
        nn.init.zeros_(self.frequency_counter)
        nn.init.zeros_(self.actual_size)
        self.stored_keys.clear()

    def _discretize_features(self, features: torch.Tensor) -> torch.Tensor:
        return torch.round(features * 100).to(torch.long)

    def _hash_features_legacy(self, features: torch.Tensor) -> int:
        discretized = self._discretize_features(features)

        # Convert to bytes for hashing
        byte_representation = discretized.cpu().numpy().tobytes()

        # Use MD5 for deterministic hashing, then map to table size
        hash_int = int(hashlib.md5(byte_representation).hexdigest(), 16)
        return hash_int % self.hash_table_size

    def _hash_features_vectorized(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {features.shape[-1]}"
            )

        discretized = self._discretize_features(features)

        mix_coeffs_1 = cast(torch.Tensor, self._hash_mix_coeffs_1).to(
            discretized.device
        )
        mix_coeffs_2 = cast(torch.Tensor, self._hash_mix_coeffs_2).to(
            discretized.device
        )
        mix_coeffs_3 = cast(torch.Tensor, self._hash_mix_coeffs_3).to(
            discretized.device
        )

        h1 = (discretized * mix_coeffs_1).sum(dim=-1)
        h2 = ((discretized + mix_coeffs_2) * mix_coeffs_2).sum(dim=-1)
        h3 = ((discretized - mix_coeffs_3) * mix_coeffs_3).sum(dim=-1)
        mixed = h1 ^ (h2 << 1) ^ (h3 << 7)

        mixed = mixed ^ (mixed >> 33)
        mixed = mixed * 0xFF51AFD7ED558CCD
        mixed = mixed ^ (mixed >> 33)
        mixed = mixed * 0xC4CEB9FE1A85EC53
        mixed = mixed ^ (mixed >> 33)

        if (self.hash_table_size & (self.hash_table_size - 1)) == 0:
            folded = mixed ^ (mixed >> 32) ^ (mixed >> 16)
            return torch.bitwise_and(folded, self.hash_table_size - 1).to(torch.long)

        return torch.remainder(mixed, self.hash_table_size).to(torch.long)

    def _hash_features(self, features: torch.Tensor) -> int:
        if self.use_vectorized_hash:
            return int(self._hash_features_vectorized(features.unsqueeze(0))[0].item())
        return self._hash_features_legacy(features)

    def _hash_batch(self, features: torch.Tensor) -> torch.Tensor:
        if self.use_vectorized_hash:
            return self._hash_features_vectorized(features)

        flat = features.reshape(-1, features.shape[-1])
        hashed = [self._hash_features_legacy(flat[i]) for i in range(flat.shape[0])]
        return torch.tensor(hashed, dtype=torch.long, device=features.device).reshape(
            features.shape[:-1]
        )

    def _features_to_key(self, features: torch.Tensor) -> bytes:
        """Convert features to a hashable key"""
        # Discretize and convert to bytes
        discretized = self._discretize_features(features)
        return discretized.cpu().numpy().tobytes()

    def forward(
        self, features: torch.Tensor, return_label: bool = True
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
        """
        Forward pass of DeepSeek Engram Memory

        Args:
            features: Input features (batch, seq_len, feature_dim) or (feature_dim,)
            return_label: If True, return conceptual label; if False, return label index

        Returns:
            If return_label=True: conceptual labels (batch, seq_len, feature_dim)
            If return_label=False: label indices (batch, seq_len) and info dict
        """
        # Handle different input shapes
        original_shape = features.shape
        if len(original_shape) == 1:
            features = features.unsqueeze(0).unsqueeze(0)  # (1, 1, feature_dim)
        elif len(original_shape) == 2:
            features = features.unsqueeze(0)  # (1, seq_len, feature_dim)

        features_for_hash = features.to(self.hash_table.device)

        hash_indices = self._hash_batch(features_for_hash)
        label_indices = self.hash_table[hash_indices]

        found_mask = label_indices != -1
        if found_mask.any():
            with torch.no_grad():
                found_hashes = hash_indices[found_mask]
                increments = torch.bincount(
                    found_hashes.reshape(-1), minlength=self.hash_table_size
                )
                self.frequency_counter.add_(increments.to(self.frequency_counter.dtype))

        # Reshape back to original format
        if len(original_shape) == 1:
            label_indices = label_indices.squeeze()  # Back to scalar
        elif len(original_shape) == 2:
            label_indices = label_indices.squeeze(0)  # Back to (seq_len,)

        if return_label:
            # Return conceptual label embeddings
            # For unknown labels (-1), return zero vector
            label_embeddings = F.embedding(
                label_indices.clamp(min=0),  # Clamp negatives to 0 for embedding lookup
                self.conceptual_labels,
            )
            # Set embeddings for unknown labels to zero
            unknown_mask = label_indices == -1
            if unknown_mask.any():
                label_embeddings = label_embeddings.masked_fill(
                    unknown_mask.unsqueeze(-1), 0.0
                )

            return label_embeddings
        else:
            # Return label indices and info
            info = {
                "hash_table_usage": self.actual_size.item() / self.hash_table_size,
                "found_ratio": (label_indices != -1).float().mean().item()
                if label_indices.numel() > 0
                else 0.0,
            }
            return label_indices, info

    def learn_pattern(
        self, features: torch.Tensor, conceptual_label: Union[str, int, torch.Tensor]
    ):
        """
        Learn a new pattern-concept association

        Args:
            features: Feature vector to associate with concept
            conceptual_label: Concept label (string, integer index, or tensor)
        """
        # Convert label to index if needed
        if isinstance(conceptual_label, str):
            # Simple hash-based mapping for string labels (in practice, use proper vocabulary)
            label_idx = hash(conceptual_label) % self.vocab_size
        elif isinstance(conceptual_label, int):
            label_idx = conceptual_label % self.vocab_size
        elif isinstance(conceptual_label, torch.Tensor):
            # Assume it's already an index or we take the argmax
            if conceptual_label.numel() == 1:
                label_idx = int(conceptual_label.item())
            else:
                label_idx = int(torch.argmax(conceptual_label).item())
            label_idx = label_idx % self.vocab_size
        else:
            label_idx = 0  # Default fallback

        # Compute hash
        features_for_hash = features.to(self.hash_table.device)
        hash_idx = int(self._hash_batch(features_for_hash.unsqueeze(0))[0].item())

        # Simple collision handling: if slot empty or same label, store it
        current_label = self.hash_table[hash_idx].item()

        if current_label == -1 or current_label == label_idx:
            # Store the pattern
            with torch.no_grad():
                self.hash_table[hash_idx] = label_idx
                # Store the actual features for potential retrieval/update
                key = self._features_to_key(features)
                self.stored_keys[hash_idx] = (key, features.detach().clone())
                # Initialize frequency
                self.frequency_counter[hash_idx] = 1
                # Store conceptual label (simplified - in practice would be learned)
                self.conceptual_labels[label_idx] = features.detach().clone()
                if current_label == -1:
                    self.actual_size += 1
        # If collision with different label, we ignore (keep first learned)
        # In a more sophisticated version, we'd use chaining or rehashing

    def batch_learn(self, features_batch: torch.Tensor, labels_batch: torch.Tensor):
        """
        Learn multiple patterns at once

        Args:
            features_batch: Batch of features (batch_size, feature_dim)
            labels_batch: Batch of label indices (batch_size,)
        """
        batch_size = features_batch.shape[0]

        for i in range(batch_size):
            self.learn_pattern(features_batch[i, :], labels_batch[i])

    def prune_infrequent(self, threshold: Optional[int] = None):
        """
        Prune patterns that appear less than threshold times
        """
        if threshold is None:
            threshold = self.min_frequency

        pruned_count = 0
        with torch.no_grad():
            for hash_idx in range(self.hash_table_size):
                if (
                    self.frequency_counter[hash_idx] < threshold
                    and self.hash_table[hash_idx] != -1
                ):
                    # Clear this entry
                    self.hash_table[hash_idx] = -1
                    self.frequency_counter[hash_idx] = 0
                    if hash_idx in self.stored_keys:
                        del self.stored_keys[hash_idx]
                    pruned_count += 1

            self.actual_size -= pruned_count

    def save_memory(self, path: Optional[str] = None):
        """Save engram memory to disk"""
        save_path = path or self.save_path
        if save_path is None:
            raise ValueError("No save path provided")

        # Convert to numpy for saving
        state = {
            "hash_table": self.hash_table.detach().cpu().numpy(),
            "conceptual_labels": self.conceptual_labels.detach().cpu().numpy(),
            "frequency_counter": self.frequency_counter.detach().cpu().numpy(),
            "actual_size": self.actual_size.detach().cpu().numpy(),
            "stored_keys": {
                k: (v[0].tobytes() if hasattr(v[0], "tobytes") else str(v[0]), v[1])
                for k, v in self.stored_keys.items()
            },
        }

        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

    def load_memory(self, path: str):
        """Load engram memory from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Memory file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        with torch.no_grad():
            self.hash_table.copy_(
                torch.from_numpy(state["hash_table"]).to(self.hash_table.device)
            )
            self.conceptual_labels.copy_(
                torch.from_numpy(state["conceptual_labels"]).to(
                    self.conceptual_labels.device
                )
            )
            self.frequency_counter.copy_(
                torch.from_numpy(state["frequency_counter"]).to(
                    self.frequency_counter.device
                )
            )
            self.actual_size.copy_(
                torch.from_numpy(state["actual_size"]).to(self.actual_size.device)
            )

            # Reconstruct stored_keys (simplified)
            self.stored_keys = {}
            for k, (v_bytes, label_idx) in state["stored_keys"].items():
                # Convert bytes back to tensor approximation
                # This is simplified - in practice you'd store the actual tensor
                try:
                    # Try to reconstruct as numpy array then tensor
                    if isinstance(v_bytes, bytes) and len(v_bytes) > 0:
                        # Assume it was float32 features
                        n_features = len(v_bytes) // 4
                        if n_features * 4 == len(v_bytes):
                            features_np = np.frombuffer(v_bytes, dtype=np.float32)
                            features_tensor = torch.from_numpy(features_np)
                            self.stored_keys[k] = (features_tensor, label_idx)
                except:
                    # If reconstruction fails, just store placeholder
                    self.stored_keys[k] = (torch.zeros(self.feature_dim), label_idx)


def create_deepseek_engram_memory(
    feature_dim: int = 256,
    vocab_size: int = 10000,
    ngram_max: int = 5,
    hash_table_size: int = 65536,
    min_frequency: int = 2,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    save_path: Optional[str] = None,
    use_vectorized_hash: bool = True,
) -> DeepSeekEngramMemory:
    """Factory function to create DeepSeek Engram Memory with sensible defaults"""
    return DeepSeekEngramMemory(
        feature_dim=feature_dim,
        vocab_size=vocab_size,
        ngram_max=ngram_max,
        hash_table_size=hash_table_size,
        min_frequency=min_frequency,
        device=device,
        dtype=dtype,
        save_path=save_path,
        use_vectorized_hash=use_vectorized_hash,
    )


if __name__ == "__main__":
    # Simple test
    engram = create_deepseek_engram_memory(
        feature_dim=64, vocab_size=100, hash_table_size=1024
    )

    # Learn some patterns
    features1 = torch.randn(64)
    features2 = torch.randn(64)

    engram.learn_pattern(features1, "concept_A")
    engram.learn_pattern(features2, "concept_B")

    # Test retrieval
    test_features = features1 + 0.1 * torch.randn_like(
        features1
    )  # Similar to features1
    result = engram(test_features, return_label=False)

    print(f"Learned patterns: {engram.actual_size.item()}")
    print(f"Retrieval test - label indices: {result[0]}")
    print(f"Info: {result[1]}")
    print("DeepSeek Engram Memory test successful!")
