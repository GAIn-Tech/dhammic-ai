"""
Numenta SDR Tokenizer - Sense Doors / Transduction
Spiking Neural Network utilizing Sparse Distributed Representations (SDR) and Semantic Folding.
Compresses continuous raw reality into discrete, high-frequency, fault-tolerant sparse topological maps across 28 zones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math


class NumentaSDRTokenizer(nn.Module):
    """
    Numenta SDR Tokenizer - Sense Doors / Transduction
    Spiking Neural Network utilizing Sparse Distributed Representations (SDR) and Semantic Folding.
    Compresses continuous raw reality into discrete, high-frequency, fault-tolerant sparse topological maps across 28 zones.
    """

    def __init__(
        self,
        input_dim: int = 512,  # Dimension of raw sensory input
        sdr_dim: int = 2048,  # Dimension of SDR space
        sdr_sparsity: float = 0.02,  # Target sparsity (2% active bits -> ~40 bits active)
        n_topological_zones: int = 28,  # Number of topological zones for semantic folding
        sematic_folding_ratio: float = 0.1,  # Ratio for semantic folding
        threshold_adaptation_rate: float = 0.001,  # Rate for adaptive threshold
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sdr_dim = sdr_dim
        self.sdr_sparsity = sdr_sparsity
        self.n_topological_zones = n_topological_zones
        self.sematic_folding_ratio = sematic_folding_ratio
        self.threshold_adaptation_rate = threshold_adaptation_rate

        # Calculate active bits per zone
        self.active_bits_per_zone = max(
            1, int(sdr_dim * sdr_sparsity // n_topological_zones)
        )

        # Projection to SDR space
        self.sdr_projection = nn.Linear(
            input_dim, sdr_dim, bias=False, device=device, dtype=dtype
        )

        # Semantic folding weights (maps input features to topological zones)
        self.semantic_folding = nn.Parameter(
            torch.randn(input_dim, n_topological_zones, device=device, dtype=dtype)
            * math.sqrt(2.0 / (input_dim + n_topological_zones))
        )

        # Adaptive threshold for sparsity enforcement
        self.register_buffer("threshold", torch.tensor(0.5, device=device, dtype=dtype))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        nn.init.xavier_uniform_(self.sdr_projection.weight)
        # Semantic folding already initialized in constructor

    def forward(
        self, x: torch.Tensor, use_vectorized: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of Numenta SDR Tokenizer

        Args:
            x: Continuous raw reality input (batch, seq_len, input_dim)
            use_vectorized: Whether to use the fast vectorized path (default: True)

        Returns:
            sdr_output: Discrete SDR representation (batch, seq_len, sdr_dim)
            info: Dictionary with diagnostic information
        """
        batch_size, seq_len, _ = x.shape

        # Project to SDR space
        sdr_raw = self.sdr_projection(x)  # (batch, seq_len, sdr_dim)

        # Apply semantic folding to create topological structure
        # Compute zone activations based on semantic folding
        zone_activations = torch.einsum(
            "bsd,dz->bsz", x, self.semantic_folding
        )  # (batch, seq_len, n_topological_zones)
        zone_activations = F.softmax(zone_activations, dim=-1)  # Normalize across zones

        # Apply competitive learning within each zone to enforce sparsity
        if use_vectorized:
            # Vectorized zone processing
            zone_size = math.ceil(self.sdr_dim / self.n_topological_zones)
            padded_sdr_dim = zone_size * self.n_topological_zones

            if padded_sdr_dim > self.sdr_dim:
                sdr_raw_padded = F.pad(sdr_raw, (0, padded_sdr_dim - self.sdr_dim))
            else:
                sdr_raw_padded = sdr_raw

            # Reshape to (batch, seq_len, n_topological_zones, zone_size)
            sdr_zones = sdr_raw_padded.view(
                batch_size, seq_len, self.n_topological_zones, zone_size
            )

            # Expand zone activations for weighting
            # (batch, seq_len, n_topological_zones, 1)
            zone_weight = zone_activations.unsqueeze(-1)

            # Apply soft thresholding to enforce sparsity within zone
            zone_threshold = self.threshold * (
                1.0 + 0.1 * torch.randn_like(self.threshold)
            )  # Add noise for diversity

            # Vectorized softshrink across all zones
            zone_sparse = F.softshrink(sdr_zones, lambd=zone_threshold.item())

            # Weight by zone activation
            weighted_zones = zone_sparse * zone_weight

            # Reshape back to (batch, seq_len, padded_sdr_dim)
            sdr_output_padded = weighted_zones.view(batch_size, seq_len, padded_sdr_dim)

            # Truncate to original sdr_dim
            if padded_sdr_dim > self.sdr_dim:
                sdr_output = sdr_output_padded[:, :, : self.sdr_dim]
            else:
                sdr_output = sdr_output_padded
        else:
            # Legacy fallback path (Python loop)
            sdr_output = torch.zeros_like(sdr_raw)
            zone_size = self.sdr_dim // self.n_topological_zones

            for zone in range(self.n_topological_zones):
                # Get zone-specific activation weights
                zone_weight = zone_activations[
                    :, :, zone : zone + 1
                ]  # (batch, seq_len, 1)

                # Extract zone-specific features from SDR raw
                start_idx = zone * zone_size
                end_idx = min((zone + 1) * zone_size, self.sdr_dim)
                zone_features = sdr_raw[:, :, start_idx:end_idx]

                # Apply soft thresholding to enforce sparsity within zone
                zone_threshold = self.threshold * (
                    1.0 + 0.1 * torch.randn_like(self.threshold)
                )  # Add noise for diversity
                zone_sparse = F.softshrink(zone_features, lambd=zone_threshold.item())

                # Weight by zone activation and accumulate
                # We need to pad zone_sparse back to full sdr_dim to accumulate
                padded_zone_sparse = torch.zeros_like(sdr_raw)
                padded_zone_sparse[:, :, start_idx:end_idx] = zone_sparse
                sdr_output += padded_zone_sparse * zone_weight

        # Enforce global sparsity constraint
        sdr_output = self._enforce_sparsity(sdr_output)

        # Convert to binary spikes (discrete representation)
        sdr_binary = (sdr_output > 0).float()

        # Calculate diagnostic information
        active_ratio = sdr_binary.mean().item()
        info = {
            "active_ratio": active_ratio,
            "target_sparsity": self.sdr_sparsity,
            "sdr_mean": sdr_output.mean().item(),
            "sdr_std": sdr_output.std().item(),
            "zone_entropy": self._compute_zone_entropy(zone_activations),
        }

        return sdr_binary, info

    def _enforce_sparsity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce sparsity constraint using soft thresholding with adaptive threshold
        """
        batch_size, seq_len, _ = x.shape

        # Flatten for threshold calculation
        x_flat = x.reshape(-1)

        # Calculate threshold to achieve target sparsity
        k = int(self.sdr_sparsity * x_flat.shape[0])
        if k > 0:
            _, indices = torch.topk(x_flat.abs(), k, sorted=False)
            threshold = torch.zeros_like(x_flat)
            threshold[indices] = 1.0
            threshold = threshold.view_as(x)

            # Apply adaptive threshold update
            with torch.no_grad():
                current_sparsity = (x_flat.abs() > 0).float().mean()
                error = self.sdr_sparsity - current_sparsity.item()
                self.threshold += self.threshold_adaptation_rate * error
                self.threshold.clamp_(min=0.01, max=0.9)

            return x * threshold
        else:
            return torch.zeros_like(x)

    def _compute_zone_entropy(self, zone_activations: torch.Tensor) -> float:
        """
        Compute entropy of zone activations as measure of semantic folding effectiveness
        """
        # Average across batch and sequence
        avg_activations = zone_activations.mean(dim=(0, 1))  # (n_topological_zones,)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        entropy = -(avg_activations * torch.log(avg_activations + epsilon)).sum().item()
        return entropy

    def compute_overlap(self, sdr1: torch.Tensor, sdr2: torch.Tensor) -> torch.Tensor:
        """
        Compute overlap between two binary SDRs using bitwise AND.
        Vectorized across batch dimension.

        Args:
            sdr1: Binary SDR tensor (batch, seq_len, sdr_dim) or (batch, sdr_dim)
            sdr2: Binary SDR tensor (batch, seq_len, sdr_dim) or (batch, sdr_dim)

        Returns:
            overlap: Number of overlapping active bits
        """
        # Ensure boolean tensors for bitwise operations
        sdr1_bool = sdr1.bool()
        sdr2_bool = sdr2.bool()

        # Bitwise AND for overlap
        overlap = torch.bitwise_and(sdr1_bool, sdr2_bool)

        # Sum across the SDR dimension
        return overlap.sum(dim=-1).float()

    def update_threshold(self, target_sparsity: Optional[float] = None):
        """
        Manually update threshold for homeostasis (Sati function)
        """
        if target_sparsity is not None:
            self.sdr_sparsity = target_sparsity


def create_numenta_sdr_tokenizer(
    input_dim: int = 512,
    sdr_dim: int = 2048,
    sdr_sparsity: float = 0.02,
    n_topological_zones: int = 28,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> NumentaSDRTokenizer:
    """Factory function to create Numenta SDR Tokenizer with sensible defaults"""
    return NumentaSDRTokenizer(
        input_dim=input_dim,
        sdr_dim=sdr_dim,
        sdr_sparsity=sdr_sparsity,
        n_topological_zones=n_topological_zones,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    # Simple test
    tokenizer = create_numenta_sdr_tokenizer(
        input_dim=64, sdr_dim=256, sdr_sparsity=0.02
    )
    x = torch.randn(2, 10, 64)  # batch=2, seq_len=10, input_dim=64
    sdr_output, info = tokenizer(x)
    print(f"Input shape: {x.shape}")
    print(f"SDR output shape: {sdr_output.shape}")
    print(f"Active ratio: {info['active_ratio']:.4f}")
    print(f"Target sparsity: {info['target_sparsity']}")
    print("Numenta SDR Tokenizer test successful!")
