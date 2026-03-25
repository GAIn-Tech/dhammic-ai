"""
Tests for Physical and Thermodynamic Optimizations component
"""

import torch
import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from physical_thermodynamic_optimizations import (
    PrincipleOfLeastAction,
    VectorizationNumentaSDR,
    ParallelizationSmallWorldNetworks,
    DataFlowAsynchronousSpiking,
    GradientsEnergyBasedModels,
    apply_physical_thermodynamic_optimizations,
    sparsify_tensor,
    sparse_overlap,
    local_attention_window,
    event_driven_update,
)


def test_principle_of_least_action_sparsity():
    """Test sparsity application"""
    x = torch.randn(100)
    x_sparse = PrincipleOfLeastAction.apply_sparsity(x, 0.02)

    nonzero_orig = torch.count_nonzero(x).item()
    nonzero_sparse = torch.count_nonzero(x_sparse).item()

    # Should be much fewer nonzeros in sparse version
    assert nonzero_sparse < nonzero_orig
    # Should be approximately 2% of original
    expected_nonzero = int(100 * 0.02)
    assert abs(nonzero_sparse - expected_nonzero) <= 5  # Allow some variance
    print("✓ Principle of Least Action sparsity test passed")


def test_principle_of_least_action_bitwise():
    """Test bitwise approximation"""
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    x_binarized = PrincipleOfLeastAction.bitwise_approximation(x, 0.0)

    expected = torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0])
    assert torch.allclose(x_binarized, expected)
    print("✓ Principle of Least Action bitwise approximation test passed")


def test_vectorization_numenta_sdr_overlap():
    """Test sparse topological overlap"""
    # Create two sparse vectors with some overlap
    a = torch.zeros(100)
    b = torch.zeros(100)
    a[::10] = 1.0  # Indices 0,10,20,30,40,50,60,70,80,90
    b[::15] = 1.0  # Indices 0,15,30,45,60,75,90

    overlap = VectorizationNumentaSDR.sparse_topological_overlap(a, b)
    # Overlap at indices 0,30,60,90 = 4 elements
    assert overlap.item() == 4.0
    print("✓ Vectorization Numenta SDR overlap test passed")


def test_vectorization_numenta_sdr_folding():
    """Test semantic topological folding"""
    # Create a vector that can be evenly divided
    x = torch.randn(96)  # Divisible by 24, 32, 48, etc.
    folded = VectorizationNumentaSDR.sematic_topological_folding(x, n_folds=24)

    # Should have length 96/24 = 4
    assert folded.shape == (4,)
    print("✓ Vectorization Numenta SDR folding test passed")


def test_parallelization_small_world_local_attention():
    """Test local attention"""
    batch_size, seq_len, dim = 2, 10, 8
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    v = torch.randn(batch_size, seq_len, dim)

    local_out = ParallelizationSmallWorldNetworks.local_attention(
        q, k, v, window_size=4
    )

    assert local_out.shape == (batch_size, seq_len, dim)
    print("✓ Parallelization Small World Networks local attention test passed")


def test_parallelization_small_world_clustering_mask():
    """Test fractal clustering mask"""
    mask = ParallelizationSmallWorldNetworks.fractal_clustering_mask(
        size=16, cluster_sizes=[2, 4, 8]
    )

    assert mask.shape == (16, 16)
    # Should be binary (0s and 1s)
    assert torch.all((mask == 0) | (mask == 1))
    # Should have some connections (not all zeros)
    assert torch.sum(mask) > 0
    print("✓ Parallelization Small World Networks clustering mask test passed")


def test_dataflow_asynchronous_spiking_event_driven():
    """Test event-driven update detection"""
    state1 = torch.tensor([1.0, 2.0, 3.0])
    state2 = torch.tensor([1.0, 2.0, 3.0])  # Identical

    should_update, change = DataFlowAsynchronousSpiking.event_driven_update(
        state2, state1, threshold=0.1
    )
    assert not should_update
    assert change == 0.0

    state3 = torch.tensor([1.0, 2.0, 4.0])  # Changed
    should_update, change = DataFlowAsynchronousSpiking.event_driven_update(
        state3, state1, threshold=0.1
    )
    assert should_update
    assert change == 1.0
    print("✓ Data Flow Asynchronous Spiking event-driven update test passed")


def test_dataflow_asynchronous_spiking_spike_generation():
    """Test sparse spike generation"""
    membrane = torch.tensor([0.5, 1.2, 0.8, 1.5])
    spikes, new_potential = DataFlowAsynchronousSpiking.sparse_spike_generation(
        membrane, threshold=1.0, decay=0.9
    )

    # Should spike at indices 1 and 3 (values >= 1.0)
    expected_spikes = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.allclose(spikes, expected_spikes)

    # Those that spiked should be reset to 0, others decayed
    expected_potential = torch.tensor(
        [
            0.5 * 0.9,  # 0.45 - decayed
            0.0,  # 0.0 - reset after spike
            0.8 * 0.9,  # 0.72 - decayed
            0.0,  # 0.0 - reset after spike
        ]
    )
    assert torch.allclose(new_potential, expected_potential)
    print("✓ Data Flow Asynchronous Spiking spike generation test passed")


def test_gradients_energy_based_models_hopfield():
    """Test Hopfield network energy"""
    states = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]])  # 2 batches, 3 units
    weights = torch.tensor([[0.0, 0.5, -0.3], [0.5, 0.0, 0.2], [-0.3, 0.2, 0.0]])

    energy = GradientsEnergyBasedModels.hopfield_energy(states, weights)

    # Manual calculation for first batch [1, -1, 1]:
    # E = -1/2 * [1,-1,1] * [[0,0.5,-0.3],[0.5,0,0.2],[-0.3,0.2,0]] * [1,-1,1]^T
    # E = -1/2 * [1,-1,1] * [0*1 + 0.5*-1 + -0.3*1, 0.5*1 + 0*-1 + 0.2*1, -0.3*1 + 0.2*-1 + 0*1]
    # E = -1/2 * [1,-1,1] * [-0.8, 0.7, -0.5]
    # E = -1/2 * (1*-0.8 + -1*0.7 + 1*-0.5) = -1/2 * (-0.8 -0.7 -0.5) = -1/2 * (-2.0) = 1.0

    assert len(energy) == 2
    assert abs(energy[0].item() - 1.0) < 1e-5
    print("✓ Gradients Energy Based Models Hopfield energy test passed")


def test_gradients_energy_based_models_ising():
    """Test Ising model energy"""
    spins = torch.tensor([[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]])  # 2 batches
    coupling = torch.tensor([[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
    external_field = torch.tensor([0.2, -0.3, 0.1])

    energy = GradientsEnergyBasedModels.ising_model_energy(
        spins, coupling, external_field
    )

    # Manual calculation for first batch [1, -1, 1]:
    # Interaction: -0.5 * sum_{i,j} J_ij * s_i * s_j (excluding i=j)
    # J_01*s0*s1 = 0.5*1*-1 = -0.5
    # J_02*s0*s2 = 0.0*1*1 = 0.0
    # J_10*s1*s0 = 0.5*-1*1 = -0.5
    # J_12*s1*s2 = 0.5*-1*1 = -0.5
    # J_20*s2*s0 = 0.0*1*1 = 0.0
    # J_21*s2*s1 = 0.5*1*-1 = -0.5
    # Sum = -0.5 + 0.0 -0.5 -0.5 + 0.0 -0.5 = -2.0
    # Interaction energy = -0.5 * (-2.0) = 1.0

    # Field: -sum h_i * s_i = -(0.2*1 + -0.3*-1 + 0.1*1) = -(0.2 + 0.3 + 0.1) = -0.6

    # Total = 1.0 - 0.6 = 0.4

    assert len(energy) == 2
    assert abs(energy[0].item() - 0.4) < 1e-5
    print("✓ Gradients Energy Based Models Ising energy test passed")


def test_gradients_energy_based_models_gradient_descent():
    """Test energy gradient descent"""

    # Simple quadratic energy function: E(x) = x^2
    # Minimum at x = 0
    def energy_func(x):
        return torch.sum(x**2, dim=-1)

    start = torch.tensor([5.0])  # Start far from minimum
    result = GradientsEnergyBasedModels.energy_gradient_descent(
        start, energy_func, learning_rate=0.1, steps=20
    )

    # Should be close to 0
    assert abs(result.item()) < 0.1
    print("✓ Gradients Energy Based Models gradient descent test passed")


def test_utility_functions():
    """Test utility functions"""
    # Test sparsify_tensor
    x = torch.randn(50)
    x_sparse = sparsify_tensor(x, 0.1)
    nonzero_orig = torch.count_nonzero(x).item()
    nonzero_sparse = torch.count_nonzero(x_sparse).item()
    assert nonzero_sparse < nonzero_orig

    # Test sparse_overlap
    a = torch.zeros(20)
    b = torch.zeros(20)
    a[::5] = 1.0
    b[::4] = 1.0
    overlap = sparse_overlap(a, b)
    # Overlap at indices 0,20 (but 20 is out of bounds for size 20) so just 0
    # Actually: a has indices 0,5,10,15; b has 0,4,8,12,16
    # Overlap only at index 0
    assert overlap.item() == 1.0

    # Test local_attention_window
    q = torch.randn(1, 6, 8)
    k = torch.randn(1, 6, 8)
    v = torch.randn(1, 6, 8)
    local_out = local_attention_window(q, k, v, window_size=3)
    assert local_out.shape == (1, 6, 8)

    # Test event_driven_update
    state1 = torch.ones(3)
    state2 = torch.ones(3) * 2.0
    should_update, change = event_driven_update(state2, state1, threshold=0.5)
    assert should_update
    assert change == 1.0

    print("✓ Utility functions test passed")


def test_apply_physical_thermodynamic_optimizations():
    """Test applying optimizations to a model"""
    # Simple linear model
    model = torch.nn.Linear(10, 5)
    original_weight = model.weight.data.clone()

    # Apply optimizations
    optimized_model = apply_physical_thermodynamic_optimizations(model, "moderate")

    # Should still be a linear model
    assert isinstance(optimized_model, torch.nn.Linear)
    assert optimized_model.in_features == 10
    assert optimized_model.out_features == 5

    # Weights should be unchanged (since it's conceptual)
    assert torch.allclose(optimized_model.weight.data, original_weight)
    print("✓ Apply physical and thermodynamic optimizations test passed")


if __name__ == "__main__":
    test_principle_of_least_action_sparsity()
    test_principle_of_least_action_bitwise()
    test_vectorization_numenta_sdr_overlap()
    test_vectorization_numenta_sdr_folding()
    test_parallelization_small_world_local_attention()
    test_parallelization_small_world_clustering_mask()
    test_dataflow_asynchronous_spiking_event_driven()
    test_dataflow_asynchronous_spiking_spike_generation()
    test_gradients_energy_based_models_hopfield()
    test_gradients_energy_based_models_ising()
    test_gradients_energy_based_models_gradient_descent()
    test_utility_functions()
    test_apply_physical_thermodynamic_optimizations()
    print("All Physical and Thermodynamic Optimizations tests passed!")
