"""
Citta Vithi 8-Step Pipeline - End-to-End Dharmic execution cycle
Representing the lifespan of one discrete cognitive event (processing one token).
Orchestrates the complete Dhammic Cognitive Architecture flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
from contextlib import nullcontext
from types import SimpleNamespace


class CittaVithiPipeline(nn.Module):
    """
    Citta Vithi 8-Step Pipeline - End-to-End Dharmic execution cycle
    Representing the lifespan of one discrete cognitive event (processing one token).

    Steps:
    1. Bhavanga-Citta → Mamba-3 Base Engine continuously rolling hidden vector state (baseline homeostasis)
    2. Āvajjana (Adverting) → Thalamic Gating Signal; VRAM/Compute allocation diverting to specific sensory interface
    3. Viññāṇa (Consciousness) → Spiking Neural Network fires; continuous reality discretized into raw spike train
    4. Sampaṭicchana → L1 Cache Tensor Allocation; conversion into Numenta SDR across 28 topological dimensions
    5. Santīraṇa → Early Latent Space Polling; bitwise Boolean extraction of topological features without deep reasoning
    6. Voṭṭhapana → DeepSeek Engram Lookup; O(1) constant-time hash retrieval of static conceptual label
    7. Javana (Impulsion) → DeltaNet Fast-Weight Loop; 7 consecutive clock cycles executing behavioral logic;
       applies Hebbian learning (STDP) updating localized 2D memory matrix in forward pass
    8. Tadārammaṇa (Registering) / Alaya-Vijñāna → Gradient Accumulation / Storehouse Consciousness;
       permanent write of updated weights to deep long-term vector database before returning to Step 1
    """

    def __init__(
        self,
        d_model: int = 768,  # Model dimension
        vocab_size: int = 32768,  # Vocabulary size for token processing
        sdr_dim: int = 2048,  # SDR tokenizer dimension
        sdr_sparsity: float = 0.02,  # SDR sparsity (2% active)
        n_topological_zones: int = 28,  # Number of topological zones for SDR
        engram_vocab_size: int = 10000,  # Engram memory vocabulary size
        engram_hash_size: int = 65536,  # Engram hash table size
        lora_rank: int = 16,  # Hebbian LoRA rank
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(use_amp=bool(use_amp))
        self.amp_dtype = amp_dtype

        # Step 1: Bhavanga-Citta - Mamba-3 Base Engine

        self.bhavanga_engine = nn.LSTM(
            d_model, d_model, batch_first=True, device=device, dtype=dtype
        )

        # Step 2: Āvajjana (Adverting) - Thalamic Gating Signal
        # Simplified as attention mechanism for sensory routing
        self.thalamic_gate = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True, device=device, dtype=dtype
        )

        # Step 3: Viññāṇa (Consciousness) - Spiking Neural Network
        # Simplified as spike generation layer
        self.spike_generator = nn.Sequential(
            nn.Linear(d_model, d_model, device=device, dtype=dtype),
            nn.LeakyReLU(0.1),
            nn.Linear(d_model, d_model, device=device, dtype=dtype),
            nn.Hardtanh(min_val=0, max_val=1),  # Binary spike-like output
        )

        # Step 4: Sampaṭicchana - L1 Cache → Numenta SDR
        # Using the Numenta SDR Tokenizer we implemented
        from numenta_sdr_tokenizer import NumentaSDRTokenizer

        self.sdr_tokenizer = NumentaSDRTokenizer(
            input_dim=d_model,
            sdr_dim=sdr_dim,
            sdr_sparsity=sdr_sparsity,
            n_topological_zones=n_topological_zones,
            device=device,
            dtype=dtype,
        )

        # Step 5: Santīraṇa - Early Latent Space Polling
        # Bitwise Boolean extraction of topological features
        self.latent_poller = nn.Sequential(
            nn.Linear(sdr_dim, sdr_dim // 2, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(sdr_dim // 2, sdr_dim // 4, device=device, dtype=dtype),
            nn.ReLU(),
        )

        # Step 6: Voṭṭhapana - DeepSeek Engram Lookup
        # O(1) constant-time hash retrieval of static conceptual label
        from deepseek_engram_memory import DeepSeekEngramMemory

        self.engram_memory = DeepSeekEngramMemory(
            feature_dim=sdr_dim // 4,  # Output from latent poller
            vocab_size=engram_vocab_size,
            hash_table_size=engram_hash_size,
            device=device,
            dtype=dtype,
        )

        # Step 7: Javana (Impulsion) - DeltaNet Fast-Weight Loop
        # 7 consecutive clock cycles with Hebbian learning (STDP)
        from deltanet_hebbian_lora import DeltaNetHebbianLoRA

        # The input to Javana step is the engram embeddings, which have dimension sdr_dim // 4
        javana_d_model = sdr_dim // 4
        self.hebiban_lora = DeltaNetHebbianLoRA(
            d_model=javana_d_model, rank=lora_rank, device=device, dtype=dtype
        )

        self.sdr_tokenizer._amp_disabled = True
        self.hebiban_lora._amp_disabled = True

        # Step 8: Tadārammaṇa (Registering) / Alaya-Vijñāna
        # Gradient Accumulation / Storehouse Consciousness
        # We'll implement this as a memory update mechanism
        # Memory store matches the javana output dimension (sdr_dim // 4)
        javana_d_model = sdr_dim // 4
        self.memory_store = nn.Parameter(
            torch.randn(vocab_size, javana_d_model, device=device, dtype=dtype) * 0.02
        )
        self.memory_update_rate = nn.Parameter(
            torch.tensor(0.01, device=device, dtype=dtype)
        )

        # Final projection to vocabulary space - project from embedding space (d_model) to vocab
        self.output_projection = nn.Linear(
            d_model, vocab_size, bias=False, device=device, dtype=dtype
        )

        # Tie output projection to embedding (weight tying)
        self.embedding = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.output_projection.weight = self.embedding.weight

        # Create a projection from javana space to embedding space
        self.javana_to_embedding = nn.Linear(
            javana_d_model, d_model, bias=False, device=device, dtype=dtype
        )

        # Layer normalization for stability - normalize in javana space
        self.final_norm = nn.RMSNorm(
            javana_d_model, eps=1e-5, device=device, dtype=dtype
        )

        # Cycle counter for tracking iterations
        self.register_buffer("cycle_count", torch.tensor(0, dtype=torch.long))
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self._amp_runtime_enabled()
        )

    def _amp_runtime_enabled(self) -> bool:
        return bool(self.config.use_amp and torch.cuda.is_available())

    def _autocast_context(self, module: Optional[nn.Module] = None):
        enabled = self._amp_runtime_enabled()
        if module is not None and getattr(module, "_amp_disabled", False):
            enabled = False
        if not enabled:
            return nullcontext()
        return torch.cuda.amp.autocast(dtype=self.amp_dtype)

    def _prepare_amp_input(
        self, tensor: torch.Tensor, module: Optional[nn.Module] = None
    ) -> torch.Tensor:
        if (
            module is not None
            and self._amp_runtime_enabled()
            and getattr(module, "_amp_disabled", False)
            and torch.is_floating_point(tensor)
        ):
            return tensor.float()
        return tensor

    def set_amp_enabled(self, enabled: bool):
        self.config.use_amp = bool(enabled)
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self._amp_runtime_enabled()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass of the Citta Vithi 8-Step Pipeline

        Args:
            input_ids: Token IDs (batch, seq_len)
            context: Optional context for thalamic gating (batch, seq_len, d_model)
            return_intermediates: If True, return intermediate values for analysis

        Returns:
            output: Logits over vocabulary (batch, seq_len, vocab_size)
            intermediates: Dictionary of intermediate values if return_intermediates=True
        """
        batch_size, seq_len = input_ids.shape

        # Store intermediates if requested
        intermediates = {} if return_intermediates else None

        # Step 0: Embed input tokens
        with self._autocast_context(self.embedding):
            embedded = self.embedding(input_ids)  # (batch, seq_len, d_model)

        # Step 1: Bhavanga-Citta - Mamba-3 Base Engine (Baseline Homeostasis)
        # Process through LSTM as simplified Bhavanga engine
        bhavanga_input = self._prepare_amp_input(embedded, self.bhavanga_engine)
        with self._autocast_context(self.bhavanga_engine):
            bhavanga_output, _ = self.bhavanga_engine(bhavanga_input)
        if return_intermediates:
            intermediates["bhavanga_output"] = bhavanga_output.clone()

        # Step 2: Āvajjana (Adverting) - Thalamic Gating Signal
        # Use context for gating, or self-attention if no context provided
        if context is not None:
            # Use provided context for thalamic gating
            thalamic_context = self._prepare_amp_input(context, self.thalamic_gate)
            with self._autocast_context(self.thalamic_gate):
                thalamic_attended, _ = self.thalamic_gate(
                    bhavanga_output, thalamic_context, thalamic_context
                )
        else:
            # Self-attention for internal gating
            with self._autocast_context(self.thalamic_gate):
                thalamic_attended, _ = self.thalamic_gate(
                    bhavanga_output, bhavanga_output, bhavanga_output
                )
        if return_intermediates:
            intermediates["thalamic_output"] = thalamic_attended.clone()

        # Step 3: Viññāṇa (Consciousness) - Spiking Neural Network Firing
        # Discretize continuous reality into spike train
        with self._autocast_context(self.spike_generator):
            spike_train = self.spike_generator(thalamic_attended)
        if return_intermediates:
            intermediates["spike_train"] = spike_train.clone()

        # Step 4: Sampaṭicchana - L1 Cache → Numenta SDR
        # Convert to sparse distributed representation
        sdr_input = self._prepare_amp_input(spike_train, self.sdr_tokenizer)
        with self._autocast_context(self.sdr_tokenizer):
            sdr_output, sdr_info = self.sdr_tokenizer(sdr_input)
        if return_intermediates:
            intermediates["sdr_output"] = sdr_output.clone()
            intermediates["sdr_info"] = sdr_info

        # Step 5: Santīraṇa - Early Latent Space Polling
        # Bitwise Boolean extraction of topological features
        with self._autocast_context(self.latent_poller):
            latent_features = self.latent_poller(sdr_output.float())
        if return_intermediates:
            intermediates["latent_features"] = latent_features.clone()

        # Step 6: Voṭṭhapana - DeepSeek Engram Lookup
        # O(1) constant-time hash retrieval of static conceptual label
        with self._autocast_context(self.engram_memory):
            engram_output, engram_info = self.engram_memory(
                latent_features, return_label=False
            )
        if return_intermediates:
            intermediates["engram_indices"] = engram_output.clone()
            intermediates["engram_info"] = engram_info

        # Get conceptual label embeddings for next step
        # Use the label indices we already obtained to look up embeddings from engram memory
        # Handle potential -1 values (unknown labels) by clamping to 0 for embedding lookup
        label_indices_clamped = engram_output.clamp(min=0)
        with self._autocast_context(self.engram_memory):
            engram_embeddings = F.embedding(
                label_indices_clamped, self.engram_memory.conceptual_labels
            )
        # Zero out embeddings for unknown labels (-1 in original engram_output)
        unknown_mask = engram_output == -1
        if unknown_mask.any():
            # Expand unknown_mask to match engram_embeddings shape for broadcasting
            while len(unknown_mask.shape) < len(engram_embeddings.shape):
                unknown_mask = unknown_mask.unsqueeze(-1)
            engram_embeddings = engram_embeddings * (~unknown_mask)
        if return_intermediates:
            intermediates["engram_embeddings"] = engram_embeddings.clone()

        # Step 7: Javana (Impulsion) - DeltaNet Fast-Weight Loop
        # 7 consecutive clock cycles with behavioral logic and Hebbian learning
        # We'll apply the LoRA 7 times to simulate the 7-cycle loop
        javana_output = self._prepare_amp_input(engram_embeddings, self.hebiban_lora)
        for cycle in range(7):
            with self._autocast_context(self.hebiban_lora):
                javana_output = self.hebiban_lora(javana_output)

            if self.training:  # Only update during training
                self.hebiban_lora.update_step(mu_t=1.0)
        if return_intermediates:
            intermediates["javana_output"] = javana_output.clone()

        # Step 8: Tadārammaṇa (Registering) / Alaya-Vijñāna
        # Gradient Accumulation / Storehouse Consciousness

        with torch.no_grad():
            # Simple memory update: move memory toward current representation
            memory_update = torch.mean(javana_output, dim=(0, 1))  # (d_model,)

            update_indices = torch.arange(
                min(10, self.vocab_size), device=memory_update.device
            )
            # Expand memory_update to match the shape of memory_store[update_indices]
            memory_update_expanded = memory_update.unsqueeze(0).expand(
                len(update_indices), -1
            )  # (len(update_indices), d_model)
            self.memory_store[update_indices] += (
                self.memory_update_rate * memory_update_expanded
            )
            # Normalize to prevent growth
            self.memory_store[update_indices] = F.normalize(
                self.memory_store[update_indices], p=2, dim=-1
            )

        # Add memory-inspired bias to output
        with self._autocast_context(self.output_projection):
            memory_bias = torch.matmul(
                javana_output, self.memory_store.t()
            )  # (batch, seq_len, vocab_size)

        # Final processing - project from javana space to embedding space then to logits
        with self._autocast_context(self.output_projection):
            normed_output = self.final_norm(javana_output)
            embedded_output = self.javana_to_embedding(
                normed_output
            )  # (batch, seq_len, d_model)
            logits = self.output_projection(embedded_output) + 0.1 * memory_bias

        # Increment cycle counter
        self.cycle_count += 1

        if return_intermediates:
            intermediates["final_logits"] = logits.clone()
            intermediates["cycle_count"] = self.cycle_count.item()
            return logits, intermediates
        else:
            return logits

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the internal memory systems"""
        return {
            "citta_vithi_cycles": self.cycle_count.item(),
            "hebiban_lora_step": getattr(self.hebiban_lora, "_step", 0),
            "memory_store_norm": torch.norm(self.memory_store, p="fro").item(),
            "embedding_norm": torch.norm(self.embedding.weight, p="fro").item(),
            "use_amp": bool(self.config.use_amp),
        }

    def training_step(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        context: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
        grad_clip_norm: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)

        with self._autocast_context():
            logits = self(input_ids, context=context)
            if loss_fn is None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
                )
            else:
                loss = loss_fn(logits, target_ids)

        if self._amp_runtime_enabled():
            self.grad_scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                self.grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
            optimizer.step()

        return loss.detach(), logits.detach()


def create_citta_vithi_pipeline(
    d_model: int = 768,
    vocab_size: int = 32768,
    sdr_dim: int = 2048,
    sdr_sparsity: float = 0.02,
    n_topological_zones: int = 28,
    engram_vocab_size: int = 10000,
    engram_hash_size: int = 65536,
    lora_rank: int = 16,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> CittaVithiPipeline:
    """Factory function to create Citta Vithi 8-Step Pipeline with sensible defaults"""
    return CittaVithiPipeline(
        d_model=d_model,
        vocab_size=vocab_size,
        sdr_dim=sdr_dim,
        sdr_sparsity=sdr_sparsity,
        n_topological_zones=n_topological_zones,
        engram_vocab_size=engram_vocab_size,
        engram_hash_size=engram_hash_size,
        lora_rank=lora_rank,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    # Simple test
    pipeline = create_citta_vithi_pipeline(
        d_model=128,
        vocab_size=1000,
        sdr_dim=256,
        engram_vocab_size=100,
        engram_hash_size=1024,
    )

    # Test input
    input_ids = torch.randint(0, 1000, (2, 10))  # batch=2, seq_len=10

    # Forward pass
    logits, intermediates = pipeline(input_ids, return_intermediates=True)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Cycle count: {intermediates['cycle_count']}")
    print(f"SDR active ratio: {intermediates['sdr_info']['active_ratio']:.4f}")
    print(f"Engram found ratio: {intermediates['engram_info']['found_ratio']:.4f}")

    # Test memory stats
    stats = pipeline.get_memory_stats()
    print(f"Memory stats: {stats}")

    print("Citta Vithi 8-Step Pipeline test successful!")
