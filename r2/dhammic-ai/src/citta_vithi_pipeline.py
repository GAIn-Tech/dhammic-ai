"""
Citta Vithi Pipeline v4 — Fully-Fused Dharmic Architecture.

Architecture: SDR Embedding+Engram → [Mamba-3 SSD + SwiGLU]×N + HTM(layer K) → LM Head

All hot-path ops go through fused Triton kernels:
  - SDR top-k / Gumbel STE     : fused_sdr_topk (K7)
  - Mamba-3 block              : fused inproj/RoPE/trapezoidal/SSD (K1-K5)
  - SwiGLU MLP                 : FusedSwiGLUMLP (K6)
  - Engram column gather       : FusedEngramGatherModule (K8)
  - LM-head + cross-entropy    : FusedLMHeadXentModule (K9, train.py)

Production-only: no eager fallback paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
from types import SimpleNamespace

from mamba_base_engine import Mamba3Block, Mamba3Config, Mamba3BaseEngine
from kernels import (
    FusedRMSNormModule,
    FusedSDRTopKModule,
    FusedSwiGLUMLP,
    FusedEngramGatherModule,  # legacy HTMEngram support — kept for back-compat
)
from htm_layer import FaithfulHTM
from engram_layer import FaithfulEngram


# ── SDR Embedding (Sampaticchana — Dense→Sparse→Dense) ────────────────────
class SDREmbedding(nn.Module):
    """
    Dharmic Step 4 (Sampaticchana): Dense→Sparse→Dense with fused top-k.
    Uses FusedSDRTopKModule (K7) for the Gumbel-STE / eval-topk path —
    same operator at training and eval, no Python dual-branch.
    """

    def __init__(self, vocab_size: int, d_model: int, sdr_dim: int = 2048,
                 k_active: int = 40, chunk_size: int = 256,
                 engram: Optional["FaithfulEngram"] = None):
        super().__init__()
        self.dense_embed = nn.Embedding(vocab_size, d_model)
        self.sdr_dim = sdr_dim
        self.k_active = k_active
        self.chunk_size = chunk_size
        self.to_sdr = nn.Linear(d_model, sdr_dim, bias=False)
        self.from_sdr = nn.Linear(sdr_dim, d_model, bias=False)
        # K7: fused top-k + Gumbel + STE / eval-topk in one operator
        self.topk = FusedSDRTopKModule(k_active=k_active, temperature=0.5)
        # Optional DeepSeek Engram (n-gram hash, token-id-keyed) injected here
        # because token_ids are available at the embedding layer and Engram
        # *needs* them. Faithful to arXiv:2601.07372 (see docs/engram_fidelity.md).
        self.engram = engram

    def _chunk_branch(self, dense_chunk: torch.Tensor) -> torch.Tensor:
        """Per-chunk sdr-width branch. Encapsulated so checkpoint() can
        recompute the (B, chunk, sdr_dim) tensors at backward instead of
        storing them — kills the dominant per-token VRAM leak.
        """
        logits = self.to_sdr(dense_chunk)
        sdr = self.topk(logits)
        return self.from_sdr(sdr)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        dense = self.dense_embed(token_ids)
        T = dense.shape[1]
        # SDR branch — single-shot if short, otherwise chunk+checkpoint so the
        # (B,T,sdr_dim) intermediate is never held full-width.
        if T <= self.chunk_size:
            sdr_out = self._chunk_branch(dense)
        else:
            outs = []
            for s in range(0, T, self.chunk_size):
                e = min(s + self.chunk_size, T)
                ck = dense[:, s:e]
                outs.append(torch.utils.checkpoint.checkpoint(
                    self._chunk_branch, ck, use_reentrant=False,
                ))
            sdr_out = torch.cat(outs, dim=1)
        out = dense + sdr_out
        # DeepSeek Engram residual (token-id n-gram hash) — always applied
        # when present, regardless of branch.
        if self.engram is not None:
            out = out + self.engram(out, token_ids)
        return out


# ── HTM Layer (Numenta HTM — Spatial Pooler + Temporal Memory) ────────────
class HTMLayer(nn.Module):
    """
    Numenta-faithful HTM (SP + TM) wrapped with a gated residual head.

    NOT a hybrid: the retrieval core is FaithfulHTM (soft-permanence Numenta
    HTM, see docs/htm_fidelity.md). The surrounding gate + out_proj + residual
    add is the integration layer that lets HTM compose with the LM stream.

    Naming note: the old `HTMEngram` class conflated HTM (Numenta) with
    Engram (DeepSeek arXiv:2601.07372) which are SEPARATE concepts. They are
    now implemented separately — HTM here, Engram in SDREmbedding.
    """

    def __init__(self, d_model: int, n_columns: int = 4096,
                 cells_per_column: int = 8, k_active_columns: int = 20,
                 segments_per_cell: int = 2, synapses_per_segment: int = 16,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.d_model = d_model
        self.htm = FaithfulHTM(
            d_input=d_model,
            n_columns=n_columns,
            cells_per_column=cells_per_column,
            k_active_columns=k_active_columns,
            segments_per_cell=segments_per_cell,
            synapses_per_segment=synapses_per_segment,
        )
        # Residual head: gate + out_proj (composes HTM with LM stream)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.gate.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        retrieved = self.htm(x)             # FaithfulHTM (B, T, D)
        g = torch.sigmoid(self.gate(x))
        return x + g * self.out_proj(retrieved)


# ── Hebbian LoRA (Javana — lightweight side-channel) ──────────────────────
class HebbianLoRA(nn.Module):
    """
    Dharmic Step 7 (Javana): Single-pass LoRA with Hebbian trace update.
    Lightweight: one matmul pair + trace update. No multi-cycle loop.
    """

    def __init__(self, d_model: int, rank: int = 8):
        super().__init__()
        self.W_A = nn.Parameter(torch.randn(d_model, rank) * (1.0 / math.sqrt(d_model)))
        self.W_B = nn.Parameter(torch.zeros(rank, d_model))
        self.gate = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.gate.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W_A  # (B, T, rank)
        lora_out = h @ self.W_B  # (B, T, d_model)
        g = torch.sigmoid(self.gate(x))
        return x + g * lora_out


# ── MambaBlock (backbone layer) ───────────────────────────────────────────
class MambaBlock(nn.Module):
    """Mamba-3 SSD + fused SwiGLU + optional fused Engram / Hebbian injection."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 3,
                 n_heads: int = 8, chunk_size: int = 256,
                 engram: Optional["HTMLayer"] = None,
                 hebbian: Optional[HebbianLoRA] = None):
        super().__init__()
        # K1: fused RMSNorms
        self.norm1 = FusedRMSNormModule(d_model, eps=1e-5)
        self.norm2 = FusedRMSNormModule(d_model, eps=1e-5)
        self.mamba = Mamba3BaseEngine(Mamba3Config(
            d_model=d_model, d_state=d_state, d_conv=4, expand=expand,
            n_heads=n_heads, chunk_size=chunk_size,
        ))
        # K6: fused SwiGLU MLP (single op replaces up + chunk + silu*val + down)
        mlp_dim = int(d_model * 2.5)
        self.mlp = FusedSwiGLUMLP(d_model=d_model, mlp_dim=mlp_dim, bias=False)
        self.engram = engram
        self.hebbian = hebbian

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mamba SSM
        x = x + self.mamba(self.norm1(x))
        # SwiGLU MLP (K6 fused)
        x = x + self.mlp(self.norm2(x))
        # Dharmic injection (engram + hebbian at this layer only)
        if self.engram is not None:
            x = self.engram(x)
        if self.hebbian is not None:
            x = self.hebbian(x)
        return x


# ── CittaVithiPipeline v3 ────────────────────────────────────────────────
class CittaVithiPipeline(nn.Module):
    """
    Citta Vithi v3 — Inline Dharmic Architecture

    Flow: SDR Embedding → [Mamba-3 SSD + SwiGLU] × N → LM Head
    Engram injected at layer `engram_layer`. Hebbian LoRA at same layer.
    All dharmic components are residual additions — zero extra pathway.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        mamba_expand: int = 3,
        n_heads: int = 8,
        chunk_size: int = 256,
        vocab_size: int = 32768,
        sdr_dim: int = 2048,
        sdr_k_active: int = 40,
        engram_n_columns: int = 4096,
        engram_cells_per_col: int = 8,
        engram_k_active: int = 20,
        engram_layer: int = 2,      # which layer gets engram injection
        lora_rank: int = 8,
        use_grad_checkpoint: bool = False,
        # Legacy params (ignored, kept for API compat)
        sdr_sparsity: float = 0.02,
        n_topological_zones: int = 14,
        engram_vocab_size: int = 4096,
        engram_hash_size: int = 16384,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.use_grad_checkpoint = use_grad_checkpoint
        self.config = SimpleNamespace(use_amp=False)

        # ── DeepSeek Engram (token-id keyed, lives in embedding) ─────────
        # Engram is built first so it can be wired into SDREmbedding.
        # n-gram size 3 is the paper default; multi-head primes ensure low
        # collision rate. Per-ngram vocab is conservative for small models —
        # 8x the LM vocab gives ~1% collision rate at HALF-fill.
        deepseek_engram = FaithfulEngram(
            d_model=d_model,
            tokenizer_vocab_size=vocab_size,
            d_value=d_model,
            max_ngram_size=3,
            n_heads_per_ngram=max(2, d_model // 32),  # 2 heads at d=64, 4 at d=128
            engram_vocab_size_per_ngram=[8 * vocab_size, 8 * vocab_size],
        )

        # ── SDR Embedding (with Engram residual injected at the embedding)
        self.embedding = SDREmbedding(
            vocab_size=vocab_size, d_model=d_model,
            sdr_dim=sdr_dim, k_active=sdr_k_active,
            chunk_size=chunk_size,
            engram=deepseek_engram,
        )

        # ── Numenta HTM (hidden-state keyed, lives at one backbone layer)
        # Faithful Spatial Pooler + Temporal Memory. Separate from Engram.
        htm = HTMLayer(
            d_model=d_model, n_columns=engram_n_columns,
            cells_per_column=engram_cells_per_col,
            k_active_columns=engram_k_active,
        )
        hebbian = HebbianLoRA(d_model=d_model, rank=lora_rank)

        # ── Backbone: N layers of Mamba-3 SSD + SwiGLU ──────────────────
        # HTM injected at one layer (existing engram_layer position); Engram
        # already injected at the embedding step. The two memory systems are
        # SEPARATE: Engram = content-addressable via token-id n-gram hash,
        # HTM = sequence-predictive via cortical minicolumns. Faithful to
        # their respective papers — no longer the HTMEngram hybrid.
        layers = []
        for i in range(n_layers):
            inject_htm = htm if i == min(engram_layer, n_layers - 1) else None
            inject_hebbian = hebbian if i == min(engram_layer, n_layers - 1) else None
            layers.append(MambaBlock(
                d_model=d_model, d_state=d_state, expand=mamba_expand,
                n_heads=n_heads, chunk_size=chunk_size,
                engram=inject_htm, hebbian=inject_hebbian,
            ))
        self.backbone = nn.ModuleList(layers)

        # ── Output ───────────────────────────────────────────────────────
        # K1: fused RMSNorm. lm_head is owned by FusedLMHeadXentModule
        # in train.py — the model returns the post-norm *hidden* tensor.
        # Use ``compute_logits()`` to materialize logits when needed
        # (autoregressive generation, eval).
        self.final_norm = FusedRMSNormModule(d_model, eps=1e-5)

        # Stats
        self.register_buffer("cycle_count", torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        input_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """Forward pass returning post-norm *hidden* (B, T, D).

        train.py pairs this with ``FusedLMHeadXentModule`` for the
        train-time loss (K9), which never materializes the (M, V) logits
        tensor. For inference paths that need logits, call
        ``self.compute_logits(hidden)``.
        """
        # SDR Embedding (Dharmic step 0+4 fused)
        x = self.embedding(input_ids)

        # Backbone: fused Mamba-3 + SwiGLU + (optional) engram/hebbian
        for layer in self.backbone:
            x = layer(x)

        # Final norm — return hidden, not logits (see docstring)
        hidden = self.final_norm(x)

        self.cycle_count += 1

        if return_intermediates:
            return hidden, {
                "final_hidden": hidden.clone(),
                "cycle_count": self.cycle_count.item(),
            }
        return hidden

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Materialize logits = hidden @ tied_embedding_weight.T.

        Used by inference paths (autoregressive generation, eval). The
        training loss path uses ``FusedLMHeadXentModule`` instead, which
        avoids materializing the (M, V) logits tensor.
        """
        return F.linear(hidden, self.embedding.dense_embed.weight)

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "citta_vithi_cycles": self.cycle_count.item(),
            "n_layers": self.n_layers,
            "d_model": self.d_model,
        }


def create_citta_vithi_pipeline(
    d_model: int = 128, n_layers: int = 4, d_state: int = 16,
    mamba_expand: int = 3, n_heads: int = 8, chunk_size: int = 256,
    vocab_size: int = 32768, sdr_dim: int = 2048, sdr_k_active: int = 40,
    engram_n_columns: int = 4096, engram_cells_per_col: int = 8,
    engram_k_active: int = 20, engram_layer: int = 2, lora_rank: int = 8,
    use_grad_checkpoint: bool = False, **kwargs,
) -> CittaVithiPipeline:
    return CittaVithiPipeline(
        d_model=d_model, n_layers=n_layers, d_state=d_state,
        mamba_expand=mamba_expand, n_heads=n_heads, chunk_size=chunk_size,
        vocab_size=vocab_size, sdr_dim=sdr_dim, sdr_k_active=sdr_k_active,
        engram_n_columns=engram_n_columns, engram_cells_per_col=engram_cells_per_col,
        engram_k_active=engram_k_active, engram_layer=engram_layer,
        lora_rank=lora_rank, use_grad_checkpoint=use_grad_checkpoint,
    )


if __name__ == "__main__":
    pipeline = create_citta_vithi_pipeline(
        d_model=128, n_layers=2, d_state=8, n_heads=4,
        vocab_size=1000, sdr_dim=256, sdr_k_active=10,
        engram_n_columns=512, engram_cells_per_col=4, engram_k_active=10,
        chunk_size=256,
    ).cuda()
    input_ids = torch.randint(0, 1000, (2, 256), device="cuda")
    hidden = pipeline(input_ids)
    print(f"Input: {input_ids.shape}, Hidden: {hidden.shape}")
    print(f"Params: {sum(p.numel() for p in pipeline.parameters()) / 1e6:.2f}M")

    # Gradient test via fused LM head + xent
    from kernels import FusedLMHeadXentModule
    pipeline.train()
    target = torch.randint(0, 1000, (2, 256), device="cuda")
    head = FusedLMHeadXentModule(
        d_model=128, vocab_size=1000,
        weight=pipeline.embedding.dense_embed.weight,
    ).cuda()
    loss = head(hidden, target)
    loss.backward()
    print(f"SDR to_sdr grad: {pipeline.embedding.to_sdr.weight.grad.norm():.4f}")
    print("Citta Vithi v4 test PASSED!")
