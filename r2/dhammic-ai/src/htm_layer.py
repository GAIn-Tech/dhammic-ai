"""
Faithful Numenta HTM (Spatial Pooler + Temporal Memory) as a PyTorch nn.Module.

This is a faithful adaptation of Numenta's HTM algorithm
(see ``htm.core`` community fork — github.com/htm-community/htm.core —
``py/htm/algorithms/spatial_pooler.py`` and ``temporal_memory.py``;
Hawkins & Ahmad 2016 "Why Neurons Have Thousands of Synapses") into a
differentiable layer suitable for end-to-end LM pretraining.

What is FAITHFUL:
  - Spatial Pooler: potential-synapse permanences, overlap, boost factor
    homeostasis, k-Winners-Take-All over columns.
  - Temporal Memory: per-column cells, per-cell dendritic segments,
    per-segment synapses to a sample of "other" cells, predicted-vs-burst
    activation dynamics, Hebbian-style permanence update direction.

What is SIMPLIFIED for differentiability:
  - Discrete synapse permanence threshold replaced with
    ``sigmoid((perm - thr) / tau)`` so the LM-loss gradient flows through
    permanences. This is the standard "soft-HTM" approach
    (see HTM × deep-learning hybrid papers).
  - The hard predicted-cell / burst-column decision is implemented as
    softmax-pooling over segments + sigmoid gate. The same operator is
    used at training and eval — no Python dual-branch.
  - Hebbian permanence increments are exposed as a *Triton kernel*
    (``tm_hebbian_delta``) but applied as an *auxiliary regulariser*
    (gradient drives main learning). See docs/htm_fidelity.md.

API:
    htm = FaithfulHTM(
        d_input=128,
        n_columns=256,
        cells_per_column=8,
        k_active_columns=20,
        segments_per_cell=2,
        synapses_per_segment=16,
    )
    out = htm(x, reset_state=False)   # x: (B, T, d_input), out: (B, T, d_input)

State (carried across time):
    htm.boost              : (n_columns,) fp32 — homeostatic boost
    htm.prev_active_cells  : (B, n_columns * cells_per_column) fp32 [persistent]
                             (registered buffer; auto-reshapes when batch size
                              changes via reset_state=True)

Production-only, bf16 in / bf16 out, fp32 internal compute. No eager fallback.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

try:
    from src.kernels.htm import sp_boosted_overlap, tm_predict, tm_hebbian_delta
except ImportError:
    from kernels.htm import sp_boosted_overlap, tm_predict, tm_hebbian_delta


class FaithfulHTM(nn.Module):
    """Faithful HTM layer (Spatial Pooler + Temporal Memory) as a differentiable
    nn.Module backed by fused Triton kernels.

    Args:
        d_input:                Input feature dim.
        n_columns:              Number of cortical mini-columns.
        cells_per_column:       Cells per column (HTM standard: 8-32).
        k_active_columns:       k for k-WTA over columns (HTM sparsity).
        segments_per_cell:      Distal dendritic segments per cell.
        synapses_per_segment:   Synapses per segment (sampled from other cells).
        d_output:               Output dim (default = d_input via learned proj).
        perm_thr:               Soft synapse permanence threshold.
        perm_tau:               Soft-threshold temperature.
        softmax_beta:           Segment softmax pooling sharpness.
        boost_strength:         Homeostatic boost strength (Numenta default 2.0).
        boost_decay:            EMA decay for boost duty-cycle (Numenta default
                                 ~0.99).
        target_density:         Target column duty cycle (k/n_columns).
        hebbian_lr:             If > 0, Hebbian aux regulariser strength.
        dtype:                  Activation dtype (bf16 default).
    """

    def __init__(
        self,
        d_input: int,
        n_columns: int = 256,
        cells_per_column: int = 8,
        k_active_columns: int = 20,
        segments_per_cell: int = 2,
        synapses_per_segment: int = 16,
        d_output: int | None = None,
        perm_thr: float = 0.5,
        perm_tau: float = 0.1,
        softmax_beta: float = 4.0,
        boost_strength: float = 2.0,
        boost_decay: float = 0.99,
        target_density: float | None = None,
        hebbian_lr: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if d_output is None:
            d_output = d_input

        self.d_input = d_input
        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.k_active = k_active_columns
        self.n_cells = n_columns * cells_per_column
        self.n_segs = segments_per_cell
        self.n_syns = synapses_per_segment
        self.d_output = d_output

        self.perm_thr = float(perm_thr)
        self.perm_tau = float(perm_tau)
        self.softmax_beta = float(softmax_beta)
        self.boost_strength = float(boost_strength)
        self.boost_decay = float(boost_decay)
        self.target_density = (
            float(target_density)
            if target_density is not None
            else k_active_columns / n_columns
        )
        self.hebbian_lr = float(hebbian_lr)
        self.act_dtype = dtype

        # -------- Spatial Pooler parameters --------
        # Potential-synapse permanences (n_columns, d_input). Initialised
        # around perm_thr so ~half synapses are "open" at init.
        perm_init = perm_thr + 0.05 * torch.randn(n_columns, d_input)
        self.sp_permanence = nn.Parameter(perm_init.to(torch.float32))

        # Boost factor (homeostatic state — not trained via gradient)
        self.register_buffer(
            "boost", torch.ones(n_columns, dtype=torch.float32)
        )
        # EMA of column activation duty cycle, used to update boost
        self.register_buffer(
            "duty_cycle", torch.full((n_columns,), self.target_density, dtype=torch.float32)
        )

        # -------- Temporal Memory parameters --------
        # Per-segment permanences (n_cells, n_segs, n_syns) fp32
        tm_perm_init = perm_thr + 0.05 * torch.randn(
            self.n_cells, self.n_segs, self.n_syns
        )
        self.tm_permanence = nn.Parameter(tm_perm_init.to(torch.float32))
        # Per-segment synapse target cell indices (static topology — sampled
        # uniformly once at init). Each synapse listens to one cell in [0,n_cells).
        seg_idx = torch.randint(
            low=0, high=self.n_cells,
            size=(self.n_cells, self.n_segs, self.n_syns),
            dtype=torch.int32,
        )
        self.register_buffer("seg_idx", seg_idx)

        # -------- TM state (carried across time) --------
        # prev_active_cells: (B, n_cells). Allocated lazily on first forward.
        self.register_buffer(
            "prev_active_cells", torch.zeros(1, self.n_cells, dtype=torch.float32)
        )

        # -------- Output projection --------
        # cell-vector to d_output (learned)
        self.cell_proj = nn.Linear(self.n_cells, d_output, bias=False)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _kwta(self, scores: Tensor) -> Tensor:
        """Differentiable k-WTA over columns. Returns:
          gated   (M, n_columns): sigmoid(scores) * mask  — gradient-carrying
                                  per-column activation strength.
          mask    (M, n_columns): binary 0/1 — strict top-k indicator.

        Gradient flows through `gated` (which depends on scores), so
        sp_permanence is trained. The strict binary `mask` is used for boost
        homeostasis (which is a buffer-only update, no grad needed there).
        """
        k = self.k_active
        topk_scores, topk_idx = torch.topk(scores, k=k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, topk_idx, 1.0)
        # Normalise scores within the top-k set so the gate is in [0,1] and
        # differentiable. `mask` zeroes everything else.
        gated = torch.sigmoid(scores) * mask
        return gated, mask

    def _update_boost(self, active_col_mask: Tensor):
        """Numenta homeostatic boost update — in-place on buffers.

        active_col_mask: (M, n_columns) in {0, 1}.
        """
        with torch.no_grad():
            # average activation over rows
            inst_duty = active_col_mask.mean(dim=0)  # (n_columns,)
            self.duty_cycle.mul_(self.boost_decay).add_(
                inst_duty, alpha=1.0 - self.boost_decay
            )
            # Numenta boost formula:
            #   boost = exp(boost_strength * (target_density - duty_cycle))
            self.boost.copy_(
                torch.exp(self.boost_strength * (self.target_density - self.duty_cycle))
            )

    def _reset_prev_active(self, batch_size: int, device: torch.device):
        self.prev_active_cells = torch.zeros(
            batch_size, self.n_cells, dtype=torch.float32, device=device
        )

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def _apply(self, fn, recurse=True):
        """Override _apply so model.to(bf16) does NOT cast HTM permanences.

        The fused Triton kernels (sp_boosted_overlap, tm_predict) require
        permanence parameters in fp32 because the soft-sigmoid threshold is
        numerically unstable in bf16 (small (perm - thr)/tau differences
        get truncated). Inputs/outputs are bf16 — only the permanence
        parameters stay in fp32.
        """
        # Apply to children first
        super()._apply(fn, recurse=recurse)
        # Force fp32 on all params/buffers required by the Triton kernels.
        # (boost, duty_cycle, prev_active_cells all fp32; seg_idx int32.)
        with torch.no_grad():
            for name in ("sp_permanence", "tm_permanence"):
                p = getattr(self, name)
                if p.dtype != torch.float32:
                    p.data = p.data.to(torch.float32)
            for name in ("boost", "duty_cycle", "prev_active_cells"):
                if hasattr(self, name):
                    b = getattr(self, name)
                    if b.dtype != torch.float32:
                        setattr(self, name, b.to(torch.float32))
        return self

    def forward(self, x: Tensor, reset_state: bool = False) -> Tensor:
        """
        x: (B, T, d_input)  bf16/fp32
        Returns: (B, T, d_output)
        """
        assert x.dim() == 3, f"expected (B, T, d_input), got {x.shape}"
        B, T, D = x.shape
        assert D == self.d_input
        device = x.device

        # Reset / lazy-allocate state if batch size changed
        if reset_state or self.prev_active_cells.shape[0] != B or \
                self.prev_active_cells.device != device:
            self._reset_prev_active(B, device)

        out_steps = []
        prev_active = self.prev_active_cells  # (B, n_cells)

        # We iterate over time, carrying the (B, n_cells) state. Each step
        # processes the whole batch as one "M = B" row block — both kernels
        # tile over M.
        x_f = x.to(torch.float32)
        for t in range(T):
            x_t = x_f[:, t, :]  # (B, d_input)
            # ---- Spatial Pooler: boosted soft-overlap -> k-WTA ----
            sp_scores = sp_boosted_overlap(
                x_t.contiguous(),
                self.sp_permanence,
                self.boost,
                self.perm_thr,
                self.perm_tau,
            )  # (B, n_columns)
            sp_gated, col_mask = self._kwta(sp_scores)  # (B, n_columns)

            # Update boost homeostasis (buffer-only, no autograd)
            self._update_boost(col_mask)

            # ---- Temporal Memory: predicted-cell scoring ----
            cell_score = tm_predict(
                prev_active.contiguous(),
                self.tm_permanence,
                self.seg_idx,
                self.perm_thr,
                self.perm_tau,
                self.softmax_beta,
            )  # (B, n_cells)

            # ---- Predicted-vs-burst dynamics ----
            # For each active column, cells with high cell_score become active.
            # If no cell is predicted, the column "bursts" (all cells active).
            # We implement this as: active_cells = sp_gated_repeated *
            #   ( sigmoid(cell_score) if any-predicted else uniform-burst ).
            # sp_gated carries gradient to sp_permanence; col_mask is buffer-only.
            col_mask_rep = sp_gated.unsqueeze(-1).expand(
                -1, -1, self.cells_per_column
            ).reshape(B, self.n_cells)  # (B, n_cells)

            cell_pred = torch.sigmoid(cell_score - self.perm_thr)  # (B, n_cells) in [0,1]

            # Per-column max predicted score — used to detect "any predicted"
            cell_pred_col = cell_pred.view(B, self.n_columns, self.cells_per_column)
            col_max_pred = cell_pred_col.max(dim=-1).values  # (B, n_columns)
            # any_predicted gate (soft): 1 if max > 0.5
            any_pred = torch.sigmoid(8.0 * (col_max_pred - 0.5))  # (B, n_columns)
            any_pred_rep = any_pred.unsqueeze(-1).expand(
                -1, -1, self.cells_per_column
            ).reshape(B, self.n_cells)

            burst = (1.0 - any_pred_rep) * (1.0 / self.cells_per_column)
            active_cells = col_mask_rep * (any_pred_rep * cell_pred + burst)

            # ---- Optional Hebbian regulariser (no grad path) ----
            if self.training and self.hebbian_lr > 0.0:
                with torch.no_grad():
                    delta = tm_hebbian_delta(
                        prev_active.detach().contiguous(),
                        active_cells.detach().contiguous(),
                        self.tm_permanence.detach(),
                        self.seg_idx,
                    )
                    self.tm_permanence.data.add_(delta, alpha=self.hebbian_lr)

            # Roll state
            prev_active = active_cells
            out_steps.append(active_cells)

        # Store final state for next call
        with torch.no_grad():
            self.prev_active_cells = prev_active.detach()

        # Stack and project
        cell_seq = torch.stack(out_steps, dim=1)  # (B, T, n_cells)
        # cell_proj weights live in act_dtype-compatible memory; convert
        out = self.cell_proj(cell_seq.to(self.cell_proj.weight.dtype))
        return out.to(self.act_dtype)

    def extra_repr(self) -> str:
        return (
            f"d_input={self.d_input}, n_columns={self.n_columns}, "
            f"cells_per_column={self.cells_per_column}, k_active={self.k_active}, "
            f"segments_per_cell={self.n_segs}, synapses_per_segment={self.n_syns}"
        )
