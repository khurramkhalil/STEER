"""Differentiable Sudoku constraint signals for the STEER regularizer.

These signals turn a (soft) distribution over Sudoku cell values into scalar
quantities that downstream Signal Temporal Logic (STL) properties operate on:

* ``violation`` -- how badly the grid breaks Sudoku's all-different constraints,
* ``progress``  -- how much of the grid is still unfilled (blank mass),
* ``stability`` -- how much the prediction changed between two reasoning steps.

All signals are computed from *probabilities* (post-softmax) so they are smooth
and differentiable with respect to the model's logits.

Token / vocabulary layout (per cell, ``VOCAB_SIZE`` classes)::

    index 0       -> PAD
    index 1       -> BLANK (empty cell)
    index 2..10   -> digits 1..9
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# --- Sudoku / vocabulary geometry -------------------------------------------
GRID_SIZE = 81          # 9 x 9 cells, flattened
SIDE = 9                # cells per row / column / box
NUM_DIGITS = 9          # digits 1..9
VOCAB_SIZE = 11         # PAD + BLANK + 9 digits
PAD_TOKEN = 0
BLANK_TOKEN = 1
FIRST_DIGIT_TOKEN = 2   # token index of digit "1"


class SudokuSignals:
    """Computes differentiable constraint signals for batches of Sudoku grids.

    The 27 Sudoku "units" (9 rows, 9 columns, 9 boxes) are pre-computed once as
    a ``(27, 9)`` index tensor so that violation scores can be evaluated with a
    single gather.
    """

    def __init__(self) -> None:
        units = self._row_indices() + self._col_indices() + self._box_indices()
        # (27, 9): 27 units, each referencing 9 flat cell indices.
        self.unit_indices = torch.tensor(units, dtype=torch.long)

    # -- unit definitions ----------------------------------------------------
    @staticmethod
    def _row_indices() -> list[list[int]]:
        return [[r * SIDE + c for c in range(SIDE)] for r in range(SIDE)]

    @staticmethod
    def _col_indices() -> list[list[int]]:
        return [[r * SIDE + c for r in range(SIDE)] for c in range(SIDE)]

    @staticmethod
    def _box_indices() -> list[list[int]]:
        boxes = []
        for box_row in range(3):
            for box_col in range(3):
                cells = [
                    (box_row * 3 + r) * SIDE + (box_col * 3 + c)
                    for r in range(3)
                    for c in range(3)
                ]
                boxes.append(cells)
        return boxes

    def to(self, device: torch.device | str) -> "SudokuSignals":
        """Move the cached index tensor to ``device`` (in place) and return self."""
        self.unit_indices = self.unit_indices.to(device)
        return self

    # -- signals -------------------------------------------------------------
    def compute_violation_score(self, probs: torch.Tensor) -> torch.Tensor:
        """Expected number of duplicate digits across all 27 units.

        For each unit and digit we sum the probability mass assigned to that
        digit across the unit's 9 cells. A valid unit places each digit exactly
        once, so any expected mass above 1.0 is a (soft) duplication. We sum
        ``relu(count - 1)`` over all units and digits.

        Args:
            probs: ``(batch, GRID_SIZE, VOCAB_SIZE)`` probabilities.

        Returns:
            ``(batch,)`` non-negative violation scores (0.0 == constraint-satisfying).
        """
        batch_size = probs.shape[0]

        # Keep only digit probabilities (drop PAD and BLANK): (batch, 81, 9)
        digit_probs = probs[:, :, FIRST_DIGIT_TOKEN:]

        # Gather the 9 cells of each of the 27 units: (batch, 27, 9, 9)
        # dims -> (batch, unit, cell_in_unit, digit)
        flat_indices = self.unit_indices.reshape(-1)  # (243,)
        gathered = digit_probs[:, flat_indices, :].view(
            batch_size, 27, SIDE, NUM_DIGITS
        )

        # Expected count of each digit within each unit: (batch, 27, 9)
        digit_counts = gathered.sum(dim=2)

        # Penalise expected mass beyond the single allowed occurrence.
        excess = F.relu(digit_counts - 1.0)
        return excess.sum(dim=(1, 2))

    def compute_progress_score(self, probs: torch.Tensor) -> torch.Tensor:
        """Expected number of still-blank cells (lower == more progress).

        Args:
            probs: ``(batch, GRID_SIZE, VOCAB_SIZE)`` probabilities.

        Returns:
            ``(batch,)`` expected blank-cell count in ``[0, GRID_SIZE]``.
        """
        return probs[:, :, BLANK_TOKEN].sum(dim=1)

    def compute_stability_score(
        self, current_probs: torch.Tensor, prev_probs: torch.Tensor
    ) -> torch.Tensor:
        """Sum of squared change between consecutive reasoning steps.

        Args:
            current_probs: ``(batch, GRID_SIZE, VOCAB_SIZE)`` at step ``t``.
            prev_probs:    ``(batch, GRID_SIZE, VOCAB_SIZE)`` at step ``t - 1``.

        Returns:
            ``(batch,)`` non-negative change scores (0.0 == identical predictions).
        """
        return ((current_probs - prev_probs) ** 2).sum(dim=(1, 2))
