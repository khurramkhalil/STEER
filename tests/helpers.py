"""Shared helpers for building Sudoku probability tensors in tests."""

from __future__ import annotations

import torch

from steer.signals import BLANK_TOKEN, FIRST_DIGIT_TOKEN, GRID_SIZE, SIDE, VOCAB_SIZE


def solved_grid() -> list[int]:
    """Return a valid, fully-solved 9x9 Sudoku as 81 digits (1..9), row-major.

    Uses the standard shifted-pattern construction
    ``value(r, c) = ((r * 3 + r // 3 + c) mod 9) + 1`` which is guaranteed to
    satisfy every row, column and box constraint.
    """
    return [
        ((r * 3 + r // 3 + c) % SIDE) + 1
        for r in range(SIDE)
        for c in range(SIDE)
    ]


def grid_to_probs(grid: list[int | None]) -> torch.Tensor:
    """One-hot encode a grid into ``(1, GRID_SIZE, VOCAB_SIZE)`` probabilities.

    Each entry is a digit ``1..9`` or ``None`` for a blank cell.
    """
    assert len(grid) == GRID_SIZE
    probs = torch.zeros(1, GRID_SIZE, VOCAB_SIZE)
    for i, value in enumerate(grid):
        token = BLANK_TOKEN if value is None else FIRST_DIGIT_TOKEN + (value - 1)
        probs[0, i, token] = 1.0
    return probs


def blank_probs() -> torch.Tensor:
    """A fully-blank grid as ``(1, GRID_SIZE, VOCAB_SIZE)`` probabilities."""
    return grid_to_probs([None] * GRID_SIZE)
