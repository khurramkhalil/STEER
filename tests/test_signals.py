"""Unit tests for the differentiable Sudoku signals."""

import torch

from steer.signals import GRID_SIZE, SudokuSignals
from tests.helpers import blank_probs, grid_to_probs, solved_grid


def test_unit_indices_shape_and_coverage():
    signals = SudokuSignals()
    # 27 units (9 rows + 9 cols + 9 boxes), 9 cells each.
    assert signals.unit_indices.shape == (27, 9)
    # Every cell appears in exactly 3 units (its row, column and box).
    counts = torch.bincount(signals.unit_indices.reshape(-1), minlength=GRID_SIZE)
    assert torch.all(counts == 3)


def test_valid_solution_has_zero_violation():
    signals = SudokuSignals()
    probs = grid_to_probs(solved_grid())
    violation = signals.compute_violation_score(probs)
    assert torch.allclose(violation, torch.zeros(1), atol=1e-5)


def test_blank_grid_has_zero_violation():
    # Blanks carry no digit mass, so there is nothing to violate.
    signals = SudokuSignals()
    assert torch.allclose(signals.compute_violation_score(blank_probs()), torch.zeros(1))


def test_duplicate_in_row_is_penalised():
    signals = SudokuSignals()
    grid = [None] * GRID_SIZE
    grid[0] = 5
    grid[1] = 5  # duplicate 5 in row 0 (and box 0)
    violation = signals.compute_violation_score(grid_to_probs(grid))
    # relu(count-1) = 1 in the shared row AND in the shared box -> 2.0 total.
    assert torch.allclose(violation, torch.tensor([2.0]), atol=1e-5)


def test_violation_is_differentiable():
    signals = SudokuSignals()
    logits = torch.randn(2, GRID_SIZE, 11, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    signals.compute_violation_score(probs).sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_progress_counts_blanks():
    signals = SudokuSignals()
    grid = [None] * GRID_SIZE
    for c in range(9):  # fill row 0 -> 9 non-blank cells, 72 blanks
        grid[c] = c + 1
    progress = signals.compute_progress_score(grid_to_probs(grid))
    assert torch.allclose(progress, torch.tensor([72.0]))


def test_progress_zero_when_full():
    signals = SudokuSignals()
    progress = signals.compute_progress_score(grid_to_probs(solved_grid()))
    assert torch.allclose(progress, torch.zeros(1))


def test_stability_zero_for_identical_steps():
    signals = SudokuSignals()
    probs = grid_to_probs(solved_grid())
    assert torch.allclose(signals.compute_stability_score(probs, probs), torch.zeros(1))


def test_stability_mse_for_single_cell_change():
    signals = SudokuSignals()
    grid = [None] * GRID_SIZE
    grid[0] = 1
    prev = grid_to_probs(grid)
    grid[0] = 2
    curr = grid_to_probs(grid)
    # One cell flips one-hot bucket: (1-0)^2 + (0-1)^2 = 2.0.
    assert torch.allclose(signals.compute_stability_score(curr, prev), torch.tensor([2.0]))
