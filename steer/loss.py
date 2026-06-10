"""STEER regularization loss.

``STEERLoss`` consumes a full reasoning trajectory (the per-step logits emitted
by a recursive reasoner) and returns a scalar regularization loss plus a
dictionary of diagnostic metrics.

The loss is the hinge on the negative STL robustness of the three STEER
properties (see :mod:`steer.stl`): for each property we pay ``relu(-rho)``, i.e.
zero cost when the property is satisfied and a linearly growing penalty
proportional to how badly it is violated.

IMPORTANT: the input trajectory must carry gradients (do **not** pass a detached
tensor) -- otherwise this loss is a no-op. See ``tests/test_loss.py`` for the
regression test that enforces this.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from steer.signals import SudokuSignals
from steer.stl import STEERProperties


class STEERLoss(nn.Module):
    """Self-Terminating Efficient Reasoning regularizer for Sudoku trajectories.

    Args:
        epsilon_viol: Tolerance for the path-validity property.
        epsilon_stab: Convergence threshold for the eventual-convergence property.
    """

    def __init__(self, epsilon_viol: float = 0.1, epsilon_stab: float = 0.01) -> None:
        super().__init__()
        self.signals = SudokuSignals()
        self.properties = STEERProperties(
            epsilon_viol=epsilon_viol, epsilon_stab=epsilon_stab
        )

    def forward(
        self, trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the STEER loss and diagnostics for a batch of trajectories.

        Args:
            trajectory: ``(batch, steps, seq_len, vocab)`` logits, one slice per
                recursive reasoning step. Must be gradient-carrying.

        Returns:
            ``(loss, metrics)`` where ``loss`` is a scalar tensor and ``metrics``
            is a dict of detached diagnostic tensors (keys prefixed ``steer/``).
        """
        batch_size, steps, seq_len, vocab = trajectory.shape

        self.signals.to(trajectory.device)

        # Differentiable per-cell distributions.
        probs = F.softmax(trajectory, dim=-1)  # (batch, steps, seq_len, vocab)

        # Per-step signals, evaluated by flattening (batch, steps) together.
        flat_probs = probs.view(-1, seq_len, vocab)
        violations = self.signals.compute_violation_score(flat_probs).view(batch_size, steps)
        progress = self.signals.compute_progress_score(flat_probs).view(batch_size, steps)

        # Stability at step t is the change from t-1 (stability[:, 0] := 0).
        stability = torch.zeros(batch_size, steps, device=trajectory.device, dtype=probs.dtype)
        if steps > 1:
            curr = probs[:, 1:].reshape(-1, seq_len, vocab)
            prev = probs[:, :-1].reshape(-1, seq_len, vocab)
            stab_scores = self.signals.compute_stability_score(curr, prev)
            stability[:, 1:] = stab_scores.view(batch_size, steps - 1)

        # STL robustness -> hinge loss (zero when each property is satisfied).
        robustness = self.properties.compute_robustness(violations, progress, stability)
        loss_improve = F.relu(-robustness["rho_improve"]).mean()
        loss_valid = F.relu(-robustness["rho_valid"]).mean()
        loss_converge = F.relu(-robustness["rho_converge"]).mean()
        total_loss = loss_improve + loss_valid + loss_converge

        metrics = self._build_metrics(
            total_loss, robustness, violations, progress, stability, batch_size, steps
        )
        return total_loss, metrics

    @torch.no_grad()
    def _build_metrics(
        self,
        total_loss: torch.Tensor,
        robustness: Dict[str, torch.Tensor],
        violations: torch.Tensor,
        progress: torch.Tensor,
        stability: torch.Tensor,
        batch_size: int,
        steps: int,
    ) -> Dict[str, torch.Tensor]:
        """Build a dict of detached diagnostics for logging (no gradient impact)."""
        metrics: Dict[str, torch.Tensor] = {}

        # Per-step trajectory diagnostics.
        for t in range(steps):
            metrics[f"steer/viol_step_{t}"] = violations[:, t].mean()
            metrics[f"steer/prog_step_{t}"] = progress[:, t].mean()
            if t > 0:
                metrics[f"steer/stab_step_{t}"] = stability[:, t].mean()

        # Fraction of samples that are (nearly) constraint-satisfying at the end.
        metrics["steer/validity_rate"] = (violations[:, -1] < 0.01).float().mean()

        # Convergence step: first step (>=1) whose change falls below eps_stab.
        # A sentinel column guarantees an index even when a sample never settles.
        if steps > 1:
            stable = stability[:, 1:] < self.properties.epsilon_stab
            sentinel = torch.ones(batch_size, 1, dtype=torch.bool, device=stability.device)
            converged_idx = torch.argmax(
                torch.cat([stable, sentinel], dim=1).int(), dim=1
            ).float() + 1.0  # +1: column 0 (t=0) was skipped
            metrics["steer/convergence_step"] = converged_idx.mean()
            metrics["steer/compute_savings"] = 1.0 - converged_idx.mean() / steps

        metrics.update(
            {
                "steer/loss": total_loss.detach(),
                "steer/rho_improve": robustness["rho_improve"].mean(),
                "steer/rho_valid": robustness["rho_valid"].mean(),
                "steer/rho_converge": robustness["rho_converge"].mean(),
                "steer/viol_final": violations[:, -1].mean(),
                "steer/prog_final": progress[:, -1].mean(),
            }
        )
        return metrics
