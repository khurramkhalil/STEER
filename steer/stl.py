"""Signal Temporal Logic (STL) robustness for STEER reasoning trajectories.

STEER encodes three desiderata about a reasoning trajectory as STL formulae and
scores each one with its quantitative *robustness* :math:`\\rho` (positive when
the property holds, negative when it is violated, magnitude = margin):

* **Monotonic progress** ``G (progress(t) <= progress(t-1))`` -- the number of
  blank cells never increases. Robustness ``min_t [progress(t-1) - progress(t)]``.
* **Path validity** ``G (violation(t) < eps_viol)`` -- the grid stays (nearly)
  constraint-satisfying throughout. Robustness ``min_t [eps_viol - violation(t)]``.
* **Eventual convergence** ``F (stability(t) < eps_stab)`` -- at some step the
  prediction stops changing. Robustness ``max_t [eps_stab - stability(t)]``.

``G`` (globally / always) maps to ``min`` over time; ``F`` (eventually) maps to
``max`` over time, following the standard STL robustness semantics.
"""

from __future__ import annotations

from typing import Dict

import torch


class STEERProperties:
    """Evaluates the three STEER STL properties over a batch of trajectories.

    Args:
        epsilon_viol: Tolerance for the path-validity property.
        epsilon_stab: Tolerance (convergence threshold) for eventual convergence.
    """

    def __init__(self, epsilon_viol: float = 0.1, epsilon_stab: float = 0.01) -> None:
        self.epsilon_viol = epsilon_viol
        self.epsilon_stab = epsilon_stab

    def compute_robustness(
        self,
        violations: torch.Tensor,
        progress: torch.Tensor,
        stability: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample robustness for each STEER property.

        Args:
            violations: ``(batch, T+1)`` violation signal at each step.
            progress:   ``(batch, T+1)`` progress signal at each step.
            stability:  ``(batch, T+1)`` stability signal. Index 0 is a sentinel
                (no predecessor step exists), so convergence is evaluated only
                over the real transitions ``t = 1..T``.

        Returns:
            Dict with ``(batch,)`` robustness tensors under keys
            ``rho_improve``, ``rho_valid`` and ``rho_converge``. Positive values
            mean the property is satisfied with that margin. When there are no
            transitions (a single-step trajectory) the time-quantified
            properties are vacuously satisfied (robustness 0).
        """
        batch_size = progress.shape[0]
        has_transitions = progress.shape[1] >= 2

        # Monotonic progress: progress should be non-increasing over time.
        if has_transitions:
            prog_diff = progress[:, :-1] - progress[:, 1:]      # (batch, T)
            rho_improve = prog_diff.min(dim=1).values
        else:
            rho_improve = progress.new_zeros(batch_size)        # vacuously true

        # Path validity: violation must stay below eps_viol at every step.
        rho_valid = (self.epsilon_viol - violations).min(dim=1).values

        # Eventual convergence: stability must drop below eps_stab at some real
        # transition. The sentinel at t=0 (always 0) is excluded so this term is
        # not trivially satisfied.
        if has_transitions:
            rho_converge = (self.epsilon_stab - stability[:, 1:]).max(dim=1).values
        else:
            rho_converge = stability.new_zeros(batch_size)      # vacuously true

        return {
            "rho_improve": rho_improve,
            "rho_valid": rho_valid,
            "rho_converge": rho_converge,
        }
