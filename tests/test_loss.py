"""Unit tests for STEERLoss, including the gradient-flow regression test."""

import torch

from steer.loss import STEERLoss
from steer.signals import GRID_SIZE, VOCAB_SIZE


def make_trajectory(batch=2, steps=4, requires_grad=False):
    return torch.randn(batch, steps, GRID_SIZE, VOCAB_SIZE, requires_grad=requires_grad)


def test_loss_is_nonnegative_scalar():
    loss, _ = STEERLoss()(make_trajectory())
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_metrics_dict_is_populated():
    # Regression: a previous version referenced `metrics` before defining it,
    # which raised NameError the moment STEER was active.
    _, metrics = STEERLoss()(make_trajectory(steps=4))
    assert metrics, "metrics dict must not be empty"
    for key in (
        "steer/loss",
        "steer/rho_improve",
        "steer/rho_valid",
        "steer/rho_converge",
        "steer/validity_rate",
        "steer/convergence_step",
        "steer/compute_savings",
    ):
        assert key in metrics, f"missing metric: {key}"
    # Per-step diagnostics exist for every step.
    assert "steer/viol_step_0" in metrics
    assert "steer/stab_step_3" in metrics


def test_metrics_are_detached():
    _, metrics = STEERLoss()(make_trajectory(requires_grad=True))
    assert all(not v.requires_grad for v in metrics.values())


def test_gradient_flows_to_trajectory():
    # THE key regression: STEER must be a real, gradient-carrying loss, not a
    # no-op. If the trajectory were detached upstream this would fail.
    traj = make_trajectory(requires_grad=True)
    loss, _ = STEERLoss()(traj)
    loss.backward()
    assert traj.grad is not None
    assert torch.isfinite(traj.grad).all()
    assert traj.grad.abs().sum().item() > 0.0


def test_single_step_trajectory_is_handled():
    # With steps==1 there is no stability/progress-diff; loss must still be finite.
    loss, metrics = STEERLoss()(make_trajectory(steps=1))
    assert torch.isfinite(loss)
    # Convergence diagnostics are only defined for steps > 1.
    assert "steer/convergence_step" not in metrics


def test_compute_savings_in_unit_range():
    _, metrics = STEERLoss()(make_trajectory(steps=6))
    savings = metrics["steer/compute_savings"].item()
    assert 0.0 <= savings <= 1.0


def test_loss_runs_in_float_dtypes():
    for dtype in (torch.float32, torch.float64):
        traj = make_trajectory().to(dtype).requires_grad_(True)
        loss, _ = STEERLoss()(traj)
        loss.backward()
        assert traj.grad.dtype == dtype
