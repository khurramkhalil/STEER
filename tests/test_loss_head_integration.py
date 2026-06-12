"""Integration tests for STEER wiring inside ACTLossHead.

These guard the regression where STEER was registered but never actually applied
during training (the train loop passed a detached / empty trajectory, so the
regularizer silently contributed nothing).
"""

import torch
import torch.nn as nn

from models.losses import ACTLossHead
from steer.loss import STEERLoss
from steer.signals import GRID_SIZE, VOCAB_SIZE

BATCH, STEPS = 2, 4


class _FakeCarry:
    def __init__(self, labels, halted, steps):
        self.current_data = {"labels": labels}
        self.halted = halted
        self.steps = steps


class _FakeRecursiveModel(nn.Module):
    """Minimal stand-in exposing the interface ACTLossHead relies on.

    Produces a deterministic, parameter-dependent trajectory so we can assert on
    exact loss values and on gradient flow.
    """

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(VOCAB_SIZE))
        torch.manual_seed(0)
        self.register_buffer("base", torch.randn(BATCH, STEPS, GRID_SIZE, VOCAB_SIZE))

    def forward(self, carry, batch):
        trajectory = self.base * self.w  # (B, STEPS, GRID_SIZE, VOCAB_SIZE)
        logits = trajectory[:, -1]       # final reasoning step is the prediction
        outputs = {
            "logits": logits,
            "q_halt_logits": torch.zeros(BATCH),
            "q_continue_logits": torch.zeros(BATCH),
            "trajectory": trajectory,
        }
        new_carry = _FakeCarry(
            labels=torch.zeros(BATCH, GRID_SIZE, dtype=torch.long),
            halted=torch.ones(BATCH, dtype=torch.bool),
            steps=torch.ones(BATCH, dtype=torch.int32),
        )
        return new_carry, outputs


def _run(steer_lambda):
    model = _FakeRecursiveModel()
    head = ACTLossHead(
        model,
        "stablemax_cross_entropy",
        steer_loss_fn=STEERLoss() if steer_lambda > 0 else None,
        steer_lambda=steer_lambda,
    )
    _, loss, metrics, _, _ = head(return_keys=[], carry=None, batch=None)
    return loss, metrics


def test_steer_disabled_emits_no_steer_metrics():
    _, metrics = _run(steer_lambda=0.0)
    assert not any(k.startswith("steer/") for k in metrics)


def test_steer_skipped_in_eval_mode():
    # STEER is a training-time regularizer; in eval mode it must not run (its
    # per-batch-mean metrics do not fit the eval aggregation pipeline).
    model = _FakeRecursiveModel()
    head = ACTLossHead(model, "stablemax_cross_entropy", steer_loss_fn=STEERLoss(), steer_lambda=1.0)
    head.eval()
    _, loss, metrics, _, _ = head(return_keys=[], carry=None, batch=None)
    assert not any(k.startswith("steer/") for k in metrics)
    base_loss, _ = _run(steer_lambda=0.0)
    assert torch.allclose(loss, base_loss, atol=1e-5)


def test_steer_enabled_adds_to_total_loss():
    base_loss, _ = _run(steer_lambda=0.0)
    total_loss, metrics = _run(steer_lambda=1.0)

    assert "steer/loss" in metrics
    # Total loss == task loss + lambda * steer loss (lambda = 1 here).
    expected = base_loss + metrics["steer/loss"]
    assert torch.allclose(total_loss, expected, atol=1e-5)
    # Random trajectory almost surely violates a property -> non-trivial penalty.
    assert metrics["steer/loss"].item() > 0


def test_lambda_scales_the_penalty():
    _, m1 = _run(steer_lambda=1.0)
    base, _ = _run(steer_lambda=0.0)
    total10, _ = _run(steer_lambda=10.0)
    # 10x lambda applies 10x the same steer penalty on top of the task loss.
    assert torch.allclose(total10, base + 10.0 * m1["steer/loss"], atol=1e-4)


def test_gradient_reaches_model_parameter_through_steer():
    model = _FakeRecursiveModel()
    head = ACTLossHead(model, "stablemax_cross_entropy", steer_loss_fn=STEERLoss(), steer_lambda=1.0)
    _, loss, _, _, _ = head(return_keys=[], carry=None, batch=None)
    loss.backward()
    assert model.w.grad is not None
    assert torch.isfinite(model.w.grad).all()
    assert model.w.grad.abs().sum().item() > 0
