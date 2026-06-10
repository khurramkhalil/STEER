"""Unit tests for STL robustness semantics of the three STEER properties."""

import torch

from steer.stl import STEERProperties


def make_props(eps_viol=0.1, eps_stab=0.01):
    return STEERProperties(epsilon_viol=eps_viol, epsilon_stab=eps_stab)


def test_monotonic_progress_positive_when_decreasing():
    # progress strictly decreases -> property holds -> rho_improve > 0.
    props = make_props()
    progress = torch.tensor([[3.0, 2.0, 1.0]])
    rho = props.compute_robustness(
        violations=torch.zeros(1, 3), progress=progress, stability=torch.zeros(1, 3)
    )
    assert rho["rho_improve"].item() > 0


def test_monotonic_progress_negative_when_increasing():
    # An increase at any step violates G(prog(t) <= prog(t-1)).
    props = make_props()
    progress = torch.tensor([[1.0, 2.0, 1.5]])
    rho = props.compute_robustness(
        violations=torch.zeros(1, 3), progress=progress, stability=torch.zeros(1, 3)
    )
    assert rho["rho_improve"].item() < 0


def test_path_validity_sign_tracks_epsilon():
    props = make_props(eps_viol=0.1)
    progress = torch.zeros(1, 3)
    stability = torch.zeros(1, 3)

    valid = props.compute_robustness(torch.tensor([[0.0, 0.05, 0.02]]), progress, stability)
    assert valid["rho_valid"].item() > 0  # always below eps_viol

    invalid = props.compute_robustness(torch.tensor([[0.0, 0.5, 0.02]]), progress, stability)
    assert invalid["rho_valid"].item() < 0  # one step exceeds eps_viol


def test_eventual_convergence_positive_if_any_step_settles():
    props = make_props(eps_stab=0.01)
    progress = torch.zeros(1, 4)
    violations = torch.zeros(1, 4)
    # Never settles below eps_stab -> property fails.
    never = props.compute_robustness(violations, progress, torch.tensor([[0.0, 0.5, 0.4, 0.3]]))
    assert never["rho_converge"].item() < 0
    # Settles at the last step -> property holds.
    settles = props.compute_robustness(violations, progress, torch.tensor([[0.0, 0.5, 0.4, 0.005]]))
    assert settles["rho_converge"].item() > 0


def test_robustness_is_per_sample():
    props = make_props()
    progress = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])  # row0 good, row1 bad
    rho = props.compute_robustness(torch.zeros(2, 3), progress, torch.zeros(2, 3))
    assert rho["rho_improve"].shape == (2,)
    assert rho["rho_improve"][0].item() > 0 > rho["rho_improve"][1].item()
