import numpy as np
import pytest
import torch

from wasman.controllers.valve_mode_router import ValveModeRouter
from wasman.learning.valve_mode_tree import fit_mode_tree


def test_tree_learns_splits_and_router_uses_only_physical_history():
    x = np.zeros((1024, 8))
    x[:, 0] = np.linspace(0, 1, len(x))
    y = np.minimum((x[:, 0] * 4).astype(int), 3)
    tree = fit_mode_tree(x[::2], y[::2], min_leaf=4, max_depth=6)
    router = ValveModeRouter(tree)
    prediction = router.classify(torch.tensor(x[1::2]))
    assert (prediction.numpy() == y[1::2]).mean() > 0.99
    observations = torch.zeros(4, 50)
    observations[:, 38] = torch.tensor([0.1, 0.4, 0.7, 0.9])
    assert torch.equal(router(observations), torch.arange(4))
    observations[:, 27:38] = torch.randn(4, 11) * 100
    assert torch.equal(router(observations), torch.arange(4))
    observations[:, 38] = 0
    assert torch.equal(router(observations), torch.arange(4))
    router.reset(torch.tensor([2]))
    assert torch.equal(router(observations), torch.tensor([0, 1, 0, 3]))


def test_adaptive_unloading_waits_for_target_and_preserves_force_free_opening():
    from wasman.controllers.valve import adaptive_release_stroke

    phase = torch.tensor([4, 5, 5, 5, 5])
    force = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.2], [0.0, 0.04], [1.0, 1.0]])
    current = torch.tensor([0.002, 0.001, 0.002, 0.002, 0.004])
    reference = torch.tensor([0.002, 0.002, 0.002, 0.002, 0.004])
    result = adaptive_release_stroke(phase, force, current, reference)
    torch.testing.assert_close(result, torch.tensor([0.002, 0.002, 0.002025, 0.002, 0.004]))


def test_contact_admittance_is_release_only_rate_limited_and_bounded():
    from wasman.controllers.valve import contact_release_offset

    offset = torch.zeros(4, 3)
    offset[3, 0] = 0.00999
    phase = torch.tensor([4, 5, 5, 5])
    forces = torch.zeros(4, 2, 3)
    forces[:, 0, 0] = 100
    forces[2, 1, 0] = -100
    result = contact_release_offset(offset, phase, forces, 1 / 30)
    torch.testing.assert_close(result[:, 0], torch.tensor([0, 0.0001, 0, 0.01]))
    assert not result[:, 1:].any()


def test_expert_reference_cannot_place_release_window_outside_goal():
    from wasman.controllers.valve import ValveExpert

    with pytest.raises(ValueError, match="unchanged task goal"):
        ValveExpert(None, hold_angle_deg=175)
    with pytest.raises(ValueError, match="unchanged task goal"):
        ValveExpert(None, hold_angle_deg=170)
