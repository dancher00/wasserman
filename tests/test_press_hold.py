from types import SimpleNamespace

import pytest
import torch

from wasman.controllers.press_hold import PressHoldController, normal_press_force


@pytest.mark.unit
def test_normal_force_has_correct_bias_braking_and_limits():
    zero = torch.zeros(1)
    assert normal_press_force(torch.tensor([0.0055]), zero, bias=1.0).item() == pytest.approx(1.0)
    assert normal_press_force(torch.tensor([0.009]), zero).item() < 0
    assert normal_press_force(torch.tensor([0.0055]), torch.tensor([0.1])).item() < 0
    assert normal_press_force(torch.tensor([-1.0]), zero).item() <= 15
    assert normal_press_force(torch.tensor([1.0]), zero).item() >= -15


@pytest.mark.unit
def test_hold_latches_only_valid_contact_and_resets_independently():
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        _base_target_pos_w=torch.zeros(2, 3),
        _base_target_quat_w=torch.zeros(2, 4),
        _arm_targets=torch.zeros(2, 4),
        _button_travel=torch.tensor([0.001, 0.001]),
        _distance=torch.tensor([0.05, 0.5]),
        _alignment=torch.ones(2),
        cfg=SimpleNamespace(tool_contact_distance=0.13),
    )
    hold = PressHoldController(env)
    arm = torch.ones(2, 4)
    hold.update(torch.ones(2, dtype=torch.bool), env._base_target_pos_w, env._base_target_quat_w, arm)
    assert hold.active.tolist() == [True, False]
    arm[:] = 2
    hold.update(torch.ones(2, dtype=torch.bool), env._base_target_pos_w, env._base_target_quat_w, arm)
    assert env._arm_targets[0].tolist() == [1] * 4
    hold.reset([0])
    assert not hold.active.any()
