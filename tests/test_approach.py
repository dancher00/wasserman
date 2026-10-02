import pytest
import torch

from wasman.controllers.approach import ApproachGate


@pytest.mark.unit
def test_arm_requires_position_and_continuously_settled_motion():
    gate = ApproachGate(5, "cpu", hold_steps=3)
    error = torch.zeros(5, 3)
    error[0, 0] = 0.2
    speed = torch.tensor([0.0, 0.1, 0.0, 0.0, 0.0])
    angular = torch.tensor([0.0, 0.0, 0.2, 0.0, 0.0])
    attitude = torch.tensor([0.0, 0.0, 0.0, 0.3, 0.0])
    for _ in range(2):
        assert not gate.update(error, speed, angular, attitude).any()
    assert gate.update(error, speed, angular, attitude).tolist() == [False] * 4 + [True]
    # Enabled manipulation stays latched, even when contact moves the base.
    speed[:] = 0.2
    assert gate.update(error, speed, angular, attitude)[4]
    gate.reset([4])
    assert not gate.enabled.any()
    assert not gate.stable_steps.any()


@pytest.mark.unit
def test_unstable_frame_restarts_hold_and_reset_is_per_environment():
    gate = ApproachGate(2, "cpu", hold_steps=3)
    error, zero = torch.zeros(2, 3), torch.zeros(2)
    gate.update(error, zero, zero, zero)
    gate.update(error, torch.tensor([0.1, 0.0]), zero, zero)
    assert gate.update(error, zero, zero, zero).tolist() == [False, True]
    gate.reset([1])
    assert gate.stable_steps.tolist() == [1, 0]
    gate.reset()
    assert not gate.enabled.any() and not gate.stable_steps.any()
