import pytest
import torch

from wasman.controllers.trajectory import advance_target


@pytest.mark.unit
def test_reference_limits_and_no_overshoot():
    position = torch.zeros(4, 3)
    velocity = torch.zeros_like(position)
    desired = torch.ones_like(position)
    for _ in range(400):
        previous = velocity.clone()
        position, velocity = advance_target(position, velocity, desired, 0.1, 0.2, 1 / 30)
        assert (position <= desired).all()
        assert (velocity.abs() <= 0.100001).all()
        # Arrival clamp may stop immediately; accelerating never exceeds the bound.
        assert (velocity.abs() - previous.abs() <= 0.2 / 30 + 1e-6).all()
    assert torch.allclose(position, desired)


@pytest.mark.unit
def test_reference_stays_still_at_goal():
    position = torch.ones(2, 3)
    result, speed = advance_target(position, torch.zeros_like(position), position, 0.1, 0.2, 1 / 30)
    assert torch.equal(result, position)
    assert not speed.any()
