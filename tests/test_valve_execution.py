import pytest
import torch

from wasman.controllers.valve_execution import execute_valve_action

pytestmark = pytest.mark.unit


def test_execution_is_rate_limited_level_and_independent_of_task_state():
    obs = torch.zeros(3, 50)
    action = torch.ones(3, 11)
    result = execute_valve_action(action, obs, 1 / 30)
    assert torch.all(result[:, 3:6] == 0)
    assert torch.allclose(result[:, 9], torch.full((3,), 0.35 / 3.2 / 30))
    assert torch.allclose(result[:, 10], torch.full((3,), 0.8 / 30))
    obs[:, :27] = torch.randn(3, 27)
    obs[:, 38:] = torch.randn(3, 12)
    assert torch.equal(result, execute_valve_action(action, obs, 1 / 30))


def test_execution_uses_applied_reference_and_does_not_mutate_inputs():
    obs = torch.zeros(2, 50)
    obs[:, 27:38] = 0.4
    action = torch.full((2, 11), -10.)
    result = execute_valve_action(action, obs, 1 / 30)
    assert (result[:, :3] > 0.39).all()
    assert (action == -10).all()
    assert (obs[:, 27:38] == 0.4).all()
    with pytest.raises(ValueError):
        execute_valve_action(action, obs, 0)
