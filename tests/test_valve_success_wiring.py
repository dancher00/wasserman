"""Check the real environment's scoring/logger wiring without simulating physics."""

import math
from types import SimpleNamespace

import torch

from wasman.controllers.valve_contract import ValveContract
from wasman.controllers.valve_success import AMB_VALVE_SUCCESS
from wasman.tasks.underwater_panel.valve import UnderwaterPressButtonEnv, UnderwaterRotateValveEnv


def test_angle_gate_overrides_parent_gates_and_episode_log_survives_angle_drop(monkeypatch):
    env = object.__new__(UnderwaterRotateValveEnv)
    env.cfg = SimpleNamespace(
        valve_success_contract=AMB_VALVE_SUCCESS,
        success_max_attitude_error=0.25,
        success_max_angular_speed=0.35,
        tool_contact_alignment=0.7,
    )
    angle = torch.tensor([[math.radians(170)], [0.0]])
    env.button = SimpleNamespace(
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=angle), joint_vel=SimpleNamespace(torch=torch.ones(2, 1)))
    )
    env._button_joint_id = 0
    env._base_attitude_error = env._base_angular_speed = torch.ones(2)
    env._alignment = torch.zeros(2)
    env.valve_contract = ValveContract(2, "cpu", 1 / 30)
    env._valve_episode_succeeded = torch.zeros(2, dtype=torch.bool)
    env._episode_succeeded = torch.zeros(2, dtype=torch.bool)
    env.extras = {"wasman_tool_speed": torch.ones(2)}
    env.finger_forces = lambda: torch.zeros(2, 2)
    env.opposing_contacts = lambda: torch.zeros(2, dtype=torch.bool)
    env.wheel_clearance = lambda: torch.zeros(2)

    def parent_dones(self):
        # Deliberately contaminate the parent's button counter.
        self._episode_succeeded[:] = True
        return torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)

    monkeypatch.setattr(UnderwaterPressButtonEnv, "_get_dones", parent_dones)
    terminated, truncated = env._get_dones()
    assert not terminated.any() and not truncated.any()
    assert env.extras["wasman_success"].tolist() == [True, False]
    assert not env.extras["wasman_strict_valve_success"].any()
    assert env._episode_succeeded.tolist() == [True, False]
    angle[:] = 0
    env._get_dones()
    assert not env.extras["wasman_success"].any()
    assert env._episode_succeeded.tolist() == [True, False]

    logged = []

    def parent_reset(self, ids):
        logged.append(self._episode_succeeded[ids].clone())
        self._episode_succeeded[ids] = False

    monkeypatch.setattr(UnderwaterPressButtonEnv, "_reset_idx", parent_reset)
    env._reset_idx(torch.tensor([0]))
    assert logged[0].item()
    assert not env._valve_episode_succeeded.any()
    env._get_dones()
    assert not env._episode_succeeded.any()
