"""Angle-only task success with separate physical grasp/turn/release diagnostics."""

import math

import torch
from isaaclab.utils.math import quat_apply

from wasman.controllers.valve_contract import ValveContract
from wasman.controllers.valve_success import select_valve_success, valve_angle_criteria, valve_success_metadata
from wasman.tasks.underwater_press_button.config.bluerov2_alpha.env import UnderwaterPressButtonEnv


class UnderwaterRotateValveEnv(UnderwaterPressButtonEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        valve_success_metadata(self.cfg.valve_success_contract)  # Fail early on unknown contract versions.
        self.valve_contract = ValveContract(self.num_envs, self.device, self.step_dt)
        self._valve_episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("approach", "alignment", "grasp", "turn", "hold", "success", "attitude", "action_rate")
        }

    def finger_forces(self):
        return self.finger_force_vectors().norm(dim=-1)

    def finger_force_vectors(self):
        return torch.stack(
            [
                self.scene[name].data.normal_force_matrix_w.torch.reshape(self.num_envs, -1, 3).sum(1)
                for name in ("left_contact", "right_contact")
            ],
            1,
        )

    def opposing_contacts(self):
        forces = self.finger_force_vectors()
        unit = forces / forces.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (unit[:, 0] * unit[:, 1]).sum(-1) < -0.25

    def wheel_clearance(self):
        # Axial distance is conservative for a withdrawal along -X: require the
        # tool behind the wheel plane, not merely far from one rotating marker.
        wheel = self.button.data.body_link_pos_w.torch[:, self._button_body_id]
        tool = self.robot.data.body_link_pos_w.torch[:, self._tool_body_id]
        return wheel[:, 0] - tool[:, 0]

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        angle = self.button.data.joint_pos.torch[:, self._button_joint_id]
        speed = self.button.data.joint_vel.torch[:, self._button_joint_id]
        stable = self._base_attitude_error < self.cfg.success_max_attitude_error
        stable &= self._base_angular_speed < self.cfg.success_max_angular_speed
        stable &= self._alignment > self.cfg.tool_contact_alignment
        strict_success = self.valve_contract.update(
            angle,
            speed,
            self.finger_forces(),
            self.wheel_clearance(),
            self.extras["wasman_tool_speed"],
            stable,
            self.opposing_contacts(),
        )
        success = select_valve_success(angle, strict_success, self.cfg.valve_success_contract)
        # The parent has just updated its button-specific counter. Replace that
        # result using valve-only history, preserving success through the episode.
        self._valve_episode_succeeded |= success
        self._episode_succeeded.copy_(self._valve_episode_succeeded)
        self.extras["wasman_success"] = success.clone()
        self.extras["wasman_strict_valve_success"] = strict_success.clone()
        self.extras["success_criteria"] = {k: v.clone() for k, v in valve_angle_criteria(angle).items()}
        self.extras["wasman_success_contract"] = self.cfg.valve_success_contract
        self.extras["wasman_valve_angle"] = angle.clone()
        self.extras["wasman_valve_angular_speed"] = speed.clone()
        self.extras["wasman_valve_held"] = self.valve_contract.held.clone()
        self.extras["wasman_grasp_turn"] = self.valve_contract.grasp_turn.clone()
        self.extras["wasman_ungrasped_motion"] = self.valve_contract.ungrasped_motion.clone()
        self.extras["wasman_contact_opposition"] = self.opposing_contacts().clone()
        self.extras["wasman_finger_forces"] = self.finger_forces().clone()
        self.extras["wasman_grasped"] = self.valve_contract.previous_bilateral.clone()
        return terminated, truncated

    def _get_observations(self):
        legacy = super()._get_observations()["policy"]
        angle = self.button.data.joint_pos.torch[:, self._button_joint_id]
        speed = self.button.data.joint_vel.torch[:, self._button_joint_id]
        grip = self.robot.data.joint_pos.torch[:, self._gripper_joint_ids] / 0.0098
        grip_speed = self.robot.data.joint_vel.torch[:, self._gripper_joint_ids] / 0.05
        tool_y = quat_apply(
            self.robot.data.body_link_quat_w.torch[:, self._tool_body_id],
            legacy.new_tensor([0.0, 1.0, 0.0]).expand(self.num_envs, -1),
        )
        contract = self.valve_contract
        extra = torch.cat(
            (
                angle[:, None] / math.pi,
                speed[:, None],
                grip,
                grip_speed,
                self.finger_forces().clamp(max=10) / 5,
                tool_y,
                contract.grasp_turn[:, None] / math.pi,
                contract.held[:, None].float(),
                contract.hold_steps[:, None].clamp(max=30).float() / 30,
            ),
            -1,
        )
        return {"policy": torch.cat((legacy, extra), -1)}

    def _get_rewards(self):
        # A one-finger push earns no turn reward. Log the actual returned terms,
        # not the parent's button-specific shaping.
        bilateral = self.valve_contract.previous_bilateral
        angle = self.button.data.joint_pos.torch[:, self._button_joint_id]
        angle_error = (angle - math.radians(173)).abs()
        grasp = bilateral.float()
        terms = {
            "approach": 3 * torch.exp(-12 * self._distance),
            "alignment": self._alignment,
            "grasp": 4 * grasp,
            "turn": 8 * grasp * (1 - angle_error / math.pi).clamp(0, 1),
            "hold": 15 * self.valve_contract.held.float(),
            "success": 30 * self.extras["wasman_success"].float(),
            "attitude": -2 * self._base_attitude_error.square(),
            "action_rate": -0.02 * (self.actions - self.previous_actions).square().sum(-1),
        }
        reward = torch.zeros(self.num_envs, device=self.device)
        for name, value in terms.items():
            scaled = torch.nan_to_num(value * self.step_dt)
            self._episode_reward_sums[name] += scaled
            reward += scaled
        return torch.where(self._invalid_state, self.cfg.invalid_state_penalty, reward)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "valve_contract"):
            self.valve_contract.reset(env_ids)
        if hasattr(self, "_valve_episode_succeeded"):
            self._valve_episode_succeeded[env_ids] = False
