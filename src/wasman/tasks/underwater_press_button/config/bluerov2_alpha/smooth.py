"""Smooth press-and-hold variant. Old benchmark/checkpoints remain unchanged."""

import torch
from isaaclab.utils import configclass

from wasman.controllers.trajectory import advance_target

from .env import UnderwaterPressButtonEnv
from .env_cfg import UnderwaterPressButtonEnvCfg


@configclass
class SmoothPressButtonEnvCfg(UnderwaterPressButtonEnvCfg):
    success_hold_steps = 30
    success_max_travel = 0.007
    success_max_tool_speed = 0.035
    reward_press_progress = 0.0
    reward_success = 0.0
    reward_press = 0.0
    penalty_action_rate = -0.10
    episode_length_s = 16.0


class SmoothPressButtonEnv(UnderwaterPressButtonEnv):
    """Rate-limited approach and a reward for steady depth, not repeated impacts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_reference_velocity = torch.zeros_like(self._base_target_pos_w)
        self._arm_reference_velocity = torch.zeros_like(self._arm_targets)
        self._smooth_previous_depth = torch.zeros(self.num_envs, device=self.device)

    def _pre_physics_step(self, actions):
        previous_base = self._base_target_pos_w.clone()
        previous_arm = self._arm_targets.clone()
        # Low-pass requested pose, followed by physical-unit reference limits.
        super()._pre_physics_step(self.actions + 0.12 * (actions.clamp(-1, 1) - self.actions))
        self._limit_requested_targets()
        base_speed, arm_speed = self._reference_limits()
        self._base_target_pos_w[:], self._base_reference_velocity[:] = advance_target(
            previous_base, self._base_reference_velocity, self._base_target_pos_w, base_speed, 0.18, self.step_dt
        )
        self._arm_targets[:], self._arm_reference_velocity[:] = advance_target(
            previous_arm, self._arm_reference_velocity, self._arm_targets, arm_speed, 0.6, self.step_dt
        )

    def _limit_requested_targets(self):
        """Optional variant-specific interlocks, before reference rate limiting."""

    def _reference_limits(self):
        near = (self._distance < 0.10).unsqueeze(-1)
        return torch.where(near, 0.015, 0.10), torch.where(near, 0.10, 0.35)

    def _get_rewards(self):
        reward = super()._get_rewards()
        depth = self._button_travel
        engaged = (self._distance < self.cfg.tool_contact_distance) & (self._alignment > 0.70)
        steady = torch.exp(-((depth - 0.0055) / 0.0015).square()) * engaged
        tool_speed = self.robot.data.body_com_lin_vel_w.torch[:, self._tool_body_id].norm(dim=-1)
        overtravel = ((depth - 0.007).clamp_min(0.0) / 0.003).square()
        # Signed potential progress: releasing and pressing again cannot farm it.
        progress = (depth - self._smooth_previous_depth) / self.cfg.button_pressed_threshold
        self._smooth_previous_depth.copy_(depth)
        extra = 65.0 * steady - 60.0 * overtravel - 120.0 * engaged * tool_speed.square() + 3.0 * progress
        return reward + extra * self.step_dt

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        indices = slice(None) if env_ids is None else env_ids
        self._base_target_pos_w[indices] = self.robot.data.root_pos_w.torch[indices]
        self._base_reference_velocity[indices] = 0.0
        self._arm_reference_velocity[indices] = 0.0
        self._smooth_previous_depth[indices] = 0.0
