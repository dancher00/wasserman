"""Object-free diagnostics sharing the released robot's actual hydro/motor/controller path.

Not a registered learning task and no manipulation success is defined here.
"""

import torch

from wasman.tasks.underwater_press_button.config.bluerov2_alpha.env import UnderwaterPressButtonEnv


class ControllerDiagnosticEnv(UnderwaterPressButtonEnv):
    diagnostic_only = True

    def __init__(self, cfg, **kwargs):
        if cfg.scene.panel is not None or cfg.scene.button is not None:
            raise ValueError("Controller diagnostics must contain no manipulation objects")
        cfg.observation_space = 13
        super().__init__(cfg, **kwargs)

    def _get_observations(self):
        body = self._base_body_id
        return {
            "policy": torch.cat(
                (
                    self.robot.data.body_link_pos_w.torch[:, body],
                    self.robot.data.body_link_quat_w.torch[:, body],
                    self.robot.data.body_com_vel_w.torch[:, body],
                ),
                dim=-1,
            )
        }

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        position = self.robot.data.body_link_pos_w.torch
        velocity = self.robot.data.body_com_vel_w.torch
        finite = torch.isfinite(position).all(dim=(1, 2))
        finite &= torch.isfinite(self.robot.data.body_link_quat_w.torch).all(dim=(1, 2))
        finite &= torch.isfinite(velocity).all(dim=(1, 2))
        finite &= torch.isfinite(self.robot.data.joint_pos.torch).all(dim=1)
        finite &= torch.isfinite(self.robot.data.joint_vel.torch).all(dim=1)
        valid = velocity[..., :3].abs().amax(dim=(1, 2)) < self.cfg.max_valid_body_linear_speed
        valid &= velocity[..., 3:].abs().amax(dim=(1, 2)) < self.cfg.max_valid_body_angular_speed
        valid &= self.robot.data.joint_vel.torch.abs().amax(dim=1) < self.cfg.max_valid_joint_speed
        self._invalid_state.copy_(~(finite & valid))
        base = position[:, self._base_body_id] - self.scene.env_origins
        escaped = base.norm(dim=-1) > self.cfg.max_base_distance
        escaped |= (base[:, 2] < self.cfg.min_base_height) | (base[:, 2] > self.cfg.max_base_height)
        return self._invalid_state | escaped, self.episode_length_buf >= self.max_episode_length
