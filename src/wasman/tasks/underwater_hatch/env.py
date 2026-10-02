"""Passive hatch opening with measured contact, opening angle and stable hold."""

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from wasman.controllers.hatch_contract import HatchContract
from wasman.tasks.underwater_press_button.config.bluerov2_alpha.env import UnderwaterPressButtonEnv


class UnderwaterHatchEnv(UnderwaterPressButtonEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hatch_contract = HatchContract(self.num_envs, self.device, self.step_dt)

    def finger_force_vectors(self):
        return torch.stack(
            [
                self.scene[name].data.normal_force_matrix_w.torch.reshape(self.num_envs, -1, 3).sum(1)
                for name in ("left_contact", "right_contact")
            ],
            1,
        )

    def _update_task_state(self):
        super()._update_task_state()
        # Approach the handle normal from above, following the hinged lid.
        normal = self._target_delta_b.new_tensor((0.0, 0.0, -1.0)).expand(self.num_envs, 3)
        desired = quat_apply(self.button.data.body_link_quat_w.torch[:, self._button_body_id], normal)
        tool_axis = self._target_delta_b.new_tensor((1.0, 0.0, 0.0)).expand(self.num_envs, 3)
        actual = quat_apply(self.robot.data.body_link_quat_w.torch[:, self._tool_body_id], tool_axis)
        self._alignment.copy_((desired * actual).sum(-1))
        self._alignment_error_b.copy_(
            quat_apply_inverse(self.robot.data.body_link_quat_w.torch[:, self._base_body_id], desired - actual)
        )

    def _get_dones(self):
        done, timeout = super()._get_dones()
        success = self.hatch_contract.update(
            self.button.data.joint_pos.torch[:, self._button_joint_id],
            self.button.data.joint_vel.torch[:, self._button_joint_id],
            self.finger_force_vectors(),
            self._distance < self.cfg.tool_contact_distance,
            self.extras["wasman_tool_speed"],
            (self._base_attitude_error < self.cfg.success_max_attitude_error)
            & (self._base_angular_speed < self.cfg.success_max_angular_speed),
        )
        self._episode_succeeded.copy_(success)
        self.extras["wasman_success"] = success.clone()
        self.extras["wasman_hatch_grasp_motion"] = self.hatch_contract.grasp_motion.clone()
        self.extras["wasman_hatch_ungrasped_motion"] = self.hatch_contract.ungrasped_motion.clone()
        self.extras["wasman_hatch_opposing_grasp"] = self.hatch_contract.previous_grasp.clone()
        self.extras["wasman_hatch_angle_rad"] = self.button.data.joint_pos.torch[:, self._button_joint_id].clone()
        self.extras["wasman_hatch_speed_rad_s"] = self.button.data.joint_vel.torch[:, self._button_joint_id].clone()
        return done, timeout

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "hatch_contract"):
            self.hatch_contract.reset(env_ids)
