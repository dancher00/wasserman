"""Development state-feedback hatch expert; commands only the robot actuators."""

import math

import torch

from wasman.controllers.tool_pose import ToolPoseController


class HatchExpert:
    phase_names = ("standoff", "descend", "grasp", "lift", "hold", "complete")

    def __init__(self, env):
        self.env = env
        self.ik = ToolPoseController(env, posture_gain=0.5, posture_joint_target=(math.pi, -0.5, 2.64, math.pi / 6))
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.settled = torch.zeros_like(self.phase)
        self.reference = torch.zeros(env.num_envs, device=env.device)
        self.gripper = torch.ones_like(self.reference)
        self.grasp_angle = torch.zeros_like(self.reference)
        self.grasp_offset = torch.zeros(env.num_envs, 3, device=env.device)
        self.grasp_tool_offset = self.grasp_offset.clone()
        self.grasp_quaternion = torch.zeros(env.num_envs, 4, device=env.device)
        self.target = self.grasp_offset.clone()
        self.navigation_ready = torch.zeros_like(self.phase, dtype=torch.bool)
        self.navigation_settled = torch.zeros_like(self.phase)
        self.transit_reference = env.robot.data.body_link_pos_w.torch[:, env._base_body_id].clone()

    def finger_forces(self):
        return torch.stack(
            [
                self.env.scene[name].data.normal_force_matrix_w.torch.reshape(self.env.num_envs, -1, 3).sum(1)
                for name in ("left_contact", "right_contact")
            ],
            1,
        ).norm(dim=-1)

    def actions(self):
        from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

        env = self.env
        base = env.robot.data.body_link_pos_w.torch[:, env._base_body_id]
        staging = env.scene.env_origins + env._base_target_nominal
        delta = staging - self.transit_reference
        self.transit_reference += delta * (0.10 * env.step_dt / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(
            max=1
        )
        base_speed = env.robot.data.body_com_lin_vel_w.torch[:, env._base_body_id].norm(dim=-1)
        settled = ((base - staging).norm(dim=-1) < 0.035) & (base_speed < 0.025)
        settled &= env._base_attitude_error < 0.07
        self.navigation_settled[:] = torch.where(settled, self.navigation_settled + 1, 0)
        self.navigation_ready |= self.navigation_settled >= 15
        pivot = env.button.data.body_link_pos_w.torch[:, env._button_body_id]
        angle = env.button.data.joint_pos.torch[:, env._button_joint_id]
        speed = env.button.data.joint_vel.torch[:, env._button_joint_id]
        tool = env.robot.data.body_link_pos_w.torch[:, env._tool_body_id]
        tool_q = env.robot.data.body_link_quat_w.torch[:, env._tool_body_id]
        zero = torch.zeros_like(angle)
        target_q = quat_from_euler_xyz(zero + math.pi / 2, zero + math.pi / 2, zero)
        offset = pivot.new_tensor(env.cfg.mechanism_contact_offset).expand_as(pivot).clone()
        offset[:, 2] -= 0.008
        target = pivot + quat_apply(env.button.data.body_link_quat_w.torch[:, env._button_body_id], offset)
        target[self.phase == 0, 2] += 0.24
        forces = self.finger_forces()
        bilateral = (forces > 0.15).all(-1)
        advancing = (self.phase == 3) & bilateral
        self.reference[advancing] += 0.08 * env.step_dt
        self.reference[:] = torch.minimum(self.reference, (angle - self.grasp_angle + 0.10).clamp_min(0))
        self.reference.clamp_(max=math.radians(85))
        rotation = quat_from_euler_xyz(zero, self.reference, zero)
        lifting = self.phase >= 3
        # The cylindrical bar can rotate in the fingers. Keep a top-down grasp
        # instead of rotating the whole robot/arm into the rising lid's swept volume.
        target[lifting] = (pivot + quat_apply(rotation, self.grasp_offset) + self.grasp_tool_offset)[lifting]
        target_q[lifting] = self.grasp_quaternion[lifting]
        request_grip = torch.where(self.phase < 2, 1.0, -1.0)
        self.gripper += (request_grip - self.gripper).clamp(-env.step_dt, env.step_dt)
        error = (tool - target).norm(dim=-1)
        tool_speed = env.robot.data.body_com_lin_vel_w.torch[:, env._tool_body_id].norm(dim=-1)
        orientation_error = 1 - (tool_q * target_q).sum(-1).abs()
        eligible = (error < 0.012) & (tool_speed < 0.025) & (orientation_error < 0.008)
        eligible = torch.where(self.phase == 2, bilateral & (self.gripper < -0.99), eligible)
        eligible = torch.where(self.phase == 3, (angle > math.radians(82)) & (speed.abs() < 0.05), eligible)
        eligible = torch.where(self.phase == 4, env._episode_succeeded, eligible)
        eligible &= self.navigation_ready
        self.settled[:] = torch.where(eligible, self.settled + 1, 0)
        transition = (self.settled >= 12) & (self.phase < 5)
        grasp = transition & (self.phase == 2)
        handle_offset = quat_apply(
            env.button.data.body_link_quat_w.torch[:, env._button_body_id],
            pivot.new_tensor(env.cfg.mechanism_contact_offset).expand_as(pivot),
        )
        self.grasp_offset[grasp] = handle_offset[grasp]
        self.grasp_tool_offset[grasp] = (tool - pivot - handle_offset)[grasp]
        self.grasp_quaternion[grasp] = tool_q[grasp]
        self.grasp_angle[grasp] = angle[grasp]
        self.phase[transition] += 1
        self.settled[transition] = 0
        self.target.copy_(target)
        actions = self.ik.actions(target, target_q, self.gripper)
        transit = ~self.navigation_ready
        actions[transit, :3] = (
            (self.transit_reference - env.scene.env_origins - env._base_target_nominal)
            / env._base_target_position_scale
        )[transit]
        actions[transit, 6:10] = 0  # Keep the physical arm folded until the vehicle settles.
        actions[transit, 10] = -1
        self.gripper[transit] = -1
        return actions.clamp(-1, 1)
