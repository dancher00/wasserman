"""Measured-state approach/grasp/trajectory expert; robot commands only."""

import math

import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from wasman.controllers.tool_pose import ToolPoseController


class MarineMechanismExpert:
    def __init__(self, env, mechanism, *, approach_offset_x=0.0):
        self.env, self.mechanism = env, mechanism
        if not -0.04 <= approach_offset_x <= 0.04:
            raise ValueError("Development approach offset must be within 40 mm")
        self.approach_offset_x = approach_offset_x
        self.ik = ToolPoseController(env, posture_gain=0.08)
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.settled = torch.zeros_like(self.phase)
        self.reference = torch.zeros(env.num_envs, device=env.device)
        self.gripper = torch.zeros_like(self.reference)
        self.target = torch.zeros(env.num_envs, 3, device=env.device)
        self.anchor = self.target.clone()
        self.grasp_q = torch.zeros(env.num_envs, 4, device=env.device)
        self.grasp_q[:, 3] = 1
        self.grasp_joint = self.reference.clone()

    def finger_forces(self):
        return torch.stack(
            [
                self.env.scene[n].data.normal_force_matrix_w.torch.reshape(self.env.num_envs, -1, 3).sum(1).norm(dim=-1)
                for n in ["left_contact", "right_contact"]
            ],
            -1,
        )

    def actions(self):
        e = self.env
        q = e.button.data.joint_pos.torch[:, e._button_joint_id]
        bp = e.button.data.body_link_pos_w.torch[:, e._button_body_id]
        bq = e.button.data.body_link_quat_w.torch[:, e._button_body_id]
        tool = e.robot.data.body_link_pos_w.torch[:, e._tool_body_id]
        tip = bp + quat_apply(bq, bp.new_tensor(e.cfg.mechanism_contact_offset).expand_as(bp))
        target = tip.clone()
        target[:, 0] += self.approach_offset_x
        if self.mechanism == "PullLever":
            # Grasp the crossbar away from the T-junction with the lever stem.
            # The center grasp lost one finger near35degrees in the RGB pilot.
            target[:, 1] += 0.024
        zero = torch.zeros_like(q)
        tq = quat_from_euler_xyz(zero + math.pi / 2, zero, zero)
        target[self.phase == 0, 0] -= 0.10
        forces = self.finger_forces()
        bilateral = (forces > 0.12).all(-1)
        moving = self.phase >= 3
        if self.mechanism == "PushSlider":
            self.reference += 0.015 * e.step_dt * ((self.phase == 3) & bilateral)
            self.reference[:] = torch.minimum(self.reference, (q - self.grasp_joint + 0.015).clamp_min(0))
            self.reference.clamp_(max=0.295)
            target[moving] = self.anchor[moving]
            # Source fixed root is rotated180 degrees: local +Y is world -Y.
            target[moving, 1] -= self.reference[moving]
        else:
            self.reference += 0.07 * e.step_dt * ((self.phase == 3) & bilateral)
            self.reference[:] = torch.minimum(self.reference, (q - self.grasp_joint + 0.08).clamp_min(0))
            self.reference[:] = torch.minimum(self.reference, (math.radians(44) - self.grasp_joint).clamp_min(0))
            rotation = quat_from_euler_xyz(zero, self.reference, zero)
            target[moving] = (bp + quat_apply(rotation, self.anchor))[moving]
        requested = torch.where(self.phase < 2, 0.5, -1.0)
        self.gripper += (requested - self.gripper).clamp(-e.step_dt, e.step_dt)
        err = (tool - target).norm(dim=-1)
        speed = e.robot.data.body_com_lin_vel_w.torch[:, e._tool_body_id].norm(dim=-1)
        eligible = (err < (0.012 if self.mechanism == "PushSlider" else 0.008)) & (speed < 0.03)
        eligible = torch.where(self.phase == 2, bilateral & (self.gripper < -0.99), eligible)
        eligible = torch.where(
            self.phase == 3, q - e.cfg.mechanism_initial_position >= e.cfg.button_pressed_threshold, eligible
        )
        self.settled[:] = torch.where(eligible, self.settled + 1, 0)
        transition = (self.settled >= 10) & (self.phase < 4)
        capture = transition & (self.phase == 2)
        self.anchor[capture] = (tool if self.mechanism == "PushSlider" else tool - bp)[capture]
        self.grasp_joint[capture] = q[capture]
        self.phase[transition] += 1
        self.settled[transition] = 0
        self.target.copy_(target)
        return self.ik.actions(target, tq, self.gripper)
