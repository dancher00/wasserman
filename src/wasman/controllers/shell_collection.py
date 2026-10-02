"""Measured-state shell expert: commands robot only, never moves or attaches a shell."""

import math

import torch

from wasman.controllers.tool_pose import ToolPoseController


class ShellCollectionExpert:
    phase_names = ("standoff", "descend", "grasp", "lift", "carry", "lower", "release", "withdraw", "complete")

    def __init__(self, env):
        self.env = env
        self.ik = ToolPoseController(env, posture_gain=0.5, posture_joint_target=(math.pi, -0.5, 2.64, math.pi / 6))
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.object_id = torch.zeros_like(self.phase)
        self.settled = torch.zeros_like(self.phase)
        self.gripper = torch.full((env.num_envs,), -1.0, device=env.device)
        self.grasp_tool_position = torch.zeros(env.num_envs, 3, device=env.device)
        self.grasp_shell_offset = torch.zeros_like(self.grasp_tool_position)
        self.transport_reference = torch.zeros_like(self.grasp_tool_position)
        self.lost_grasp_steps = torch.zeros_like(self.phase)
        self.retry_count = torch.zeros_like(self.phase)
        self.navigation_ready = torch.zeros_like(self.phase, dtype=torch.bool)
        self.navigation_settled = torch.zeros_like(self.phase)
        self.transit_reference = env.robot.data.body_link_pos_w.torch[:, env._base_body_id].clone()
        self.target = self.grasp_tool_position.clone()

    def actions(self):
        from isaaclab.utils.math import quat_from_euler_xyz

        env = self.env
        ids = torch.arange(env.num_envs, device=env.device)
        base = env.robot.data.body_link_pos_w.torch[:, env._base_body_id]
        staging = env.scene.env_origins + env._base_target_nominal
        delta = staging - self.transit_reference
        self.transit_reference += delta * (0.1 * env.step_dt / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(
            max=1
        )
        base_speed = env.robot.data.body_com_lin_vel_w.torch[:, env._base_body_id].norm(dim=-1)
        ready = ((base - staging).norm(dim=-1) < 0.035) & (base_speed < 0.025)
        self.navigation_settled[:] = torch.where(ready, self.navigation_settled + 1, 0)
        self.navigation_ready |= self.navigation_settled >= 15
        tool = env.robot.data.body_link_pos_w.torch[:, env._tool_body_id]
        tool_q = env.robot.data.body_link_quat_w.torch[:, env._tool_body_id]
        shell = env.shell_positions()[ids, self.object_id]
        grasped = torch.zeros_like(self.phase, dtype=torch.bool)
        if env.collection_state is not None:
            grasped = env.collection_state["grasped"][ids, self.object_id]
        expecting_grasp = ((self.phase == 2) & (self.gripper < -0.99)) | (self.phase == 3) | (self.phase == 4)
        self.lost_grasp_steps[:] = torch.where(expecting_grasp & ~grasped, self.lost_grasp_steps + 1, 0)
        retry = self.lost_grasp_steps >= torch.where(self.phase == 2, 30, 12)
        self.phase[retry] = 0
        self.settled[retry] = 0
        self.lost_grasp_steps[retry] = 0
        self.retry_count[retry] += 1
        zero = torch.zeros(env.num_envs, device=env.device)
        # Pinch across the short axis: the open fingers cannot descend around
        # the 77 mm long axis. TCP clearance includes the 7 mm fingertip offset.
        target_q = quat_from_euler_xyz(zero + math.pi / 2, zero + math.pi / 2, zero + math.pi / 2)
        target = shell.clone()
        target[:, 2] -= 0.003
        target[self.phase == 0, 2] += 0.18
        holding = self.phase >= 3
        target[holding] = self.grasp_tool_position[holding]
        target[holding, 2] += 0.15
        slots = tool.new_tensor([[-0.10, -0.04, 0.0125], [0.10, -0.04, 0.0125], [0.0, 0.10, 0.0125]])
        destination = env.hoop_center + slots[self.object_id] + self.grasp_shell_offset
        if env.collection_state is not None:
            support_height = shell[:, 2] - env.collection_state["bottom_clearance_m"][ids, self.object_id]
            # A below-centre grasp can put the fingers below the shell's bottom.
            # Keep their measured CAD extent (7 mm below TCP) clear of the bed,
            # then release near it and let ordinary gravity/contact settle it.
            destination[:, 2] = (support_height + self.grasp_shell_offset[:, 2] + 0.002).clamp_min(0.012)
        target[self.phase >= 4] = destination[self.phase >= 4]
        target[self.phase == 4, 2] += 0.15
        target[self.phase >= 7, 2] += 0.20
        request_grip = torch.where((self.phase >= 2) & (self.phase <= 5), -1.0, 1.0)
        self.gripper += (request_grip - self.gripper).clamp(-env.step_dt, env.step_dt)
        forces = env.finger_force_vectors()[ids, self.object_id].norm(dim=-1)
        tool_speed = env.robot.data.body_com_lin_vel_w.torch[:, env._tool_body_id].norm(dim=-1)
        close = (tool - target).norm(dim=-1) < 0.009
        oriented = 1 - (tool_q * target_q).sum(-1).abs() < 0.008
        eligible = close & oriented & (tool_speed < 0.025)
        # The earlier 9 mm pose gate began descent with an 8 mm lateral error,
        # pushing the next shell before closure. Centre first, then descend slowly.
        eligible = torch.where(
            self.phase == 0, eligible & ((tool[:, :2] - target[:, :2]).norm(dim=-1) < 0.004), eligible
        )
        eligible = torch.where(self.phase == 2, grasped & (self.gripper < -0.99), eligible)
        eligible = torch.where(self.phase == 3, eligible & grasped & env.contract.lifted[ids, self.object_id], eligible)
        eligible = torch.where(self.phase == 6, (forces < 0.05).all(-1) & (self.gripper > 0.99), eligible)
        if env.collection_state is not None:
            bottom = env.collection_state["bottom_clearance_m"][ids, self.object_id]
            criteria = env.contract.criteria
            supported = (bottom >= -criteria.penetration_tolerance_m) & (bottom <= criteria.resting_gap_m)
            supported &= env.contract.transported[ids, self.object_id] & (tool_speed < 0.025)
            # Seed5200: the shell was already resting inside the hoop, while a
            # closed finger pressed it (~20 N) and prevented the exact TCP pose.
            # Open once physically supported; release/settle are still scored
            # independently by the unchanged task contract.
            eligible = torch.where(
                self.phase == 5,
                (eligible | supported) & env.collection_state["inside_hoop"][ids, self.object_id],
                eligible,
            )
            eligible = torch.where(
                self.phase == 7, eligible & env.collection_state["placed"][ids, self.object_id], eligible
            )
        eligible &= self.navigation_ready
        self.settled[:] = torch.where(eligible, self.settled + 1, 0)
        transition = (self.settled >= 12) & (self.phase < 8)
        descend = transition & (self.phase == 0)
        grasp = transition & (self.phase == 2)
        self.grasp_tool_position[grasp] = tool[grasp]
        self.grasp_shell_offset[grasp] = (tool - shell)[grasp]
        next_shell = transition & (self.phase == 7) & (self.object_id < env.cfg.shell_count - 1)
        self.phase[transition] += 1
        self.phase[next_shell] = 0
        self.object_id[next_shell] += 1
        self.settled[transition] = 0
        # A discontinuous 150 mm lift produced a sharp tool acceleration and
        # immediate slip in the short-axis audit. Advance the Cartesian target
        # slowly while transporting; ordinary contact must carry the object.
        self.transport_reference[grasp | descend] = tool[grasp | descend]
        moving = (self.phase == 1) | ((self.phase >= 3) & (self.phase <= 5))
        delta = target - self.transport_reference
        speed = torch.where(self.phase == 4, 0.04, 0.02)
        increment = delta * (speed[:, None] * env.step_dt / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(
            max=1
        )
        self.transport_reference[moving] += increment[moving]
        command_target = torch.where(moving[:, None], self.transport_reference, target)
        self.target.copy_(command_target)
        action = self.ik.actions(command_target, target_q, self.gripper)
        transit = ~self.navigation_ready
        action[transit, :3] = (
            (self.transit_reference - env.scene.env_origins - env._base_target_nominal)
            / env._base_target_position_scale
        )[transit]
        action[transit, 6:10] = 0
        action[transit, 10] = -1
        self.gripper[transit] = -1
        return action.clamp(-1, 1)
