"""Rate-limited Cartesian reference controller for a station-kept UVMS.

Base attitude stays with the station keeper; translation + four arm joints
solve a damped, weighted IK problem. Commands go through ordinary actuators.
"""

import torch


def damped_step(jacobian, error, weights, damping=0.04):
    weighted = jacobian * weights
    identity = torch.eye(jacobian.shape[-2], device=jacobian.device, dtype=jacobian.dtype)
    dual = torch.linalg.solve(weighted @ jacobian.transpose(-1, -2) + damping**2 * identity, error.unsqueeze(-1))
    return (weighted.transpose(-1, -2) @ dual).squeeze(-1)


class ToolPoseController:
    def __init__(self, env, posture_gain=0.0, posture_joint_target=None):
        self.env = env
        self.posture_gain = posture_gain
        self.posture_joint_target = (
            None if posture_joint_target is None else torch.tensor(posture_joint_target, device=env.device)
        )
        self.weights = torch.tensor([0.35, 0.35, 0.35, 1.0, 1.0, 1.0, 1.0], device=env.device)
        self.max_speed = torch.tensor([0.10, 0.10, 0.10, 0.35, 0.35, 0.35, 0.35], device=env.device)

    def actions(self, target_position_w, target_quaternion_w, gripper):
        from isaaclab.utils.math import compute_pose_error

        env = self.env
        position = env.robot.data.body_link_pos_w.torch[:, env._tool_body_id]
        quaternion = env.robot.data.body_link_quat_w.torch[:, env._tool_body_id]
        dp, dr = compute_pose_error(position, quaternion, target_position_w, target_quaternion_w)
        error = torch.cat((dp, dr), -1)
        # Fixed attitude base: only its three world-translation columns are used.
        base_j = torch.zeros(env.num_envs, 6, 3, device=env.device)
        base_j[:, :3] = torch.eye(3, device=env.device)
        arm_columns = [6 + j for j in env._arm_joint_ids]
        arm_j = env.robot.data.body_link_jacobian_w.torch[:, env._tool_body_id, :, arm_columns]
        jacobian = torch.cat((base_j, arm_j), -1)
        delta = damped_step(jacobian, error, self.weights)
        base_actual = env.robot.data.body_link_pos_w.torch[:, env._base_body_id]
        arm_actual = env.robot.data.joint_pos.torch[:, env._arm_joint_ids]
        if self.posture_gain:
            posture = env._arm_nominal_targets if self.posture_joint_target is None else self.posture_joint_target
            rest = torch.cat((torch.zeros_like(base_actual), self.posture_gain * (posture - arm_actual)), -1)
            null_task_error = (jacobian @ rest.unsqueeze(-1)).squeeze(-1)
            delta += rest - damped_step(jacobian, null_task_error, self.weights)
        desired = torch.cat((base_actual, arm_actual), -1) + delta
        previous = torch.cat((env._base_target_pos_w, env._arm_targets), -1)
        reference = previous + (desired - previous).clamp(-self.max_speed * env.step_dt, self.max_speed * env.step_dt)
        base, arm = reference[:, :3], reference[:, 3:]
        result = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
        result[:, :3] = (base - env.scene.env_origins - env._base_target_nominal) / env._base_target_position_scale
        result[:, 6:10] = (arm - env._arm_nominal_targets) / env._arm_target_scale
        if not env.cfg.policy_gripper:
            raise ValueError("Tool-pose controller requires the gripper-enabled task interface")
        result[:, 10] = gripper
        return result.clamp(-1.0, 1.0)
