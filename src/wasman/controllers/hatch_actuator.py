"""Lossless, phase-free eight-channel interface to the existing hatch controller.

The three base attitude channels are zero for every expert action. The eight
remaining absolute normalized targets retain both folded transit and deployed
manipulation without consulting the object, contact sensors or expert phase.
"""

import torch


def pack_action(action):
    if action[..., 3:6].abs().max() > 1e-6:
        raise ValueError("Eight-channel interface requires level-attitude targets")
    return torch.cat((action[..., :3], action[..., 6:]), -1)


def unpack_action(command):
    if command.shape[-1] != 8 or not torch.isfinite(command).all():
        raise ValueError("Expected eight finite normalized target channels")
    result = command.new_zeros((*command.shape[:-1], 11))
    result[..., :3] = command[..., :3]
    result[..., 6:] = command[..., 3:]
    return result


def measured_state(env):
    base = (
        env.robot.data.body_link_pos_w.torch[:, env._base_body_id] - env.scene.env_origins - env._base_target_nominal
    ) / env._base_target_position_scale
    arm = (env.robot.data.joint_pos.torch[:, env._arm_joint_ids] - env._arm_nominal_targets) / env._arm_target_scale
    limits = env.robot.data.soft_joint_pos_limits.torch[:, env._gripper_joint_ids[0]]
    jaw = env.robot.data.joint_pos.torch[:, env._gripper_joint_ids[0]]
    jaw = 2 * (jaw - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1
    return torch.cat((base, arm, jaw[:, None]), -1)
