"""Contact interlock and normal-force regulation for a spring-loaded button."""

import torch


def normal_press_force(depth, normal_speed, bias=0.0, target_depth=0.0055):
    """Measured depth feedback and normal damping, in newtons.

    This request goes through real T200 motors, not directly to the tool or
    button. Reverse thrust brakes overtravel.
    """
    return (bias + 2000.0 * (target_depth - depth) - 300.0 * normal_speed).clamp(-15.0, 15.0)


class PressHoldController:
    """Latch the reached arm pose; maintain contact with normal-force control."""

    def __init__(self, env):
        self.env = env
        self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.base_position = torch.zeros_like(env._base_target_pos_w)
        self.base_quaternion = torch.zeros_like(env._base_target_quat_w)
        self.arm_target = torch.zeros_like(env._arm_targets)

    def update(self, eligible, previous_base_position, previous_base_quaternion, previous_arm_target):
        env = self.env
        touch = eligible & (env._button_travel >= 0.0005)
        touch &= (env._distance < env.cfg.tool_contact_distance) & (env._alignment > 0.70)
        newly_active = touch & ~self.active
        self.base_position[newly_active] = previous_base_position[newly_active]
        self.base_quaternion[newly_active] = previous_base_quaternion[newly_active]
        self.arm_target[newly_active] = previous_arm_target[newly_active]
        self.active |= newly_active
        env._arm_targets[self.active] = self.arm_target[self.active]
        env._base_target_pos_w[self.active] = self.base_position[self.active]
        env._base_target_quat_w[self.active] = self.base_quaternion[self.active]

    def reset(self, indices=None):
        self.active[slice(None) if indices is None else indices] = False
