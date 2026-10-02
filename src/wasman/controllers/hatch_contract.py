"""Measured opening under an opposing grasp; independent of expert phase and commands."""

import math

import torch


class HatchContract:
    def __init__(self, num_envs, device, dt):
        self.dt = dt
        self.previous_angle = torch.zeros(num_envs, device=device)
        self.grasp_motion = torch.zeros_like(self.previous_angle)
        self.ungrasped_motion = torch.zeros_like(self.previous_angle)
        self.previous_grasp = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success = torch.zeros_like(self.previous_grasp)
        self.hold_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    def update(self, angle, speed, forces, near_handle, tool_speed, stable):
        magnitudes = forces.norm(dim=-1)
        unit = forces / magnitudes[..., None].clamp_min(1e-6)
        opposing = (unit[:, 0] * unit[:, 1]).sum(-1) < -0.25
        grasp = (magnitudes > 0.1).all(-1) & opposing & near_handle
        delta = angle - self.previous_angle
        engaged = grasp & self.previous_grasp
        self.grasp_motion += torch.where(engaged, delta, 0.0)
        self.ungrasped_motion += torch.where(~engaged, delta.abs(), 0.0)
        self.previous_angle.copy_(angle)
        self.previous_grasp.copy_(grasp)
        hold = (angle >= math.radians(80)) & (angle <= math.radians(105))
        hold &= (self.grasp_motion >= math.radians(75)) & (self.ungrasped_motion <= math.radians(5))
        hold &= grasp & (speed.abs() <= 0.05) & (tool_speed <= 0.04) & stable
        self.hold_steps[:] = torch.where(hold, self.hold_steps + 1, 0)
        self.success |= self.hold_steps >= round(1 / self.dt)
        return self.success
