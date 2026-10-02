"""Measured valve completion, independent of the controller commanding the robot."""

import math

import torch


class ValveContract:
    """Require a two-finger turn, settled hold, then a contact-free withdrawal.

    Inputs are measured physical quantities. In particular a spun wheel alone,
    a commanded gripper closure, or an expert phase is never success evidence.
    """

    def __init__(self, num_envs, device, dt):
        self.dt = dt
        self.contact = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.grasp_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.hold_steps = torch.zeros_like(self.grasp_steps)
        self.release_steps = torch.zeros_like(self.grasp_steps)
        self.grasp_turn = torch.zeros(num_envs, device=device)
        self.previous_angle = torch.zeros_like(self.grasp_turn)
        self.grasp_started = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ungrasped_motion = torch.zeros_like(self.grasp_turn)
        self.previous_bilateral = torch.zeros_like(self.grasp_started)
        self.held = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success = torch.zeros_like(self.held)

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    def update(self, angle, angular_speed, finger_forces, tool_clearance, tool_speed, stable_base, opposed=None):
        previous_bilateral = self.previous_bilateral.clone()
        self.contact = torch.where(self.contact, finger_forces > 0.05, finger_forces >= 0.1)
        bilateral = self.contact.all(-1)
        if opposed is not None:
            bilateral &= opposed
        self.grasp_started |= bilateral
        self.grasp_steps = torch.where(bilateral, self.grasp_steps + 1, 0)
        # Signed progress prevents accumulating success by wiggling the wheel.
        delta = angle - self.previous_angle
        self.ungrasped_motion += torch.where(
            self.grasp_started & ~(bilateral & previous_bilateral), delta.abs(), torch.zeros_like(delta)
        )
        self.grasp_turn += torch.where(bilateral & previous_bilateral, delta, torch.zeros_like(delta))
        self.previous_angle.copy_(angle)
        self.previous_bilateral.copy_(bilateral)
        in_goal = (angle >= math.radians(170)) & (angle <= math.radians(175))
        turned_in_grasp = self.grasp_turn >= math.radians(165)
        turned_in_grasp &= self.ungrasped_motion <= math.radians(5)
        settled = angular_speed.abs() <= 0.05
        hold = in_goal & turned_in_grasp & bilateral & settled & stable_base & (tool_speed <= 0.025)
        self.hold_steps = torch.where(hold, self.hold_steps + 1, 0)
        self.held |= self.hold_steps >= round(1.0 / self.dt)
        released = self.held & in_goal & settled & ~self.contact.any(-1)
        released &= (tool_clearance >= 0.08) & stable_base
        self.release_steps = torch.where(released, self.release_steps + 1, 0)
        self.success |= self.release_steps >= round(0.5 / self.dt)
        return self.success
