"""Measured-state interlock: settle the vehicle before enabling arm commands."""

import torch


class ApproachGate:
    def __init__(self, num_envs: int, device, hold_steps: int = 12):
        self.hold_steps = hold_steps
        self.enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stable_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, position_error, linear_speed, angular_speed, attitude_error):
        ready = (position_error[:, 0].abs() < 0.05) & (position_error[:, 1:].norm(dim=-1) < 0.08)
        ready &= (linear_speed < 0.04) & (angular_speed < 0.12) & (attitude_error < 0.22)
        self.stable_steps[:] = torch.where(ready, self.stable_steps + 1, 0)
        self.enabled |= self.stable_steps >= self.hold_steps
        return self.enabled

    def reset(self, indices=None):
        indices = slice(None) if indices is None else indices
        self.enabled[indices] = False
        self.stable_steps[indices] = 0
