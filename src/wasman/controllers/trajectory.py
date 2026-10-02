"""Bounded target motion, shared by policy control and deterministic tests."""

import torch


def advance_target(position, velocity, desired, max_speed, acceleration, dt):
    """Per-axis speed/acceleration-limited tracking; clamp arrival without overshoot."""
    error = desired - position
    requested = (error / dt).clamp(-max_speed, max_speed)
    velocity = velocity + (requested - velocity).clamp(-acceleration * dt, acceleration * dt)
    step = velocity * dt
    arrived = (step * error >= 0) & (step.abs() >= error.abs())
    return torch.where(arrived, desired, position + step), torch.where(arrived, 0.0, velocity)
