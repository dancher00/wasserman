"""Declared actuator-capacity intervention; preserve the native static curve and lag."""

import torch


def capped_allocation(motors, wrench, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("Capacity fraction must lie in (0,1]")
    requested = wrench @ motors.inverse.T
    limits = torch.where(requested >= 0, motors.force_knots[-1], -motors.force_knots[0]) * fraction
    scale = (limits / requested.abs().clamp_min(1e-6)).amin(dim=-1, keepdim=True).clamp(max=1.0)
    return requested, scale
