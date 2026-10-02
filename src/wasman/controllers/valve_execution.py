"""Stage-independent actuator execution; no object state or success feedback."""

import torch


def execute_valve_action(action: torch.Tensor, observation: torch.Tensor, dt: float) -> torch.Tensor:
    """Limit reference velocities and keep the vehicle level (XYZW unaffected).

    Public previous *applied* references make this stateless across resets.
    Neither the wheel angle, contact, phase nor success enters this mapping.
    PPO retains the sampled raw action for its likelihood calculation.
    """
    if dt <= 0 or action.shape[-1] != 11 or observation.shape[-1] != 50:
        raise ValueError("Expected positive dt, 11 actions and 50 observations")
    rates = action.new_tensor([0.1 / 0.5, 0.1 / 0.35, 0.1 / 0.35,
                               0, 0, 0, 0.35 / 0.8, 0.35, 0.35, 0.35 / 3.2, 0.8])
    previous = observation[..., 27:38]
    result = (previous + (action - previous).clamp(-rates * dt, rates * dt)).clamp(-1, 1)
    return torch.cat((result[..., :3], torch.zeros_like(result[..., 3:6]), result[..., 6:]), -1)
