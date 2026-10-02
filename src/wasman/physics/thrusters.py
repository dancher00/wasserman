"""Eight independently lagged T200 thrusters, using public 16 V static data.

No ideal-wrench bypass: output is only the sum of realized motor forces and
their moment arms. Motor time constant is an exposed engineering assumption.
"""

import json
import math
from pathlib import Path

import torch

from wasman.controllers.thruster_visuals import DIRECTIONS, HANDEDNESS, POSITIONS, allocation_matrix


def interpolate(x, knots, values):
    """Clamped piecewise-linear interpolation, fully batched on the device."""
    x = x.clamp(knots[0], knots[-1]).contiguous()
    hi = torch.searchsorted(knots, x).clamp(1, len(knots) - 1)
    lo = hi - 1
    alpha = (x - knots[lo]) / (knots[hi] - knots[lo])
    return values[lo] + alpha * (values[hi] - values[lo])


class BatchedT200:
    def __init__(self, num_envs, device, *, dt, time_constant=0.08, command_delay_steps=2, positions=None):
        if dt <= 0 or time_constant < 0 or command_delay_steps < 0:
            raise ValueError("Invalid actuator timing")
        self.dt = dt
        self.alpha = 1.0 if time_constant == 0 else 1 - math.exp(-dt / time_constant)
        rows = json.loads((Path(__file__).parent / "data/t200_16v.json").read_text())["samples"]
        table = torch.tensor(rows, device=device)
        self.pwm_knots, magnitude, self.force_knots = table.T.contiguous()
        self.rpm_knots = magnitude * self.force_knots.sign()
        # Inverse force lookup uses the correct edge of the neutral deadband
        # separately for each branch. Forward PWM map retains the full deadband.
        neg = self.force_knots < 0
        pos = self.force_knots > 0
        zero = table.new_zeros(1)
        self.negative_force = torch.cat((self.force_knots[neg], zero))
        self.negative_pwm = torch.cat((self.pwm_knots[neg], table.new_tensor([1472.0])))
        self.positive_force = torch.cat((zero, self.force_knots[pos]))
        self.positive_pwm = torch.cat((table.new_tensor([1528.0]), self.pwm_knots[pos]))
        unique = (self.force_knots != 0) | (self.pwm_knots == 1500)
        self.spinning_rpm = self.rpm_knots[unique].contiguous()
        self.spinning_force = self.force_knots[unique].contiguous()
        self.matrix = allocation_matrix(device, positions=positions)
        self.inverse = torch.linalg.pinv(self.matrix)
        self.directions = table.new_tensor(DIRECTIONS)
        self.offsets = table.new_tensor(POSITIONS if positions is None else positions) - table.new_tensor(
            (0.0, 0.0, 0.011)
        )
        self.hand = table.new_tensor(HANDEDNESS)
        self.force = torch.zeros(num_envs, 8, device=device)
        self.rpm = torch.zeros_like(self.force)  # signed thrust convention, before CW/CCW handedness
        self.pwm = torch.full_like(self.force, 1500.0)
        self.requested_force = torch.zeros_like(self.force)
        self.command_force = torch.zeros_like(self.force)
        self.realized_wrench = torch.zeros(num_envs, 6, device=device)
        self.requested_wrench = torch.zeros_like(self.realized_wrench)
        self.saturation_scale = torch.ones(num_envs, 1, device=device)
        self.delay = torch.zeros(command_delay_steps + 1, num_envs, 8, device=device)
        self.delay_index = 0

    def allocate(self, wrench):
        requested = wrench @ self.inverse.T
        # Uniform desaturation preserves the requested wrench direction, though
        # not optimal use of the redundant actuators. No hidden residual wrench.
        limits = torch.where(requested >= 0, self.force_knots[-1], -self.force_knots[0])
        scale = (limits / requested.abs().clamp_min(1e-6)).amin(dim=-1, keepdim=True).clamp(max=1.0)
        return requested, scale

    def step(self, force_b, torque_b):
        self.requested_wrench.copy_(torch.cat((force_b, torque_b), dim=-1))
        request, scale = self.allocate(self.requested_wrench)
        self.requested_force.copy_(request)
        self.saturation_scale.copy_(scale)
        command = request * scale
        command = torch.where(command.abs() < 0.02 * 9.80665, 0.0, command)
        self.command_force.copy_(command)
        self.pwm.copy_(
            torch.where(
                command > 0,
                interpolate(command, self.positive_force, self.positive_pwm),
                interpolate(command, self.negative_force, self.negative_pwm),
            )
        )
        self.pwm[command == 0] = 1500.0
        target_rpm = interpolate(self.pwm, self.pwm_knots, self.rpm_knots)
        self.delay[self.delay_index].copy_(target_rpm)
        self.delay_index = (self.delay_index + 1) % len(self.delay)
        self.rpm.lerp_(self.delay[self.delay_index], self.alpha)
        self.force.copy_(interpolate(self.rpm, self.spinning_rpm, self.spinning_force))
        self.realized_wrench.copy_(self.force @ self.matrix.T)
        return self.force.unsqueeze(-1) * self.directions

    @property
    def shaft_omega(self):
        return self.rpm * self.hand * (2 * math.pi / 60)

    def reset(self, indices):
        for buffer in (
            self.force,
            self.rpm,
            self.command_force,
            self.requested_force,
            self.realized_wrench,
            self.requested_wrench,
        ):
            buffer[indices] = 0
        self.pwm[indices] = 1500.0
        self.saturation_scale[indices] = 1.0
        self.delay[:, indices] = 0
