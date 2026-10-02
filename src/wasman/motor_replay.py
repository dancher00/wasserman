"""Exact input-wrench tape around the real T200 actuator model, not fake forces."""

from contextlib import contextmanager

import torch


class MotorCommandTape:
    """Capture or replay the six-component input of each native motor-model step.

    Replay ignores live feedback wrenches, but calls the unchanged real step:
    allocation, saturation, PWM inversion, delay and rotor lag remain operative.
    """

    def __init__(self, motor, input_wrenches=None):
        self.motor = motor
        self.input_wrenches = input_wrenches
        if input_wrenches is not None:
            if input_wrenches.ndim != 3 or input_wrenches.shape[1:] != (motor.force.shape[0], 6):
                raise ValueError("Tape shape must be (physics_steps, environments, 6)")
            if not torch.isfinite(input_wrenches).all():
                raise ValueError("Nonfinite input-wrench tape")
        self.inputs, self.forces = [], []
        self._original = None

    def step(self, force, torque):
        wrench = torch.cat((force, torque), dim=-1)
        if self.input_wrenches is not None:
            if len(self.inputs) >= len(self.input_wrenches):
                raise RuntimeError("Replay exhausted; never fall back to live feedback")
            wrench = self.input_wrenches[len(self.inputs)]
        if not torch.isfinite(wrench).all():
            raise RuntimeError("Nonfinite motor input")
        result = self._original(wrench[:, :3], wrench[:, 3:])
        self.inputs.append(wrench.detach().clone())
        self.forces.append(self.motor.force.detach().clone())
        return result

    @contextmanager
    def installed(self):
        if self._original is not None:
            raise RuntimeError("Tape already installed")
        self._original = self.motor.step
        self.motor.step = self.step
        try:
            yield self
        finally:
            self.motor.step = self._original
            self._original = None

    def tensors(self):
        if not self.inputs:
            raise RuntimeError("Empty motor tape")
        return torch.stack(self.inputs), torch.stack(self.forces)


def verify_matched_tapes(first, second):
    """Bit-identical inputs AND pre-boundary forces, not an approximate match."""
    inputs0, forces0 = first.tensors()
    inputs1, forces1 = second.tensors()
    return {
        "input_wrenches_exact_match": torch.equal(inputs0, inputs1),
        "pre_boundary_motor_forces_exact_match": torch.equal(forces0, forces1),
        "physics_steps": len(inputs0),
    }
