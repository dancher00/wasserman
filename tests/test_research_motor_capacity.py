"""Capacity intervention must be identity at100% and bound actual native motor forces."""

import importlib.util
from pathlib import Path
from types import MethodType

import torch

from wasman.physics.thrusters import BatchedT200

spec = importlib.util.spec_from_file_location(
    "conditions", Path(__file__).parents[1] / "scripts/research_conditions.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reference_is_native_allocation_and_reduction_bounds_actual_force():
    motors = BatchedT200(4, "cpu", dt=1 / 240)
    wrench = torch.tensor(
        [[100.0, 20, 20, 2, 3, 4], [-100, -20, -20, -2, -3, -4], [0, 0, 0, 0, 0, 0], [400, 200, 80, 15, 5, 5]]
    )
    reference = motors.allocate(wrench)
    intervention = module.capped_allocation(motors, wrench, 1.0)
    assert all(torch.equal(a, b) for a, b in zip(reference, intervention, strict=True))
    original_curve = motors.force_knots.clone()
    motors.allocate = MethodType(lambda self, w: module.capped_allocation(self, w, 0.6), motors)
    for _ in range(300):
        motors.step(wrench[:, :3], wrench[:, 3:])
    assert motors.force.min() >= original_curve[0] * 0.6 - 1e-4
    assert motors.force.max() <= original_curve[-1] * 0.6 + 1e-4
    assert torch.equal(motors.force_knots, original_curve)
    assert torch.equal(motors.force[2], torch.zeros(8))
