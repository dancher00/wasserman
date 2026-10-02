"""Presentation must preserve physical motors and stop when they are off."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pxr import Usd, UsdGeom, UsdPhysics

spec = importlib.util.spec_from_file_location(
    "rex_rotor_presentation", Path(__file__).parents[1] / "scripts/rex_rotor_presentation.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(collider=False):
    stage = Usd.Stage.CreateInMemory()
    for i in range(6):
        prim = UsdGeom.Xform.Define(stage, f"/World/envs/env_0/Robot/Geometry/base_link/thruster_{i}")
        mesh = UsdGeom.Mesh.Define(stage, str(prim.GetPath()) + "/mesh")
        if collider:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    parameters = SimpleNamespace(thruster_directions=np.tile([1.0, 0.0, 0.0], (6, 1)), max_thrust=100)
    return stage, parameters


def test_motor_direction_stop_and_unchanged_force():
    stage, parameters = fixture()
    display = module.RexRotorPresentation(stage, parameters)
    force = torch.tensor([[100.0, -100.0, 0.0, 25.0, -25.0, 0.0]])
    motors = SimpleNamespace(force=force, parameters=parameters, dt=1 / 30)
    before = force.clone()
    display.update(motors)
    assert torch.equal(force, before)
    assert display.phase == pytest.approx([48, 312, 0, 24, 336, 0])
    motors.force = torch.zeros_like(force)
    previous = display.phase.copy()
    display.update(motors)
    assert display.phase == previous


def test_refuses_to_animate_collision_geometry():
    stage, parameters = fixture(collider=True)
    with pytest.raises(ValueError, match="collider or body"):
        module.RexRotorPresentation(stage, parameters)
