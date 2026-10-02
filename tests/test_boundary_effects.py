import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wasman.physics.boundary_effects import (
    BlueROVBoundaryEffect,
    BoundaryEffectCfg,
    plane_jet_loss,
    validate_boundary_scene,
)
from wasman.physics.hydrodynamics import quat_apply_xyzw
from wasman.physics.thrusters import BatchedT200


def plane_case(distance=0.2, direction=(0, 0, -1), x=0.0):
    pos = torch.tensor([[[x, 0.0, distance]]])
    direction = pos.new_tensor([[direction]])
    zero = torch.zeros_like(pos)
    return plane_jet_loss(
        pos,
        direction,
        zero,
        pos.new_tensor([[[0, 0, 1]]]),
        pos.new_tensor([[[1, 0, 0]]]),
        pos.new_tensor([[[0, 1, 0]]]),
        pos.new_tensor([[0.5, 0.5]]),
        coefficient=pos.new_tensor([0.2]),
        reach=0.762,
    )


def test_downstream_reversal_parallel_zero_and_behind_wall():
    assert plane_case()[0].item() > 0
    for direction in ((0, 0, 1), (1, 0, 0), (0, 0, 0)):
        assert plane_case(direction=direction)[0].item() == 0
    assert plane_case(distance=-0.1)[0].item() == 0


def test_distance_monotonic_and_finite_surface_extent():
    losses = [plane_case(distance=d)[0].item() for d in (0, 0.05, 0.2, 0.5, 0.762, 2)]
    assert losses == sorted(losses, reverse=True)
    assert losses[0] == pytest.approx(0.2)
    assert losses[-2:] == [0, 0]
    loss, distance = plane_case(x=0.501)
    assert loss.item() == 0 and math.isinf(distance.item())


def test_oblique_momentum_is_weaker():
    assert 0 < plane_case(direction=(0.6, 0, -0.8))[0] < plane_case()[0]


def test_native_motors_reverse_seabed_and_zero_force():
    model = BlueROVBoundaryEffect()
    pose = torch.tensor([[0, 0, 0.2, 0, 0, 0, 1.0]])
    force = torch.full((1, 8), -10.0)  # Vertical axes -Z; negative force has down-going exhaust.
    applied, gain, _ = model.apply(pose, force)
    assert (gain[:, :4] == 1).all() and (gain[:, 4:] < 1).all()
    assert torch.allclose(applied, force[..., None] * model.directions * gain[..., None])
    assert (model.apply(pose, -force)[1] == 1).all()
    zero, gain, _ = model.apply(pose, force * 0)
    assert (zero == 0).all() and (gain == 1).all()


def test_rigid_rotation_covariance_with_rotated_finite_panel():
    model = BlueROVBoundaryEffect(cfg=BoundaryEffectCfg(seabed_loss=0))
    pose = torch.tensor([[0, 0, 1, 0, 0, 0, 1.0]])
    panel = torch.tensor([[0.5, 0, 1, 0, 0, 0, 1.0]])
    force = torch.full((1, 8), 10.0)
    reference = model.apply(pose, force, panel)
    q = torch.tensor([[0, 0, math.sin(0.6), math.cos(0.6)]])
    rotated = pose.clone()
    rotated[:, :3] = quat_apply_xyzw(q, pose[:, :3])
    rotated[:, 3:] = q
    other = panel.clone()
    other[:, :3] = quat_apply_xyzw(q, panel[:, :3])
    other[:, 3:] = q
    result = model.apply(rotated, force, other)
    assert torch.allclose(reference[1], result[1], atol=1e-6)
    assert (reference[1] < 1).any()


def test_no_virtual_infinite_wall_and_no_cross_environment_interaction():
    model = BlueROVBoundaryEffect(cfg=BoundaryEffectCfg(seabed_loss=0))
    pose = torch.tensor([[0, 0, 1, 0, 0, 0, 1.0], [30, 30, 1, 0, 0, 0, 1.0]])
    panel = pose.clone()
    panel[:, 0] += 0.5
    force = torch.full((2, 8), 10.0)
    gain = model.apply(pose, force, panel)[1]
    assert torch.allclose(gain[0], gain[1], atol=1e-6)
    panel[:, 2] += 4  # Exact same X standoff, but jets miss its finite rectangle.
    assert (model.apply(pose, force, panel)[1] == 1).all()


def test_zero_coefficients_exact_identity_and_multiple_planes_bounded():
    pose = torch.tensor([[0, 0, 0.1, 0, 0, 0, 1.0]])
    panel = torch.tensor([[0.3, 0, 0.1, 0, 0, 0, 1.0]])
    force = torch.tensor([[10, -10, 10, -10, -10, -10, -10, -10.0]])
    model = BlueROVBoundaryEffect(cfg=BoundaryEffectCfg(seabed_loss=0, wall_loss=0))
    applied, gain, _ = model.apply(pose, force, panel)
    assert torch.equal(applied, force[..., None] * model.directions)
    assert torch.equal(gain, torch.ones_like(gain))
    gain = BlueROVBoundaryEffect().apply(pose, force, panel)[1]
    assert ((gain >= 0.8) & (gain <= 1)).all()


def test_actuator_pipeline_retains_delay_lag_and_saturation_before_boundary():
    motors = BatchedT200(1, "cpu", dt=1 / 120)
    model = BlueROVBoundaryEffect()
    pose = torch.tensor([[0, 0, 0.1, 0, 0, 0, 1.0]])
    force, torque = torch.tensor([[0, 0, 1000.0]]), torch.zeros(1, 3)
    for _ in range(2):
        assert torch.equal(motors.step(force, torque), torch.zeros(1, 8, 3))
    motors.step(force, torque)
    assert (motors.force.abs() > 0).any() and (motors.saturation_scale < 1).all()
    before_force, before_rpm = motors.force.clone(), motors.rpm.clone()
    applied, gain, _ = model.apply(pose, motors.force)
    assert (gain[:, 4:] < 1).all()
    assert torch.equal(motors.force, before_force) and torch.equal(motors.rpm, before_rpm)
    assert (applied.abs() <= (motors.force[..., None] * motors.directions).abs() + 1e-8).all()
    motors.reset([0])
    assert (motors.rpm == 0).all() and (motors.delay == 0).all()


def test_asymmetric_mount_loss_induces_correct_lever_arm_torque_and_far_identity():
    model = BlueROVBoundaryEffect(cfg=BoundaryEffectCfg(seabed_loss=0))
    # One angled jet, finite panel shifted toward its actual exhaust intersection.
    pose = torch.tensor([[0, 0, 1, 0, 0, 0, 1.0]])
    panel = torch.tensor([[0.5, 0.25, 1, 0, 0, 0, 1.0]])
    force = torch.zeros(1, 8)
    force[0, 0] = 10
    applied, gain, _ = model.apply(pose, force, panel)
    before = force[..., None] * model.directions
    offset = model.positions - torch.tensor((0, 0, 0.011))
    delta_torque = torch.linalg.cross(offset[None], applied - before).sum(1)
    expected = torch.linalg.cross(offset[0], before[0, 0]) * (gain[0, 0] - 1)
    assert delta_torque.norm() > 1e-4
    assert torch.allclose(delta_torque[0], expected, atol=1e-7)
    # Torque follows the actual mount, rather than an invented wall-attraction force.
    assert delta_torque[0, 2] > 0
    panel[:, 0] += 2
    far = model.apply(pose, force, panel)[0]
    assert torch.equal(far, before) and not torch.equal(applied, far)


@pytest.mark.parametrize(
    "params", [{"wall_loss": -0.1}, {"seabed_loss": 1}, {"diameter_m": 0}, {"range_diameters": float("nan")}]
)
def test_invalid_assumptions_rejected(params):
    with pytest.raises(ValueError):
        BoundaryEffectCfg(**params)


def test_only_native_panel_geometry_supported_no_phantom_hatch_floor():
    root = Path(__file__).parents[1] / "src/wasman/assets/data"

    def asset(path):
        return SimpleNamespace(
            spawn=SimpleNamespace(usd_path=str(root / path), scale=None),
            init_state=SimpleNamespace(pos=(0, 0, 0), rot=(0, 0, 0, 1)),
        )

    panel, seabed = asset("panels/ship_green/panel.usda"), asset("seabed/sand.usda")
    cfg = SimpleNamespace(use_physical_thrusters=True, scene=SimpleNamespace(panel=panel, seabed=seabed))
    validate_boundary_scene(cfg)
    cfg.scene.panel = None
    validate_boundary_scene(cfg)  # Object-free diagnostics deliberately omit the panel.
    cfg.scene.sand_apron = object()  # Hatch has a floor aperture, not the native flat slab.
    with pytest.raises(ValueError, match="Hatch"):
        validate_boundary_scene(cfg)
    cfg.scene.sand_apron = None
    cfg.scene.panel = panel
    cfg.scene.seabed.init_state.pos = (0, 0, -0.4)
    with pytest.raises(ValueError, match="z=0"):
        validate_boundary_scene(cfg)
    cfg.scene.seabed.init_state.pos = (0, 0, 0)
    panel.spawn.scale = (2, 1, 1)
    with pytest.raises(ValueError, match="scaled"):
        validate_boundary_scene(cfg)


def test_outside_actual_sand_rectangle_has_no_bottom_loss():
    pose = torch.tensor([[101, 0, 0.1, 0, 0, 0, 1.0]])
    gain = BlueROVBoundaryEffect().apply(pose, torch.full((1, 8), -10.0))[1]
    assert (gain == 1).all()
