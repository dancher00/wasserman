import torch

from wasman.controllers.thruster_visuals import NOMINAL_POSITIONS, POSITIONS
from wasman.physics.thrusters import BatchedT200, interpolate


def test_t200_data_deadband_asymmetry_and_clamps():
    model = BatchedT200(1, "cpu", dt=1 / 120)
    assert 51 < model.force_knots[-1] < 52
    assert -41 < model.force_knots[0] < -39
    rpm = interpolate(torch.tensor([1100.0, 1472.0, 1500.0, 1528.0, 1900.0]), model.pwm_knots, model.rpm_knots)
    assert rpm[0] < -3400 and rpm[-1] > 3500
    assert not rpm[1:4].any()


def test_allocation_direction_saturation_and_moments():
    model = BatchedT200(2, "cpu", dt=1 / 120)
    w = torch.tensor([[3.0, 4.0, 5.0, 0.2, 0.3, 0.4], [300.0, -400.0, 700.0, 50.0, 20.0, -60.0]])
    requested, scale = model.allocate(w)
    allocated = requested * scale
    torch.testing.assert_close(allocated @ model.matrix.T, w * scale, atol=1e-4, rtol=1e-4)
    assert scale[0] == 1 and scale[1] < 1
    assert allocated.max() <= model.force_knots[-1] + 1e-5
    assert allocated.min() >= model.force_knots[0] - 1e-5
    vectors = allocated.unsqueeze(-1) * model.directions
    moment = torch.linalg.cross(model.offsets.expand_as(vectors), vectors).sum(1)
    torch.testing.assert_close(moment, (allocated @ model.matrix.T)[:, 3:])


def test_delay_lag_reversal_and_partial_reset():
    model = BatchedT200(2, "cpu", dt=1 / 120, command_delay_steps=2)
    force = torch.tensor([[0.0, 0.0, -20.0], [0.0, 0.0, -20.0]])
    torque = torch.zeros_like(force)
    model.step(force, torque)
    model.step(force, torque)
    assert not model.rpm.any()
    model.step(force, torque)
    assert model.rpm.abs().max() > 0
    for _ in range(240):
        model.step(force, torque)
    torch.testing.assert_close(model.realized_wrench[:, :3], force, atol=0.05, rtol=0.01)
    before = model.rpm.clone()
    model.step(-force, torque)
    torch.testing.assert_close(model.rpm, before, atol=1e-3, rtol=1e-4)
    for _ in range(240):
        model.step(-force, torque)
    torch.testing.assert_close(model.realized_wrench[:, :3], -force, atol=0.05, rtol=0.01)
    model.reset(torch.tensor([0]))
    assert not model.force[0].any() and not model.delay[:, 0].any()
    assert model.force[1].abs().max() > 0


def test_disabled_delay_and_idle():
    model = BatchedT200(1, "cpu", dt=1 / 120, time_constant=0, command_delay_steps=0)
    force = torch.tensor([[0.0, 0.0, -20.0]])
    model.step(force, torch.zeros_like(force))
    torch.testing.assert_close(model.realized_wrench[:, :3], force, atol=0.05, rtol=0.01)
    model.step(torch.zeros_like(force), torch.zeros_like(force))
    assert not model.force.any()


def test_versioned_mounts_change_allocation_and_moments_without_mutating_legacy():
    original = BatchedT200(1, "cpu", dt=1 / 120)
    registered = BatchedT200(1, "cpu", dt=1 / 120, positions=NOMINAL_POSITIONS)
    torch.testing.assert_close(original.offsets + torch.tensor([0.0, 0.0, 0.011]), torch.tensor(POSITIONS))
    torch.testing.assert_close(registered.offsets + torch.tensor([0.0, 0.0, 0.011]), torch.tensor(NOMINAL_POSITIONS))
    assert not torch.allclose(original.matrix, registered.matrix)
    forces = torch.tensor([[2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0]])
    vectors = forces[..., None] * registered.directions
    actual = torch.cat((vectors.sum(1), torch.linalg.cross(registered.offsets[None], vectors).sum(1)), -1)
    torch.testing.assert_close(forces @ registered.matrix.T, actual)
    fresh = BatchedT200(1, "cpu", dt=1 / 120)
    torch.testing.assert_close(fresh.matrix, original.matrix, rtol=0, atol=0)
