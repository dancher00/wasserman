"""Unit tests for the simulator-independent hydrodynamic model."""

import pytest
import torch

from wasman.physics import BatchedHydrodynamics, LinkHydrodynamics


def make_link(
    *,
    volume: float = 0.0,
    cob: tuple[float, float, float] = (0.0, 0.0, 0.0),
    added: tuple[float, ...] = (0.0,) * 6,
    linear: tuple[float, ...] = (0.0,) * 6,
    quadratic: tuple[float, ...] = (0.0,) * 6,
) -> LinkHydrodynamics:
    return LinkHydrodynamics(
        name="test_link",
        volume=volume,
        center_of_buoyancy=cob,
        added_mass=added,
        linear_damping=linear,
        quadratic_damping=quadratic,
    )


def identity_quaternion(num_envs: int = 1) -> torch.Tensor:
    quaternion = torch.zeros((num_envs, 1, 4))
    quaternion[..., 3] = 1.0
    return quaternion


@pytest.mark.unit
def test_linear_and_quadratic_drag_oppose_motion() -> None:
    model = BatchedHydrodynamics(
        [make_link(linear=(2.0,) * 6, quadratic=(3.0,) * 6)], num_envs=1, dt=0.1, device="cpu"
    )
    twist = torch.tensor([[[2.0, -1.0, 0.5, -0.25, 0.1, -2.0]]])
    force, torque, _ = model.compute(twist, identity_quaternion())
    wrench = torch.cat((force, torque), dim=-1)
    expected = -(2.0 * twist + 3.0 * twist.abs() * twist)
    torch.testing.assert_close(wrench, expected)
    assert torch.sum(wrench * twist) < 0.0


@pytest.mark.unit
def test_added_mass_opposes_acceleration_after_history_is_initialized() -> None:
    model = BatchedHydrodynamics(
        [make_link(added=(2.0, 0.0, 0.0, 0.0, 0.0, 0.0))],
        num_envs=1,
        dt=0.1,
        device="cpu",
        acceleration_filter=1.0,
    )
    model.compute(torch.zeros((1, 1, 6)), identity_quaternion())
    twist = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    force, torque, acceleration = model.compute(twist, identity_quaternion())
    torch.testing.assert_close(acceleration[..., 0], torch.tensor([[10.0]]))
    torch.testing.assert_close(force[..., 0], torch.tensor([[-20.0]]))
    torch.testing.assert_close(torque, torch.zeros_like(torque))


@pytest.mark.unit
def test_buoyancy_and_center_of_buoyancy_torque_use_body_frame() -> None:
    model = BatchedHydrodynamics(
        [make_link(volume=0.01, cob=(0.1, 0.0, 0.0))],
        num_envs=1,
        dt=0.01,
        device="cpu",
        water_density=1000.0,
        gravity=10.0,
    )
    force, torque, _ = model.compute(torch.zeros((1, 1, 6)), identity_quaternion())
    torch.testing.assert_close(force, torch.tensor([[[0.0, 0.0, 100.0]]]))
    torch.testing.assert_close(torque, torch.tensor([[[0.0, -10.0, 0.0]]]))


@pytest.mark.unit
def test_reset_suppresses_spurious_added_mass_impulse() -> None:
    model = BatchedHydrodynamics(
        [make_link(added=(4.0,) * 6)], num_envs=2, dt=0.1, device="cpu", acceleration_filter=1.0
    )
    model.compute(torch.ones((2, 1, 6)), identity_quaternion(2))
    model.reset(torch.tensor([0]))
    next_twist = torch.full((2, 1, 6), 2.0)
    force, torque, acceleration = model.compute(next_twist, identity_quaternion(2))
    torch.testing.assert_close(acceleration[0], torch.zeros_like(acceleration[0]))
    assert torch.all(acceleration[1] == 10.0)
    torch.testing.assert_close(torch.cat((force[0], torque[0]), dim=-1), torch.zeros((1, 6)))


@pytest.mark.unit
def test_parameter_randomization_scales_buoyancy() -> None:
    model = BatchedHydrodynamics(
        [make_link(volume=0.01)], num_envs=2, dt=0.1, device="cpu", water_density=1000.0, gravity=10.0
    )
    model.set_parameter_scales(
        torch.tensor([0, 1]), volume=torch.tensor([0.5, 1.5]), damping=1.0, added_mass=1.0
    )
    force, _, _ = model.compute(torch.zeros((2, 1, 6)), identity_quaternion(2))
    torch.testing.assert_close(force[:, 0, 2], torch.tensor([50.0, 150.0]))
