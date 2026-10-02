"""Unit tests for the simulator-independent station-keeping controller."""

import math

import pytest
import torch

from wasman.controllers import BatchedStationKeepingController, StationKeepingGains
from wasman.controllers.station_keeping import quaternion_angle_error_xyzw


def make_controller(num_envs: int = 1) -> BatchedStationKeepingController:
    return BatchedStationKeepingController(
        num_envs=num_envs,
        dt=0.1,
        device="cpu",
        gains=StationKeepingGains(
            position_kp=(2.0, 2.0, 2.0),
            position_kd=(1.0, 1.0, 1.0),
            position_ki=(0.5, 0.5, 0.5),
            rotation_kp=(3.0, 3.0, 3.0),
            rotation_kd=(1.0, 1.0, 1.0),
            rotation_ki=(0.25, 0.25, 0.25),
            max_force=(10.0, 10.0, 10.0),
            max_torque=(10.0, 10.0, 10.0),
            position_integral_limit=(1.0, 1.0, 1.0),
            rotation_integral_limit=(1.0, 1.0, 1.0),
        ),
    )


def identity_quaternion(num_envs: int = 1) -> torch.Tensor:
    quaternion = torch.zeros((num_envs, 4))
    quaternion[:, 3] = 1.0
    return quaternion


@pytest.mark.unit
def test_equilibrium_produces_zero_wrench() -> None:
    controller = make_controller()
    zeros = torch.zeros((1, 3))
    identity = identity_quaternion()
    force, torque, position_error, orientation_error = controller.compute(
        position_w=zeros,
        quaternion_w=identity,
        linear_velocity_w=zeros,
        angular_velocity_w=zeros,
        target_position_w=zeros,
        target_quaternion_w=identity,
    )
    torch.testing.assert_close(force, zeros)
    torch.testing.assert_close(torque, zeros)
    torch.testing.assert_close(position_error, zeros)
    torch.testing.assert_close(orientation_error, zeros)


@pytest.mark.unit
def test_world_position_force_is_rotated_into_body_frame() -> None:
    controller = make_controller()
    zeros = torch.zeros((1, 3))
    yaw_90 = torch.tensor([[0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    force, _, _, _ = controller.compute(
        position_w=zeros,
        quaternion_w=yaw_90,
        linear_velocity_w=zeros,
        angular_velocity_w=zeros,
        target_position_w=target,
        target_quaternion_w=yaw_90,
    )
    # Kp plus one integral update gives 2.05 N along world X, or body -Y.
    torch.testing.assert_close(force, torch.tensor([[0.0, -2.05, 0.0]]), atol=1.0e-5, rtol=1.0e-5)


@pytest.mark.unit
def test_positive_yaw_error_commands_restoring_torque() -> None:
    controller = make_controller()
    zeros = torch.zeros((1, 3))
    yaw = 0.2
    current = torch.tensor([[0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]])
    _, torque, _, orientation_error = controller.compute(
        position_w=zeros,
        quaternion_w=current,
        linear_velocity_w=zeros,
        angular_velocity_w=zeros,
        target_position_w=zeros,
        target_quaternion_w=identity_quaternion(),
    )
    assert orientation_error[0, 2] > 0.0
    assert torque[0, 2] < 0.0
    torch.testing.assert_close(
        quaternion_angle_error_xyzw(current, identity_quaternion()), torch.tensor([yaw]), atol=1.0e-6, rtol=1.0e-6
    )


@pytest.mark.unit
def test_reset_clears_integral_state_selectively() -> None:
    controller = make_controller(num_envs=2)
    zeros = torch.zeros((2, 3))
    target = torch.ones((2, 3))
    controller.compute(
        position_w=zeros,
        quaternion_w=identity_quaternion(2),
        linear_velocity_w=zeros,
        angular_velocity_w=zeros,
        target_position_w=target,
        target_quaternion_w=identity_quaternion(2),
    )
    controller.reset(torch.tensor([0]))
    torch.testing.assert_close(controller.position_integral[0], torch.zeros(3))
    assert torch.all(controller.position_integral[1] > 0.0)
