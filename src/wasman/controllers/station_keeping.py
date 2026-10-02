"""Batched geometric PID station keeping for free-floating underwater robots.

The control architecture follows the separation used by AM-Bench: a policy
selects a bounded absolute pose target and a low-level controller converts the
SE(3) tracking error into a body-frame wrench.  The implementation is native
to WASMAN, uses Isaac Lab 3.x XYZW quaternions, and has no simulator imports so
that it can be unit tested on CPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from wasman.physics import quat_apply_inverse_xyzw

Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class StationKeepingGains:
    """Per-axis gains, saturation, and anti-windup limits in SI units."""

    position_kp: Vector3
    position_kd: Vector3
    position_ki: Vector3
    rotation_kp: Vector3
    rotation_kd: Vector3
    rotation_ki: Vector3
    max_force: Vector3
    max_torque: Vector3
    position_integral_limit: Vector3
    rotation_integral_limit: Vector3

    def __post_init__(self) -> None:
        for name, values in (
            ("position_kp", self.position_kp),
            ("position_kd", self.position_kd),
            ("position_ki", self.position_ki),
            ("rotation_kp", self.rotation_kp),
            ("rotation_kd", self.rotation_kd),
            ("rotation_ki", self.rotation_ki),
            ("max_force", self.max_force),
            ("max_torque", self.max_torque),
            ("position_integral_limit", self.position_integral_limit),
            ("rotation_integral_limit", self.rotation_integral_limit),
        ):
            if len(values) != 3:
                raise ValueError(f"{name} must contain three axes.")
            if any(value < 0.0 for value in values):
                raise ValueError(f"{name} entries must be non-negative.")


def _rotation_matrix_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Return body-to-world rotation matrices for normalized XYZW quaternions."""
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-9)
    x, y, z, w = quaternion.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def geometric_orientation_error_xyzw(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the SO(3) error vector in the current body frame.

    The sign follows geometric attitude control: a positive current yaw error
    relative to the target produces a positive error about body Z, which the
    controller opposes with negative torque.
    """
    current_rotation = _rotation_matrix_xyzw(current)
    target_rotation = _rotation_matrix_xyzw(target)
    relative = torch.matmul(target_rotation.transpose(-1, -2), current_rotation)
    skew = relative - relative.transpose(-1, -2)
    return 0.5 * torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)


def quaternion_angle_error_xyzw(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the shortest unsigned orientation distance in radians."""
    current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp_min(1.0e-9)
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(1.0e-9)
    cosine_half_angle = torch.sum(current * target, dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(cosine_half_angle)


class BatchedStationKeepingController:
    """Geometric PID pose controller that returns a saturated body-frame wrench."""

    def __init__(
        self,
        *,
        num_envs: int,
        dt: float,
        device: str | torch.device,
        gains: StationKeepingGains,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        self.num_envs = num_envs
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.gains = gains

        for name in (
            "position_kp",
            "position_kd",
            "position_ki",
            "rotation_kp",
            "rotation_kd",
            "rotation_ki",
            "max_force",
            "max_torque",
            "position_integral_limit",
            "rotation_integral_limit",
        ):
            setattr(
                self,
                f"_{name}",
                torch.tensor(getattr(gains, name), device=self.device, dtype=self.dtype).view(1, 3),
            )

        self.position_integral = torch.zeros((num_envs, 3), device=self.device, dtype=self.dtype)
        self.rotation_integral = torch.zeros_like(self.position_integral)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Clear integral state for all or selected environments."""
        if env_ids is None:
            self.position_integral.zero_()
            self.rotation_integral.zero_()
            return
        indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.position_integral[indices] = 0.0
        self.rotation_integral[indices] = 0.0

    def compute(
        self,
        *,
        position_w: torch.Tensor,
        quaternion_w: torch.Tensor,
        linear_velocity_w: torch.Tensor,
        angular_velocity_w: torch.Tensor,
        target_position_w: torch.Tensor,
        target_quaternion_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute force/torque in body axes plus position/orientation errors."""
        vector_shape = (self.num_envs, 3)
        quaternion_shape = (self.num_envs, 4)
        for name, value in (
            ("position_w", position_w),
            ("linear_velocity_w", linear_velocity_w),
            ("angular_velocity_w", angular_velocity_w),
            ("target_position_w", target_position_w),
        ):
            if value.shape != vector_shape:
                raise ValueError(f"{name} must have shape {vector_shape}.")
        for name, value in (("quaternion_w", quaternion_w), ("target_quaternion_w", target_quaternion_w)):
            if value.shape != quaternion_shape:
                raise ValueError(f"{name} must have shape {quaternion_shape}.")

        position_error_w = target_position_w - position_w
        orientation_error_b = geometric_orientation_error_xyzw(quaternion_w, target_quaternion_w)
        angular_velocity_b = quat_apply_inverse_xyzw(quaternion_w, angular_velocity_w)

        self.position_integral.add_(position_error_w * self.dt)
        self.position_integral.clamp_(-self._position_integral_limit, self._position_integral_limit)
        self.rotation_integral.add_(orientation_error_b * self.dt)
        self.rotation_integral.clamp_(-self._rotation_integral_limit, self._rotation_integral_limit)

        force_w = (
            self._position_kp * position_error_w
            - self._position_kd * linear_velocity_w
            + self._position_ki * self.position_integral
        )
        torque_b = (
            -self._rotation_kp * orientation_error_b
            - self._rotation_kd * angular_velocity_b
            - self._rotation_ki * self.rotation_integral
        )
        force_b = quat_apply_inverse_xyzw(quaternion_w, force_w)
        force_b = torch.nan_to_num(force_b).clamp(-self._max_force, self._max_force)
        torque_b = torch.nan_to_num(torque_b).clamp(-self._max_torque, self._max_torque)
        return force_b, torque_b, position_error_w, orientation_error_b
