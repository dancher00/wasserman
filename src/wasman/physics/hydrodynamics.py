"""Batched six-degree-of-freedom underwater link hydrodynamics.

The model intentionally has no Isaac Sim imports. It can therefore be unit
tested on CPU and used by an Isaac Lab environment only at the final wrench
application boundary. All quaternions use Isaac Lab 3.x's XYZW convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

Vector3 = tuple[float, float, float]
Vector6 = tuple[float, float, float, float, float, float]


def quat_apply_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit quaternions stored as ``(x, y, z, w)``."""
    xyz = quaternion[..., :3]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., 3:4] * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


def quat_apply_inverse_xyzw(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Inverse-rotate vectors by unit quaternions stored as ``(x, y, z, w)``."""
    xyz = quaternion[..., :3]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector - quaternion[..., 3:4] * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


@dataclass(frozen=True, slots=True)
class LinkHydrodynamics:
    """Hydrodynamic parameters for one rigid link in SI units.

    ``added_mass`` is the positive diagonal magnitude of ``-M_A`` and follows
    the twist order ``(surge, sway, heave, roll, pitch, yaw)``. Damping values
    are also positive magnitudes; the implementation applies the opposing sign.
    The center of buoyancy is expressed from the center of mass in body axes.
    """

    name: str
    volume: float
    center_of_buoyancy: Vector3
    added_mass: Vector6
    linear_damping: Vector6
    quadratic_damping: Vector6

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A hydrodynamic link must have a non-empty name.")
        if self.volume < 0.0:
            raise ValueError(f"{self.name}: volume must be non-negative.")
        for label, values, size in (
            ("center_of_buoyancy", self.center_of_buoyancy, 3),
            ("added_mass", self.added_mass, 6),
            ("linear_damping", self.linear_damping, 6),
            ("quadratic_damping", self.quadratic_damping, 6),
        ):
            if len(values) != size:
                raise ValueError(f"{self.name}: {label} must have {size} entries.")
            if label != "center_of_buoyancy" and any(value < 0.0 for value in values):
                raise ValueError(f"{self.name}: {label} entries must be non-negative.")


class BatchedHydrodynamics:
    """Vectorized per-link buoyancy, damping, and added-mass model.

    The returned force and torque are expressed in each link's body frame and
    act at its center of mass. Acceleration is estimated from relative twist and
    low-pass filtered to avoid numerically amplifying solver noise.
    """

    def __init__(
        self,
        links: Sequence[LinkHydrodynamics],
        num_envs: int,
        dt: float,
        device: str | torch.device,
        *,
        water_density: float = 1025.0,
        gravity: float = 9.81,
        acceleration_filter: float = 0.2,
        max_linear_acceleration: float = 30.0,
        max_angular_acceleration: float = 50.0,
    ) -> None:
        if not links:
            raise ValueError("At least one link is required.")
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if water_density <= 0.0 or gravity <= 0.0:
            raise ValueError("water_density and gravity must be positive.")
        if not 0.0 < acceleration_filter <= 1.0:
            raise ValueError("acceleration_filter must be in (0, 1].")

        self.link_names = tuple(link.name for link in links)
        if len(set(self.link_names)) != len(self.link_names):
            raise ValueError("Link names must be unique.")
        self.num_envs = num_envs
        self.num_links = len(links)
        self.dt = float(dt)
        self.water_density = float(water_density)
        self.gravity = float(gravity)
        self.acceleration_filter = float(acceleration_filter)
        self.device = torch.device(device)
        self.dtype = torch.float32

        self.volume = self._tensor([link.volume for link in links]).view(1, self.num_links, 1)
        self.center_of_buoyancy = self._tensor([link.center_of_buoyancy for link in links]).view(
            1, self.num_links, 3
        )
        self.added_mass = self._tensor([link.added_mass for link in links]).view(1, self.num_links, 6)
        self.linear_damping = self._tensor([link.linear_damping for link in links]).view(1, self.num_links, 6)
        self.quadratic_damping = self._tensor([link.quadratic_damping for link in links]).view(
            1, self.num_links, 6
        )
        self.acceleration_limit = self._tensor(
            [max_linear_acceleration] * 3 + [max_angular_acceleration] * 3
        ).view(1, 1, 6)

        scale_shape = (num_envs, 1, 1)
        self.volume_scale = torch.ones(scale_shape, device=self.device, dtype=self.dtype)
        self.damping_scale = torch.ones(scale_shape, device=self.device, dtype=self.dtype)
        self.added_mass_scale = torch.ones(scale_shape, device=self.device, dtype=self.dtype)
        self.previous_relative_twist = torch.zeros(
            (num_envs, self.num_links, 6), device=self.device, dtype=self.dtype
        )
        self.filtered_acceleration = torch.zeros_like(self.previous_relative_twist)
        self.history_valid = torch.zeros((num_envs, 1, 1), device=self.device, dtype=torch.bool)

    def _tensor(self, values: object) -> torch.Tensor:
        return torch.as_tensor(values, device=self.device, dtype=self.dtype)

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Clear acceleration history for selected environments."""
        if env_ids is None:
            self.previous_relative_twist.zero_()
            self.filtered_acceleration.zero_()
            self.history_valid.zero_()
            return
        indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.previous_relative_twist[indices] = 0.0
        self.filtered_acceleration[indices] = 0.0
        self.history_valid[indices] = False

    def set_parameter_scales(
        self,
        env_ids: torch.Tensor | Sequence[int],
        *,
        volume: torch.Tensor | float,
        damping: torch.Tensor | float,
        added_mass: torch.Tensor | float,
    ) -> None:
        """Set coherent vehicle-level randomization scales for selected environments."""
        indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for destination, value, label in (
            (self.volume_scale, volume, "volume"),
            (self.damping_scale, damping, "damping"),
            (self.added_mass_scale, added_mass, "added_mass"),
        ):
            source = torch.as_tensor(value, device=self.device, dtype=self.dtype).reshape(-1)
            if source.numel() == 1:
                source = source.expand(indices.numel())
            if source.numel() != indices.numel():
                raise ValueError(f"{label} scale must be scalar or have one value per selected environment.")
            if not torch.all(torch.isfinite(source) & (source > 0.0)):
                raise ValueError(f"{label} scale must be finite and positive.")
            destination[indices, 0, 0] = source

    def compute(
        self,
        relative_twist_b: torch.Tensor,
        body_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute body-frame wrench and filtered relative acceleration.

        Args:
            relative_twist_b: Relative water velocity in body axes, shaped
                ``(num_envs, num_links, 6)`` with linear components first.
            body_quat_w: Body orientations in world axes as XYZW quaternions,
                shaped ``(num_envs, num_links, 4)``.
        """
        expected_twist = (self.num_envs, self.num_links, 6)
        expected_quat = (self.num_envs, self.num_links, 4)
        if relative_twist_b.shape != expected_twist:
            raise ValueError(f"relative_twist_b must have shape {expected_twist}.")
        if body_quat_w.shape != expected_quat:
            raise ValueError(f"body_quat_w must have shape {expected_quat}.")

        raw_acceleration = (relative_twist_b - self.previous_relative_twist) / self.dt
        raw_acceleration = torch.clamp(raw_acceleration, -self.acceleration_limit, self.acceleration_limit)
        raw_acceleration = torch.where(self.history_valid, raw_acceleration, torch.zeros_like(raw_acceleration))
        filtered = self.acceleration_filter * raw_acceleration + (1.0 - self.acceleration_filter) * (
            self.filtered_acceleration
        )
        filtered = torch.where(self.history_valid, filtered, torch.zeros_like(filtered))

        effective_added_mass = self.added_mass * self.added_mass_scale
        effective_linear_damping = self.linear_damping * self.damping_scale
        effective_quadratic_damping = self.quadratic_damping * self.damping_scale

        damping_wrench = effective_linear_damping * relative_twist_b
        damping_wrench += effective_quadratic_damping * relative_twist_b.abs() * relative_twist_b
        added_mass_wrench = effective_added_mass * filtered

        linear_velocity = relative_twist_b[..., :3]
        angular_velocity = relative_twist_b[..., 3:]
        linear_added_momentum = effective_added_mass[..., :3] * linear_velocity
        angular_added_momentum = effective_added_mass[..., 3:] * angular_velocity
        coriolis_force = torch.cross(angular_velocity, linear_added_momentum, dim=-1)
        coriolis_torque = torch.cross(linear_velocity, linear_added_momentum, dim=-1)
        coriolis_torque += torch.cross(angular_velocity, angular_added_momentum, dim=-1)
        coriolis_wrench = torch.cat((coriolis_force, coriolis_torque), dim=-1)

        hydrodynamic_wrench = -(damping_wrench + added_mass_wrench + coriolis_wrench)

        buoyancy_w = torch.zeros(
            (self.num_envs, self.num_links, 3), device=self.device, dtype=self.dtype
        )
        buoyancy_w[..., 2] = (
            self.water_density * self.gravity * (self.volume * self.volume_scale).squeeze(-1)
        )
        buoyancy_b = quat_apply_inverse_xyzw(body_quat_w, buoyancy_w)
        buoyancy_torque_b = torch.cross(self.center_of_buoyancy.expand_as(buoyancy_b), buoyancy_b, dim=-1)

        force_b = hydrodynamic_wrench[..., :3] + buoyancy_b
        torque_b = hydrodynamic_wrench[..., 3:] + buoyancy_torque_b

        self.previous_relative_twist.copy_(relative_twist_b)
        self.filtered_acceleration.copy_(filtered)
        self.history_valid.fill_(True)
        return force_b, torque_b, filtered
