"""Measured pick/carry/release/settle contract for the new shell-hoop task.

No simulator or expert dependency. Bounds are transformed collision-mesh vertices;
requiring every vertex inside the circle includes their entire convex hull.
All positions use the same world frame. A reset must clear the relevant rows.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShellCollectionCriteria:
    lift_clearance_m: float = 0.04
    carry_distance_m: float = 0.10
    hoop_margin_m: float = 0.01
    resting_gap_m: float = 0.005
    penetration_tolerance_m: float = 0.003
    grasp_force_n: float = 0.10
    opposing_dot: float = -0.25
    release_force_n: float = 0.05
    grasp_distance_m: float = 0.06
    release_distance_m: float = 0.08
    linear_speed_m_s: float = 0.02
    angular_speed_rad_s: float = 0.20
    hold_s: float = 1.0


class ShellCollectionContract:
    def __init__(self, num_envs, num_shells, device, dt, criteria=None):
        self.criteria = criteria or ShellCollectionCriteria()
        self.dt = dt
        self.lifted = torch.zeros((num_envs, num_shells), dtype=torch.bool, device=device)
        self.transported = torch.zeros_like(self.lifted)
        self.previous_position_xy = torch.zeros((num_envs, num_shells, 2), device=device)
        self.previous_lifting = torch.zeros_like(self.lifted)
        self.carried_distance = torch.zeros((num_envs, num_shells), device=device)
        self.hold_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    def update(
        self,
        *,
        positions,
        bounds,
        linear_velocity,
        angular_velocity,
        finger_forces,
        tool_distance,
        hoop_center_xy,
        hoop_inner_radius,
        floor_z,
    ):
        """Per-shell arrays are [env,shell,...]; finger forces [env,shell,2,3]."""
        c = self.criteria
        force = finger_forces.norm(dim=-1)
        unit = finger_forces / force[..., None].clamp_min(1e-6)
        opposing = (unit[..., 0, :] * unit[..., 1, :]).sum(-1) < c.opposing_dot
        grasp = (force > c.grasp_force_n).all(-1) & opposing & (tool_distance < c.grasp_distance_m)
        bottom = bounds[..., 2].amin(-1) - floor_z
        lifting = grasp & (bottom >= c.lift_clearance_m)
        self.lifted |= lifting
        delta = (positions[..., :2] - self.previous_position_xy).norm(dim=-1)
        self.carried_distance += torch.where(lifting & self.previous_lifting, delta, 0.0)
        self.transported |= self.carried_distance >= c.carry_distance_m
        self.previous_position_xy.copy_(positions[..., :2])
        self.previous_lifting.copy_(lifting)
        radial = (bounds[..., :2] - hoop_center_xy[:, None, None, :]).norm(dim=-1)
        inside = (radial <= hoop_inner_radius - c.hoop_margin_m).all(-1)
        grounded = (bottom >= -c.penetration_tolerance_m) & (bottom <= c.resting_gap_m)
        released = (force < c.release_force_n).all(-1) & (tool_distance >= c.release_distance_m)
        settled = linear_velocity.norm(dim=-1) <= c.linear_speed_m_s
        settled &= angular_velocity.norm(dim=-1) <= c.angular_speed_rad_s
        placed = self.transported & inside & grounded & released & settled
        hold = placed.all(-1)
        self.hold_steps[:] = torch.where(hold, self.hold_steps + 1, 0)
        self.success |= self.hold_steps >= round(c.hold_s / self.dt)
        return dict(
            success=self.success.clone(),
            grasped=grasp,
            lifted=self.lifted.clone(),
            transported=self.transported.clone(),
            placed=placed,
            bottom_clearance_m=bottom,
            inside_hoop=inside,
        )
