"""Opt-in, uncalibrated jet/boundary thrust-loss sensitivity model.

NOT a validated T200 wall/ground model. See docs/underwater-physical-effects.md.
No Isaac imports; positions are metres and quaternions are XYZW.
"""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import torch

from wasman.assets.pool_geometry import POOL_GEOMETRY, POOL_USD_PATH
from wasman.controllers.thruster_visuals import DIRECTIONS, POSITIONS
from wasman.physics.hydrodynamics import quat_apply_xyzw


def validate_boundary_scene(cfg):
    """Reject unsupported geometry before creating a simulator scene."""
    if not cfg.use_physical_thrusters:
        raise ValueError("Experimental boundary effects require physical T200 thrusters")
    scene = cfg.scene
    root = Path(__file__).parents[1] / "assets/data"
    panel = getattr(scene, "panel", None)
    seabed = getattr(scene, "seabed", None)
    allowed = {
        str((root / "panels" / name / "panel.usda").resolve())
        for name in ("ship_green", "harbor_concrete", "oxidized_steel")
    }
    if seabed is None or getattr(scene, "sand_apron", None) is not None:
        raise ValueError("Boundary effects support standard panel scenes only, not Hatch/cutout or ship scenes")
    if panel is not None and str(Path(panel.spawn.usd_path).resolve()) not in allowed:
        raise ValueError("Boundary effects require the native standard panel geometry")
    is_pool = Path(seabed.spawn.usd_path).resolve() == POOL_USD_PATH.resolve()
    if not is_pool and str(Path(seabed.spawn.usd_path).resolve()) != str((root / "seabed/sand.usda").resolve()):
        raise ValueError("Boundary effects require the standard sand surface or native finite pool")
    if is_pool:
        if not all(isfinite(v) for v in seabed.init_state.pos) or seabed.init_state.pos[2] != 0:
            raise ValueError("Pool boundary geometry requires finite translation and floor z=0")
        if tuple(seabed.init_state.rot) != (0, 0, 0, 1):
            raise ValueError("Pool boundary geometry does not support rotated pools")
    elif tuple(seabed.init_state.pos) != (0, 0, 0) or tuple(seabed.init_state.rot) != (0, 0, 0, 1):
        raise ValueError("Boundary effects require untransformed seabed at z=0")
    for asset in (panel, seabed):
        if asset is None:
            continue
        if getattr(asset.spawn, "scale", None) not in (None, (1, 1, 1)):
            raise ValueError("Boundary effects do not support scaled scene geometry")
    return POOL_GEOMETRY if is_pool else None


@dataclass(frozen=True)
class BoundaryEffectCfg:
    seabed_loss: float = 0.20
    wall_loss: float = 0.20
    diameter_m: float = 0.0762
    range_diameters: float = 10.0
    seabed_z: float = 0.0

    def __post_init__(self):
        for value in (self.seabed_loss, self.wall_loss):
            if not isfinite(value) or not 0 <= value < 1:
                raise ValueError("Boundary loss must be finite in [0, 1)")
        if not all(isfinite(v) and v > 0 for v in (self.diameter_m, self.range_diameters)):
            raise ValueError("Diameter and interaction range must be finite and positive")
        if not isfinite(self.seabed_z):
            raise ValueError("Seabed height must be finite")


def plane_jet_loss(position, exhaust, center, normal, tangent_u, tangent_v, half_size, *, coefficient, reach):
    """Loss on each N motor jet for E environments and P oriented rectangles.

    Input motors (E,N,3), rectangles (E,P,3), half_size (P,2), coefficient (P,).
    Normals point from solid into water, tangent bases must be orthonormal.
    Exhaust points OPPOSITE actual signed force. Only downstream/front-facing
    intersections count; finite panel extents are not infinite virtual walls.
    Returned loss and ray distance have shape (E,N,P); misses have +inf distance.
    """
    incidence = -(exhaust[:, :, None] * normal[:, None]).sum(-1)
    clearance = ((position[:, :, None] - center[:, None]) * normal[:, None]).sum(-1)
    distance = clearance / incidence.clamp_min(1e-8)
    intersection = position[:, :, None] + distance[..., None] * exhaust[:, :, None]
    offset = intersection - center[:, None]
    u = (offset * tangent_u[:, None]).sum(-1).abs()
    v = (offset * tangent_v[:, None]).sum(-1).abs()
    hit = (incidence > 1e-6) & (clearance >= 0) & (distance < reach)
    hit &= (u <= half_size[None, None, :, 0]) & (v <= half_size[None, None, :, 1])
    # Squared incidence follows normal momentum-flux scaling. The compact
    # smooth kernel and its coefficient/range are explicit stress assumptions.
    kernel = (1 - (distance / reach).square()).clamp_min(0).square()
    loss = torch.where(hit, coefficient * incidence.clamp(0, 1).square() * kernel, 0.0)
    return loss, torch.where(hit, distance, torch.inf)


class BlueROVBoundaryEffect:
    """BlueROV jets against native sand or finite pool, plus the standard panel.

    Uses the nearest downstream surface only: no double counting at corners.
    Native panel half extents (0.04,0.575,0.575)m match all current PBR panels.
    The opt-in pool uses the same finite wet rectangles as its collision asset.
    Custom/scaled walls, ship hulls, inlet blockage and mutual jets are outside scope.
    """

    def __init__(
        self, device="cpu", cfg=None, *, pool_geometry=None, pool_center=(0, 0, 0), env_origins=None, positions=None
    ):
        self.cfg = BoundaryEffectCfg() if cfg is None else cfg
        self.positions = torch.tensor(POSITIONS if positions is None else positions, device=device)
        if self.positions.shape != (8, 3) or not torch.isfinite(self.positions).all():
            raise ValueError("Expected eight finite thruster positions")
        self.directions = torch.tensor(DIRECTIONS, device=device)
        self.panel_normals = self.positions.new_tensor(
            [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        )
        self.panel_u = self.positions.new_tensor([(0, 1, 0)] * 2 + [(1, 0, 0)] * 4)
        self.panel_v = torch.linalg.cross(self.panel_normals, self.panel_u)
        self.panel_center = self.panel_normals * self.positions.new_tensor((0.04, 0.575, 0.575))
        self.panel_half_size = self.positions.new_tensor([(0.575, 0.575)] * 2 + [(0.04, 0.575)] * 4)
        self.pool_geometry = pool_geometry
        self.pool_center = self.positions.new_tensor(pool_center)
        self.env_origins = env_origins
        if pool_geometry is not None:
            rectangles = pool_geometry.wet_boundary_rectangles()
            self.pool_centers = self.positions.new_tensor([r[0] for r in rectangles])
            self.pool_normals = self.positions.new_tensor([r[1] for r in rectangles])
            self.pool_u = self.positions.new_tensor([r[2] for r in rectangles])
            self.pool_v = self.positions.new_tensor([r[3] for r in rectangles])
            self.pool_half_size = self.positions.new_tensor([r[4] for r in rectangles])
            self.pool_coeff = self.positions.new_tensor(
                [self.cfg.seabed_loss if r[5] == "floor" else self.cfg.wall_loss for r in rectangles]
            )

    def apply(self, base_pose_w, axial_force, panel_pose_w=None):
        count = base_pose_w.shape[0]
        q = base_pose_w[:, None, 3:].expand(-1, 8, -1)
        positions = base_pose_w[:, None, :3] + quat_apply_xyzw(q, self.positions.expand(count, -1, -1))
        force_b = axial_force[..., None] * self.directions
        axes_w = quat_apply_xyzw(q, self.directions.expand(count, -1, -1))
        exhaust = -axial_force.sign()[..., None] * axes_w
        center = self.positions.new_tensor((0, 0, self.cfg.seabed_z)).view(1, 1, 3).expand(count, 1, 3)
        normal = self.positions.new_tensor((0, 0, 1)).view(1, 1, 3).expand(count, 1, 3)
        u = self.positions.new_tensor((1, 0, 0)).view(1, 1, 3).expand(count, 1, 3)
        v = self.positions.new_tensor((0, 1, 0)).view(1, 1, 3).expand(count, 1, 3)
        half_size = self.positions.new_full((1, 2), 100.0)  # Exact native Sand collision rectangle.
        coeff = self.positions.new_tensor([self.cfg.seabed_loss])
        if self.pool_geometry is not None:
            origins = self.positions.new_zeros((count, 3)) if self.env_origins is None else self.env_origins
            if origins.shape != (count, 3):
                raise ValueError("Pool environment origins must match the motor batch")
            center = self.pool_centers[None] + self.pool_center + origins[:, None]
            normal = self.pool_normals[None].expand(count, -1, -1)
            u = self.pool_u[None].expand(count, -1, -1)
            v = self.pool_v[None].expand(count, -1, -1)
            half_size, coeff = self.pool_half_size, self.pool_coeff
        if panel_pose_w is not None:
            pq = panel_pose_w[:, None, 3:].expand(-1, 6, -1)
            center = torch.cat(
                (center, panel_pose_w[:, None, :3] + quat_apply_xyzw(pq, self.panel_center.expand(count, -1, -1))), 1
            )
            normal = torch.cat((normal, quat_apply_xyzw(pq, self.panel_normals.expand(count, -1, -1))), 1)
            u = torch.cat((u, quat_apply_xyzw(pq, self.panel_u.expand(count, -1, -1))), 1)
            v = torch.cat((v, quat_apply_xyzw(pq, self.panel_v.expand(count, -1, -1))), 1)
            half_size = torch.cat((half_size, self.panel_half_size))
            coeff = torch.cat((coeff, coeff.new_full((6,), self.cfg.wall_loss)))
        loss, distance = plane_jet_loss(
            positions,
            exhaust,
            center,
            normal,
            u,
            v,
            half_size,
            coefficient=coeff,
            reach=self.cfg.diameter_m * self.cfg.range_diameters,
        )
        nearest = distance.argmin(-1, keepdim=True)
        selected = loss.gather(-1, nearest).squeeze(-1)
        gain = 1.0 - selected
        return force_b * gain[..., None], gain, loss
