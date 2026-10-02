"""Orientation-aware thrust allocation for visualization, not motor dynamics.

Mounts and CW/CCW meshes follow evan-palmer/blue's Heavy Reach description.
Legacy tasks apply an ideal body wrench. The physical T200 variants reuse this
mount geometry and supply their realized forces to the illustrative animation.
"""

import math

import torch

NOMINAL_POSITIONS = (
    (0.088, -0.102, -0.04),
    (0.088, 0.102, -0.04),
    (-0.088, -0.102, -0.04),
    (-0.088, 0.102, -0.04),
    (0.118, -0.215, 0.064),
    (0.118, 0.215, 0.064),
    (-0.118, -0.215, 0.064),
    (-0.118, 0.215, 0.064),
)
# Registration of the vendored CAD housing to its nominal mount description.
# Measured from the four vertical housing axes and horizontal motor center Z.
CAD_OFFSET = (0.0978, -0.0244, 0.0115)
# Mounted local joint axis (0, 0, -1) expressed in base_link (FLU).
SQRT3_2 = math.sqrt(3) / 2
DIRECTIONS = (
    (-SQRT3_2, -0.5, 0),
    (-SQRT3_2, 0.5, 0),
    (SQRT3_2, -0.5, 0),
    (SQRT3_2, 0.5, 0),
    (0, 0, -1),
    (0, 0, -1),
    (0, 0, -1),
    (0, 0, -1),
)
# The cup-shaped rotor surrounds the stator; it must NOT sit end-to-end on its
# nose. The previous extra 36–38 mm shift exposed a second tier outside the duct.
SHAFT_OFFSETS = (0.0,) * 8
POSITIONS = tuple(
    tuple(p[i] + CAD_OFFSET[i] - DIRECTIONS[j][i] * SHAFT_OFFSETS[j] for i in range(3))
    for j, p in enumerate(NOMINAL_POSITIONS)
)
HANDEDNESS = (1, 1, -1, -1, 1, -1, -1, 1)


def allocation_matrix(device="cpu", dtype=torch.float32, *, positions=None):
    """Map signed axial forces to force/torque at the base COM (z=0.011 m)."""
    positions = torch.tensor(POSITIONS if positions is None else positions, device=device, dtype=dtype)
    if positions.shape != (8, 3) or not torch.isfinite(positions).all():
        raise ValueError("Expected eight finite thruster positions")
    positions = positions - positions.new_tensor((0, 0, 0.011))
    directions = torch.tensor(DIRECTIONS, device=device, dtype=dtype)
    return torch.cat((directions, torch.linalg.cross(positions, directions)), dim=1).T


class ThrusterDisplayState:
    """Integrate illustrative rotor speeds from a minimum-norm wrench mix.

    Omega is NOT calibrated T200 RPM. Use 0–4 rev/s to keep motion readable at
    30 fps; handedness and thrust reversal are preserved. No time remapping of
    the robot, no extra rigid bodies, collisions, or physical forces.
    """

    def __init__(self, num_envs, device, *, positions=None):
        self.inverse = torch.linalg.pinv(allocation_matrix(device, positions=positions))
        self.hand = torch.tensor(HANDEDNESS, device=device)
        self.force = torch.zeros(num_envs, 8, device=device)
        self.omega = torch.zeros_like(self.force)
        self.phase = torch.zeros_like(self.force)

    def update(self, force_b, torque_b, dt, realized_force=None):
        self.force.copy_(
            torch.cat((force_b, torque_b), dim=-1) @ self.inverse.T if realized_force is None else realized_force
        )
        demand = (self.force.abs() / 40.0).clamp(max=1.0).sqrt()
        target = self.hand * self.force.sign() * demand * (2 * math.pi * 4)
        self.omega.lerp_(target, 1 - math.exp(-dt / 0.10))
        self.phase.add_(self.omega * dt).remainder_(2 * math.pi)

    def reset(self, indices):
        self.force[indices] = 0
        self.omega[indices] = 0
        self.phase[indices] = 0


class ThrusterVisuals:
    """Rotate only the eight propeller visual Xforms in each displayed robot."""

    def __init__(self, stage, num_envs, device, max_visible_envs=16, *, positions=None):
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        material = UsdShade.Material.Define(stage, "/World/Looks/T200Graphite")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/T200Graphite/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.055, 0.06, 0.065))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.48)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        self.state = ThrusterDisplayState(min(num_envs, max_visible_envs), device, positions=positions)
        self.operations = []
        for index in range(min(num_envs, max_visible_envs)):
            ops = []
            for rotor in range(1, 9):
                path = f"/World/envs/env_{index}/Robot/Geometry/base_link/thruster{rotor}_visual"
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise RuntimeError(f"Missing original T200 propeller visual: {path}")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
                transform = UsdGeom.Xformable(prim).GetLocalTransformation()
                shaft = transform.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
                expected = Gf.Vec3d(*DIRECTIONS[rotor - 1])
                if (shaft - expected).GetLength() > 1e-5:
                    raise RuntimeError(f"Converted T200 shaft axis differs from allocator: {path}: {shaft}")
                # Collada has an internal +pi/2 X node transform, composed by
                # the importer into this Xform. Converted mesh shaft is Z:
                # use -Z, NOT the pre-conversion URDF mesh's +Y.
                ops.append(UsdGeom.Xformable(prim).AddRotateZOp(opSuffix="propeller_spin"))
            self.operations.append(ops)

    def update(self, force_b, torque_b, dt, render, realized_force=None):
        count = len(self.operations)
        self.state.update(
            force_b[:count], torque_b[:count], dt, None if realized_force is None else realized_force[:count]
        )
        if render:
            degrees = self.state.phase.rad2deg().cpu().tolist()
            for row, ops in zip(degrees, self.operations, strict=True):
                for angle, op in zip(row, ops, strict=True):
                    op.Set(-angle)

    def reset(self, env_ids):
        if env_ids is not None:
            env_ids = torch.as_tensor(env_ids, device=self.state.phase.device, dtype=torch.long)
        env_ids = slice(None) if env_ids is None else env_ids[env_ids < len(self.operations)]
        self.state.reset(env_ids)
