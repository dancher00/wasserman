"""Camera-readable native Rex propellers, driven by realized motors, visual only."""

import math


class RexRotorPresentation:
    def __init__(self, stage, parameters):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        self.ops = []
        self.phase = [0.0] * 6
        self.turns = [0.0] * 6
        self.samples = []
        self.paths = []
        for i in range(6):
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/Geometry/base_link/thruster_{i}")
            if not prim.IsValid():
                raise ValueError(f"Missing native Rex rotor visual {i}")
            if any(p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(prim)):
                raise ValueError("Rotor presentation cannot modify a collider or body")
            xform = UsdGeom.Xformable(prim)
            axis = xform.GetLocalTransformation().TransformDir(Gf.Vec3d(1, 0, 0)).GetNormalized()
            if (axis - Gf.Vec3d(*parameters.thruster_directions[i])).GetLength() > 1e-5:
                raise ValueError("Imported rotor axis differs from physical motor axis")
            self.ops.append(xform.AddRotateXOp(opSuffix="presentation_spin"))
            self.paths.append(str(prim.GetPath()))

    def update(self, motors):
        # Native physical omega remains untouched. The monotone display mapping
        # is bounded at four turns/s to avoid the stationary-wheel effect at30fps.
        forces = motors.force[0].detach().cpu().tolist()
        for i, force in enumerate(forces):
            speed = math.copysign(4 * math.sqrt(min(abs(force) / motors.parameters.max_thrust, 1)), force)
            increment = speed * motors.dt
            self.turns[i] += abs(increment)
            self.phase[i] = (self.phase[i] + increment * 360) % 360
            self.ops[i].Set(self.phase[i])

    def record(self):
        self.samples.append(self.phase.copy())

    def report(self):
        return dict(
            native_rotor_paths=self.paths,
            axis="local +X, checked against physical motor axes",
            display_max_revolutions_s=4,
            physical_rpm=False,
            driven_by="realized signed motor force",
            accumulated_absolute_turns=self.turns,
            frame_angles_deg=self.samples,
            modifies="six noncolliding visual transforms only",
        )
