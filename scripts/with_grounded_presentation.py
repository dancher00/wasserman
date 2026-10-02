# ruff: noqa: E402
"""Render existing demos with grounded visual surfaces; preserve all physics opinions.

This wrapper never edits a source USD or task configuration. It is intentionally
outside the frozen benchmark source set. New films are separate presentation takes.
"""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--audit-path", type=Path, required=True)
parser.add_argument("script", type=Path)
parser.add_argument("arguments", nargs=argparse.REMAINDER)
args = parser.parse_args()
from isaaclab.app import AppLauncher

original_launcher = AppLauncher.__init__
audit = dict(
    version="grounded-presentation-v1",
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    changes=[],
    physics_unchanged=False,
    scope="new presentation takes, not replacement benchmark trials",
)
applied = False


def physics_signature(stage):
    from pxr import UsdPhysics

    signature = {}
    for prim in stage.Traverse():
        relevant = {}
        collider = prim.HasAPI(UsdPhysics.CollisionAPI)
        body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name.startswith(("physics:", "physx")) or (
                (collider or body)
                and (
                    name.startswith("xformOp")
                    or name in ["points", "faceVertexCounts", "faceVertexIndices", "size", "radius", "height", "axis"]
                )
            ):
                relevant[name] = str(attr.Get())
        for rel in prim.GetRelationships():
            if rel.GetName().startswith(("physics:", "physx")):
                relevant[rel.GetName()] = [str(x) for x in rel.GetTargets()]
        if relevant:
            signature[str(prim.GetPath())] = relevant
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def dress(stage):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    before = physics_signature(stage)
    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        # Every standard panel style separates its visible mesh from its collider.
        if path.endswith("/Panel/Surface") and prim.IsA(UsdGeom.Mesh):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError("Panel visual unexpectedly carries collisions")
            mesh = UsdGeom.Mesh(prim)
            points = list(mesh.GetPointsAttr().Get())
            low = min(p[2] for p in points)
            if low < -0.8:
                continue
            target = min(-2.0, low)
            updated = [Gf.Vec3f(p[0], p[1], target if abs(p[2] - low) < 1e-5 else p[2]) for p in points]
            mesh.GetPointsAttr().Set(updated)
            st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
            if st and st.Get():
                uv = list(st.Get())
                # Standard slab sides have a vertical second texture coordinate.
                for i in range(min(16, len(uv))):
                    if abs(points[i][2] - low) < 1e-5:
                        uv[i] = Gf.Vec2f(uv[i][0], uv[i][1] + target - low)
                st.Set(uv)
            audit["changes"].append(
                dict(prim=path, change="extend visible slab down", old_bottom=low, new_bottom=target)
            )
        if path.endswith("/SandApron/Sand") and prim.IsA(UsdGeom.Mesh):
            original = UsdGeom.Mesh(prim)
            points = list(original.GetPointsAttr().Get())
            # Retain the128-sided aperture; extend only the outer boundary beyond every view.
            updated = [
                Gf.Vec3f(p[0] * 100 / 1.5, p[1] * 100 / 1.5, p[2]) if max(abs(p[0]), abs(p[1])) > 1.49 else p
                for p in points
            ]
            visual = UsdGeom.Mesh.Define(stage, str(prim.GetParent().GetPath()) + "/ContinuousVisibleSand")
            visual.CreatePointsAttr(updated)
            visual.CreateFaceVertexCountsAttr(original.GetFaceVertexCountsAttr().Get())
            visual.CreateFaceVertexIndicesAttr(original.GetFaceVertexIndicesAttr().Get())
            visual.CreateSubdivisionSchemeAttr("none")
            visual.CreateDoubleSidedAttr(True)
            UsdGeom.PrimvarsAPI(visual).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
                [(p[0] / 2, p[1] / 2) for p in updated]
            )
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
            original.CreatePurposeAttr("guide")
            audit["changes"].append(
                dict(
                    prim=path,
                    change="continuous visible sand around unchanged aperture",
                    original_collider_preserved=True,
                    old_half_extent_m=1.5,
                    new_visual_half_extent_m=100,
                )
            )
    after = physics_signature(stage)
    if before != after:
        raise ValueError("Presentation changed physical geometry/properties")
    audit.update(physics_unchanged=True, physics_before_sha256=before, physics_after_sha256=after)


def launch(self, *a, **kw):
    original_launcher(self, *a, **kw)
    import isaaclab.sim as sim

    reset = sim.SimulationContext.reset

    def presentation_reset(context, *r, **k):
        global applied
        if not applied:
            dress(sim.get_current_stage())
            applied = True
            # SimulationApp.close uses os._exit: persist proof before Kit can exit.
            args.audit_path.parent.mkdir(parents=True, exist_ok=True)
            args.audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        return reset(context, *r, **k)

    sim.SimulationContext.reset = presentation_reset


AppLauncher.__init__ = launch
sys.argv = [str(args.script), *args.arguments]
try:
    runpy.run_path(str(args.script), run_name="__main__")
finally:
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    args.audit_path.write_text(json.dumps(audit, indent=2) + "\n")
