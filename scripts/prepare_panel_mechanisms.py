"""Versioned authored CAD dressing; upstream collision/joint definitions preserved."""

import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/wasman/assets/data/objects/ambench"
DEST = SOURCE.parent / "marine_mechanisms_v1"


def material(stage, path, color, metal, rough):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def box(stage, path, pos, size, mat):
    obj = UsdGeom.Cube.Define(stage, path)
    obj.CreateSizeAttr(1)
    obj.AddTranslateOp().Set(Gf.Vec3d(*pos))
    obj.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
    return obj


def cylinder(stage, path, pos, radius, length, axis, mat, sides=64):
    # Explicit smooth-sided mesh avoids low-resolution cylinder tessellation.
    theta = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    points = np.concatenate(
        [
            np.column_stack([np.full(sides, h), radius * np.cos(theta), radius * np.sin(theta)])
            for h in [-length / 2, length / 2]
        ]
    )
    if axis == "Y":
        points = points[:, [1, 0, 2]]
    if axis == "Z":
        points = points[:, [1, 2, 0]]
    faces = [[i, (i + 1) % sides, (i + 1) % sides + sides, i + sides] for i in range(sides)]
    faces += [list(range(sides - 1, -1, -1)), list(range(sides, 2 * sides))]
    obj = UsdGeom.Mesh.Define(stage, path)
    obj.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    obj.CreateFaceVertexCountsAttr([len(f) for f in faces])
    obj.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    obj.CreateSubdivisionSchemeAttr("none")
    obj.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
    return obj


def signature(stage):
    rows = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint):
            rows[str(prim.GetPath())] = {
                a.GetName(): str(a.Get())
                for a in prim.GetAttributes()
                if a.GetName().startswith(
                    ("physics:", "physx", "xformOp:", "points", "faceVertex", "size", "radius", "height", "axis")
                )
            }
            rows[str(prim.GetPath())].update(
                {
                    r.GetName(): str(r.GetTargets())
                    for r in prim.GetRelationships()
                    if r.GetName().startswith("physics:")
                }
            )
    return rows


def build(name):
    source = SOURCE / (name + ".usd")
    original = Usd.Stage.Open(str(source))
    dest = DEST / (name + ".usda")
    original.Flatten().Export(str(dest))
    stage = Usd.Stage.Open(str(dest))
    root = str(stage.GetDefaultPrim().GetPath())
    mats = {
        "steel": material(stage, root + "/Looks/Steel", (0.43, 0.49, 0.51), 0.82, 0.29),
        "dark": material(stage, root + "/Looks/Anodized", (0.055, 0.085, 0.095), 0.65, 0.38),
        "rubber": material(stage, root + "/Looks/Grip", (0.045, 0.055, 0.050), 0.02, 0.72),
        "yellow": material(stage, root + "/Looks/SafetyOchre", (0.72, 0.40, 0.045), 0.25, 0.43),
        "black": material(stage, root + "/Looks/Socket", (0.009, 0.014, 0.017), 0.05, 0.7),
    }
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Gprim):
            namepart = prim.GetName()
            key = "rubber" if namepart in ("tip_collision", "knob_handle") else "steel"
            if namepart in ("base_collision", "base", "knob_base"):
                key = "dark"
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mats[key])
    if name == "lever":
        base, moving = root + "/lever/base", root + "/lever/lever"
        # Mounting plate lies behind the existing pivot; rotation maps +local Z
        # into the approach direction. Decorative pieces have no collision API.
        box(stage, base + "/Mount", (0, 0, -0.015), (0.09, 0.08, 0.008), mats["dark"])
        for x in [-0.035, 0.035]:
            for y in [-0.029, 0.029]:
                tag = str(len(list(stage.Traverse())))
                cylinder(stage, base + "/Bolt" + tag, (x, y, -0.009), 0.006, 0.006, "Z", mats["steel"], 6)
                cylinder(stage, base + "/Socket" + tag, (x, y, -0.0055), 0.0027, 0.001, "Z", mats["black"], 6)
        cylinder(stage, base + "/Bearing", (0, 0, 0), 0.023, 0.042, "Y", mats["steel"])
        for y in [-0.024, 0.024]:
            cylinder(stage, base + ("/HubA" if y < 0 else "/HubB"), (0, y, 0), 0.015, 0.009, "Y", mats["dark"])
        box(stage, moving + "/SafetyInlay", (-0.0151, 0, 0.078), (0.0006, 0.009, 0.105), mats["yellow"])
        for i, y in enumerate(np.linspace(-0.032, 0.032, 13)):
            cylinder(stage, moving + f"/GripRib{i}", (0, float(y), 0.15), 0.0112, 0.0015, "Y", mats["rubber"])
        for i, y in enumerate([-0.038, 0.038]):
            cylinder(stage, moving + f"/GripCap{i}", (0, y, 0.15), 0.011, 0.003, "Y", mats["yellow"])
    else:
        base, moving = root + "/Xform/Rail", root + "/Xform/Slider/Xform"
        box(stage, base + "/Mount", (-0.016, 0, 0), (0.012, 1.09, 0.16), mats["dark"])
        for i, y in enumerate([-0.525, 0.525]):
            box(stage, base + f"/EndBracket{i}", (0.003, y, 0), (0.065, 0.045, 0.13), mats["steel"])
            for j, z in enumerate([-0.051, 0.051]):
                cylinder(stage, base + f"/Bolt{i}{j}", (0.040, y, z), 0.011, 0.006, "X", mats["steel"], 6)
                cylinder(stage, base + f"/Socket{i}{j}", (0.0435, y, z), 0.005, 0.001, "X", mats["black"], 6)
        for i, y in enumerate(np.linspace(-0.48, 0.48, 25)):
            box(
                stage,
                base + f"/Tick{i}",
                (-0.009, float(y), 0.059),
                (0.001, 0.002, 0.016 if i % 4 == 0 else 0.008),
                mats["steel"],
            )
        box(stage, moving + "/GripInlay", (0.055, 0, 0), (0.001, 0.026, 0.047), mats["yellow"])
    assert signature(stage) == signature(original), "Appearance update changed original physical definitions"
    stage.GetRootLayer().Save()
    return {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "asset": str(dest.relative_to(ROOT)),
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        "physical_definitions_unchanged": True,
    }


def grounded_panel():
    source = ROOT / "src/wasman/assets/data/panels/ship_green/panel.usda"
    original = Usd.Stage.Open(str(source))
    dest = DEST / "grounded_panel.usda"
    # Reference keeps PBR texture resolution relative to its original directory.
    stage = Usd.Stage.CreateNew(str(dest))
    panel = UsdGeom.Xform.Define(stage, "/Panel").GetPrim()
    stage.SetDefaultPrim(panel)
    panel.GetReferences().AddReference("../../panels/ship_green/panel.usda")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    mat = UsdShade.Material(stage.GetPrimAtPath("/Panel/Material"))
    # The lower part is buried below z=0 for all registered fixture reset heights.
    # Original upper slab stays identical. Extension is a real fixed collider.
    extension = box(stage, "/Panel/Foundation", (0, 0, -0.79), (0.08, 1.15, 0.45), mat)
    UsdPhysics.CollisionAPI.Apply(extension.GetPrim())
    extension.CreatePurposeAttr("guide")
    surface = UsdGeom.Mesh(original.GetPrimAtPath("/Panel/Surface"))
    mesh = UsdGeom.Mesh.Define(stage, "/Panel/FoundationSurface")
    points = np.asarray(surface.GetPointsAttr().Get()).copy()
    points[:, 2] = (points[:, 2] + 0.575) * (0.45 / 1.15) - 1.015
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(surface.GetFaceVertexCountsAttr().Get())
    mesh.CreateFaceVertexIndicesAttr(surface.GetFaceVertexIndicesAttr().Get())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    uv = np.asarray(UsdGeom.PrimvarsAPI(surface).GetPrimvar("st").Get()).copy()
    uv[:16, 1] = uv[:16, 1] * (0.45 / 1.15) - 0.44
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        Vt.Vec2fArray.FromNumpy(uv.astype(np.float32))
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    trim = material(stage, "/Panel/Trim", (0.13, 0.18, 0.17), 0.65, 0.48)
    for i, y in enumerate([-0.565, 0.565]):
        box(stage, f"/Panel/Edge{i}", (-0.044, y, -0.22), (0.009, 0.016, 1.59), trim)
    stage.GetRootLayer().Save()
    assert signature(stage)["/Panel/Slab"] == signature(original)["/Panel/Slab"]
    return str(dest.relative_to(ROOT))


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "marine-mechanisms-v1",
        "visuals": "Original authored industrial CAD dressing; not a sourced scan or calibrated hardware model.",
        "assets": [build(n) for n in ["lever", "slider_with_rail"]],
        "panel": grounded_panel(),
    }
    (DEST / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
