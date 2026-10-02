"""Finite cargo-hold scenery built from timber architecture and real CC0 PBR maps.

Original procedural geometry, not a wreck scan. Opt-in visual set dressing only:
no task collision/grasp claims, no default task environment changes.
"""

from pathlib import Path

HOLD_BOUNDS_M = ((-3.1, -2.07, -0.16), (3.1, 2.07, 2.53))


def build_ship_interior(stage, root="/World/ShipInterior"):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    asset_dir = Path(__file__).resolve().parent / "data/environment_variants/deck"
    UsdGeom.Xform.Define(stage, root)
    material = UsdShade.Material.Define(stage, root + "/Timber")
    surface = UsdShade.Shader.Define(stage, root + "/Timber/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    uv = UsdShade.Shader.Define(stage, root + "/Timber/UV")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    for channel, input_name, kind in (
        ("diffuse", "diffuseColor", Sdf.ValueTypeNames.Color3f),
        ("normal", "normal", Sdf.ValueTypeNames.Normal3f),
        ("roughness", "roughness", Sdf.ValueTypeNames.Float),
    ):
        texture = UsdShade.Shader.Define(stage, f"{root}/Timber/{channel}")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(asset_dir / f"{channel}.jpg")))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if channel == "diffuse" else "raw")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        for axis in ("S", "T"):
            texture.CreateInput(f"wrap{axis}", Sdf.ValueTypeNames.Token).Set("repeat")
        if channel == "normal":
            texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        surface.CreateInput(input_name, kind).ConnectToSource(
            texture.ConnectableAPI(), "r" if channel == "roughness" else "rgb"
        )

    def box(name, center, size, rotate_x=0):
        # Independent UVs on all six faces, in physical metres, no stretched rim.
        sx, sy, sz = (v / 2 for v in size)
        vertices = [
            (-sx, -sy, -sz),
            (sx, -sy, -sz),
            (sx, sy, -sz),
            (-sx, sy, -sz),
            (-sx, -sy, sz),
            (sx, -sy, sz),
            (sx, sy, sz),
            (-sx, sy, sz),
        ]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
        mesh = UsdGeom.Mesh.Define(stage, f"{root}/{name}")
        mesh.CreatePointsAttr(vertices)
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        texture_uv = []
        for u, v in (
            (size[0], size[1]),
            (size[0], size[1]),
            (size[0], size[2]),
            (size[0], size[2]),
            (size[1], size[2]),
            (size[1], size[2]),
        ):
            texture_uv.extend([(0, 0), (u / 2, 0), (u / 2, v / 2), (0, v / 2)])
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(texture_uv)
        mesh.AddTranslateOp().Set(Gf.Vec3d(*center))
        if rotate_x:
            mesh.AddRotateXOp().Set(rotate_x)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    box("Deck", (0, 0, -0.08), (6.2, 3.6, 0.16))
    # Flared lower hull and vertical upper hull, visibly enclosing the robot.
    for side, sign in (("Port", 1), ("Starboard", -1)):
        box(side + "Bilge", (0, sign * 1.82, 0.42), (6.2, 0.14, 0.90), -sign * 22)
        box(side + "Hull", (0, sign * 2.0, 1.63), (6.2, 0.14, 1.75))
        box(side + "Shelf", (0, sign * 1.85, 0.9), (6.2, 0.28, 0.13))
        for index, x in enumerate((-2.7, -1.7, -0.7, 0.3, 1.3, 2.3)):
            box(f"{side}Frame{index}", (x, sign * 1.86, 1.65), (0.13, 0.20, 1.7))
            box(f"{side}Knee{index}", (x, sign * 1.70, 0.45), (0.13, 0.22, 0.92), -sign * 22)
    box("AftBulkhead", (-3.0, 0, 1.25), (0.18, 4.0, 2.5))
    # Two-part far bulkhead forms an actual doorway, not an endless wood plane.
    box("ForwardBulkheadPort", (3.0, 1.35, 1.25), (0.18, 1.3, 2.5))
    box("ForwardBulkheadStarboard", (3.0, -1.35, 1.25), (0.18, 1.3, 2.5))
    box("DoorLintel", (3.0, 0, 2.23), (0.18, 1.4, 0.54))
    for index, x in enumerate((-2.7, -1.7, -0.7, 0.3, 1.3, 2.3)):
        box(f"OverheadBeam{index}", (x, 0, 2.40), (0.16, 4.0, 0.18))
    # Partial overhead deck leaves an authentic open loading hatch for key light.
    box("OverheadDeckPort", (0, 1.55, 2.49), (6.2, 0.9, 0.08))
    box("OverheadDeckStarboard", (0, -1.55, 2.49), (6.2, 0.9, 0.08))
    return {
        "bounds_m": HOLD_BOUNDS_M,
        "geometry": "original procedural finite timber cargo hold",
        "texture": "https://polyhaven.com/a/weathered_planks",
        "texture_license": "CC0-1.0",
        "collision": False,
        "scope": "visual preview; no chest task validation",
    }
