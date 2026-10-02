"""Generate finite pool architecture with sourced CC0 straight-grid ceramic PBR."""

import hashlib
import json
from dataclasses import asdict

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from wasman.assets.pool_geometry import POOL_GEOMETRY, POOL_USD_PATH
from wasman.assets.pool_markings import floor_tile_regions, partition_rectangles, wall_tile_regions

# Eight source ceramic modules per repeat, uniformly scaled to 50mm mosaic.
TILE_REPEAT_M = 0.40
TILE_TEXTURE_DIRECTORY = "tiles107"


def build_pool(stage):
    geometry = POOL_GEOMETRY
    root = UsdGeom.Xform.Define(stage, "/Pool")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    def tiled_material(name, tint, bias=(0.0, 0.0, 0.0)):
        material = UsdShade.Material.Define(stage, f"/Pool/{name}")
        surface = UsdShade.Shader.Define(stage, f"/Pool/{name}/Surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        uv = UsdShade.Shader.Define(stage, f"/Pool/{name}/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        for channel, input_name, kind in (
            ("diffuse", "diffuseColor", Sdf.ValueTypeNames.Color3f),
            ("normal", "normal", Sdf.ValueTypeNames.Normal3f),
            ("roughness", "roughness", Sdf.ValueTypeNames.Float),
        ):
            texture = UsdShade.Shader.Define(stage, f"/Pool/{name}/{channel}")
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(f"{TILE_TEXTURE_DIRECTORY}/{channel}.jpg")
            )
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
                "sRGB" if channel == "diffuse" else "raw"
            )
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
            for axis in ("S", "T"):
                texture.CreateInput(f"wrap{axis}", Sdf.ValueTypeNames.Token).Set("repeat")
            if channel == "normal":
                texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
                texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
            elif channel == "diffuse":
                texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*tint, 1))
                texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias, 0))
            surface.CreateInput(input_name, kind).ConnectToSource(
                texture.ConnectableAPI(), "r" if channel == "roughness" else "rgb"
            )
        return material

    material = tiled_material("Tile", (0.80, 0.92, 1.0))
    # Source has pale ceramic/darker grout. An authored inverse albedo variant
    # keeps light joints around dark ceramic, unlike a simple navy multiply.
    stripe = tiled_material("LaneMaterial", (-0.55, -0.55, -0.55), (0.57, 0.595, 0.635))

    def plain(name, color, roughness=0.4, opacity=1.0):
        mat = UsdShade.Material.Define(stage, f"/Pool/{name}")
        shader = UsdShade.Shader.Define(stage, f"/Pool/{name}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.333 if name == "Water" else 1.5)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    coping = plain("CopingMaterial", (0.67, 0.73, 0.75), 0.75)
    water = plain("Water", (0.035, 0.31, 0.43), 0.08, 0.12)

    def box(name, center, size, mat, collision=False, marked_faces=None):
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
        mesh = UsdGeom.Mesh.Define(stage, f"/Pool/{name}")
        marked_faces = marked_faces or {}
        points, texture_uv, indices, dark_indices = [], [], [], []
        for face_index, face in enumerate(faces):
            # Global planar coordinates keep grout and normal phase identical
            # across a light/dark tile boundary and neighbouring subdivisions.
            axes = (0, 1) if face_index < 2 else (0, 2) if face_index < 4 else (1, 2)
            u_axis, v_axis = axes
            normal_axis = ({0, 1, 2} - set(axes)).pop()
            normal_coord = vertices[face[0]][normal_axis]
            bounds = (
                min(vertices[i][u_axis] + center[u_axis] for i in face),
                max(vertices[i][u_axis] + center[u_axis] for i in face),
                min(vertices[i][v_axis] + center[v_axis] for i in face),
                max(vertices[i][v_axis] + center[v_axis] for i in face),
            )
            cells = partition_rectangles(bounds, marked_faces.get(face_index, []))
            for (a, b, c, d), dark in cells:
                quad = [(a, c), (b, c), (b, d), (a, d)]
                # Preserve outward winding for each cuboid face.
                if face_index in (0, 3, 5):
                    quad.reverse()
                if dark:
                    dark_indices.append(len(indices) // 4)
                for u, v in quad:
                    point = [0.0, 0.0, 0.0]
                    point[normal_axis] = normal_coord
                    point[u_axis], point[v_axis] = u - center[u_axis], v - center[v_axis]
                    indices.append(len(points))
                    points.append(tuple(point))
                    texture_uv.append((u / TILE_REPEAT_M, v / TILE_REPEAT_M))
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * (len(indices) // 4))
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(texture_uv)
        mesh.AddTranslateOp().Set(Gf.Vec3d(*center))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        if dark_indices:
            binding = UsdShade.MaterialBindingAPI(mesh.GetPrim())
            subset = binding.CreateMaterialBindSubset("DarkTileRegions", dark_indices)
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(stripe)
        if collision:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            # Every generated component is an exact cuboid. Use primitive box
            # approximation rather than long thin convex hulls (GPU cooking can
            # otherwise fall back to CPU for 25m wall/coping pieces).
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("boundingCube")
        return mesh

    for name, center, size in geometry.collision_boxes():
        marked = {}
        if name == "Floor":
            marked[1] = floor_tile_regions()
        else:
            inner_face = {"WestWall": 4, "EastWall": 5, "SouthWall": 3, "NorthWall": 2}[name]
            marked[inner_face] = wall_tile_regions(end_wall=name in ("WestWall", "EastWall"))
        box(name, center, size, material, collision=True, marked_faces=marked)
    x, y, h = geometry.length_m / 2, geometry.width_m / 2, geometry.wall_height_m
    for side, sign in (("West", -1), ("East", 1)):
        box(side + "Coping", (sign * (x + 0.2), 0, h + 0.06), (0.4, 2 * y + 0.8, 0.12), coping, True)
    for side, sign in (("South", -1), ("North", 1)):
        box(side + "Coping", (0, sign * (y + 0.2), h + 0.06), (2 * x, 0.4, 0.12), coping, True)
    # An explicitly finite non-collision optical surface. Dynamics use the
    # existing submerged-robot approximation, not a free-surface fluid solver.
    surface_mesh = UsdGeom.Mesh.Define(stage, "/Pool/WaterSurface")
    z = geometry.water_depth_m
    surface_mesh.CreatePointsAttr([(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)])
    surface_mesh.CreateFaceVertexCountsAttr([4])
    surface_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    surface_mesh.CreateSubdivisionSchemeAttr("none")
    surface_mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(surface_mesh.GetPrim()).Bind(water)


def main():
    POOL_USD_PATH.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    build_pool(stage)
    stage.GetRootLayer().Export(str(POOL_USD_PATH))
    provenance = {
        "geometry": "original procedural finite swimming pool; metre units",
        "dimensions": asdict(POOL_GEOMETRY),
        "dimension_source": (
            "https://resources.fina.org/fina/document/2026/02/18/"
            "e6815ecc-06d9-4f0b-98e9-4c441cf5e6a3/2026-02-18_World-Aquatics_CR-Final.pdf"
        ),
        "dimension_basis": (
            "World Aquatics Feb 2026 15.1.2:25m length;15.6.2.1:25m permanent ten-lane width;"
            "15.4.2:2.5m recommended depth. Freeboard/wall/floor thickness are design assumptions."
        ),
        "texture_source": "https://ambientcg.com/view?id=Tiles107",
        "texture_license": "CC0-1.0",
        "texture_provenance": "tiles107/provenance.json",
        "texture_origin": "ambientCG procedural PBR material, not a photograph or measured pool scan",
        "texture_repeat_m": TILE_REPEAT_M,
        "tile_module_m": 0.05,
        "dark_tile_albedo": "Authored shader scale(-.55,-.55,-.55)+bias(.57,.595,.635), same source maps",
        "collision": "floor, four walls and above-water coping; no water-surface collider",
        "boundary_geometry": "PoolGeometry.wet_boundary_rectangles(), same inner floor/walls as collision asset",
        "marking_source": "World Aquatics February 2026 sections 15.13.1–15.13.4",
        "markings": (
            "0.20m dark tile lanes,21m length,1m T-bars at2m from ends;0.5m crosses15m fromeachend;"
            "full-height end-wall targets with0.5m crosses0.3m belowwater. Materialchoice not mandated."
        ),
        "marking_construction": "Coplanar mesh face subsets sharing tile UVs, normal and roughness; no raised decals",
        "limitations": (
            "No free-surface fluid solver or optical refraction calibration; no swimming certification claim. "
            "No competition equipment/touch panels or dimensional allowance for them."
        ),
        "usd_sha256": hashlib.sha256(POOL_USD_PATH.read_bytes()).hexdigest(),
    }
    (POOL_USD_PATH.parent / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(POOL_USD_PATH)


if __name__ == "__main__":
    main()
