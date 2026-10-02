"""Presentation-only industrial support, sharing the v1 rail contact surface."""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade


def build_support(stage, root):
    material = UsdShade.Material.Define(stage, '/World/Support/WeatheredSteel')
    shader = UsdShade.Shader.Define(stage, '/World/Support/WeatheredSteel/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(.45)
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.78)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    uv = UsdShade.Shader.Define(stage, '/World/Support/WeatheredSteel/UV')
    uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    texture = UsdShade.Shader.Define(stage, '/World/Support/WeatheredSteel/Color')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(root / 'src/wasman/assets/data/panels/ship_green/diffuse.jpg'))
    texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
    texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.65, .70, .64, 1))
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')
    metal = UsdShade.Material.Define(stage, '/World/Support/FastenerSteel')
    surface = UsdShade.Shader.Define(stage, '/World/Support/FastenerSteel/Surface')
    surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.19, .20, .18))
    surface.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(.8)
    surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.48)
    metal.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')

    def tube(name, start, end, radius, segments=32, mat=material):
        a, b = np.array(start, dtype=float), np.array(end, dtype=float)
        direction = (b - a) / np.linalg.norm(b-a)
        first = np.cross(direction, [0, 1, 0])
        if np.linalg.norm(first) < .1:
            first = np.cross(direction, [0, 0, 1])
        first /= np.linalg.norm(first)
        second = np.cross(direction, first)
        ring = [radius * (np.cos(t)*first + np.sin(t)*second) for t in np.linspace(0, 2*np.pi, segments, endpoint=False)]
        points, coords, counts = [], [], []
        for i in range(segments):
            j = (i+1) % segments
            points.extend([a+ring[i], a+ring[j], b+ring[j], b+ring[i]])
            coords.extend([(i/segments, 0), ((i+1)/segments, 0), ((i+1)/segments, np.linalg.norm(b-a)*2), (i/segments, np.linalg.norm(b-a)*2)])
            counts.append(4)
        for center, reverse in [(a, True), (b, False)]:
            indices = list(range(segments))[::-1] if reverse else list(range(segments))
            points.extend(center+ring[i] for i in indices)
            coords.extend((.5+.5*np.cos(i*2*np.pi/segments), .5+.5*np.sin(i*2*np.pi/segments)) for i in indices)
            counts.append(segments)
        mesh = UsdGeom.Mesh.Define(stage, '/World/Support/' + name)
        mesh.CreatePointsAttr([Gf.Vec3f(*x) for x in points])
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'faceVarying').Set([Gf.Vec2f(*x) for x in coords])
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    # The graspable cylinder is exactly the recorded physical rail, r=18 mm.
    tube('Grip', [3, .5, .96], [3, .5, 1.24], .018, mat=metal)
    for i, z in enumerate([.96, 1.24]):
        tube(f'Collar{i}', [3, .5, z-.014], [3, .5, z+.014], .031)
        tube(f'Arm{i}', [3.0, .5, z], [4.015, .5, z], .022)
        tube(f'WallFlange{i}', [4.005, .5, z], [4.035, .5, z], .095)
        tube(f'Weld{i}', [3.97, .5, z], [4.012, .5, z], .033)
        for j, angle in enumerate(np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4):
            y, h = .5+.068*np.cos(angle), z+.068*np.sin(angle)
            tube(f'Bolt{i}_{j}', [3.99, y, h], [4.007, y, h], .011, segments=6, mat=metal)
    tube('DiagonalBrace', [3.16, .5, 1.235], [3.99, .5, .965], .014)
    return {'style': 'weathered tubular steel, reinforcing brace, welded collars and bolted wall flanges',
            'grasp_surface_unchanged': {'center_m': [3, .5, 1.1], 'radius_m': .018, 'height_m': .28},
            'physics_changes': False, 'cosmetic_geometry_only': True}
