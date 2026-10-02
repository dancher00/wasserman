"""CPU geometry/material checks for integral ceramic pool markings."""

import pytest

from wasman.assets.pool_geometry import POOL_USD_PATH
from wasman.assets.pool_markings import (
    LANE_CENTRES,
    LINE_WIDTH_M,
    floor_tile_regions,
    partition_rectangles,
    wall_tile_regions,
)


def test_marking_dimensions_and_positions_follow_swimming_rules():
    assert len(LANE_CENTRES) == 10
    assert 0.2 <= LINE_WIDTH_M <= 0.3
    regions = dict(floor_tile_regions())
    assert len(regions) == 50
    for lane, y in enumerate(LANE_CENTRES):
        a, b, c, d = regions[f"Lane{lane}"]
        assert b - a == 21
        assert d - c == pytest.approx(0.2)
        assert (c + d) / 2 == pytest.approx(y)
        for x in (-10.5, 10.5):
            a, b, c, d = regions[f"Lane{lane}T{x:+g}"]
            assert 12.5 - abs((a + b) / 2) == 2
            assert d - c == 1
        for x in (-2.5, 2.5):
            a, b, c, d = regions[f"Lane{lane}15m{x:+g}"]
            assert 12.5 + abs((a + b) / 2) == 15
            assert d - c == 0.5
    targets = dict(wall_tile_regions(end_wall=True))
    assert len(targets) == 21
    for lane in range(10):
        a, b, c, d = targets[f"Target{lane}Cross"]
        assert b - a == 0.5
        assert (c + d) / 2 == pytest.approx(2.5 - 0.3)
    assert len(wall_tile_regions(end_wall=False)) == 1


def test_surface_partition_exactly_covers_floor_and_uses_complete_mosaic_modules():
    cells = partition_rectangles((-12.5, 12.5, -12.5, 12.5), floor_tile_regions())
    assert sum((b - a) * (d - c) for (a, b, c, d), _ in cells) == pytest.approx(625)
    assert any(dark for _, dark in cells) and any(not dark for _, dark in cells)
    for bounds, _ in cells:
        for edge in bounds:
            assert edge / 0.05 == pytest.approx(round(edge / 0.05))


def test_native_materials_share_maps_uv_phase_and_integral_faces():
    usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom, UsdShade

    stage = usd.Stage.Open(str(POOL_USD_PATH))
    for channel in ("diffuse", "normal", "roughness"):
        paths = [stage.GetPrimAtPath(f"/Pool/{material}/{channel}") for material in ("Tile", "LaneMaterial")]
        assert paths[0].GetAttribute("inputs:file").Get() == paths[1].GetAttribute("inputs:file").Get()
        assert "tiles107" in str(paths[0].GetAttribute("inputs:file").Get())
    for name in ("Floor", "WestWall", "EastWall", "SouthWall", "NorthWall"):
        prim = stage.GetPrimAtPath(f"/Pool/{name}")
        subset = stage.GetPrimAtPath(f"/Pool/{name}/DarkTileRegions")
        assert subset.IsValid()
        assert UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()[0].GetPath() == "/Pool/LaneMaterial"
        mesh = UsdGeom.Mesh(prim)
        coords = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get()
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        translation = prim.GetAttribute("xformOp:translate").Get()
        # Floor's entire top, including dark regions, is at z=0 and uses the
        # same UV formula. Duplicate vertices at material edges match exactly.
        if name == "Floor":
            for offset, vertex_index in enumerate(indices):
                point = tuple(points[vertex_index][axis] + translation[axis] for axis in range(3))
                if abs(point[2]) < 1e-7:
                    u, v = coords[offset]
                    # Side faces also have top-edge vertices; only top faces
                    # have all four corners on the floor plane.
                    face_start = offset // 4 * 4
                    if all(abs(points[i][2] + translation[2]) < 1e-7 for i in indices[face_start : face_start + 4]):
                        assert (u, v) == pytest.approx((point[0] / 0.4, point[1] / 0.4), abs=3e-6)
