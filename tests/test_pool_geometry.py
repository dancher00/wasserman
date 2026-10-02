from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wasman.assets.pool_geometry import POOL_GEOMETRY, POOL_USD_PATH, PoolGeometry
from wasman.physics.boundary_effects import BlueROVBoundaryEffect, BoundaryEffectCfg, validate_boundary_scene


def test_standard_dimensions_and_finite_wet_boundaries():
    assert POOL_GEOMETRY.length_m == POOL_GEOMETRY.width_m == 25
    assert POOL_GEOMETRY.water_depth_m == 2.5
    assert POOL_GEOMETRY.wall_height_m == 2.7
    rectangles = POOL_GEOMETRY.wet_boundary_rectangles()
    assert len(rectangles) == 5
    for _, normal, u, v, half_size, _ in rectangles:
        n, a, b = (torch.tensor(vector, dtype=torch.float32) for vector in (normal, u, v))
        assert torch.allclose(torch.linalg.cross(n, a), b)
        assert n.norm() == a.norm() == b.norm() == 1
        assert all(0 < h <= 12.5 for h in half_size)


def test_native_usd_collision_and_waterline_match_geometry():
    usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom, UsdPhysics

    stage = usd.Stage.Open(str(POOL_USD_PATH))
    collisions = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(collisions) == 9  # Floor + four walls + four above-water coping strips.
    for name, center, size in POOL_GEOMETRY.collision_boxes():
        prim = stage.GetPrimAtPath(f"/Pool/{name}")
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "boundingCube"
        assert prim.GetAttribute("xformOp:translate").Get() == pytest.approx(center)
        points = torch.tensor(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        assert tuple((points.amax(0) - points.amin(0)).tolist()) == pytest.approx(size)
        assert len(UsdGeom.PrimvarsAPI(prim).GetPrimvar("st").Get()) == len(
            UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get()
        )
    water = stage.GetPrimAtPath("/Pool/WaterSurface")
    assert not water.HasAPI(UsdPhysics.CollisionAPI)
    assert {round(p[2], 5) for p in UsdGeom.Mesh(water).GetPointsAttr().Get()} == {2.5}
    # Lane/T/wall markings are integral face materials, never raised decal boxes.
    assert not stage.GetPrimAtPath("/Pool/Lane9").IsValid()
    assert stage.GetPrimAtPath("/Pool/Floor/DarkTileRegions").IsValid()


def test_boundary_replaces_sand_plane_with_pool_finite_floor_and_walls():
    config = BoundaryEffectCfg(seabed_loss=0.2, wall_loss=0.2)
    model = BlueROVBoundaryEffect(cfg=config, pool_geometry=POOL_GEOMETRY)
    pose = torch.tensor([[0.0, 0.0, 0.1, 0, 0, 0, 1]])
    force = torch.full((1, 8), -10.0)
    _, gain, losses = model.apply(pose, force)
    assert losses.shape == (1, 8, 5)
    assert (gain[:, 4:] < 1).all()
    pose[:, 0] = 14  # Outside actual pool; no imaginary infinite-floor interaction.
    assert (model.apply(pose, force)[1] == 1).all()
    pose[:, :3] = torch.tensor([12.0, 0.0, 1.0])
    force[:] = 10
    assert (model.apply(pose, force)[1][:, :2] < 1).any()
    pose[:, 2] = 3.4  # No wall above its wet extent, no invisible solid top.
    assert (model.apply(pose, force)[1] == 1).all()


def test_pool_anchor_and_environment_batch_translation_covariance():
    origins = torch.tensor([[0.0, 0.0, 0.0], [40.0, -40.0, 0.0]])
    model = BlueROVBoundaryEffect(pool_geometry=POOL_GEOMETRY, pool_center=(-10.5, 9.5, 0), env_origins=origins)
    pose = torch.tensor([[1.5, 0, 0.4, 0, 0, 0, 1.0], [41.5, -40.0, 0.4, 0, 0, 0, 1.0]])
    force = torch.tensor([[10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]]).expand(2, -1)
    _, gain, _ = model.apply(pose, force)
    assert torch.allclose(gain[0], gain[1], atol=1e-6)
    assert (gain < 1).any()


def test_panel_and_pool_are_nearest_only_not_summed():
    model = BlueROVBoundaryEffect(pool_geometry=POOL_GEOMETRY, pool_center=(-12.0, 0, 0))
    pose = torch.tensor([[0, 0, 1.0, 0, 0, 0, 1.0]])
    panel = torch.tensor([[0.5, 0, 1.0, 0, 0, 0, 1.0]])
    _, gain, losses = model.apply(pose, torch.full((1, 8), 10.0), panel)
    assert losses.shape == (1, 8, 11)
    assert (gain >= 0.8).all()
    # A single selected-surface loss, never addition of pool and backing-panel losses.
    assert ((1 - gain)[..., None] - losses).abs().amin(-1).max() < 1e-6


def test_pool_validator_supports_native_anchor_but_rejects_rotation_scale_and_hatch():
    def asset(path, pos=(0, 0, 0)):
        return SimpleNamespace(
            spawn=SimpleNamespace(usd_path=str(path), scale=None),
            init_state=SimpleNamespace(pos=pos, rot=(0, 0, 0, 1)),
        )

    root = Path(__file__).parents[1] / "src/wasman/assets/data"
    pool = asset(POOL_USD_PATH, (-10.5, 9.5, 0))
    cfg = SimpleNamespace(
        use_physical_thrusters=True,
        scene=SimpleNamespace(panel=asset(root / "panels/ship_green/panel.usda"), seabed=pool),
    )
    assert validate_boundary_scene(cfg) == POOL_GEOMETRY
    pool.init_state.rot = (0, 0, 1, 0)
    with pytest.raises(ValueError, match="rotated"):
        validate_boundary_scene(cfg)
    pool.init_state.rot = (0, 0, 0, 1)
    pool.spawn.scale = (2, 1, 1)
    with pytest.raises(ValueError, match="scaled"):
        validate_boundary_scene(cfg)
    pool.spawn.scale = None
    cfg.scene.sand_apron = object()
    with pytest.raises(ValueError, match="Hatch"):
        validate_boundary_scene(cfg)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_bad_pool_dimensions_rejected(bad):
    with pytest.raises(ValueError):
        PoolGeometry(water_depth_m=bad)
