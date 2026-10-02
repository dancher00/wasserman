import pytest

from wasman.assets.ship_interior import HOLD_BOUNDS_M, build_ship_interior


def test_hold_is_finite_fully_textured_and_not_training_collision():
    pxr = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom, UsdPhysics, UsdShade

    stage = pxr.Stage.CreateInMemory()
    report = build_ship_interior(stage)
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert len(meshes) > 30
    assert report["collision"] is False
    for prim in meshes:
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        mesh = UsdGeom.Mesh(prim)
        assert len(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()) == 24
        assert UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath()
    assert HOLD_BOUNDS_M[1][0] - HOLD_BOUNDS_M[0][0] < 7
    assert stage.GetPrimAtPath("/World/ShipInterior/OverheadBeam0").IsValid()
    assert stage.GetPrimAtPath("/World/ShipInterior/PortBilge").IsValid()


def test_robot_spotlights_attached_and_switchable():
    pxr = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom, UsdLux

    from wasman.assets.underwater_optics import mount_robot_spotlights, set_scene_lighting

    stage = pxr.Stage.CreateInMemory()
    base = "/World/Robot/base"
    UsdGeom.Xform.Define(stage, base)
    UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    UsdLux.DomeLight.Define(stage, "/World/FillLight")
    lamps = mount_robot_spotlights(stage, base)
    assert len(lamps) == 4
    for path in lamps:
        assert stage.GetPrimAtPath(path).GetParent().GetPath() == base
        assert UsdLux.ShapingAPI(stage.GetPrimAtPath(path)).GetShapingConeAngleAttr().Get() == 42
    set_scene_lighting(stage, lamps, ambient=False, lamps=True)
    assert UsdLux.LightAPI(stage.GetPrimAtPath("/World/KeyLight")).GetIntensityAttr().Get() == 0
    assert UsdLux.LightAPI(stage.GetPrimAtPath(lamps[0])).GetIntensityAttr().Get() == 18000
    set_scene_lighting(stage, lamps, ambient=True, lamps=False)
    assert UsdLux.LightAPI(stage.GetPrimAtPath("/World/KeyLight")).GetIntensityAttr().Get() == 50000
    assert UsdLux.LightAPI(stage.GetPrimAtPath(lamps[0])).GetIntensityAttr().Get() == 0
    with pytest.raises(ValueError, match="Missing robot light"):
        set_scene_lighting(stage, ["/World/Missing"], ambient=False, lamps=True)
