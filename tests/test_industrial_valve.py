from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


def test_realistic_valve_has_separate_wheel_open_collisions_and_local_textures():
    root = Path(__file__).parents[1] / "src/wasman/assets/data/objects/industrial_valve"
    stage = Usd.Stage.Open(str(root / "valve.usdc"))
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/Valve/base_to_shaft"))
    assert joint.GetAxisAttr().Get() == "X"
    assert str(joint.GetBody1Rel().GetTargets()[0]) == "/Valve/handle_link"
    wheel = stage.GetPrimAtPath("/Valve/handle_link")
    rim = [p for p in wheel.GetChildren() if p.GetName().startswith("Rim")]
    assert len(rim) == 24
    assert all(p.HasAPI(UsdPhysics.CollisionAPI) for p in rim)
    assert all(UsdGeom.Imageable(p).GetPurposeAttr().Get() == "guide" for p in rim)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Valve/handle_link/Wheel"))
    extent = np.ptp(np.array(mesh.GetPointsAttr().Get()), axis=0)
    assert 0.12 < extent[1] < 0.13 and 0.12 < extent[2] < 0.13
    assert not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    texture_paths = [a.Get().path for p in stage.Traverse() for a in p.GetAttributes() if a.GetName() == "inputs:file"]
    assert len(texture_paths) == 4
    assert all((root / p).is_file() for p in texture_paths)
