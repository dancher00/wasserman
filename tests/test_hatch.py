"""Asset/configuration regressions; physical sensor and hinge checks live in check_hatch.py."""

import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from wasman.tasks.underwater_hatch.env_cfg import HatchCamerasEnvCfg, HatchEnvCfg
from wasman.tasks.underwater_press_button.config.bluerov2_alpha.env_cfg import UnderwaterPressButtonEnvCfg

pytestmark = pytest.mark.unit
DATA = Path(__file__).resolve().parents[1] / "src/wasman/assets/data"


def test_hatch_registration_and_isolation():
    for suffix in ("", "-Cameras"):
        assert gym.spec(f"Wasman-Underwater-OpenHatch{suffix}-Direct").entry_point.endswith(":UnderwaterHatchEnv")
    hatch, cameras, button = HatchEnvCfg(), HatchCamerasEnvCfg(), UnderwaterPressButtonEnvCfg()
    assert hatch.scene.panel is None and button.scene.panel is not None
    assert hatch.scene.button.init_state.rot == (0, 0, 0, 1)
    assert hatch.scene.button.init_state.pos == (0.75, 0, 0)
    assert hatch.button_z_range == (0, 0)
    assert hatch.mechanism_joint_name == "Hinge"
    assert hatch.require_finger_contact
    assert hatch.scene.button.actuators["passive"].stiffness == 0
    assert hatch.button_pressed_threshold == math.radians(80)
    assert hatch.mechanism_success_max_speed == 0.05
    assert hatch.success_hold_steps * hatch.decimation * hatch.sim.dt == 1.0
    assert (hatch.action_space, hatch.observation_space) == (11, 38)
    assert (button.action_space, button.observation_space) == (10, 37)
    assert button.scene.seabed.init_state.pos == (0, 0, 0)
    assert button.scene.seabed.collision_group == -1
    assert cameras.scene.num_envs == 8
    assert not hasattr(button.scene, "base_camera") and not hasattr(hatch.scene, "base_camera")
    assert cameras.scene.base_camera.prim_path.endswith("base_link/BaseCamera")
    assert cameras.scene.gripper_camera.prim_path.endswith("alpha_jaw_base_link/GripperCamera")
    for cam in (cameras.scene.base_camera, cameras.scene.gripper_camera):
        assert (cam.width, cam.height) == (384, 384)
        assert cam.data_types == ["rgb"] and cam.update_latest_camera_pose
        assert math.isclose(sum(x * x for x in cam.offset.rot), 1.0)


def test_hatch_is_passive_and_has_an_open_collision_rim():
    stage = Usd.Stage.Open(str(DATA / "objects/submarine_hatch/hatch.usda"))
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/Hatch/Hinge"))
    assert joint.GetAxisAttr().Get() == "Y"
    assert joint.GetLowerLimitAttr().Get() == 0 and joint.GetUpperLimitAttr().Get() == 105
    assert UsdPhysics.DriveAPI(joint.GetPrim(), "angular").GetStiffnessAttr().Get() == 0
    assert stage.GetPrimAtPath("/Hatch/base_link").HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not stage.GetPrimAtPath("/Hatch/base_link/Coaming").HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetPrimAtPath("/Hatch/lid/GripBar").HasAPI(UsdPhysics.CollisionAPI)
    assert len([p for p in stage.Traverse() if p.GetName().startswith("RimCollision")]) == 32
    assert len(UsdGeom.Mesh(stage.GetPrimAtPath("/Hatch/lid/Lid")).GetPointsAttr().Get()) > 500


def test_sand_sources_and_real_aperture():
    provenance = json.loads((DATA / "seabed/provenance.json").read_text())
    assert provenance["license"] == "CC0-1.0" and provenance["texture_width_m"] == 2
    for name, meta in provenance["maps"].items():
        assert hashlib.sha256((DATA / f"seabed/{name}.jpg").read_bytes()).hexdigest() == meta["sha256"]
    stage = Usd.Stage.Open(str(DATA / "objects/submarine_hatch/sand_cutout.usda"))
    points = UsdGeom.Mesh(stage.GetPrimAtPath("/Seabed/Sand")).GetPointsAttr().Get()
    assert min(math.hypot(p[0], p[1]) for p in points) > 0.262
