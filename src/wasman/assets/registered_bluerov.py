"""Explicit opt-in to the corrected, versioned BlueROV geometry.

Legacy task IDs/assets/checkpoints retain their historical registration. A new
asset must change motor allocation and proximity geometry together, not visuals
alone. This is not a new hardware calibration or a center-mounted manipulator.
"""

import hashlib
import json
from pathlib import Path

REGISTERED_DIR = Path(__file__).parent / "data/robots/bluerov2_alpha_registered_v1"


def configure_registered_bluerov(cfg):
    manifest_path = REGISTERED_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Prepare registered BlueROV geometry before selecting the new task")
    manifest = json.loads(manifest_path.read_text())
    urdf = REGISTERED_DIR / manifest["urdf"]
    if not manifest["passed"]:
        raise ValueError("Registered geometry has not passed its source audit")
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != manifest["generated_urdf_sha256"]:
        raise ValueError("Registered URDF differs from its audited manifest")
    for item in manifest["files"] + manifest["collision_components"]:
        if "path" not in item:  # Explicitly omitted zero-volume decoration.
            continue
        asset = REGISTERED_DIR / item["path"]
        if hashlib.sha256(asset.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Registered geometry asset changed: {item['path']}")
    # Grasp tasks replace coarse finger boxes with source CAD collisions and
    # repair mixed-unit mimic coupling at spawn. Preserve that task contract.
    from wasman.assets.grasp_geometry import grasp_robot_urdf, spawn_grasp_robot

    if cfg.scene.robot.spawn.func is spawn_grasp_robot:
        cfg.scene.robot.spawn.asset_path = str(grasp_robot_urdf(urdf))
    else:
        from wasman.assets.open_geometry import PROFILE, public_robot_urdf, selected_profile

        cfg.scene.robot.spawn.asset_path = str(public_robot_urdf(urdf) if selected_profile() == PROFILE else urdf)
    cfg.thruster_positions = tuple(tuple(p) for p in manifest["thruster_positions_m"])
    cfg.robot_geometry_version = "registered-v1"
    from wasman.assets.open_geometry import PROFILE, selected_profile

    if selected_profile() == PROFILE:
        cfg.robot_geometry_version = "registered-v1/" + PROFILE
    cfg.robot_camera_profile = "geometry-v2"
    return cfg


def audit_registered_stage(env):
    """Check actual converter output before recording; USD only for local geometry."""
    import numpy as np
    from isaaclab.sim import get_current_stage
    from pxr import Gf, Usd, UsdGeom

    from wasman.controllers.thruster_visuals import DIRECTIONS

    manifest = json.loads((REGISTERED_DIR / "manifest.json").read_text())
    stage = get_current_stage()
    base = stage.GetPrimAtPath("/World/envs/env_0/Robot/Geometry/base_link")
    visual = stage.GetPrimAtPath(str(base.GetPath()) + "/bluerov2_heavy_visual")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], False)
    bounds = cache.ComputeRelativeBound(visual, base).ComputeAlignedRange()
    actual = np.asarray([list(bounds.GetMin()), list(bounds.GetMax())])
    expected = np.asarray(manifest["derived_visuals"][0]["bounds_m"])
    if not np.allclose(actual, expected, rtol=0, atol=2e-5):
        raise RuntimeError(f"Converted body bounds disagree with source: {actual} vs {expected}")
    motors = []
    for i, (position, direction) in enumerate(zip(env.cfg.thruster_positions, DIRECTIONS, strict=True), 1):
        prim = stage.GetPrimAtPath(str(base.GetPath()) + f"/thruster{i}_visual")
        transform = UsdGeom.Xformable(prim).GetLocalTransformation()
        actual_position = np.asarray(transform.ExtractTranslation())
        actual_axis = np.asarray(transform.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized())
        if not np.allclose(actual_position, position, atol=2e-5, rtol=0):
            raise RuntimeError(f"Motor {i} visual and applied force origins differ")
        if not np.allclose(actual_axis, direction, atol=2e-5, rtol=0):
            raise RuntimeError(f"Motor {i} visual and applied force axes differ")
        motors.append({"origin_m": actual_position.tolist(), "axis": actual_axis.tolist()})
    return {
        "passed": True,
        "body_bounds_m": actual.tolist(),
        "motors": motors,
        "manifest_sha256": hashlib.sha256((REGISTERED_DIR / "manifest.json").read_bytes()).hexdigest(),
    }
