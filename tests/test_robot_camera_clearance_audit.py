import hashlib
import json
import runpy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
audit = runpy.run_path(str(ROOT / "scripts/audit_robot_cameras.py"))


def test_camera_inside_convex_solid_is_not_mistaken_for_surface_clearance():
    cube = trimesh.creation.box(extents=[1, 1, 1])
    result = audit["proximity"](cube, np.array([[0, 0, 0], [0.6, 0, 0], [0.505, 0, 0]]), 0.01)
    assert result["inside_solid_frames"] == [0]
    assert result["surface_envelope_overlap_frames"] == [2]
    assert result["minimum_envelope_clearance_m"] == pytest.approx(-0.005)


def test_nonfinite_arm_telemetry_is_rejected():
    robot = ET.fromstring('<robot><link name="base_link" /></robot>')
    with pytest.raises(ValueError, match="finite four-joint"):
        audit["link_transforms"](robot, [{"arm_joint_position_rad": [0, 0, np.nan, 0]}])


@pytest.mark.parametrize(
    "filename", ["camera_geometry_v2_approach_audit.json", "camera_geometry_v2_registered_take42_audit.json"]
)
def test_recorded_full_approach_has_no_known_camera_geometry_conflicts(filename):
    report = json.loads((ROOT / "artifacts" / filename).read_text())
    assert report["frames"] == 1080
    assert report["hull_collision_components_checked"] == 341
    assert report["profile"]["name"] == "geometry-v2"
    assert report["first_arm_pose_rad"][0] > 3.1  # The folded approach is included.
    for path, key in (
        (ROOT / report["urdf"], "urdf_sha256"),
        (ROOT / report["trace"], "trace_sha256"),
        ((ROOT / report["urdf"]).parent / "manifest.json", "asset_manifest_sha256"),
        (ROOT / "src/wasman/robot_camera_profiles.py", "camera_profile_source_sha256"),
        (ROOT / "scripts/audit_robot_cameras.py", "audit_source_sha256"),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report[key]
    for camera in report["cameras"].values():
        assert camera["hull_collision_conflicts"] == []
        for check in camera["visual_surfaces"].values():
            assert hashlib.sha256((ROOT / check["source"]).read_bytes()).hexdigest() == check["source_sha256"]
            assert check["minimum_envelope_clearance_m"] > 0
            assert check["surface_envelope_overlap_frames"] == []
            assert check["inside_solid_frames"] in (None, [])
    assert report["cameras"]["gripper"]["envelope_radius_m"] == 0.01
