"""CPU regression checks for the separate custom centerline Rex variant."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from wasman.controllers.rexrov2_workspace import ASSET, URDF, RexWorkspace
from wasman.physics.rexrov2 import load_parameters
from wasman.physics.rexrov2_centered import FOLDED_Q, load_centered_parameters

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def workspace():
    return RexWorkspace()


def test_only_mount_and_collision_geometry_change():
    source = ASSET.with_name("rexrov2_oberon7") / "rexrov2_oberon7.urdf"
    manifest = json.loads((ASSET / "provenance.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_urdf_sha256"]
    assert hashlib.sha256(URDF.read_bytes()).hexdigest() == manifest["generated_urdf_sha256"]
    native, centered = ET.parse(source).getroot(), ET.parse(URDF).getroot()
    for root in (native, centered):
        for element in root.iter():
            element.text = element.tail = None
    for a, b in zip(native.findall("link"), centered.findall("link"), strict=True):
        assert ET.tostring(a.find("inertial")) == ET.tostring(b.find("inertial"))
        for x, y in zip(a.findall("visual"), b.findall("visual"), strict=True):
            mesh = y.find("geometry/mesh")
            mesh.set("filename", mesh.get("filename").removeprefix("../rexrov2_oberon7/"))
            assert ET.tostring(x) == ET.tostring(y)
    for a, b in zip(native.findall("joint"), centered.findall("joint"), strict=True):
        if a.get("name") == "oberon_mount":
            assert b.find("origin").get("xyz") == "1.3 0 -0.665"
            b.find("origin").set("xyz", a.find("origin").get("xyz"))
        assert ET.tostring(a) == ET.tostring(b)


def test_component_hulls_preserve_bounds_and_hashes():
    manifest = json.loads((ASSET / "provenance.json").read_text())
    solids = [item for item in manifest["collision_components"] if "path" in item]
    assert len(solids) == 126
    for item in solids:
        assert hashlib.sha256((ASSET / item["path"]).read_bytes()).hexdigest() == item["sha256"]
        assert np.allclose(item["source_bounds_m"], item["hull_bounds_m"], atol=1e-7)


def test_compact_fold_is_native_limit_clear_and_inside_vertical_profile(workspace):
    assert ((workspace.low <= FOLDED_Q) & (workspace.high >= FOLDED_Q)).all()
    assert not workspace.collisions(FOLDED_Q)
    bounds = workspace.bounds(FOLDED_Q)
    assert np.isfinite(bounds).all()
    assert bounds[0, 2] > -0.7954
    assert np.max(np.abs(bounds[:, 1])) < 0.764
    assert bounds[1, 0] < 1.65  # Still protrudes ~0.30 m beyond front platform.
    old = workspace.bounds(np.array([0, 1.25, -1.35, 0, 0, 0, 0.25, 0.25]))
    assert bounds[0, 2] - old[0, 2] > 0.29
    assert workspace.clearance(FOLDED_Q)[0] > 0.0037


def test_collision_checker_rejects_rearward_self_intersection(workspace):
    assert workspace.collisions(np.array([0, 1.4, -1.55, 0, -1.5, 0, 0.25, 0.25]))


def test_remount_and_fold_recompute_composite_but_preserve_source_added_mass(workspace):
    native = load_parameters()
    centered = load_centered_parameters(workspace=workspace)
    assert not np.allclose(centered.rigid_composite, native.rigid_composite)
    assert np.allclose(centered.rigid_composite, centered.rigid_composite.T)
    assert np.linalg.eigvalsh(centered.rigid_composite).min() > 0
    assert np.array_equal(centered.added_mass, native.added_mass)
    assert np.array_equal(centered.mass, native.mass)
    assert np.allclose(centered.rigid_composite[:3, :3], np.eye(3) * native.mass.sum())


def test_cpu_report_pose_replay(workspace):
    report = json.loads((ROOT / "artifacts/rexrov2_centered_cpu_v2/report.json").read_text())
    assert report["passed"] and report["tested_nonadjacent_geometry_pairs"] == 4416
    assert np.array_equal(report["folded_q"], FOLDED_Q)
    for entry in report["targets"].values():
        position, rotation = workspace.fk(entry["q"])
        assert np.allclose(position, entry["target_position_base_m"], atol=1e-6)
        assert np.allclose(rotation, entry["target_rotation_base"], atol=1e-6)
        assert not workspace.collisions(entry["q"])
    for path in report["paths"]:
        assert not path["collisions"]
        assert np.all(np.array(path["smoothstep_peak_joint_velocity_rad_s"]) <= workspace.model.velocityLimit)


def test_measured_reach_evidence():
    folder = ROOT / "artifacts/rexrov2_centered_reach_v4"
    report = json.loads((folder / "report.json").read_text())
    trace = json.loads((folder / "trace.json").read_text())
    assert report["passed"] and report["self_collisions_enabled"]
    assert report["actual_collider_count"] == 126
    assert np.allclose(report["actual_collider_contact_offsets_m"], 0.001)
    assert report["initial_joint_error_rad"] == 0
    assert report["runtime_pose_or_velocity_overwrites"] == 0
    assert len(trace) == report["frames"] == 900
    assert trace[0]["t_s"] == report["telemetry_offset_s"] == 1 / 30
    assert trace[-1]["t_s"] == 30
    assert not report["actual_CAD_hull_collisions"]
    assert report["max_filtered_self_contact_force_n"] < 0.1
    assert max(report["endpoint_tracking_error_m"].values()) < 0.015
    assert np.isfinite(np.array([row["q"] for row in trace])).all()
    assert np.allclose(trace[0]["q"], FOLDED_Q, atol=1e-5)
    assert hashlib.sha256((folder / "run_script.py").read_bytes()).hexdigest() == report["run_script_sha256"]


def test_centered_floating_evidence(workspace):
    folder = ROOT / "artifacts/rexrov2_centered_stationkeeping_v1"
    report = json.loads((folder / "report.json").read_text())
    replay = json.loads((folder / "replay_audit.json").read_text())
    assert report["passed"] and replay["passed"]
    assert report["variant"] == "WASMAN centered compact-held-arm"
    assert not report["fixed_base"] and report["self_collisions_enabled"]
    assert report["imported_mass_and_com_audit_passed"]
    assert report["finite_states_and_wrenches_checked_every_physics_step"]
    assert replay["checks"]["recorded_centered_joint_poses_clear"]
    assert not replay["sampled_CAD_self_collisions"]
    assert hashlib.sha256((folder / "trace.npz").read_bytes()).hexdigest() == replay["trace_sha256"]
    assert hashlib.sha256((folder / "run_script.py").read_bytes()).hexdigest() == report["script_sha256"]
    actual_matrix = load_centered_parameters(workspace=workspace).rigid_composite
    assert np.allclose(report["nominal_held_arm_rigid_inertia"], actual_matrix)
