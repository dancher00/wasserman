"""CPU checks for a genuinely different, licensed and portable second asset."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ASSET = Path(__file__).resolve().parents[1] / "src/wasman/assets/data/robots/rexrov2_oberon7"


def test_vendor_integrity_and_license():
    metadata = json.loads((ASSET / "provenance.json").read_text())
    assert metadata["license"] == "Apache-2.0"
    assert metadata["scale"] == [1, 1, 1]
    generated_hash = hashlib.sha256((ASSET / "rexrov2_oberon7.urdf").read_bytes()).hexdigest()
    assert generated_hash == metadata["generated_urdf_sha256"]
    for entry in metadata["files"]:
        source = ASSET / entry["path"]
        assert source.stat().st_size == entry["bytes"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"]
        assert any(commit in entry["url"] for commit in metadata["revisions"].values())
    for repo in metadata["revisions"]:
        assert "Apache License" in (ASSET / f"{repo}_LICENSE.txt").read_text()


def test_connected_tree_different_arm_and_gripper():
    robot = ET.parse(ASSET / "rexrov2_oberon7.urdf").getroot()
    links = {link.get("name") for link in robot.findall("link")}
    parents = {joint.find("child").get("link"): joint.find("parent").get("link") for joint in robot.findall("joint")}
    assert links - parents.keys() == {"base_link"}
    for link in links - {"base_link"}:
        seen = set()
        while link != "base_link":
            assert link not in seen
            seen.add(link)
            link = parents[link]
    active = {joint.get("name") for joint in robot.findall("joint") if joint.get("type") != "fixed"}
    assert len(active) == 8  # six-axis arm + two driven jaws, not Alpha five-axis
    assert {"oberon_finger_left_joint", "oberon_finger_right_joint"} <= active
    assert not any("alpha" in name for name in links | active)
    assert float(robot.find("link[@name='base_link']/inertial/mass").get("value")) == 1862.87
    wrist = robot.find("joint[@name='oberon_wrist']")
    assert wrist.get("type") == "revolute"
    assert np.allclose([float(value) for value in wrist.find("origin").get("rpy").split()], [-np.pi / 2, 0, 0])
    assert float(wrist.find("limit").get("effort")) == 10.0


def test_baked_arm_meshes_preserve_source_bounds():
    import trimesh

    metadata = json.loads((ASSET / "provenance.json").read_text())
    assert len(metadata["baked_meshes"]) >= 18
    body = [
        entry
        for entry in metadata["baked_meshes"]
        if entry["path"].endswith(".obj") and entry["source"].endswith("RexROV2_no_props.dae")
    ]
    assert len(body) == 5
    assert sum(len(entry["components"]) for entry in body) == 69
    assert sum(component["triangles"] for entry in body for component in entry["components"]) == 843403
    for entry in metadata["baked_meshes"]:
        path = ASSET / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert entry["vertex_coordinates_verified"] and entry["instances_verified"]
        if path.suffix == ".obj":
            mesh = trimesh.load(path, force="scene", process=False)
            assert np.allclose(mesh.bounds, entry["source_bounds_m"], atol=1e-6)


def test_mesh_resolution_native_scale_and_positive_inertia():
    robot = ET.parse(ASSET / "rexrov2_oberon7.urdf").getroot()
    for mesh in robot.iter("mesh"):
        assert (ASSET / mesh.get("filename")).is_file()
        assert mesh.get("scale", "1 1 1") == "1 1 1"
    for link in robot.findall("link"):
        inertia = link.find("inertial/inertia")
        if inertia is None:
            assert link.get("name").startswith("thruster_")  # fixed visual only
            continue
        ixx, ixy, ixz, iyy, iyz, izz = (float(inertia.get(k)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"))
        assert np.linalg.eigvalsh([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]).min() > 0


def test_native_thruster_orientations_and_coupled_hydro_preserved():
    robot = ET.parse(ASSET / "rexrov2_oberon7.urdf").getroot()
    rotors = [
        visual
        for visual in robot.findall("link[@name='base_link']/visual")
        if visual.get("name", "").startswith("thruster_")
    ]
    assert len(rotors) == 6
    angles = np.array([[float(v) for v in visual.find("origin").get("rpy").split()] for visual in rotors])
    assert np.isclose(angles[0, 2], np.pi / 2)
    assert np.isclose(angles[1, 1], -75 * np.pi / 180)
    source = ET.parse(ASSET / "source/rexrov2.gazebo.xacro").getroot()
    matrix = np.array([float(v) for v in source.find(".//added_mass").text.split()]).reshape(6, 6)
    assert matrix[0, 2] == -103.32  # keep cross-couplings; not a diagonal BlueROV copy


def test_recorded_articulation_ranges_replay_from_actual_joint_samples():
    folder = ASSET.parents[5] / "artifacts/rexrov2_asset_smoke_v3"
    report = json.loads((folder / "report.json").read_text())
    assert report["asset_sha256"] == hashlib.sha256((ASSET / "rexrov2_oberon7.urdf").read_bytes()).hexdigest()
    positions = np.load(folder / "joint_positions.npy")
    assert positions.shape == (report["frames"], len(report["joint_names"]))
    assert np.isfinite(positions).all()
    assert np.allclose(np.ptp(positions, axis=0), report["joint_range_observed_rad"])
    robot = ET.parse(ASSET / "rexrov2_oberon7.urdf").getroot()
    for index, name in enumerate(report["joint_names"]):
        joint = robot.find(f"joint[@name='{name}']")
        bounds = joint.find("limit")
        if bounds is None:
            assert joint.get("type") == "continuous"
            continue
        assert positions[:, index].min() >= float(bounds.get("lower")) - 1e-6
        assert positions[:, index].max() <= float(bounds.get("upper")) + 1e-6


def test_actual_isaac_conversion_covers_all_source_instances_and_triangles():
    folder = ASSET.parents[5] / "artifacts/rexrov2_asset_smoke_v3"
    audit = json.loads((folder / "isaac_geometry_audit.json").read_text())
    assert audit["passed"]
    assert audit["urdf_sha256"] == hashlib.sha256((ASSET / "rexrov2_oberon7.urdf").read_bytes()).hexdigest()
    assert audit["visual_mesh_instances"] == len(audit["components"]) == 133
    assert audit["triangles"] == sum(component["triangles"] for component in audit["components"]) == 909493
    assert audit["max_vertex_error_m"] < 1e-6
    assert all(component["max_vertex_error_m"] < 1e-6 for component in audit["components"])
