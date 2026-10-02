"""Independent CPU regression checks; no Isaac or restricted CAD redistribution."""

import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wasman.bluerov2_registered import (
    ASSET_DIR,
    NATIVE_COM,
    NATIVE_DIRECTIONS,
    NATIVE_POSITIONS,
    NATIVE_RPY,
    SOURCE_DIR,
    SOURCE_SHA256,
    URDF_PATH,
    allocation_matrix,
)

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("grasp", [False, True])
def test_registered_override_preserves_task_finger_contact_geometry(grasp):
    from types import SimpleNamespace

    from wasman.assets.grasp_geometry import spawn_grasp_robot
    from wasman.assets.registered_bluerov import configure_registered_bluerov

    spawn = SimpleNamespace(func=spawn_grasp_robot if grasp else None)
    cfg = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(spawn=spawn)))
    configure_registered_bluerov(cfg)
    robot = ET.parse(spawn.asset_path)
    for side in ("left", "right"):
        collision = robot.find(f"link[@name='alpha_{side}_finger_link']/collision")
        assert (collision.find("geometry/mesh") is not None) == grasp
        assert (collision.find("geometry/box") is not None) != grasp
    assert spawn.func is (spawn_grasp_robot if grasp else None)
    assert cfg.robot_camera_profile == "geometry-v2"


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def manifest():
    return json.loads((ASSET_DIR / "manifest.json").read_text())


def test_source_unchanged_and_output_hashes(manifest):
    assert (
        manifest["source_urdf_sha256"] == hashlib.sha256((SOURCE_DIR / "bluerov2_alpha.urdf").read_bytes()).hexdigest()
    )
    assert manifest["generated_urdf_sha256"] == hashlib.sha256(URDF_PATH.read_bytes()).hexdigest()
    for name, expected in SOURCE_SHA256.items():
        assert hashlib.sha256((SOURCE_DIR / "meshes/blue" / name).read_bytes()).hexdigest() == expected
    for item in [*manifest["files"], *manifest["collision_components"]]:
        if "path" in item:
            assert hashlib.sha256((ASSET_DIR / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_full_scene_transform_scale_translation_and_mirror(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("tinyobjloader")
    generator = module("prepare_bluerov2_registered")
    mesh = trimesh.creation.box(extents=[2, 4, 6])
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3, 3] = [4, 5, 6]
    transform[0, 0] = -1
    scene.add_geometry(mesh, geom_name="shared", node_name="mirrored", transform=transform)
    second = np.eye(4)
    second[:3, 3] = [8, 9, 10]
    scene.graph.update(frame_to="second", geometry="shared", matrix=second)
    meshes, records = generator.transformed_components(scene, np.diag([0.025, 0.025, 0.025, 1]))
    assert len(meshes) == 2 and sum(r["mirrored"] for r in records) == 1
    for output, record in zip(meshes, records, strict=True):
        matrix = np.array(record["matrix_column_vector"])
        assert np.allclose(output.vertices, mesh.vertices @ matrix[:3, :3].T + matrix[:3, 3])
        assert output.is_winding_consistent and output.volume > 0
    generator.export_verified(meshes, tmp_path / "scene.obj")


def test_source_bounds_component_coverage_and_zero_origin(manifest):
    body = next(item for item in manifest["derived_visuals"] if item["source"].startswith("bluerov2"))
    assert len(body["components"]) == 51
    assert sum(c["triangles"] for c in body["components"]) == 147295
    assert np.allclose(
        body["bounds_m"],
        [[-0.2339181873, -0.2829165080, -0.1121613247], [0.2162823477, 0.2829155109, 0.1407295975]],
        atol=1e-8,
    )
    base = ET.parse(URDF_PATH).find("./link[@name='base_link']")
    visual = base.find("visual[@name='bluerov2_heavy_visual']")
    assert visual.find("origin").get("xyz") == "0 0 0"
    assert visual.find("geometry/mesh").get("scale") == "1 1 1"
    assert len(base.findall("collision")) > 300
    for item in manifest["collision_components"]:
        if "path" in item:
            assert np.allclose(item["source_bounds_m"], item["hull_bounds_m"], atol=1e-8)
            assert item["max_vertex_outside_distance_m"] <= 1e-8
        else:
            assert "zero-volume" in item["omitted_reason"]


def strip(element):
    for node in element.iter():
        node.text = node.tail = None
    return ET.tostring(element)


def test_native_joints_arm_inertias_and_restricted_assets_unchanged(manifest):
    old, new = ET.parse(SOURCE_DIR / "bluerov2_alpha.urdf"), ET.parse(URDF_PATH)
    for a, b in zip(old.findall("joint"), new.findall("joint"), strict=True):
        assert strip(a) == strip(b)
    for a, b in zip(old.findall("link"), new.findall("link"), strict=True):
        assert strip(a.find("inertial")) == strip(b.find("inertial"))
        if a.get("name") != "base_link":
            for oldmesh, newmesh in zip(a.findall(".//mesh"), b.findall(".//mesh"), strict=True):
                assert (SOURCE_DIR / oldmesh.get("filename")).resolve() == (
                    ASSET_DIR / newmesh.get("filename")
                ).resolve()
                newmesh.set("filename", oldmesh.get("filename"))
            assert strip(a) == strip(b)
    assert not manifest["restricted_arm_meshes_copied"]
    assert not (ASSET_DIR / "meshes/alpha").exists()


def test_native_rotor_axes_and_force_moments():
    base = ET.parse(URDF_PATH).find("./link[@name='base_link']")
    for index, (position, angles, direction) in enumerate(
        zip(NATIVE_POSITIONS, NATIVE_RPY, NATIVE_DIRECTIONS, strict=True), 1
    ):
        origin = base.find(f"visual[@name='thruster{index}_visual']/origin")
        assert np.allclose(np.fromstring(origin.get("xyz"), sep=" "), position)
        assert np.allclose(np.fromstring(origin.get("rpy"), sep=" "), angles)
        assert np.allclose(Rotation.from_euler("xyz", angles).apply([0, 0, -1]), direction)
    a = allocation_matrix()
    assert a.shape == (6, 8) and np.linalg.matrix_rank(a) == 6
    assert np.allclose(a[3:].T, np.cross(np.array(NATIVE_POSITIONS) - NATIVE_COM, NATIVE_DIRECTIONS))


def test_sampled_button_deployment_clearance_report():
    report = json.loads((ASSET_DIR / "cpu-clearance.json").read_text())
    assert report["passed"] and not report["collisions"]
    assert report["samples"] >= 133 and report["min_clearance_m"] > 0
    assert report["urdf_sha256"] == hashlib.sha256(URDF_PATH.read_bytes()).hexdigest()
    assert report["excluded_arm_links"] == ["alpha_m3_inline_link"]


def test_independent_live_collision_recheck():
    pytest.importorskip("pinocchio")
    report = module("audit_bluerov2_registered").audit(samples=3)
    assert report["passed"] and not report["collisions"]
