"""CPU checks against the real Alpha STL surfaces, not bounding-box stand-ins."""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from wasman.robot_camera_profiles import (
    GEOMETRY_V2,
    LEGACY_V1,
    WRIST_AUDIT_ENVELOPE_RADIUS_M,
    WRIST_V2_TARGET,
    aim_in_xz_plane,
    robot_camera_profile,
)

ASSETS = Path(__file__).resolve().parents[1] / "src/wasman/assets/data/robots/bluerov2_alpha/meshes/alpha"


@lru_cache
def wrist_meshes(angle):
    """Apply the actual URDF visual/joint transforms into jaw_base_link."""
    result = {}
    for name, file, offset, rotation in (
        ("palm", "end_effectors/RS1-124.stl", [0, 0, 0], Rotation.identity()),
        ("wrist", "RS1-100-101-123.stl", [0, 0, -0.09975], Rotation.from_euler("y", -1.5707)),
        ("left", "end_effectors/RS1-130.stl", [0, 0.0155, 0.0069], Rotation.from_euler("x", -angle)),
        ("right", "end_effectors/RS1-139.stl", [0, -0.0155, 0.0069], Rotation.from_euler("x", angle)),
    ):
        mesh = trimesh.load(ASSETS / file, force="mesh", process=False)
        mesh.vertices = rotation.apply(mesh.vertices) + offset
        result[name] = mesh
    return result


def normalized_projection(profile, points):
    """Independent pinhole projection: camera-world +X forward, +Z up."""
    local = Rotation.from_quat(profile.gripper.quaternion_xyzw).inv().apply(
        np.asarray(points) - profile.gripper.position_m
    )
    depth = local[:, 0]
    focal = profile.focal_length_mm / profile.horizontal_aperture_mm
    uv = 0.5 + focal * np.stack((-local[:, 1] / depth, -local[:, 2] / depth), axis=1)
    inside = ((uv > 0) & (uv < 1)).all(axis=1)
    inside &= (depth > profile.clipping_range_m[0]) & (depth < profile.clipping_range_m[1])
    return uv, depth, inside


def test_legacy_profile_is_exactly_reproducible():
    assert robot_camera_profile() is LEGACY_V1
    assert LEGACY_V1.base.position_m == (0.32, 0.0, 0.10)
    assert LEGACY_V1.base.quaternion_xyzw == (0.0, math.sin(math.pi / 12), 0.0, math.cos(math.pi / 12))
    assert LEGACY_V1.gripper.position_m == (0.042, 0.0, 0.035)
    assert LEGACY_V1.gripper.quaternion_xyzw == (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))
    assert LEGACY_V1.clipping_range_m == (0.015, 30.0)
    assert LEGACY_V1.orientation_convention == "world"
    uv, _, _ = normalized_projection(LEGACY_V1, [WRIST_V2_TARGET])
    assert uv[0, 1] * 256 == pytest.approx(4.856120257695, abs=1e-9)
    # This is vertex fraction inside the frustum, NOT visible image-area coverage.
    for name in ("left", "right"):
        assert normalized_projection(LEGACY_V1, wrist_meshes(0.0)[name].vertices)[2].mean() < 0.031


def test_new_profile_is_opt_in_and_not_hardware_calibration():
    assert robot_camera_profile("geometry-v2") is GEOMETRY_V2
    assert GEOMETRY_V2.hardware_calibrated is False
    assert GEOMETRY_V2.metadata()["name"] == "geometry-v2"
    assert "812fea605cd2056ba20fac504f456457594595fb" in GEOMETRY_V2.source_urls[0]
    assert GEOMETRY_V2.base.position_m == (0.21, 0.0, 0.067)
    assert GEOMETRY_V2.gripper.parent_link == "alpha_jaw_base_link"
    assert GEOMETRY_V2.base.parent_link == "base_link"
    assert np.allclose(
        Rotation.from_quat(GEOMETRY_V2.base.quaternion_xyzw).apply([1, 0, 0]), [math.sqrt(3) / 2, 0, -0.5]
    )
    with pytest.raises(ValueError, match="Unknown robot camera profile"):
        robot_camera_profile("typo")


@pytest.mark.parametrize("angle", [0.0, 0.125, 0.25, 0.375, 0.5])
def test_full_source_fingers_fit_with_margins_through_gripper_stroke(angle):
    for name in ("left", "right"):
        uv, depth, inside = normalized_projection(GEOMETRY_V2, wrist_meshes(angle)[name].vertices)
        assert inside.all()
        assert uv.min() > 0.10
        assert uv.max() < 0.80
        assert depth.min() > 0.05
    uv, _, inside = normalized_projection(GEOMETRY_V2, [WRIST_V2_TARGET])
    assert inside.all() and np.allclose(uv, [[0.5, 0.5]], atol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 0.25, 0.5])
def test_lens_and_assumed_small_envelope_clear_actual_nearby_surfaces(angle):
    position = np.array(GEOMETRY_V2.gripper.position_m)
    for mesh in wrist_meshes(angle).values():
        # Outside the mesh AABB proves this lens is not inside a closed surface.
        assert position[0] - WRIST_AUDIT_ENVELOPE_RADIUS_M > mesh.bounds[1, 0]
        _, distance, _ = trimesh.proximity.closest_point_naive(mesh, position[None])
        assert distance[0] - WRIST_AUDIT_ENVELOPE_RADIUS_M > 0.025


@pytest.mark.parametrize("angle", [0.0, 0.25, 0.5])
def test_useful_workspace_ahead_is_in_frame_and_not_hidden_by_palm(angle):
    targets = np.array([[x, y, z] for x in (-0.01, 0, 0.01) for y in (-0.01, 0, 0.01) for z in (0.12, 0.18, 0.25)])
    _, _, inside = normalized_projection(GEOMETRY_V2, targets)
    assert inside.all()
    position = np.array(GEOMETRY_V2.gripper.position_m)
    origins = np.broadcast_to(position, targets.shape)
    for name in ("palm", "wrist"):
        hits, indices, _ = wrist_meshes(angle)[name].ray.intersects_location(origins, targets - position)
        hits = hits.reshape(-1, 3)
        assert not np.any(np.linalg.norm(hits - position, axis=1) < np.linalg.norm(targets[indices] - position, axis=1))
    all_meshes = trimesh.util.concatenate(list(wrist_meshes(angle).values()))
    hits, indices, _ = all_meshes.ray.intersects_location(origins, targets - position)
    hits = hits.reshape(-1, 3)
    before_target = np.linalg.norm(hits - position, axis=1) < np.linalg.norm(targets[indices] - position, axis=1) - 1e-6
    blocked = np.unique(indices[before_target])
    # A closed finger legitimately hides one edge sample, not the whole worksite.
    assert len(blocked) <= 1
    if angle == 0.5:
        assert len(blocked) == 0


@pytest.mark.parametrize("target", [(0, 0, 0), (1, 1, 0), (math.nan, 0, 1)])
def test_bad_aims_fail_before_creating_a_camera(target):
    with pytest.raises(ValueError):
        aim_in_xz_plane((0, 0, 0), target)
