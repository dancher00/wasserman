import json

import numpy as np
import pytest

from wasman.effect_annotations import build_annotations, project_hemisphere, project_point, project_trajectory


def record():
    return {
        "base_position_w_m": [0, 0, 0.6],
        "target_position_w_m": [0, 0, 0.65],
        "current_w_m_s": [0.22, 0, 0],
        "rotor_positions_w_m": [[0.2, 0, 0.55], [-0.2, 0, 0.55]],
        "wall_plane_x_m": 0.7,
    }


def test_projection_uses_pixels_and_flips_screen_vertical():
    pixel, valid = project_point([0.5, 0.5, 0], np.eye(4), [1280, 720])
    assert valid and pixel == [960, 180]
    behind = np.eye(4)
    behind[3, 3] = -1
    assert project_point([0, 0, 0], behind, [1280, 720])[1] is False


def test_real_usd_frustum_projects_look_at_to_frame_center():
    from pxr import Gf

    camera = Gf.Camera()
    camera.SetPerspectiveFromAspectRatioAndFieldOfView(1280 / 720, 55, Gf.Camera.FOVHorizontal)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(-1, -2, 1), Gf.Vec3d(0, 0, 0.6), Gf.Vec3d(0, 0, 1))
    camera.transform = view.GetInverse()
    matrix = np.array(camera.frustum.ComputeViewMatrix() * camera.frustum.ComputeProjectionMatrix())
    pixel, valid = project_point([0, 0, 0.6], matrix, [1280, 720])
    assert valid and np.allclose(pixel, [640, 360])


@pytest.mark.parametrize("mode,axis,plane", [("seabed", 2, 0), ("wall", 0, 0.7)])
def test_clearance_segments_use_actual_finite_surface(mode, axis, plane):
    data = record()
    annotations = build_annotations(data, np.eye(4), [1280, 720], mode, pool_bounds=[-24.3, 0.7, -3, 22, 2.5])
    json.dumps(annotations, allow_nan=False)
    for ray, rotor in zip(annotations["rotor_rays"], data["rotor_positions_w_m"], strict=True):
        assert ray["start_w_m"] == rotor
        assert ray["end_w_m"][axis] == plane
        assert ray["clearance_m"] == pytest.approx(abs(rotor[axis] - plane))
        assert ray["valid"]
    outside = build_annotations(data, np.eye(4), [1280, 720], mode, pool_bounds=[-24.3, 0.7, 10, 22, 2.5])
    assert not any(r["valid"] for r in outside["rotor_rays"])


def test_current_arrows_are_one_second_water_advection_not_forces():
    data = record()
    annotations = build_annotations(data, np.eye(4), [1280, 720], "current")
    assert len(annotations["current_arrows"]) == 3
    for arrow in annotations["current_arrows"]:
        assert np.allclose(np.array(arrow["end_w_m"]) - arrow["start_w_m"], [0.22, 0, 0])
    data["current_w_m_s"] = [0, 0, 0]
    assert not build_annotations(data, np.eye(4), [1280, 720], "current")["current_arrows"]


def axial_record(positions, axes):
    return {
        **record(),
        "rotor_positions_w_m": positions,
        "rotor_axes_w": axes,
        "gripper_position_w_m": [0.4, 0, 0.4],
        "gripper_reference_w_m": [0.3, 0, 0.4],
    }


def test_tilted_floor_axis_rotates_hit_not_normal_projection():
    data = axial_record([[0, 0, 0.55]], [[1, 0, -1]])
    annotation = build_annotations(data, np.eye(4), [1280, 720], "seabed", pool_bounds=[-2, 2, -2, 2, 2.5])
    (ray,) = annotation["rotor_rays"]
    assert annotation["schema_version"] == 2
    assert np.allclose(ray["end_w_m"], [0.55, 0, 0])
    assert ray["normal_clearance_m"] == 0.55
    assert ray["ray_length_m"] == pytest.approx(0.55 * np.sqrt(2))
    assert np.allclose(ray["axis_direction_w"], [1 / np.sqrt(2), 0, -1 / np.sqrt(2)])
    # Reverse the source shaft direction: its physical line and hit are unchanged.
    data["rotor_axes_w"] = [[-1, 0, 1]]
    reversed_axis = build_annotations(data, np.eye(4), [1280, 720], "seabed", pool_bounds=[-2, 2, -2, 2, 2.5])
    assert reversed_axis["rotor_rays"] == annotation["rotor_rays"]
    # Rotate actual axis toward Y; endpoint MUST follow it, not retain X/Z guide.
    data["rotor_axes_w"] = [[0, 1, -1]]
    rotated = build_annotations(data, np.eye(4), [1280, 720], "seabed", pool_bounds=[-2, 2, -2, 2, 2.5])
    assert np.allclose(rotated["rotor_rays"][0]["end_w_m"], [0, 0.55, 0])


def test_tilted_wall_hit_follows_actual_shaft_and_positive_length():
    data = axial_record([[0.2, 0, 0.55]], [[-1, -1, 0]])
    annotation = build_annotations(data, np.eye(4), [1280, 720], "wall", pool_bounds=[-2, 0.7, -2, 2, 2.5])
    (ray,) = annotation["rotor_rays"]
    assert np.allclose(ray["end_w_m"], [0.7, 0.5, 0.55])
    assert np.allclose(ray["axis_direction_w"], [1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    assert ray["ray_length_m"] == pytest.approx(np.sqrt(0.5))
    assert ray["hemisphere"]["normal_w"] == [-1, 0, 0]


@pytest.mark.parametrize(
    "position,axis",
    [
        ([0, 0, 0.55], [1, 0, 0]),
        ([0, 0, 0.55], [0, 0, 0]),
        ([0, 0, 0], [0, 0, 1]),
        ([0, 0, -0.1], [0, 0, 1]),
        ([0, 0, 0.55], [10, 0, -1]),
    ],
)
def test_parallel_degenerate_behind_and_finite_surface_misses_have_no_ray(position, axis):
    data = axial_record([position], [axis])
    annotation = build_annotations(data, np.eye(4), [1280, 720], "seabed", pool_bounds=[-1, 1, -1, 1, 2.5])
    assert annotation["rotor_rays"] == []


def camera_matrix():
    from pxr import Gf

    camera = Gf.Camera()
    camera.SetPerspectiveFromAspectRatioAndFieldOfView(1280 / 720, 55, Gf.Camera.FOVHorizontal)
    camera.transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(-1, -2, 1), Gf.Vec3d(0, 0, 0.4), Gf.Vec3d(0, 0, 1)).GetInverse()
    return np.asarray(camera.frustum.ComputeViewMatrix() * camera.frustum.ComputeProjectionMatrix())


@pytest.mark.parametrize("normal", [[0, 0, 1], [-1, 0, 0]])
def test_hemisphere_is_metric_3d_cap_into_water_with_real_projection(normal):
    center, normal = np.array([0, 0, 0.4]), np.array(normal)
    matrix = camera_matrix()
    hemisphere = project_hemisphere(center, normal, matrix, [1280, 720])
    assert hemisphere["valid"]
    assert hemisphere["radius_m"] == 0.045
    assert hemisphere["silhouette"]["closed"] and hemisphere["base_ring"]["closed"]
    assert len(hemisphere["meridians"]) == 2
    for path in [hemisphere["silhouette"], hemisphere["base_ring"], *hemisphere["meridians"]]:
        points = np.array(path["points_w_m"])
        assert np.allclose(np.linalg.norm(points - center, axis=1), 0.045)
        assert np.min((points - center) @ normal) >= -1e-10
        expected = project_trajectory(points, matrix, [1280, 720])
        assert path["points_px"] == expected["points_px"]
        assert all(path["valid"])
    ring = np.array(hemisphere["base_ring"]["points_w_m"])
    assert np.allclose((ring - center) @ normal, 0)
    for arc in hemisphere["meridians"]:
        assert max((np.array(arc["points_w_m"]) - center) @ normal) == pytest.approx(0.045)
    # Perspective base-ring axes differ: this is not an arbitrary 2-D circle.
    extent = np.ptp(np.asarray(hemisphere["base_ring"]["points_px"]), axis=0)
    assert abs(extent[0] - extent[1]) > 1


def test_hemisphere_behind_camera_is_disabled_and_points_invalid():
    matrix = np.eye(4)
    matrix[3, 3] = -1
    result = project_hemisphere([0, 0, 0], [0, 0, 1], matrix, [1280, 720])
    assert not result["valid"] and result["silhouette"]["points_px"] == []
    assert not any(result["base_ring"]["valid"])


@pytest.mark.parametrize("mode,axis", [("seabed", [1, 0, -1]), ("wall", [1, 0, 0])])
def test_optional_small_hemisphere_reprojects_metric_geometry_without_changing_rays(mode, axis):
    data = axial_record([[0, 0, 0.55]], [axis])
    matrix = camera_matrix()
    kwargs = dict(pool_bounds=[-2, 0.7, -2, 2, 2.5])
    original = build_annotations(data, matrix, [1280, 720], mode, **kwargs)
    assert original == build_annotations(data, matrix, [1280, 720], mode, hemisphere_radius_m=0.045, **kwargs)
    small = build_annotations(data, matrix, [1280, 720], mode, hemisphere_radius_m=0.018, **kwargs)
    old_ray, ray = original["rotor_rays"][0], small["rotor_rays"][0]
    hemisphere = ray["hemisphere"]
    assert hemisphere["radius_m"] == 0.018
    assert hemisphere == project_hemisphere(ray["end_w_m"], hemisphere["normal_w"], matrix, [1280, 720], radius_m=0.018)
    assert {key: value for key, value in ray.items() if key != "hemisphere"} == {
        key: value for key, value in old_ray.items() if key != "hemisphere"
    }
    for path in [hemisphere["silhouette"], hemisphere["base_ring"], *hemisphere["meridians"]]:
        assert np.allclose(np.linalg.norm(np.asarray(path["points_w_m"]) - ray["end_w_m"], axis=1), 0.018)
    assert {key: value for key, value in small.items() if key != "rotor_rays"} == {
        key: value for key, value in original.items() if key != "rotor_rays"
    }


@pytest.mark.parametrize("radius", [0, -0.018, float("nan"), float("inf")])
def test_invalid_optional_hemisphere_radius_rejected_even_without_visible_hits(radius):
    with pytest.raises(ValueError, match="finite and positive"):
        build_annotations(axial_record([], []), np.eye(4), [1280, 720], "seabed", hemisphere_radius_m=radius)


def test_main_and_inset_gripper_history_use_actual_pose_and_independent_projection():
    data = axial_record([], [])
    history = [[0.3, 0, 0.4], [0.35, 0, 0.4], data["gripper_position_w_m"]]
    main = build_annotations(data, camera_matrix(), [1280, 720], "current", trajectory_w=history)
    inset = build_annotations(data, np.eye(4), [640, 480], "current", trajectory_w=history)
    for annotations, matrix, viewport in ((main, camera_matrix(), [1280, 720]), (inset, np.eye(4), [640, 480])):
        trajectory = annotations["gripper_trajectory"]
        assert trajectory["points_w_m"] == history
        assert trajectory["current_px"] == project_point(data["gripper_position_w_m"], matrix, viewport)[0]
        assert annotations["gripper_reference"]["label"] == "Initial tool position"
        assert annotations["gripper_reference"]["point_w_m"] == history[0]
        json.dumps(annotations, allow_nan=False)
    assert main["gripper_trajectory"]["points_px"] != inset["gripper_trajectory"]["points_px"]


def test_legacy_v2_annotations_are_exactly_reproducible_even_with_new_optional_history():
    from pathlib import Path

    root = Path(__file__).parents[1] / "artifacts/effects_v2"
    for name, mode in (
        ("current-still", "current"),
        ("seabed-near", "seabed"),
        ("wall-near", "wall"),
        ("actuator-lag", "actuator"),
    ):
        report = json.loads((root / name / "report.json").read_text())
        frames = json.loads((root / name / "trace.json").read_text())["frames"]
        bounds = None
        if report["scene"]["kind"] == "pool":
            x, y, _ = report["scene"]["center_m"]
            bounds = [x - 12.5, x + 12.5, y - 12.5, y + 12.5, 2.5]
        for frame in (frames[0], frames[-1]):
            assert "rotor_axes_w" not in frame
            actual = build_annotations(
                frame,
                frame["camera_view_projection_row_major"],
                [1280, 720],
                mode,
                pool_bounds=bounds,
                trajectory_w=[[1, 2, 3]],
            )
            assert actual == frame["annotations"]
