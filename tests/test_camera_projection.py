import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wasman.camera_projection import pinhole_view_projection
from wasman.effect_annotations import project_point

VIEWPORT = [1280, 720]
K = np.array([[600, 0, 640], [0, 600, 360], [0, 0, 1]], dtype=float)


def test_identity_ros_camera_known_pixels_and_optical_depth():
    matrix = pinhole_view_projection([0, 0, 0], [0, 0, 0, 1], K, VIEWPORT)
    for point, expected in (([0, 0, 2], [640, 360]), ([1, 0.5, 2], [940, 510]), ([-1, -0.5, 2], [340, 210])):
        pixel, valid = project_point(point, matrix, VIEWPORT)
        assert valid and np.allclose(pixel, expected)
        assert (np.r_[point, 1] @ matrix)[3] == point[2]


def test_arbitrary_live_pose_and_asymmetric_intrinsics_match_manual_ros_projection():
    position = np.array([0.8, -0.45, 3.6])
    rotation = Rotation.from_euler("xyz", [170, 20, -35], degrees=True)
    intrinsic = np.array([[530, 0, 303], [0, 570, 227], [0, 0, 1]], dtype=float)
    size = [640, 480]
    matrix = pinhole_view_projection(position, rotation.as_quat(), intrinsic, size)
    for local in ([0, 0, 2], [0.4, -0.3, 3], [-0.8, 0.7, 4], [0.01, -0.01, 0.15]):
        local = np.array(local)
        world = position + rotation.apply(local)
        actual, valid = project_point(world, matrix, size)
        expected = (intrinsic @ local)[:2] / local[2]
        assert valid and np.allclose(actual, expected, atol=1e-9)
    # Using a stale identity pose at the world origin produces a different pixel.
    stale = pinhole_view_projection([0, 0, 0], [0, 0, 0, 1], intrinsic, size)
    assert not np.allclose(project_point(world, stale, size)[0], actual)


def test_depth_clipping_and_behind_camera_rejection():
    matrix = pinhole_view_projection([0, 0, 0], [0, 0, 0, 1], K, VIEWPORT, (0.1, 10))
    for depth in (-5, 0, 0.05, 10.01):
        assert not project_point([0, 0, depth], matrix, VIEWPORT)[1]
    for depth in (0.10001, 1, 9.999):
        assert project_point([0, 0, depth], matrix, VIEWPORT)[1]
    for depth, ndc in ((0.1, -1), (10, 1)):
        clip = np.array([0, 0, depth, 1]) @ matrix
        assert clip[2] / clip[3] == pytest.approx(ndc)


def test_actual_ros_pose_agrees_with_independent_gf_camera():
    from pxr import Gf

    position = np.array([0.8, -0.45, 3.6])
    rotation = Rotation.from_euler("xyz", [165, 12, -27], degrees=True)
    near, far = 0.01, 100
    matrix = pinhole_view_projection(position, rotation.as_quat(), K, VIEWPORT, (near, far))
    gf = Gf.Camera()
    horizontal_fov = np.rad2deg(2 * np.arctan(VIEWPORT[0] / (2 * K[0, 0])))
    gf.SetPerspectiveFromAspectRatioAndFieldOfView(VIEWPORT[0] / VIEWPORT[1], horizontal_fov, Gf.Camera.FOVHorizontal)
    gf.clippingRange = Gf.Range1f(near, far)
    transform = np.eye(4)
    # Gf/OpenGL X right,Y up,Z backward vs ROS X right,Y down,Z forward.
    transform[:3, :3] = rotation.as_matrix() @ np.diag([1, -1, -1])
    transform[:3, 3] = position
    gf.transform = Gf.Matrix4d(transform.T.tolist())
    expected_matrix = np.asarray(gf.frustum.ComputeViewMatrix() * gf.frustum.ComputeProjectionMatrix())
    assert np.allclose(matrix, expected_matrix, rtol=1e-6, atol=1e-6)
    for local in ([0, 0, 2], [0.2, 0.3, 2], [-0.5, -0.2, 3]):
        world = position + rotation.apply(local)
        actual, valid = project_point(world, matrix, VIEWPORT)
        expected, gf_valid = project_point(world, expected_matrix, VIEWPORT)
        assert valid and gf_valid and np.allclose(actual, expected, atol=1e-4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_w", [0, 0]),
        ("position_w", [0, np.nan, 0]),
        ("quaternion_xyzw_ros", [0, 0, 0, 0]),
        ("quaternion_xyzw_ros", [0, 0, 0, 2]),
        ("quaternion_xyzw_ros", [0, 0, 1]),
        ("viewport", [0, 720]),
        ("viewport", [640.5, 480]),
        ("viewport", [640, np.inf]),
        ("clipping_range", [0, 100]),
        ("clipping_range", [1, 1]),
        ("clipping_range", [10, 1]),
        ("intrinsic_matrix", [[600, 1, 640], [0, 600, 360], [0, 0, 1]]),
        ("intrinsic_matrix", [[600, 0, 640], [1, 600, 360], [0, 0, 1]]),
        ("intrinsic_matrix", [[600, 0, 640], [0, -1, 360], [0, 0, 1]]),
        ("intrinsic_matrix", [[600, 0, 640], [0, 600, 360], [0, 0.1, 1]]),
    ],
)
def test_invalid_camera_parameters_are_rejected(field, value):
    arguments = dict(position_w=[0, 0, 0], quaternion_xyzw_ros=[0, 0, 0, 1], intrinsic_matrix=K, viewport=VIEWPORT)
    arguments[field] = value
    with pytest.raises(ValueError):
        pinhole_view_projection(**arguments)


def test_sensor_quaternion_roundoff_and_sign_do_not_change_projection():
    rotation = Rotation.from_euler("xyz", [2.5, 0.3, -0.7]).as_quat()
    expected = pinhole_view_projection([1, 2, 3], rotation, K, VIEWPORT)
    for quaternion in (-rotation, rotation.astype(np.float32), rotation * (1 + 1e-6)):
        assert np.allclose(pinhole_view_projection([1, 2, 3], quaternion, K, VIEWPORT), expected, atol=1e-6)
