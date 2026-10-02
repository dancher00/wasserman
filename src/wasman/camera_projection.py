"""Project a live ROS-convention camera without consulting stale USD transforms.

Pass Camera.data.pos_w, quat_w_ros and intrinsic_matrices after updating the
sensor. A Fabric pose write need not be reflected in UsdGeom.Camera.GetCamera().
All inputs describe a single camera; quaternion order is XYZW, not WXYZ.
"""

import numpy as np


def pinhole_view_projection(
    position_w,
    quaternion_xyzw_ros,
    intrinsic_matrix,
    viewport,
    clipping_range=(0.01, 100.0),
):
    """Return a row-vector world-to-clip matrix for ``project_point``.

    ROS optical axes are X right, Y down, Z forward. Pixels use top-left origin
    and exactly the supplied K (including its principal point). Clip depth is
    OpenGL-style [-1, +1]; homogeneous W is optical depth, so points behind the
    actual camera are rejected by the existing projection helper.

    Quaternion must already be unit length within sensor roundoff; it is then
    normalized precisely. Skew/projective intrinsics, malformed shapes, invalid
    clipping planes and non-finite inputs are rejected instead of falling back
    to a fabricated or stale camera.
    """
    position = np.asarray(position_w, dtype=float)
    quaternion = np.asarray(quaternion_xyzw_ros, dtype=float)
    intrinsic = np.asarray(intrinsic_matrix, dtype=float)
    size = np.asarray(viewport, dtype=float)
    clipping = np.asarray(clipping_range, dtype=float)
    for name, value, shape in (
        ("position_w", position, (3,)),
        ("quaternion_xyzw_ros", quaternion, (4,)),
        ("intrinsic_matrix", intrinsic, (3, 3)),
        ("viewport", size, (2,)),
        ("clipping_range", clipping, (2,)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{name} must have finite shape {shape}")
    if np.any(size <= 0) or not np.array_equal(size, np.floor(size)):
        raise ValueError("Viewport width and height must be positive integer pixel counts")
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, rtol=1e-4, atol=1e-6):
        raise ValueError("Camera quaternion must be unit-normalized (XYZW ROS convention)")
    quaternion = quaternion / norm
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    if fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths fx and fy must be positive")
    if not np.allclose(intrinsic[2], [0, 0, 1], rtol=0, atol=1e-8) or not np.allclose(
        [intrinsic[0, 1], intrinsic[1, 0]], 0, rtol=0, atol=1e-8
    ):
        raise ValueError("Only conventional pinhole intrinsics without skew are supported")
    near, far = clipping
    if not 0 < near < far:
        raise ValueError("Clipping range must satisfy 0 < near < far")
    x, y, z, w = quaternion
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    view = np.eye(4)
    view[:3, :3] = rotation.T
    view[:3, 3] = -rotation.T @ position
    width, height = size
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    projection = np.array(
        [
            [2 * fx / width, 0, 2 * cx / width - 1, 0],
            [0, -2 * fy / height, 1 - 2 * cy / height, 0],
            [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, 1, 0],
        ]
    )
    result = (projection @ view).T
    if not np.isfinite(result).all():
        raise ValueError("Camera parameters overflowed the projection matrix")
    return result
