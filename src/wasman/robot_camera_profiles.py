"""Versioned virtual-camera extrinsics, independent of Isaac imports.

Geometry-checked does not mean hardware-calibrated. Keep legacy-v1 unchanged:
published expert films and RGB checkpoints were collected with that profile.
"""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CameraMount:
    parent_link: str
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotCameraProfile:
    name: str
    base: CameraMount
    gripper: CameraMount
    rationale: str
    focal_length_mm: float = 12.0
    horizontal_aperture_mm: float = 20.955
    clipping_range_m: tuple[float, float] = (0.015, 30.0)
    orientation_convention: str = "world"
    hardware_calibrated: bool = False
    source_urls: tuple[str, ...] = ()

    def metadata(self):
        return asdict(self)


def aim_in_xz_plane(position, target):
    """World camera convention: +X forward, +Z up; preserve image right=-Y."""
    if not all(math.isfinite(v) for v in (*position, *target)):
        raise ValueError("Camera aim must be finite")
    dx, dy, dz = (float(b - a) for a, b in zip(position, target, strict=True))
    if abs(dy) > 1e-12 or math.hypot(dx, dz) < 1e-9:
        raise ValueError("Camera aim must be nonzero and in the local XZ plane")
    angle = math.atan2(-dz, dx)
    return (0.0, math.sin(angle / 2), 0.0, math.cos(angle / 2))


LEGACY_V1 = RobotCameraProfile(
    name="legacy-v1",
    base=CameraMount(
        "base_link", (0.32, 0.0, 0.10), (0.0, math.sin(math.pi / 12), 0.0, math.cos(math.pi / 12))
    ),
    gripper=CameraMount(
        "alpha_jaw_base_link", (0.042, 0.0, 0.035), (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))
    ),
    rationale="Original virtual sensors; wrist TCP is at the top image edge. Retained for reproducibility.",
)

WRIST_V2_POSITION = (0.055, 0.0, -0.035)
WRIST_V2_TARGET = (0.0, 0.0, 0.085)
WRIST_AUDIT_ENVELOPE_RADIUS_M = 0.010  # Clearance surrogate, not a manufactured housing.
BASE_MOUNT_SOURCE = (
    "https://github.com/Robotic-Decision-Making-Lab/blue/blob/"
    "812fea605cd2056ba20fac504f456457594595fb/"
    "blue_description/description/bluerov2_heavy_reach/urdf.xacro"
)

GEOMETRY_V2 = RobotCameraProfile(
    name="geometry-v2",
    # Upstream camera pivot, but a deliberately selected 30-degree downward
    # viewing tilt rather than its neutral +X optical axis. Not factory extrinsics.
    base=CameraMount("base_link", (0.21, 0.0, 0.067), LEGACY_V1.base.quaternion_xyzw),
    gripper=CameraMount(
        "alpha_jaw_base_link", WRIST_V2_POSITION, aim_in_xz_plane(WRIST_V2_POSITION, WRIST_V2_TARGET)
    ),
    rationale=(
        "Base uses the pinned upstream camera pivot with an engineering 30-degree downward tilt. "
        "Virtual wrist camera behind/beside the palm, aimed between the fingertips. "
        "Source-mesh frustum/clearance audit; no hardware calibration or fabricated camera housing."
    ),
    source_urls=(BASE_MOUNT_SOURCE,),
)

ROBOT_CAMERA_PROFILES = {profile.name: profile for profile in (LEGACY_V1, GEOMETRY_V2)}


def robot_camera_profile(name="legacy-v1"):
    try:
        return ROBOT_CAMERA_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"Unknown robot camera profile {name!r}; choose {tuple(ROBOT_CAMERA_PROFILES)}") from error
