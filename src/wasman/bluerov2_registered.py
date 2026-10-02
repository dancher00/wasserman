"""Opt-in, metre-baked BlueROV geometry v1; no simulator or legacy mutations.

Use these motor origins together with the derived URDF, never with the legacy
CAD-offset allocator. Coordinate origin, inertias and side-mounted arm stay native.
"""

import math
from pathlib import Path

ASSET_VERSION = "bluerov2_alpha_registered_v1"
SOURCE_DIR = Path(__file__).parent / "assets/data/robots/bluerov2_alpha"
ASSET_DIR = SOURCE_DIR.with_name(ASSET_VERSION)
URDF_PATH = ASSET_DIR / f"{ASSET_VERSION}.urdf"
BLUE_REVISION = "812fea605cd2056ba20fac504f456457594595fb"
NATIVE_POSITIONS = (
    (0.088, -0.102, -0.04),
    (0.088, 0.102, -0.04),
    (-0.088, -0.102, -0.04),
    (-0.088, 0.102, -0.04),
    (0.118, -0.215, 0.064),
    (0.118, 0.215, 0.064),
    (-0.118, -0.215, 0.064),
    (-0.118, 0.215, 0.064),
)
NATIVE_RPY = (
    (-math.pi / 2, math.pi / 2, -math.pi / 3),
    (-math.pi / 2, math.pi / 2, -2 * math.pi / 3),
    (-math.pi / 2, math.pi / 2, math.pi / 3),
    (-math.pi / 2, math.pi / 2, 2 * math.pi / 3),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)
NATIVE_DIRECTIONS = (
    (-math.sqrt(3) / 2, -0.5, 0),
    (-math.sqrt(3) / 2, 0.5, 0),
    (math.sqrt(3) / 2, -0.5, 0),
    (math.sqrt(3) / 2, 0.5, 0),
    (0, 0, -1),
    (0, 0, -1),
    (0, 0, -1),
    (0, 0, -1),
)
NATIVE_COM = (0.0, 0.0, 0.011)
SOURCE_CAMERA_MOUNT = (0.21, 0.0, 0.067)
SOURCE_SHA256 = {
    "bluerov2_heavy_reach.dae": "78bea4304d49db52e80b25c8a4580643c5c2cda4135e2069e7e0f68639bf5a92",
    "ccw_prop.dae": "e1918897f91bfa50720cec6feb3053315560878eaa5df8cc9c7519c4484f01dd",
    "cw_prop.dae": "9ac826b9c380759777869414fc79d19d880a7c66473886874010aa24101f7a05",
}


def allocation_matrix():
    """CPU ndarray mapping axial forces to force/moment about native base COM."""
    import numpy as np

    directions = np.asarray(NATIVE_DIRECTIONS)
    offsets = np.asarray(NATIVE_POSITIONS) - NATIVE_COM
    return np.concatenate((directions, np.cross(offsets, directions)), axis=1).T


def require_prepared_asset():
    """Resolve the opt-in asset without importing or changing any simulator config."""
    if not URDF_PATH.is_file():
        raise FileNotFoundError("Run scripts/prepare_bluerov2_registered.py before opting in")
    return URDF_PATH
