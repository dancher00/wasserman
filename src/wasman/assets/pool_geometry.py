"""One geometry definition for the finite pool mesh, collision and jet boundaries."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class PoolGeometry:
    length_m: float = 25.0
    width_m: float = 25.0
    water_depth_m: float = 2.5
    freeboard_m: float = 0.20
    wall_thickness_m: float = 0.20
    floor_thickness_m: float = 0.15

    def __post_init__(self):
        if not all(isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError("Pool dimensions must be finite and positive")

    @property
    def wall_height_m(self):
        return self.water_depth_m + self.freeboard_m

    def collision_boxes(self):
        """Name, local centre, XYZ dimensions. Interior floor is z=0."""
        x, y, h, t, f = (
            self.length_m / 2,
            self.width_m / 2,
            self.wall_height_m,
            self.wall_thickness_m,
            self.floor_thickness_m,
        )
        return (
            ("Floor", (0, 0, -f / 2), (2 * x + 2 * t, 2 * y + 2 * t, f)),
            ("WestWall", (-x - t / 2, 0, h / 2), (t, 2 * y + 2 * t, h)),
            ("EastWall", (x + t / 2, 0, h / 2), (t, 2 * y + 2 * t, h)),
            ("SouthWall", (0, -y - t / 2, h / 2), (2 * x, t, h)),
            ("NorthWall", (0, y + t / 2, h / 2), (2 * x, t, h)),
        )

    def wet_boundary_rectangles(self):
        """Centre, inward normal, U tangent, V tangent, half-size, loss class.

        Finite wet surfaces coincide with the rendered/collision inner surfaces.
        Wall rectangles end at the waterline: above-water coping is not a water
        jet surface. The open top is not treated as an imaginary solid lid.
        """
        x, y, h = self.length_m / 2, self.width_m / 2, self.water_depth_m
        return (
            ((0, 0, 0), (0, 0, 1), (1, 0, 0), (0, 1, 0), (x, y), "floor"),
            ((-x, 0, h / 2), (1, 0, 0), (0, 1, 0), (0, 0, 1), (y, h / 2), "wall"),
            ((x, 0, h / 2), (-1, 0, 0), (0, 1, 0), (0, 0, -1), (y, h / 2), "wall"),
            ((0, -y, h / 2), (0, 1, 0), (1, 0, 0), (0, 0, -1), (x, h / 2), "wall"),
            ((0, y, h / 2), (0, -1, 0), (1, 0, 0), (0, 0, 1), (x, h / 2), "wall"),
        )


POOL_GEOMETRY = PoolGeometry()
POOL_USD_PATH = Path(__file__).resolve().parent / "data/environments/standard_pool/pool.usda"


def configure_finite_pool(cfg, *, center=(-10.5, 9.5, 0.0)):
    """Opt-in replacement before gym.make; no infinite sand collision remains.

    Centre translates the pool relative to each environment, not the robot/task.
    Requires the generated native pool asset. Default task configurations are
    untouched unless this helper is explicitly called.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    if not POOL_USD_PATH.is_file():
        raise FileNotFoundError("Generate the pool with scripts/prepare_pool_asset.py first")
    if getattr(cfg.scene, "sand_apron", None) is not None:
        raise ValueError("Do not replace the Hatch sand/cutout scene with a pool")
    if len(center) != 3 or not all(isfinite(v) for v in center) or center[2] != 0:
        raise ValueError("Pool centre must be finite XYZ with floor at z=0")
    cfg.scene.seabed = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pool",
        spawn=sim_utils.UsdFileCfg(usd_path=str(POOL_USD_PATH)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(center)),
    )
    cfg.scene.env_spacing = max(cfg.scene.env_spacing, POOL_GEOMETRY.length_m + 2)
    return cfg
