"""Finite-pool playback variant; original approach task and scores are unchanged."""

from isaaclab.utils import configclass

from wasman.assets.pool_geometry import configure_finite_pool

from .approach_cfg import ApproachPressButtonEnvCfg


@configclass
class PoolApproachPressButtonEnvCfg(ApproachPressButtonEnvCfg):
    """Reuse the original policy contract in a separately registered, unscored scene.

    Water density and fully submerged hydrodynamics retain their benchmark
    defaults. This changes physical scenery, not freshwater calibration or the
    opt-in boundary-loss setting.
    """

    def __post_init__(self):
        super().__post_init__()
        configure_finite_pool(self)
        self.sim.default_visualizer_cfg.eye = (-2.0, -2.3, 1.65)
        self.sim.default_visualizer_cfg.lookat = (0.50, 0.0, 1.15)
