"""Long swim-in variant; the released short-start benchmark stays unchanged."""

from isaaclab.utils import configclass

from .thruster_cfg import ThrusterPressButtonEnvCfg


@configclass
class ApproachPressButtonEnvCfg(ThrusterPressButtonEnvCfg):
    # Same observation/action contract, low-level controller and contact criteria.
    # 0.85 m extra transit at the existing 0.10 m/s reference limit needs more time.
    episode_length_s = 40.0
    success_hold_steps = 90
    staging_position = (-0.08, 0.0, 0.82)
    policy_arm_nominal = (3.14, 0.15, 1.62, 0.15)

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.pos = (-0.85, 0.0, 0.82)
        # Fold the elbow upward ahead of the hull; preserve the policy's original
        # action reference separately so deployment does not redefine its actions.
        self.scene.robot.init_state.joint_pos["alpha_axis_c"] = 0.30
        self.sim.default_visualizer_cfg.eye = (-0.90, -2.5, 1.65)
        self.sim.default_visualizer_cfg.lookat = (0.0, -0.04, 0.76)
