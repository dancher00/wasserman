"""Physical shell collection into a fixed floor hoop; no granular sand model."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from wasman.assets.registered_bluerov import configure_registered_bluerov
from wasman.tasks.underwater_panel.env_cfg import UnderwaterRotateValveT200EnvCfg

ASSETS = Path(__file__).parents[2] / "assets/data/objects/shell_collection"


@configclass
class CollectShellEnvCfg(UnderwaterRotateValveT200EnvCfg):
    observation_space = 28
    episode_length_s = 240.0
    base_target_position = (0.0, 0.0, 0.60)
    base_target_position_scale = (1.60, 1.10, 0.55)
    arm_target_scale = (1.5, 2.5, 3.0, 3.2)
    valve_initial_wrist = 0.15
    shell_count = 1
    habitat = (
        ("Seagrass", (0.18, -0.57, 0.0)),
        ("RibbonKelp", (0.77, -0.60, 0.0)),
        ("FanAlgae", (1.02, -0.12, 0.0)),
        ("BranchCoral", (1.07, 0.35, 0.0)),
        ("BoulderCoral", (0.20, 0.65, 0.0)),
        ("PlateCoral", (0.17, -0.06, 0.0)),
    )
    hoop_center = (0.62, 0.45, 0.0)
    hoop_inner_radius = 0.28
    shell_positions = ((0.45, -0.30, 0.015),)
    shell_position_jitter = 0.02
    shell_yaw_range = (-0.15, 0.15)
    shell_displaced_volume = 0.080 / 2700
    shell_linear_drag = 0.25  # N / (m/s); engineering surrogate, not identified.
    shell_quadratic_drag = 0.8  # N / (m/s)^2.
    shell_angular_drag = 0.0001
    current_speed_range = (0.0, 0.0)
    current_vertical_range = (0.0, 0.0)
    turbulence_sigma = 0.0

    def __post_init__(self):
        super().__post_init__()
        configure_registered_bluerov(self)
        self.scene.panel = self.scene.button = None
        self.scene.num_envs = 4
        self.scene.env_spacing = 4.0
        self.scene.seabed.init_state.pos = (0.0, 0.0, 0.0)
        self.base_target_position = (0.0, 0.0, 0.60)
        self.base_target_position_scale = (1.60, 1.10, 0.55)
        self.arm_target_scale = (1.5, 2.5, 3.0, 3.2)
        self.scene.robot.init_state.pos = (-0.50, 0.0, 0.60)
        self.scene.robot.init_state.joint_pos["alpha_axis_c"] = 0.30
        self.scene.hoop = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Hoop",
            spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "Hoop.usda")),
            init_state=AssetBaseCfg.InitialStateCfg(pos=self.hoop_center),
        )
        for i in range(self.shell_count):
            setattr(
                self.scene,
                f"shell_{i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Shell_{i}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=str(ASSETS / "Shell.usda"),
                        activate_contact_sensors=True,
                        rigid_props=PhysxRigidBodyCfg(
                            disable_gravity=False,
                            max_depenetration_velocity=0.2,
                            solver_position_iteration_count=32,
                            solver_velocity_iteration_count=4,
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=self.shell_positions[i]),
                ),
            )
        for name in ("left_contact", "right_contact"):
            getattr(self.scene, name).filter_prim_paths_expr = [
                f"{{ENV_REGEX_NS}}/Shell_{i}" for i in range(self.shell_count)
            ]
        for i, (asset, position) in enumerate(self.habitat):
            setattr(self.scene, f"habitat_{i}", AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Habitat_{i}",
                spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / f"{asset}.usda")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=position),
            ))
        self.sim.default_visualizer_cfg.eye = (-0.55, -1.8, 1.60)
        self.sim.default_visualizer_cfg.lookat = (0.55, 0.05, 0.05)
