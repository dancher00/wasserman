"""Unlatched, pressure-equalized hatch, scaled for the Alpha5 workspace."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from wasman.assets.robot_cameras import add_robot_cameras
from wasman.tasks.underwater_panel.env_cfg import UnderwaterRotateValveT200EnvCfg
from wasman.tasks.underwater_press_button.config.bluerov2_alpha.env_cfg import BUTTON_USD_PATH


@configclass
class HatchEnvCfg(UnderwaterRotateValveT200EnvCfg):
    observation_space = 38
    episode_length_s = 60.0
    mechanism_joint_name = "Hinge"
    mechanism_body_name = "lid"
    mechanism_contact_offset = (-0.42, 0.0, 0.077)
    mechanism_joint_limits = (0.0, math.radians(105))
    mechanism_displaced_volume = 0.0018
    mechanism_success_max_speed = 0.05
    button_max_travel = math.radians(105)
    button_pressed_threshold = math.radians(80)
    success_max_travel = math.radians(105)
    success_max_tool_speed = 0.04
    success_hold_steps = 30
    tool_contact_distance = 0.06
    tool_contact_alignment = -1.01  # Handle engagement uses two real finger contacts, not button normal.
    button_y_range = (0.0, 0.0)
    button_z_range = (0.0, 0.0)
    base_target_position = (0.0, 0.0, 0.60)
    base_target_position_scale = (1.60, 0.60, 0.55)
    arm_target_scale = (1.5, 2.5, 3.0, 3.2)
    valve_initial_wrist = 0.15

    def __post_init__(self):
        super().__post_init__()
        self.scene.panel = None
        self.scene.seabed.init_state.pos = (0.0, 0.0, -0.40)
        self.scene.sand_apron = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/SandApron",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(BUTTON_USD_PATH.parent.parent / "submarine_hatch/sand_cutout.usda")
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.75, 0.0, 0.0)),
        )
        self.scene.button.spawn.usd_path = str(BUTTON_USD_PATH.parent.parent / "submarine_hatch/hatch.usda")
        self.scene.button.init_state.pos = (0.75, 0.0, 0.0)
        self.scene.button.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.scene.button.init_state.joint_pos = {"Hinge": 0.0}
        self.mechanism_joint_name = "Hinge"
        self.mechanism_body_name = "lid"
        self.mechanism_contact_offset = (-0.42, 0.0, 0.077)
        self.base_target_position = (0.0, 0.0, 0.60)
        self.base_target_position_scale = (1.60, 0.60, 0.55)
        self.arm_target_scale = (1.5, 2.5, 3.0, 3.2)
        self.scene.robot.init_state.pos = (-0.85, 0.0, 0.60)
        self.scene.robot.init_state.joint_pos["alpha_axis_c"] = 0.30
        self.scene.button.actuators["passive"].damping = 0.12
        self.scene.left_contact.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Button/lid"]
        self.scene.right_contact.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Button/lid"]
        self.sim.default_visualizer_cfg.eye = (-0.7, -1.8, 1.5)
        self.sim.default_visualizer_cfg.lookat = (0.55, 0.0, 0.10)


@configclass
class HatchCamerasEnvCfg(HatchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8  # Opt-in image collection must not inherit 512 rendered robots.
        add_robot_cameras(self)
