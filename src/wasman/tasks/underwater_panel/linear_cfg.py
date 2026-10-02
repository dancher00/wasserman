"""Versioned marine mechanisms: physical T200 robot, original task thresholds."""

import math
from pathlib import Path

from isaaclab.utils import configclass

from wasman.assets.registered_bluerov import configure_registered_bluerov
from wasman.tasks.underwater_panel.env_cfg import UnderwaterRotateValveT200EnvCfg, configure_mechanism

ASSETS = Path(__file__).resolve().parents[2] / "assets/data/objects/marine_mechanisms_v1"


@configclass
class MarineMechanismCfg(UnderwaterRotateValveT200EnvCfg):
    observation_space = 38
    episode_length_s = 60.0
    require_finger_contact = False
    success_hold_steps = 4
    mechanism_displaced_volume = 0.0
    valve_initial_wrist = 0.15
    arm_target_scale = (1.5, 1.5, 1.5, 3.2)
    base_target_position_scale = (0.5, 0.5, 0.4)

    def __post_init__(self):
        super().__post_init__()
        configure_registered_bluerov(self)
        self.scene.panel.spawn.usd_path = str(ASSETS / "grounded_panel.usda")
        self.current_speed_range = self.current_vertical_range = (0.0, 0.0)
        self.turbulence_sigma = 0.0
        self.scene.robot.init_state.pos = (-0.12, 0, 0.82)
        self.base_target_position = (-0.12, 0, 0.82)


@configclass
class MarinePushSliderCfg(MarineMechanismCfg):
    mechanism_initial_position = -0.15
    mechanism_joint_limits = (-0.15, 0.15)
    button_max_travel = 0.30
    button_pressed_threshold = 0.27
    contact_threshold = 0.005
    mechanism_progress_unit = "m"
    mechanism_contact_offset = (0.0207, 0, 0)
    button_y_range = (-0.05, 0.05)
    button_z_range = (0.73, 0.81)

    def __post_init__(self):
        super().__post_init__()
        configure_mechanism(self, "slider_with_rail.usd", "SliderJoint", "Slider", scale=0.3, rotation=(0, 0, 1, 0))
        self.scene.button.spawn.usd_path = str(ASSETS / "slider_with_rail.usda")
        self.mechanism_contact_offset = (0.0207, 0, 0)
        self.base_target_position = (-0.12, 0, 0.82)
        self.base_target_position_scale = (0.65, 0.5, 0.4)
        # This original source slider uses Y prismatic motion in its rotated root.
        for name in ["left_contact", "right_contact"]:
            getattr(self.scene, name).filter_prim_paths_expr = ["{ENV_REGEX_NS}/Button/Xform/Slider"]


@configclass
class MarinePullLeverCfg(MarineMechanismCfg):
    mechanism_initial_position = 0.0
    mechanism_joint_limits = (-math.pi / 4, math.pi / 4)
    button_max_travel = math.pi / 4
    button_pressed_threshold = math.radians(40)
    contact_threshold = math.radians(2)
    mechanism_progress_unit = "rad"
    mechanism_contact_offset = (0, 0, 0.145)
    button_y_range = (-0.12, -0.04)
    button_z_range = (0.68, 0.76)

    def __post_init__(self):
        super().__post_init__()
        configure_mechanism(self, "lever.usd", "lever_joint", "lever", rotation=(0, -math.sqrt(0.5), 0, math.sqrt(0.5)))
        self.scene.button.spawn.usd_path = str(ASSETS / "lever.usda")
        self.mechanism_contact_offset = (0, 0, 0.15)
        self.scene.button.spawn.fix_root_link = None
        self.scene.button.actuators["passive"].stiffness = 0.5
        self.base_target_position = (-0.12, 0, 0.82)
        self.base_target_position_scale = (0.65, 0.5, 0.4)
        for name in ["left_contact", "right_contact"]:
            getattr(self.scene, name).filter_prim_paths_expr = ["{ENV_REGEX_NS}/Button/lever/lever"]
