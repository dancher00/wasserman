"""Experimental AM-Bench panel mechanisms ported to Isaac Lab 3 / XYZW rotations.

These are executable development scenes, not validated benchmark results.
The shared environment keeps legacy button names internally for checkpoint compatibility.
"""

import math

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from wasman.tasks.underwater_press_button.config.bluerov2_alpha.env_cfg import (
    BUTTON_USD_PATH,
    UnderwaterPressButtonEnvCfg,
)


def configure_mechanism(cfg, asset, joint, body, *, scale=1.0, rotation=(0.0, 0.0, 1.0, 0.0)):
    """Configure a fixed-root passive mechanism, without actuator-driven success."""
    mechanism = cfg.scene.button
    mechanism.spawn.usd_path = str(BUTTON_USD_PATH.parent / asset)
    mechanism.spawn.scale = (scale, scale, scale)
    mechanism.init_state.rot = rotation
    mechanism.init_state.pos = (0.80, 0.0, 0.80)
    mechanism.init_state.joint_pos = {joint: cfg.mechanism_initial_position}
    mechanism.actuators = {
        "passive": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=1.0,
            actuator_effort_limit=20.0,
            joint_effort_limit=20.0,
            actuator_velocity_limit=2.0,
            joint_velocity_limit=2.0,
        )
    }
    cfg.mechanism_joint_name = joint
    cfg.mechanism_body_name = body
    # Neutral policy holds the initial standoff; reaching requires an action.
    cfg.base_target_position = (0.0, 0.0, 0.82)
    cfg.base_target_position_scale = (0.40, *cfg.base_target_position_scale[1:])
    cfg.tool_contact_distance = 0.06


@configclass
class UnderwaterPushSliderEnvCfg(UnderwaterPressButtonEnvCfg):
    """A 0.30 m rail, scaled from AM-Bench to the Alpha workspace."""

    action_space = 11
    observation_space = 38
    policy_gripper = True
    mechanism_initial_position = -0.15
    mechanism_joint_limits = (-0.15, 0.15)
    button_max_travel = 0.30
    button_pressed_threshold = 0.27
    contact_threshold = 0.005
    button_y_range = (-0.05, 0.05)
    button_z_range = (0.73, 0.81)
    base_target_position_scale = (0.22, 0.30, 0.20)

    def __post_init__(self):
        super().__post_init__()
        configure_mechanism(self, "slider_with_rail.usd", "SliderJoint", "Slider", scale=0.3)


@configclass
class UnderwaterPullLeverEnvCfg(UnderwaterPressButtonEnvCfg):
    """Pull a revolute handle through 40 degrees; target follows the handle tip."""

    action_space = 11
    observation_space = 38
    policy_gripper = True
    mechanism_progress_unit = "rad"
    mechanism_contact_offset = (0.0, 0.0, 0.145)
    button_max_travel = math.pi / 4
    button_pressed_threshold = math.radians(40)
    contact_threshold = math.radians(2)
    button_y_range = (-0.12, -0.04)
    button_z_range = (0.68, 0.76)
    base_target_position_scale = (0.22, 0.25, 0.25)

    def __post_init__(self):
        super().__post_init__()
        configure_mechanism(
            self, "lever.usd", "lever_joint", "lever", rotation=(0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))
        )
        self.scene.button.actuators["passive"].stiffness = 0.5
        # The upstream lever already includes a world-fixed joint under its
        # non-rigid articulation root; do not ask Isaac Lab to create another.
        self.scene.button.spawn.fix_root_link = None


@configclass
class UnderwaterRotateValveEnvCfg(UnderwaterPressButtonEnvCfg):
    """Shared valve geometry and controls; the T200 variant defines task scoring."""

    action_space = 11
    observation_space = 38
    policy_gripper = True
    mechanism_progress_unit = "rad"
    button_max_travel = math.pi
    button_pressed_threshold = math.radians(170)
    contact_threshold = math.radians(5)
    button_y_range = (-0.08, 0.0)
    button_z_range = (0.72, 0.80)
    base_target_position_scale = (0.22, 0.28, 0.25)

    def __post_init__(self):
        super().__post_init__()
        # Initialize the shared contract, then select the sourced industrial
        # fixture whose wheel axis is already local X (no container rotation).
        configure_mechanism(
            self, "valve.usd", "base_to_shaft", "handle_link", rotation=(0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
        )
        # CC0 industrial fixture: a separate wheel, with contact on its rim.
        self.scene.button.spawn.usd_path = str(BUTTON_USD_PATH.parent.parent / "industrial_valve/valve.usdc")
        self.scene.button.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.scene.button.init_state.pos = (0.81, 0.0, 0.80)
        self.mechanism_contact_offset = (0.0, -0.059, 0.0)
        # The realistic bonnet projects farther from the panel than the old
        # flat disk; retain a genuine open-water approach instead of spawning
        # the fingers against the wheel.
        self.scene.robot.init_state.pos = (-0.12, 0.0, 0.82)
        self.base_target_position = (-0.12, 0.0, 0.82)


@configclass
class UnderwaterRotateValveT200EnvCfg(UnderwaterRotateValveEnvCfg):
    """Actuated variant; contact/hold/release remain separately measured diagnostics."""

    valve_success_contract = "ambench-angle-170-v1"
    use_physical_thrusters = True
    episode_length_s = 75.0
    observation_space = 50
    decimation = 8
    thruster_command_delay_steps = 4
    arm_target_scale = (0.8, 1.0, 1.0, 3.20)
    valve_initial_wrist = 3.20
    valve_acquisition_roll = 2 * math.pi / 3 - valve_initial_wrist
    base_target_position_scale = (0.50, 0.35, 0.35)
    # The inherited ±0.18 integral budget compensates only 1.26 N vertically,
    # less than the ±3% buoyancy variation of a 13 kg vehicle. Give the PID
    # enough measured-error trim authority without reading randomized physics.
    station_position_kp = (300.0, 300.0, 350.0)
    station_position_kd = (120.0, 120.0, 130.0)
    station_position_ki = (40.0, 40.0, 45.0)
    station_position_integral_limit = (0.25, 0.25, 0.25)
    station_rotation_kp = (25.0, 28.0, 30.0)
    station_rotation_kd = (7.0, 8.0, 10.0)
    station_rotation_ki = (2.0, 2.0, 2.0)
    station_rotation_integral_limit = (0.50, 0.50, 0.50)
    success_hold_steps = 30
    require_finger_contact = True
    mechanism_displaced_volume = 0.000060

    def __post_init__(self):
        super().__post_init__()
        from wasman.assets.bluerov2_alpha import BLUEROV2_ALPHA_HYDRODYNAMICS, BLUEROV2_ALPHA_URDF_PATH
        from wasman.assets.grasp_geometry import grasp_robot_urdf, spawn_grasp_robot
        from wasman.assets.grasp_hydrodynamics import geometry_scaled_arm_hydrodynamics

        self.scene.robot.spawn.asset_path = str(grasp_robot_urdf(BLUEROV2_ALPHA_URDF_PATH))
        self.scene.robot.spawn.func = spawn_grasp_robot
        self.scene.robot.spawn.collision_type = "Convex Decomposition"
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = self.decimation
        self.link_hydrodynamics = geometry_scaled_arm_hydrodynamics(BLUEROV2_ALPHA_HYDRODYNAMICS)
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 64
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
        # Conservative stroke-side servo, not the manufacturer's jaw-force
        # rating. 50 N/(2*51 rad/m*0.07 m) ~= 7 N per fingertip at that lever.
        grip = self.scene.robot.actuators["alpha_gripper"]
        grip.stiffness = 40000.0
        grip.damping = 120.0
        grip.actuator_effort_limit = grip.joint_effort_limit = 50.0
        grip.actuator_velocity_limit = grip.joint_velocity_limit = 0.0025
        grip.armature = 0.01
        self.scene.robot.actuators["alpha_arm"].joint_names_expr = ["alpha_axis_e", "alpha_axis_d", "alpha_axis_c"]
        self.scene.robot.actuators["alpha_wrist"] = ImplicitActuatorCfg(
            joint_names_expr=["alpha_axis_b"],
            actuator_effort_limit=0.6,
            joint_effort_limit=0.6,
            actuator_velocity_limit=0.5,
            joint_velocity_limit=0.75,
            stiffness=45.0,
            damping=4.0,
            armature=0.002,
        )
        # Use the physical wrist range [0, 3.22], retaining ~10 degrees of
        # tracking-error headroom beyond the requested 173.8-degree turn.
        # The inherited 0.96 soft range plus the old action scale clipped trim.
        self.scene.robot.soft_joint_pos_limit_factor = 1.0
        self.scene.robot.init_state.joint_pos["alpha_axis_b"] = self.valve_initial_wrist
        self.mechanism_contact_offset = (
            0.0,
            -0.059 * math.cos(self.valve_acquisition_roll),
            -0.059 * math.sin(self.valve_acquisition_roll),
        )
        # The inherited button fixture disables gravity. This rotating fixture
        # needs gravity as well as the explicitly modeled handle buoyancy.
        self.scene.button.spawn.rigid_props.disable_gravity = False
        self.scene.button.actuators["passive"].damping = 0.08
        # Small, constant seal/bearing resistance; engineering assumptions,
        # not a goal-dependent brake or a measured hardware identification.
        self.scene.button.actuators["passive"].friction = 0.012
        self.scene.button.actuators["passive"].dynamic_friction = 0.008
        self.scene.robot.spawn.activate_contact_sensors = True
        finger_root = (
            "{ENV_REGEX_NS}/Robot/Geometry/base_link/alpha_m3_inline_link/alpha_m2_1_1_link/"
            "alpha_m2_joint_link/alpha_m2_1_2_link/alpha_m1_link/alpha_jaw_base_link/"
        )
        self.scene.left_contact = ContactSensorCfg(
            prim_path=finger_root + "alpha_left_finger_link",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Button/handle_link"],
        )
        self.scene.right_contact = ContactSensorCfg(
            prim_path=finger_root + "alpha_right_finger_link",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Button/handle_link"],
        )
