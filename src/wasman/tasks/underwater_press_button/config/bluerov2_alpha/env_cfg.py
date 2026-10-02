"""Configuration for the BlueROV2-Alpha underwater press-button task."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg
from isaaclab_tasks.utils import PresetCfg

from wasman.assets import BLUEROV2_ALPHA_CFG
from wasman.assets.panels import panel_path

BUTTON_USD_PATH = Path(__file__).parents[4] / "assets" / "data" / "objects" / "ambench" / "push_button.usd"


@configclass
class UnderwaterPressButtonPhysicsCfg(PresetCfg):
    """PhysX settings for stable small-displacement button contact."""

    isaacsim_physx: PhysxCfg = PhysxCfg(
        solver_type=1,
        solve_articulation_contact_last=True,
        enable_ccd=False,
        enable_stabilization=True,
    )
    default: PhysxCfg = isaacsim_physx


@configclass
class UnderwaterPressButtonSceneCfg(InteractiveSceneCfg):
    """A free-floating UVMS facing an instrument panel and spring button."""

    robot: ArticulationCfg = BLUEROV2_ALPHA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    panel: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Panel",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(panel_path()),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.86, 0.0, 0.80)),
    )
    button: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Button",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(BUTTON_USD_PATH),
            activate_contact_sensors=True,
            fix_root_link=True,
            rigid_props=PhysxRigidBodyCfg(
                disable_gravity=True,
                max_depenetration_velocity=0.5,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            articulation_props=[
                PhysxArticulationCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=2,
                )
            ],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.81, 0.0, 0.80),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={"PlungerSlideJoint": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "plunger": ImplicitActuatorCfg(
                joint_names_expr=["PlungerSlideJoint"],
                actuator_effort_limit=20.0,
                joint_effort_limit=20.0,
                actuator_velocity_limit=1.0,
                joint_velocity_limit=2.0,
                stiffness=1000.0,
                damping=80.0,
            )
        },
    )
    seabed: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/seabed",
        collision_group=-1,
        spawn=sim_utils.UsdFileCfg(usd_path=str(BUTTON_USD_PATH.parents[2] / "seabed/sand.usda")),
    )
    key_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2400.0, color=(0.42, 0.72, 0.88), angle=1.0),
    )
    fill_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.DomeLightCfg(intensity=1250.0, color=(0.30, 0.52, 0.62)),
    )


@configclass
class UnderwaterPressButtonEnvCfg(DirectRLEnvCfg):
    """AM-Bench-style instantaneous-contact task under underwater dynamics."""

    seed = 42
    decimation = 4
    episode_length_s = 12.0
    action_space = 10
    observation_space = 37
    state_space = 0

    # Shared panel mechanism contract. Defaults preserve the released checkpoint.
    mechanism_body_name = "plunger"
    mechanism_joint_name = "PlungerSlideJoint"
    mechanism_initial_position = 0.0
    mechanism_direction = 1.0
    mechanism_contact_offset = (0.0, 0.0, 0.0)
    mechanism_progress_unit = "m"
    mechanism_joint_limits = None
    policy_gripper = False
    require_finger_contact = False
    finger_contact_threshold_n = 0.1
    mechanism_displaced_volume = 0.0
    mechanism_success_max_speed = float("inf")

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics=UnderwaterPressButtonPhysicsCfg(),
    )
    scene: UnderwaterPressButtonSceneCfg = UnderwaterPressButtonSceneCfg(
        num_envs=512,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # AM-Bench-style cascaded control: the policy selects bounded absolute
    # base-pose and arm-joint targets; a geometric PID holds that pose at the
    # 120 Hz physics rate.  Zero action commands the centre of the task
    # workspace rather than an unregulated wrench.
    base_target_position = (0.18, 0.0, 0.82)
    base_target_position_scale = (0.22, 0.16, 0.14)
    base_target_rpy_scale = (0.08, 0.10, 0.15)
    arm_target_scale = (0.40, 0.45, 0.45, 0.55)

    station_position_kp = (70.0, 70.0, 90.0)
    station_position_kd = (40.0, 40.0, 48.0)
    station_position_ki = (5.0, 5.0, 7.0)
    station_rotation_kp = (7.0, 8.0, 10.0)
    station_rotation_kd = (3.0, 3.5, 4.0)
    station_rotation_ki = (0.25, 0.25, 0.30)
    station_max_force = (55.0, 55.0, 55.0)
    station_max_torque = (10.0, 10.0, 10.0)
    station_position_integral_limit = (0.18, 0.18, 0.18)
    station_rotation_integral_limit = (0.12, 0.12, 0.12)

    use_physical_thrusters = False
    thruster_time_constant = 0.08
    thruster_command_delay_steps = 2
    # None preserves the historical asset and all existing checkpoint results.
    thruster_positions = None
    robot_geometry_version = "legacy-v1"
    robot_camera_profile = "legacy-v1"
    # Experimental rear-jet/boundary loss stress model, NOT calibrated T200 data.
    # Off in every existing task/checkpoint/evaluation unless explicitly enabled.
    enable_boundary_effects = False
    boundary_seabed_loss = 0.20
    boundary_wall_loss = 0.20
    boundary_range_diameters = 10.0

    water_density = 1025.0
    acceleration_filter = 0.2
    max_relative_linear_speed = 4.0
    max_relative_angular_speed = 8.0
    max_hydrodynamic_force = 250.0
    max_hydrodynamic_torque = 40.0
    max_valid_body_linear_speed = 7.0
    max_valid_body_angular_speed = 25.0
    max_valid_joint_speed = 6.0
    current_speed_range = (0.0, 0.22)
    current_vertical_range = (-0.025, 0.025)
    turbulence_theta = 0.8
    turbulence_sigma = 0.045
    turbulence_limit = 0.12
    volume_scale_range = (0.97, 1.03)
    damping_scale_range = (0.85, 1.15)
    added_mass_scale_range = (0.90, 1.10)

    # The panel workspace is centred on the Alpha 5 tool frame; the ranges span
    # positions that require both arm correction and vehicle station keeping.
    button_y_range = (-0.17, -0.03)
    button_z_range = (0.70, 0.82)
    button_pressed_threshold = 0.004
    button_max_travel = 0.010
    contact_threshold = 0.0004
    approach_threshold = 0.09
    tool_contact_distance = 0.13
    tool_contact_alignment = 0.70
    success_max_attitude_error = 0.25
    success_max_angular_speed = 0.35
    success_hold_steps = 4
    success_max_travel = float("inf")
    success_max_tool_speed = float("inf")
    max_base_distance = 2.5
    min_base_height = 0.25
    max_base_height = 2.0
    max_base_attitude_error = 0.75
    invalid_state_penalty = -1.0

    reward_position = 4.0
    reward_precision = 8.0
    reward_alignment = 2.0
    reward_contact = 10.0
    reward_press_progress = 18.0
    reward_press = 12.0
    reward_success = 35.0
    reward_distance_progress = 18.0
    penalty_action = -0.025
    penalty_action_rate = -0.015
    penalty_base_velocity = -0.04
    penalty_joint_velocity = -0.01
    penalty_attitude = -0.40

    def __post_init__(self) -> None:
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(-1.10, -1.60, 1.70),
            lookat=(0.38, -0.05, 0.72),
            background_color=(0.005, 0.045, 0.075),
        )
