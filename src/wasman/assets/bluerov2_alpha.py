"""BlueROV2 Heavy with a Reach Alpha 5 underwater manipulator."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from wasman.assets.open_geometry import PROFILE, public_robot_urdf, selected_profile
from wasman.physics import LinkHydrodynamics

BLUEROV2_ALPHA_ASSET_DIR = Path(__file__).parent / "data" / "robots" / "bluerov2_alpha"
BLUEROV2_ALPHA_URDF_PATH = BLUEROV2_ALPHA_ASSET_DIR / "bluerov2_alpha.urdf"
if selected_profile() == PROFILE:
    BLUEROV2_ALPHA_URDF_PATH = public_robot_urdf(BLUEROV2_ALPHA_URDF_PATH)

BLUEROV2_ALPHA_ARM_JOINT_NAMES = (
    "alpha_axis_e",
    "alpha_axis_d",
    "alpha_axis_c",
    "alpha_axis_b",
)
BLUEROV2_ALPHA_GRIPPER_JOINT_NAME = "alpha_axis_a"

BLUEROV2_ALPHA_BODY_NAMES = (
    "base_link",
    "alpha_m3_inline_link",
    "alpha_m2_1_1_link",
    "alpha_m2_joint_link",
    "alpha_m2_1_2_link",
    "alpha_m1_link",
    "alpha_jaw_base_link",
    "alpha_push_rod_link",
    "alpha_left_finger_link",
    "alpha_right_finger_link",
    "alpha_tool_link",
)


def _neutral_volume(mass: float) -> float:
    return mass / 1025.0


def _arm_hydrodynamics(name: str, mass: float, length: float = 0.08) -> LinkHydrodynamics:
    """Conservative diagonal coefficients for a slender flooded arm link."""
    lateral = max(0.04, 4.0 * length)
    return LinkHydrodynamics(
        name=name,
        volume=_neutral_volume(mass),
        center_of_buoyancy=(0.0, 0.0, 0.0),
        added_mass=(0.08 * mass, 0.45 * mass, 0.45 * mass, 0.01, 0.02, 0.02),
        linear_damping=(0.12, lateral, lateral, 0.01, 0.03, 0.03),
        quadratic_damping=(0.25, 2.5 * lateral, 2.5 * lateral, 0.02, 0.08, 0.08),
    )


# Base values are BlueROV-class coefficients; arm coefficients are geometric
# estimates.  All ranges are intentionally exposed to physics randomization.
BLUEROV2_ALPHA_HYDRODYNAMICS = (
    LinkHydrodynamics(
        name="base_link",
        volume=_neutral_volume(13.0),
        center_of_buoyancy=(0.0, 0.0, 0.035),
        added_mass=(5.5, 12.7, 14.6, 0.12, 0.12, 0.20),
        linear_damping=(4.0, 6.0, 7.0, 0.20, 0.25, 0.30),
        quadratic_damping=(18.0, 32.0, 38.0, 0.80, 1.00, 1.20),
    ),
    _arm_hydrodynamics("alpha_m3_inline_link", 0.341, 0.15),
    _arm_hydrodynamics("alpha_m2_1_1_link", 0.194, 0.04),
    _arm_hydrodynamics("alpha_m2_joint_link", 0.429, 0.15),
    _arm_hydrodynamics("alpha_m2_1_2_link", 0.115, 0.04),
    _arm_hydrodynamics("alpha_m1_link", 0.200, 0.10),
    _arm_hydrodynamics("alpha_jaw_base_link", 0.023, 0.03),
    _arm_hydrodynamics("alpha_push_rod_link", 0.0001, 0.01),
    _arm_hydrodynamics("alpha_left_finger_link", 0.055, 0.09),
    _arm_hydrodynamics("alpha_right_finger_link", 0.055, 0.09),
    _arm_hydrodynamics("alpha_tool_link", 0.0001, 0.01),
)


BLUEROV2_ALPHA_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(BLUEROV2_ALPHA_URDF_PATH),
        fix_base=False,
        fix_root_link=False,
        merge_fixed_joints=False,
        self_collision=False,
        robot_type="Mobile Manipulators",
        run_multi_physics_conversion=False,
        articulation_props=PhysxArticulationCfg(
            articulation_enabled=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False),
            PhysxRigidBodyCfg(
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=5.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.5,
                enable_gyroscopic_forces=True,
                disable_gravity=False,
            ),
        ],
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.82),
        joint_pos={
            "alpha_axis_e": 3.14,
            "alpha_axis_d": 0.15,
            "alpha_axis_c": 1.62,
            "alpha_axis_b": 0.15,
            "alpha_axis_a": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    joint_ordering=(
        *BLUEROV2_ALPHA_ARM_JOINT_NAMES,
        BLUEROV2_ALPHA_GRIPPER_JOINT_NAME,
        "alpha_left_finger_joint",
        "alpha_right_finger_joint",
    ),
    body_ordering=BLUEROV2_ALPHA_BODY_NAMES,
    soft_joint_pos_limit_factor=0.96,
    actuators={
        "alpha_arm": ImplicitActuatorCfg(
            joint_names_expr=list(BLUEROV2_ALPHA_ARM_JOINT_NAMES),
            actuator_effort_limit=9.0,
            joint_effort_limit=9.0,
            actuator_velocity_limit=0.5,
            joint_velocity_limit=0.75,
            stiffness=45.0,
            damping=4.0,
            armature=0.002,
        ),
        "alpha_gripper": ImplicitActuatorCfg(
            joint_names_expr=[BLUEROV2_ALPHA_GRIPPER_JOINT_NAME],
            actuator_effort_limit=10.0,
            joint_effort_limit=10.0,
            actuator_velocity_limit=0.05,
            joint_velocity_limit=0.08,
            stiffness=600.0,
            damping=20.0,
            armature=0.0001,
        ),
        "alpha_gripper_mimic": ImplicitActuatorCfg(
            joint_names_expr=["alpha_left_finger_joint", "alpha_right_finger_joint"],
            actuator_effort_limit=1.0,
            joint_effort_limit=1.0,
            actuator_velocity_limit=2.0,
            joint_velocity_limit=2.0,
            stiffness=0.0,
            damping=0.0,
            armature=0.0001,
        ),
    },
)
