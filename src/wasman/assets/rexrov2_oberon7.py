"""Native-scale work-class ROV with a different six-axis arm and parallel jaws.

The fixed-base config is an asset-validation fixture, NOT a benchmark platform
controller. Do not substitute it into BlueROV tasks: masses, reach, action
dimensions, thrust allocation and the coupled added-mass matrix differ.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

REXROV2_OBERON7_ASSET_DIR = Path(__file__).parent / "data/robots/rexrov2_oberon7"
REXROV2_OBERON7_URDF_PATH = REXROV2_OBERON7_ASSET_DIR / "rexrov2_oberon7.urdf"
REXROV2_OBERON7_ARM_JOINTS = tuple(
    "oberon_" + name for name in ("azimuth", "shoulder", "elbow", "roll", "pitch", "wrist")
)
REXROV2_OBERON7_GRIPPER_JOINTS = ("oberon_finger_left_joint", "oberon_finger_right_joint")

# Limits from freshly expanded pinned xacro, not the stale model.urdf snapshot.
_EFFORT = dict(zip(REXROV2_OBERON7_ARM_JOINTS, (200.0, 5000.0, 300.0, 1000.0, 50.0, 10.0), strict=True))
_VELOCITY = dict(zip(REXROV2_OBERON7_ARM_JOINTS, (0.17, 0.17, 0.15, 0.25, 0.30, 0.15), strict=True))

REXROV2_OBERON7_PREVIEW_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(REXROV2_OBERON7_URDF_PATH),
        fix_base=True,
        fix_root_link=True,
        merge_fixed_joints=False,
        self_collision=False,
        robot_type="Mobile Manipulators",
        run_multi_physics_conversion=False,
        articulation_props=PhysxArticulationCfg(
            articulation_enabled=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=2,
        ),
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False),
            PhysxRigidBodyCfg(disable_gravity=True),
        ],
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # At the native zero configuration the wrist points down: leave the
        # original fingers above the floor, rather than spawning in contact.
        pos=(0.0, 0.0, 1.8),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    joint_ordering=(*REXROV2_OBERON7_ARM_JOINTS, *REXROV2_OBERON7_GRIPPER_JOINTS),
    soft_joint_pos_limit_factor=0.96,
    actuators={
        "oberon_arm": ImplicitActuatorCfg(
            joint_names_expr=list(REXROV2_OBERON7_ARM_JOINTS),
            actuator_effort_limit=_EFFORT,
            joint_effort_limit=_EFFORT,
            actuator_velocity_limit=_VELOCITY,
            joint_velocity_limit=_VELOCITY,
            # The native hydraulic-arm URDF retains up to 150 Nm Coulomb
            # friction; a tiny-robot gain would stall this asset smoke.
            stiffness=3000.0,
            damping=120.0,
        ),
        "oberon_parallel_jaws": ImplicitActuatorCfg(
            joint_names_expr=list(REXROV2_OBERON7_GRIPPER_JOINTS),
            actuator_effort_limit=30.0,
            joint_effort_limit=30.0,
            actuator_velocity_limit=0.15,
            joint_velocity_limit=0.15,
            stiffness=400.0,
            damping=30.0,
        ),
    },
)
