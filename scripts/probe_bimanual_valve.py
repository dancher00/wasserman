"""Development-only twin-arm RotateValve probe; native physical contacts."""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--isolate-fixture", action="store_true")
parser.add_argument("--dry", action="store_true")
parser.add_argument("--fixture", choices=["native", "large"], default="large")
parser.add_argument("--mode", choices=["free", "support", "two-hands"], default="free")
parser.add_argument("--seeds", type=int, nargs="+", required=True)
parser.add_argument("--seconds", type=float, default=180.0)
parser.add_argument("--dt", type=float, default=1 / 240)
parser.add_argument("--solver-type", type=int, choices=[0, 1], default=0, help="Numerical solver: 0 PGS, 1 TGS")
parser.add_argument("--grasp-effort", type=float, help="Command-level jaw closing effort in Nm; default is position closure")
parser.add_argument("--feedback-frame", choices=["world", "level"], default="world")
parser.add_argument("--solver-position-iterations", type=int, default=12)
parser.add_argument("--solver-velocity-iterations", type=int, default=2)
parser.add_argument("--balance-initial-load", action="store_true")
parser.add_argument("--hold-reset-commands", action="store_true", help="Isolate startup dynamics with constant joint targets")
parser.add_argument("--spawn-fixture-at-reset-pose", action="store_true", help="Diagnostic: avoid teleporting fixed/kinematic fixtures after warmup")
parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bimanual_valve_development"))
parser.add_argument("--audit-only", action="store_true", help="Check imported masses/COM, without integration")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["none"], enable_cameras=False)
args = parser.parse_args()
launcher = AppLauncher(args)

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import wasman.tasks  # noqa: E402, F401
from wasman.controllers.station_keeping import (  # noqa: E402
    BatchedStationKeepingController,
    StationKeepingGains,
    quaternion_angle_error_xyzw,
)
from wasman.physics.hydrodynamics import quat_apply_inverse_xyzw, quat_apply_xyzw  # noqa: E402
from wasman.physics.rexrov2 import RexThrusters  # noqa: E402


def main():
    if args.seconds < 15 or args.dt <= 0 or args.dt > 1 / 60:
        raise ValueError("Use >=15 seconds and dt<=1/60")
    if not (1 <= args.solver_position_iterations <= 255 and 0 <= args.solver_velocity_iterations <= 255):
        raise ValueError("Invalid PhysX iteration count")
    if (args.output_dir / "report.json").exists():
        raise FileExistsError("Preserve the previous diagnostic; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_script.py").write_bytes(Path(__file__).read_bytes())
    source_root = Path(__file__).resolve().parents[1]
    source_files = [
        "scripts/probe_bimanual_valve.py",
        "src/wasman/controllers/bimanual_turn_plan.py",
        "src/wasman/controllers/bimanual_control.py",
        "src/wasman/controllers/bimanual_kinematics.py",
        "src/wasman/physics/rexrov2_bimanual.py",
        "src/wasman/physics/contact_wrench.py",
        "src/wasman/assets/rexrov2_bimanual.py",
        "src/wasman/physics/rexrov2_articulated.py",
        "src/wasman/physics/rexrov2.py",
        "src/wasman/controllers/station_keeping.py",
        "configs/studies/rex-centered-button-pose.json",
    ]
    source_hashes = {}
    for source in source_files:
        # Keep the plan at the root for independent phase replay, and retain
        # full relative paths so asset/model modules cannot overwrite each other.
        if Path(source).name == "bimanual_turn_plan.py":
            (args.output_dir / Path(source).name).write_bytes((source_root / source).read_bytes())
        saved = args.output_dir / "source" / source
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes((source_root / source).read_bytes())
        source_hashes[source] = hashlib.sha256(saved.read_bytes()).hexdigest()
    from wasman.assets.rexrov2_bimanual import BIMANUAL_REX_CFG
    from wasman.physics.rexrov2_bimanual import URDF, load_bimanual_parameters

    one = json.loads(Path("configs/studies/rex-centered-button-pose.json").read_text())
    qone = one["targets"]["wall"]["q"]
    names = one["joint_names"]
    init = {side + "_" + n: v for side in ["left", "right"] for n, v in zip(names, qone)}
    from wasman.controllers.bimanual_kinematics import BimanualKinematics
    from wasman.controllers.bimanual_turn_plan import BASE_POSITION, OPEN_JAW, RELEASE_JAW, ContactPlanClock, mode_plan

    w = BimanualKinematics()
    q = np.zeros(w.model.nq)
    for n, v in init.items():
        q[w.model.joints[w.model.getJointId(n)].idx_q] = v
    poses, rots, _, _ = mode_plan(0.0, args.mode)
    if args.fixture == "native":
        poses = {"left": np.array([2.9, 0.5, 1.1]), "right": np.array([2.9, -0.5, 1.1])}
    if args.mode != "two-hands":
        poses["left"] = np.array([2.9, 0.5, 1.1])
    base_x = float(BASE_POSITION[0]) if args.fixture == "large" else 0.4
    for side in ["left", "right"]:
        q, _ = w.solve_arm(side, poses[side] - np.array([base_x, 0, 1.5]), rots[side], q)
    for side in w.indices:
        q[w.indices[side][-2:]] = OPEN_JAW
    init = {n: float(q[w.model.joints[w.model.getJointId(n)].idx_q]) for n in init}
    params = load_bimanual_parameters(init)
    source_urdf = URDF
    prototype_cfg = BIMANUAL_REX_CFG.copy()
    prototype_cfg.spawn.articulation_props.solver_position_iteration_count = args.solver_position_iterations
    prototype_cfg.spawn.articulation_props.solver_velocity_iteration_count = args.solver_velocity_iterations
    prototype_cfg.init_state.joint_pos = init
    cfg, _ = resolve_task_config("Wasman-Underwater-OpenHatch-Direct", "", overrides=("physics=isaacsim_physx",))
    cfg.sim.dt, cfg.sim.visualizer_cfgs = args.dt, []
    cfg.sim.physics.solver_type = args.solver_type
    sim = sim_utils.SimulationContext(cfg.sim)
    num_envs = len(args.seeds) + 1
    for index in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{index}", "Xform", translation=(0, index * 12.0, 0))
    robot_cfg = prototype_cfg.copy()
    robot_cfg.prim_path = "/World/envs/env_[0-9]+/Robot"
    robot_cfg.spawn.fix_base = robot_cfg.spawn.fix_root_link = False
    for properties in robot_cfg.spawn.rigid_props:
        if hasattr(properties, "disable_gravity"):
            properties.disable_gravity = False
            properties.linear_damping = properties.angular_damping = 0.0
            properties.enable_gyroscopic_forces = True
    robot = Articulation(robot_cfg)
    button_cfg, _ = resolve_task_config(
        "Wasman-Underwater-RotateValve-T200-Direct", "", overrides=("physics=isaacsim_physx",)
    )
    fixture = button_cfg.scene.button.copy()
    fixture.prim_path = "/World/envs/env_[0-9]+/Button"
    if args.fixture == "large":
        fixture.spawn.usd_path = str(Path(fixture.spawn.usd_path).with_name("valve_bimanual_4x.usda"))
        # Explicit large-fixture engineering assumptions: viscous/seal torque
        # follows shaft radius cubed (same shear stress), not an outcome fit.
        fixture.actuators["passive"].damping = 0.08 * 4**3
        fixture.actuators["passive"].friction = 0.012 * 4**3
        fixture.actuators["passive"].dynamic_friction = 0.008 * 4**3
    fixture_reset_position = (3 + 0.19385145 * 4, 0, 1.1) if args.fixture == "large" else (3.19385145, -0.441, 1.1)
    fixture_reset_position = (fixture_reset_position[0] + (10 if args.isolate_fixture else 0), *fixture_reset_position[1:])
    if args.spawn_fixture_at_reset_pose:
        fixture.init_state.pos = fixture_reset_position
        fixture.init_state.rot = (0, 0, 0, 1)
    button = Articulation(fixture)
    from isaaclab.assets import RigidObject, RigidObjectCfg

    rail = RigidObject(
        RigidObjectCfg(
            prim_path="/World/envs/env_[0-9]+/Rail",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(3.0 + (10 if args.isolate_fixture else 0), 2.5 if args.mode == "two-hands" else 0.5, 1.1)
                if args.spawn_fixture_at_reset_pose else (0, 0, 0),
            ),
            spawn=sim_utils.CylinderCfg(
                radius=0.018,
                height=0.28,
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.6, 0.65)),
            ),
        )
    )

    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import Usd, UsdPhysics, UsdGeom

    paths = {
        p.GetName(): str(p.GetPath())
        for p in Usd.PrimRange(sim_utils.get_current_stage().GetPrimAtPath("/World/envs/env_0/Robot"))
        if p.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    sensors = [
        ContactSensor(
            ContactSensorCfg(
                prim_path=paths[name].replace("env_0", "env_[0-9]+"),
                filter_prim_paths_expr=[
                    "/World/envs/env_[0-9]+/Button/handle_link",
                    "/World/envs/env_[0-9]+/Button/base_link",
                    "/World/envs/env_[0-9]+/Rail",
                ],
                track_contact_points=True,
                track_friction_forces=True,
                update_period=0,
                max_contact_data_count_per_prim=1024,
            )
        )
        for name in params.names
    ]
    stage = sim_utils.get_current_stage()
    def collider_snapshot():
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        result = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                result.append({"path": str(prim.GetPath()), "min": list(box.GetMin()), "max": list(box.GetMax())})
        return result
    spawn_audit = {"colliders_before_warmup": collider_snapshot()}
    sim.reset()
    spawn_audit["root_positions_after_warmup"] = robot.data.root_link_pos_w.torch.cpu().tolist()
    spawn_audit["root_velocities_after_warmup"] = robot.data.root_com_vel_w.torch.cpu().tolist()
    if robot.is_fixed_base:
        raise RuntimeError("Diagnostic must have a genuinely floating base")
    device = sim.device

    def tensor(value):
        return torch.tensor(value, dtype=torch.float32, device=device)

    origins = tensor([[0, index * 12, 0] for index in range(num_envs)])
    button_positions = origins + tensor(
        [3 + 0.19385145 * 4, 0, 1.1] if args.fixture == "large" else [3.19385145, -0.441, 1.1]
    )
    rail_pose = torch.cat((origins + tensor([3.0, 0.5, 1.1]), tensor([[0, 0, 0, 1]]).repeat(num_envs, 1)), -1)
    if args.mode == "two-hands":
        rail_pose[:, 1] += 2
    rail.write_root_pose_to_sim_index(root_pose=rail_pose)
    rail.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num_envs, 6, device=device))

    fixture_pose = torch.cat((button_positions, tensor([[0, 0, 0, 1]]).repeat(num_envs, 1)), -1)
    if args.isolate_fixture:
        fixture_pose[:, 0] += 10
        rail_pose[:, 0] += 10
        rail.write_root_pose_to_sim_index(root_pose=rail_pose)
    button.write_root_pose_to_sim_index(root_pose=fixture_pose)
    button.write_joint_position_to_sim_index(position=torch.zeros(num_envs, 1, device=device))
    button.write_joint_velocity_to_sim_index(velocity=torch.zeros(num_envs, 1, device=device))
    button.reset()
    targets = origins + tensor([base_x, 0, 1.5])
    target_quaternion = tensor([[0, 0, 0, 1]]).repeat(num_envs, 1)
    initial_pose = torch.cat((targets.clone(), target_quaternion.clone()), dim=-1)
    from wasman.controllers.held_arm_button_study import reset_offsets

    initial_pose[:-1, :3] += tensor(reset_offsets(args.seeds))
    initial_pose[:-1, 3:] = tensor(Rotation.from_euler("xyz", [0.0, 0.0, 0.0]).as_quat())
    robot.write_root_pose_to_sim_index(root_pose=initial_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num_envs, 6, device=device))
    arm_targets = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=arm_targets)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(arm_targets))
    robot.reset()
    robot.update(args.dt)
    (args.output_dir / "spawn_audit.json").write_text(json.dumps(spawn_audit, indent=2) + "\n")
    body_ids = robot.find_bodies(params.names, preserve_order=True)[0]
    base_id = body_ids[0]
    imported_mass = robot.data.body_mass.torch[:, body_ids].cpu().numpy()
    imported_com = robot.data.body_com_pose_b.torch[:, body_ids, :3].cpu().numpy()
    mass_matches = bool(np.allclose(imported_mass, params.mass[None], rtol=2e-6, atol=1e-6))
    com_matches = bool(np.allclose(imported_com, params.com[None], atol=1e-6))
    base_com_zero = bool(np.allclose(imported_com[:, 0], 0, atol=1e-7))
    import_audit = {
        "kind": "independent actual PhysX mass/COM import audit; no dynamics acceptance",
        "passed": mass_matches and com_matches and base_com_zero,
        "mass_matches_native_urdf": mass_matches,
        "com_matches_native_urdf": com_matches,
        "base_com_at_link_origin": base_com_zero,
        "body_names": params.names,
        "actual_physx_mass_kg": imported_mass.tolist(),
        "actual_physx_com_in_link_m": imported_com.tolist(),
        "native_mass_kg": params.mass.tolist(),
        "native_com_in_link_m": params.com.tolist(),
        "nominal_held_arm_rigid_inertia": params.rigid_composite.tolist(),
        "fixed_base": bool(robot.is_fixed_base),
        "urdf_sha256": hashlib.sha256(source_urdf.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    iteration_attributes = {}
    for prim in Usd.PrimRange(sim_utils.get_current_stage().GetPrimAtPath("/World/envs/env_0/Robot")):
        position_count = prim.GetAttribute("physxArticulation:solverPositionIterationCount")
        velocity_count = prim.GetAttribute("physxArticulation:solverVelocityIterationCount")
        if position_count and position_count.Get() is not None:
            iteration_attributes[str(prim.GetPath())] = [position_count.Get(), velocity_count.Get()]
    import_audit["authored_solver_iterations"] = iteration_attributes
    import_audit["effective_joint_properties"] = {
        key: getattr(robot.data, key).torch[0].cpu().tolist()
        for key in ("joint_stiffness", "joint_damping", "joint_friction_coeff",
                    "joint_dynamic_friction_coeff", "joint_viscous_friction_coeff")
    }
    if not iteration_attributes or not all(
        v == [args.solver_position_iterations, args.solver_velocity_iterations] for v in iteration_attributes.values()
    ):
        raise RuntimeError(f"Solver iteration authoring mismatch: {iteration_attributes}")
    # At the identity base orientation, public PhysX root columns are the same
    # world/body axes used by the nominal Pinocchio composite inertia.
    actual_base_mass = robot.data.mass_matrix.torch[:, :6, :6].cpu().numpy()
    import_audit["base_mass_matrix_max_error"] = float(np.abs(actual_base_mass - params.rigid_composite).max())
    import_audit["base_mass_matrix_matches"] = bool(
        np.allclose(actual_base_mass, params.rigid_composite, rtol=2e-5, atol=0.03)
    )
    import_audit["passed"] &= import_audit["base_mass_matrix_matches"]
    (args.output_dir / "import_audit.json").write_text(json.dumps(import_audit, indent=2) + "\n")
    if not import_audit["passed"]:
        raise RuntimeError("Imported mass/COM do not support the nominal added-inertia closure")
    wheel_id = button.find_bodies("handle_link")[0][0]
    expected_wheel_mass = 0.18 * (64 if args.fixture == "large" else 1)
    wheel_mass = float(button.data.body_mass.torch[0, wheel_id])
    fixture_audit = {
        "mode": args.mode,
        "mass_kg": wheel_mass,
        "expected_mass_kg": expected_wheel_mass,
        "actual_link_inertia": button.data.body_inertia.torch[0, wheel_id].cpu().tolist(),
        "fixture_sha256": hashlib.sha256(Path(fixture.spawn.usd_path).read_bytes()).hexdigest(),
        "passed": abs(wheel_mass - expected_wheel_mass) < 1e-4,
    }
    (args.output_dir / "fixture_audit.json").write_text(json.dumps(fixture_audit, indent=2) + "\n")
    if not fixture_audit["passed"]:
        raise RuntimeError("Fixture mass import mismatch")
    # Independent native Jacobian/kinematics check for both branches.
    import pinocchio as pin

    pm = pin.buildModelFromUrdf(str(source_urdf), pin.JointModelFreeFlyer())
    pd = pm.createData()
    pq = pin.neutral(pm)
    pq[:3] = robot.data.body_link_pos_w.torch[0, base_id].cpu().numpy()
    pq[3:7] = robot.data.body_link_quat_w.torch[0, base_id].cpu().numpy()
    for j, n in enumerate(robot.joint_names):
        pq[pm.joints[pm.getJointId(n)].idx_q] = float(robot.data.joint_pos.torch[0, j])
    jac_audit = {}
    for side in ["left", "right"]:
        name = side + "_oberon_end_effector"
        frame = pm.getFrameId(name)
        bid = robot.find_bodies(name)[0][0]
        actual = robot.data.body_link_jacobian_w.torch[0, bid, :, 6:].cpu().numpy()
        pred = pin.computeFrameJacobian(pm, pd, pq, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[
            :, [pm.joints[pm.getJointId(n)].idx_v for n in robot.joint_names]
        ]
        jac_audit[side] = {
            "max_error": float(abs(actual - pred).max()),
            "actual": actual.tolist(),
            "predicted": pred.tolist(),
        }
    (args.output_dir / "jacobian_audit.json").write_text(json.dumps(jac_audit, indent=2) + "\n")
    if max(v["max_error"] for v in jac_audit.values()) > 1e-4:
        raise RuntimeError("Simulator/URDF Jacobians disagree")
    if args.audit_only:
        print(json.dumps(import_audit), flush=True)
        return
    from wasman.physics.rexrov2_articulated import ArticulatedRexHydrodynamics

    hydro = ArticulatedRexHydrodynamics(params, device, robot.joint_names, urdf=source_urdf)
    initial_arm_targets = arm_targets.clone()
    thrusters = RexThrusters(params, num_envs, args.dt, device)
    controller = BatchedStationKeepingController(
        num_envs=num_envs,
        dt=args.dt,
        device=device,
        gains=StationKeepingGains(
            position_kp=(1400, 1400, 2200),
            position_kd=(3000, 3600, 6500),
            position_ki=(60, 60, 80),
            rotation_kp=(2500, 3500, 2200),
            rotation_kd=(1700, 2300, 1300),
            rotation_ki=(40, 40, 35),
            max_force=(4000, 4000, 5000),
            max_torque=(3000, 3000, 3000),
            position_integral_limit=(1, 1, 1),
            rotation_integral_limit=(0.5, 0.5, 0.5),
        ),
    )
    currents = torch.zeros(num_envs, 3, device=device)
    gravity = torch.zeros(num_envs, len(body_ids), 3, device=device)
    gravity[..., 2] = -hydro.mass * params.gravity
    motor_positions = tensor(params.thruster_positions)
    motor_directions = tensor(params.thruster_directions)
    records = {
        key: []
        for key in (
            "time",
            "position",
            "quaternion",
            "position_error",
            "attitude_error",
            "speed",
            "angular_speed",
            "arm_drift",
            "joint_position",
            "motor_force",
            "motor_scale",
            "added_wrench",
        )
    }
    for key in (
        "button_travel",
        "tool_position",
        "tool_alignment",
        "contact_force",
        "success_counter",
        "distance",
        "target_tool",
    ):
        records[key] = []
    success_counter = torch.zeros(num_envs, device=device, dtype=torch.long)
    ever_success = torch.zeros(num_envs, device=device, dtype=torch.bool)
    max_travel = torch.zeros(num_envs, device=device)
    tool_id = robot.find_bodies("right_oberon_end_effector")[0][0]
    plunger_id = button.find_bodies("handle_link")[0][0]
    initial_tool_position = (
        (
            robot.data.body_link_pos_w.torch[:, tool_id]
            + quat_apply_xyzw(robot.data.body_link_quat_w.torch[:, tool_id], tensor([0.16, 0, 0]).expand(num_envs, 3))
            - origins
        )
        .cpu()
        .tolist()
    )
    from wasman.controllers.bimanual_control import BimanualController

    tools = {s: robot.find_bodies(s + "_oberon_end_effector")[0][0] for s in ["left", "right"]}
    ik = BimanualController(robot, base_id, tools, targets, arm_targets, args.mode,
                           grasp_effort=args.grasp_effort, feedback_frame=args.feedback_frame,
                           balance_initial_load=args.balance_initial_load)
    plan_clock = ContactPlanClock(num_envs, args.mode)
    from wasman.physics.contact_wrench import ResolvedContactWrench
    contact_readers = [ResolvedContactWrench(sensor) for sensor in sensors]
    hand_body_ids = [[j for j, name in enumerate(params.names) if name.startswith(side + "_")]
                     for side in ["left", "right"]]
    for key in [
        "joint_target",
        "joint_velocity",
        "arm_excursion",
        "base_acceleration_predicted",
        "body_twist",
        "added_reaction_bias",
        "plunger_contact_force",
        "normal_contact_by_body",
        "friction_contact_by_body",
        "contact_latched",
        "planned_tool",
        "base_reference",
    ]:
        records[key] = []
    for key in ["valve_angle", "left_tool_position", "rail_contact_force"]:
        records[key] = []
    for key in ["plan_time", "plan_waiting", "ik_residual", "left_target_tool", "jaw_command", "valve_angular_speed"]:
        records[key] = []
    records["unfiltered_normal_contact_by_body"] = []
    records["wheel_torque_by_hand_Nm"] = []
    started = time.monotonic()

    def total_at_base(force_world, torque_world, lever, base_quaternion):
        force = force_world.sum(1)
        torque = (torque_world + torch.cross(lever, force_world, dim=-1)).sum(1)
        return torch.cat(
            (quat_apply_inverse_xyzw(base_quaternion, force), quat_apply_inverse_xyzw(base_quaternion, torque)), dim=-1
        )

    micro = []
    for step in range(round(args.seconds / args.dt)):
        now = step * args.dt
        angle_ref = np.deg2rad(175) * min(1.0, max(0.0, (now - 20) / 35))
        R = Rotation.from_rotvec([angle_ref, 0, 0]).as_matrix()
        approach = min(1.0, max(0.0, (now - 5) / 7))
        desired_tool = origins + tensor([3.0, -0.441, 1.1]) + tensor(R @ np.array([0, -0.059, 0]))
        desired_tool[:, 0] -= 0.10 * (1 - approach)
        left_goal = origins + tensor([3.0 - 0.10 * (1 - approach), 0.5, 1.1])
        if args.mode == "free":
            left_goal = origins + tensor([2.9, 0.5, 1.1])
        button.update(args.dt)
        for sensor in sensors:
            sensor.update(args.dt)
        position = robot.data.body_link_pos_w.torch[:, base_id]
        wrenches = [reader.evaluate(position) for reader in contact_readers]
        force_by_target = torch.stack([value[0] for value in wrenches], 1)
        torque_by_target = torch.stack([value[1] for value in wrenches], 1)
        contact_f = force_by_target.sum(2)
        contact_t = torque_by_target.sum(2)
        wheel_lever = position - button.data.body_com_pos_w.torch[:, plunger_id]
        wheel_body_torque = -(torque_by_target[:, :, 0] + torch.cross(
            wheel_lever[:, None].expand(-1, len(sensors), -1), force_by_target[:, :, 0], dim=-1))
        wheel_torque = torch.stack([wheel_body_torque[:, ids].sum(1) for ids in hand_body_ids], 1)
        if step % round(1 / 30 / args.dt) == 0:
            reported_force = torch.stack([
                (sensor.data.normal_force_matrix_w.torch + sensor.data.friction_force_matrix_w.torch)[:, 0]
                for sensor in sensors], 1)
            if (reported_force - force_by_target).abs().max() > 0.01:
                raise RuntimeError("Resolved contact buffers disagree with aggregate sensor forces")
        quaternion = robot.data.body_link_quat_w.torch[:, body_ids]
        base_quaternion = quaternion[:, 0]
        velocity = robot.data.body_com_lin_vel_w.torch[:, body_ids]
        angular = robot.data.body_com_ang_vel_w.torch[:, body_ids]
        centers = robot.data.body_com_pos_w.torch[:, body_ids]
        joints = robot.data.joint_pos.torch
        for state in (position, quaternion, velocity, angular, centers, joints):
            if not torch.isfinite(state).all():
                np.savez_compressed(
                    args.output_dir / "failure-steps.npz", **{k: np.asarray([r[k] for r in micro]) for k in micro[0]}
                )
                raise RuntimeError(f"Non-finite state at step {step}")
        relative = torch.cat(
            (
                quat_apply_inverse_xyzw(quaternion, velocity - currents[:, None]),
                quat_apply_inverse_xyzw(quaternion, angular),
            ),
            dim=-1,
        )
        fluid = hydro.local_fluid(relative, quaternion)
        force_world = quat_apply_xyzw(quaternion, fluid[..., :3])
        torque_world = quat_apply_xyzw(quaternion, fluid[..., 3:])
        lever = centers - position[:, None]
        static_fluid = hydro.local_fluid(torch.zeros_like(relative), quaternion)
        static_force = quat_apply_xyzw(quaternion, static_fluid[..., :3])
        static_torque = quat_apply_xyzw(quaternion, static_fluid[..., 3:])
        static_wrench = total_at_base(static_force + gravity, static_torque, lever, base_quaternion)
        # Hold the requested world tool trajectory despite measured arm sag.
        if step % round(1 / 30 / args.dt) == 0:
            grip = {"right": 0.5 if now < 12 else 0.0, "left": 0.5 if args.mode == "free" or now < 12 else 0.0}
            goals = {"left": left_goal, "right": desired_tool}
            rots = {"left": np.eye(3), "right": R}
            if args.fixture == "large":
                actual_tcp = {
                    side: (
                        robot.data.body_link_pos_w.torch[:, bid]
                        + quat_apply_xyzw(
                            robot.data.body_link_quat_w.torch[:, bid], tensor([0.16, 0, 0]).expand(num_envs, 3)
                        )
                        - origins
                    )
                    .cpu()
                    .numpy()
                    for side, bid in tools.items()
                }
                actual_rot = {
                    side: Rotation.from_quat(robot.data.body_link_quat_w.torch[:, bid].cpu().numpy()).as_matrix()
                    for side, bid in tools.items()
                }
                normal = (
                    torch.stack([sensor.data.normal_force_matrix_w.torch[:, 0] for sensor in sensors], 1).cpu().numpy()
                )
                joint_cpu = joints.cpu().numpy()

                def ready(n, boundary, kind, sides):
                    positions, orientations, _, _ = mode_plan(boundary - 1e-6, args.mode)
                    for side in sides:
                        if kind == "pose":
                            if np.linalg.norm(actual_tcp[side][n] - positions[side]) > 0.015:
                                return False
                            if Rotation.from_matrix(orientations[side].T @ actual_rot[side][n]).magnitude() > 0.08:
                                return False
                        elif kind == "open":
                            if np.min(joint_cpu[n, ik.grip[side]]) < RELEASE_JAW - 0.03:
                                return False
                        else:
                            target = 2 if side == "left" and args.mode == "support" else 0
                            for finger in ["left", "right"]:
                                ids = [
                                    params.names.index(side + "_oberon_finger_" + finger),
                                    params.names.index(side + "_oberon_finger_tip_" + finger),
                                ]
                                if np.linalg.norm(normal[n, ids, target].sum(0)) < 5:
                                    return False
                    return True

                times = plan_clock.advance(ready)
                plans = [mode_plan(t, args.mode) for t in times]
                goals = {side: origins + tensor(np.array([p[0][side] for p in plans])) for side in tools}
                rots = {side: np.array([p[1][side] for p in plans]) for side in tools}
                grip = {side: np.array([p[2][side] for p in plans]) for side in tools}
                desired_tool = goals["right"]
                left_goal = goals["left"]
            targets, arm_targets = ik.targets(now, goals, rots, grip)
            arm_targets[-1] = initial_arm_targets[-1]
            if args.hold_reset_commands:
                arm_targets[:] = initial_arm_targets
        control_force, control_torque, _, _ = controller.compute(
            position_w=position,
            quaternion_w=base_quaternion,
            linear_velocity_w=velocity[:, 0],
            angular_velocity_w=angular[:, 0],
            target_position_w=targets,
            target_quaternion_w=target_quaternion,
        )
        command = torch.cat((control_force, control_torque), dim=-1) - static_wrench
        command[-1] = 0  # independent zero-thruster drift control, no stabilizing bypass
        realized = thrusters.step(command)
        non_added = total_at_base(force_world + gravity, torque_world, lever, base_quaternion) + realized
        # One-physics-step contact-wrench estimate; convergence must be audited.
        contact_wrench = torch.cat(
            (
                quat_apply_inverse_xyzw(base_quaternion, contact_f.sum(1)),
                quat_apply_inverse_xyzw(base_quaternion, contact_t.sum(1)),
            ),
            -1,
        )
        non_added += contact_wrench
        tool_position = robot.data.body_link_pos_w.torch[:, tool_id] + quat_apply_xyzw(
            robot.data.body_link_quat_w.torch[:, tool_id], tensor([0.16, 0, 0]).expand(num_envs, 3)
        )
        tool_axis = quat_apply_xyzw(
            robot.data.body_link_quat_w.torch[:, tool_id], tensor([1.0, 0, 0]).expand(num_envs, 3)
        )
        travel = button.data.joint_pos.torch[:, 0].clone()
        max_travel = torch.maximum(max_travel, travel)
        distance = (tool_position - button.data.body_link_pos_w.torch[:, plunger_id]).norm(dim=-1)
        pressed = travel >= np.deg2rad(170)
        # Preserve the signed 170-degree RotateValve criterion at 30 Hz.
        if step % round(1 / 30 / args.dt) == 0:
            success_counter = torch.where(pressed, success_counter + 1, 0)
            ever_success |= pressed
        body_twist = torch.cat(
            (
                quat_apply_inverse_xyzw(base_quaternion, velocity[:, 0]),
                quat_apply_inverse_xyzw(base_quaternion, angular[:, 0]),
            ),
            dim=-1,
        )
        hydro.update_articulation(joints, robot.data.joint_vel.torch, body_twist, args.dt)
        added, predicted_acceleration = hydro.close_added_mass(
            body_twist, quat_apply_inverse_xyzw(base_quaternion, currents), non_added
        )
        for wrench in (fluid, command, realized, non_added, added, thrusters.force):
            if not torch.isfinite(wrench).all():
                raise RuntimeError(f"Non-finite wrench at step {step}")
        if thrusters.force.abs().max() > params.max_thrust + 0.001:
            raise RuntimeError(f"Native motor force cap exceeded at step {step}")
        if step < 10:
            micro.append(
                {
                    k: v.detach().cpu().numpy().copy()
                    for k, v in dict(
                        position=position,
                        quaternion=base_quaternion,
                        joints=joints,
                        vel=velocity,
                        angular=angular,
                        joint_velocity=robot.data.joint_vel.torch,
                        joint_target=arm_targets,
                        unfiltered_contact=torch.stack(
                            [sensor.data.net_normal_forces_w.torch[:, 0] for sensor in sensors], 1
                        ),
                        fluid=fluid,
                        contact=contact_wrench,
                        added=added,
                        realized=realized,
                        predicted=predicted_acceleration,
                    ).items()
                }
            )
        force_world[:, 0] += quat_apply_xyzw(base_quaternion, added[:, :3])
        torque_world[:, 0] += quat_apply_xyzw(base_quaternion, added[:, 3:])
        if args.dry:
            force_world.zero_()
            torque_world.zero_()
            thrusters.force.zero_()
        composer = robot.instantaneous_wrench_composer
        composer.set_forces_and_torques_index(
            forces=force_world, torques=torque_world, body_ids=body_ids, is_global=True
        )
        for motor in range(6):
            direction_world = quat_apply_xyzw(base_quaternion, motor_directions[motor].expand(num_envs, 3))
            mount_world = position + quat_apply_xyzw(base_quaternion, motor_positions[motor].expand(num_envs, 3))
            composer.add_forces_and_torques_index(
                forces=(thrusters.force[:, motor : motor + 1] * direction_world).unsqueeze(1),
                positions=mount_world.unsqueeze(1),
                body_ids=[base_id],
                is_global=True,
            )
        if step % round(1 / 30 / args.dt) == 0:
            records["time"].append(now)
            values = {
                "distance": distance,
                "target_tool": desired_tool - origins,
                "planned_tool": desired_tool - origins,
                "base_reference": targets - origins,
                "button_travel": travel,
                "valve_angle": travel,
                "valve_angular_speed": button.data.joint_vel.torch[:, 0],
                "plan_time": tensor(plan_clock.time),
                "plan_waiting": tensor(plan_clock.waiting),
                "ik_residual": tensor(ik.ik_residual),
                "left_target_tool": left_goal - origins,
                "jaw_command": tensor(np.array([grip[s] for s in ["left", "right"]]).T)
                if args.fixture == "large"
                else tensor([[grip["left"], grip["right"]]] * num_envs),
                "left_tool_position": robot.data.body_link_pos_w.torch[:, tools["left"]]
                + quat_apply_xyzw(
                    robot.data.body_link_quat_w.torch[:, tools["left"]], tensor([0.16, 0, 0]).expand(num_envs, 3)
                )
                - origins,
                "rail_contact_force": torch.stack(
                    [sensor.data.normal_force_matrix_w.torch[:, 0, 2] for sensor in sensors], 1
                ),
                "tool_position": tool_position - origins,
                "tool_alignment": tool_axis[:, 0],
                "contact_force": contact_f.sum(1),
                "success_counter": success_counter,
                "position": position - origins,
                "quaternion": base_quaternion,
                "position_error": torch.linalg.vector_norm(position - targets, dim=-1),
                "attitude_error": quaternion_angle_error_xyzw(base_quaternion, target_quaternion),
                "speed": torch.linalg.vector_norm(velocity[:, 0], dim=-1),
                "angular_speed": torch.linalg.vector_norm(angular[:, 0], dim=-1),
                "arm_drift": (joints - arm_targets).abs().amax(-1),
                "joint_position": joints,
                "motor_force": thrusters.force,
                "motor_scale": thrusters.scale.squeeze(-1),
                "added_wrench": added,
                "normal_contact_by_body": torch.stack(
                    [sensor.data.normal_force_matrix_w.torch[:, 0] for sensor in sensors], 1
                ),
                "unfiltered_normal_contact_by_body": torch.stack(
                    [sensor.data.net_normal_forces_w.torch[:, 0] for sensor in sensors], 1
                ),
                "wheel_torque_by_hand_Nm": wheel_torque,
                "friction_contact_by_body": torch.stack(
                    [sensor.data.friction_force_matrix_w.torch[:, 0] for sensor in sensors], 1
                ),
                "contact_latched": ik.contact_latched,
                "joint_target": arm_targets,
                "joint_velocity": robot.data.joint_vel.torch,
                "arm_excursion": (joints - initial_arm_targets).abs().amax(-1),
                "base_acceleration_predicted": predicted_acceleration,
                "body_twist": body_twist,
                "added_reaction_bias": hydro.rigid_bias,
                "plunger_contact_force": torch.stack(
                    [sensor.data.normal_force_matrix_w.torch[:, 0, 0] for sensor in sensors], 1
                ).sum(1),
            }
            for key, value in values.items():
                records[key].append(value.detach().cpu().numpy().copy())
        if step % round(5 / args.dt) == 0:
            np.savez_compressed(
                args.output_dir / "progress-next.npz", **{key: np.asarray(value) for key, value in records.items()}
            )
            (args.output_dir / "progress-next.npz").replace(args.output_dir / "progress.npz")
            print(
                f"t={now:.1f}s travel={travel.tolist()} tool={tool_position[0].tolist()} "
                f"drift={(joints - arm_targets).abs().max().item()}",
                flush=True,
            )
        robot.actuators.target_command.set_position_index(value=arm_targets)
        # Passive valve: no motor torque/position target. Native gravity plus
        # the existing handle displaced volume at its COM.
        buoyancy = torch.zeros(num_envs, 1, 3, device=device)
        buoyancy[:, :, 2] = 1028 * 9.81 * 0.000060 * (4**3 if args.fixture == "large" else 1)
        button.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=buoyancy, torques=torch.zeros_like(buoyancy), body_ids=[plunger_id], is_global=True
        )
        button.write_data_to_sim()
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(args.dt)
    trace = {key: np.asarray(value) for key, value in records.items()}
    np.savez_compressed(args.output_dir / "startup-steps.npz",
                        **{key: np.asarray([row[key] for row in micro]) for key in micro[0]})
    report = dict(
        kind="Twin Oberon RotateValve development",
        mode=args.mode,
        fixture=args.fixture,
        control_mode=("measured-base IK" if args.feedback_frame == "world" else "nominal level-base IK")
        + " with bounded TCP feedback",
        grasp_effort_command_Nm=args.grasp_effort,
        gripper_command_model="shared mean-aperture closing bias with native differential stiffness and damping"
        if args.grasp_effort is not None else "common position closure",
        feedback_frame=args.feedback_frame,
        robot_solver_iterations=[args.solver_position_iterations, args.solver_velocity_iterations],
        initial_load_balanced=args.balance_initial_load,
        hold_reset_commands=args.hold_reset_commands,
        dry_diagnostic=args.dry,
        isolated_fixture_diagnostic=args.isolate_fixture,
        fixtures_spawned_at_reset_pose=args.spawn_fixture_at_reset_pose,
        hand_torque_note="Reaction torque on wheel from each arm, summing normal and friction forces at their separate reported points; world XYZ, shaft along +X. Sampled at 30 Hz; no hardware validation.",
        contact_moment_model="resolved normal/friction application points; preserves force couples",
        initial_holding_torque_Nm=ik.initial_holding_torque.tolist(),
        closure="articulated",
        base_reference_m=[base_x, 0, 1.5],
        plan_clock="per-environment measured pose/contact guards",
        joint_names=robot.joint_names,
        contact_body_names=params.names,
        contact_targets=["handle_link", "base_link", "rail"],
        limits="base-only added mass; lagged measured joint acceleration/contact; no hardware calibration",
        source_files=source_hashes,
        trace_units={
            "valve_angle": "rad",
            "button_travel": "rad (legacy alias)",
            "position": "m",
            "plan_time": "virtual seconds",
            "time": "physical seconds",
        },
        successes=ever_success.tolist(),
        max_angle_rad=max_travel.cpu().tolist(),
        max_arm_drift_rad=trace["arm_drift"].max(0).tolist(),
        contact_closure="one-step measured contact force and resolved per-point moment; needs timestep convergence",
        dt=args.dt,
        solver_type=args.solver_type,
        seconds=args.seconds,
        fixed_base=False,
        gravity_enabled=True,
        runtime_pose_overwrites=0,
        zero_motor_control_index=num_envs - 1,
        seeds=args.seeds,
        initial_offsets_m=reset_offsets(args.seeds).tolist(),
        control_hz=30,
        success_sampling="any sampled signed shaft angle >=170 degrees at 30 Hz",
        initial_tool_position_m=initial_tool_position,
        protocol="docs/studies/bimanual-valve-v1-protocol.md",
        criterion="signed shaft angle >=170 degrees; ambench-angle-170-v1",
        native_motor_limit_N=params.max_thrust,
        elapsed_s=time.monotonic() - started,
    )
    np.savez_compressed(args.output_dir / "trace.npz", **trace)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    launcher.app.close(exit_code=exit_code)
