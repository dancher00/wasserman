"""Scripted articulated PressButton evaluation with source-bound telemetry."""

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
parser.add_argument("--control-mode",choices=["held","ik","joint-probe"],default="held")
parser.add_argument("--closure",choices=["held","articulated"],default="articulated")
parser.add_argument("--ik-mode",choices=["feedback","reference"],default="feedback")
parser.add_argument("--offset-frame",choices=["world","level"],default="world")
parser.add_argument("--stroke",type=float,default=.10)
parser.add_argument("--hold-travel",type=float,default=0.,help="Latch load compensation after this button travel; zero selects the first-touch diagnostic")
parser.add_argument("--seeds", type=int, nargs="+", required=True)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--dt", type=float, default=1 / 240)
parser.add_argument("--solver-type",type=int,choices=[0,1],default=1,help="Opt-in numerical solver: 0 PGS, 1 native TGS")
parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rex_button_feasibility"))
parser.add_argument("--audit-only", action="store_true", help="Check imported masses/COM, without integration")
parser.add_argument("--centered", action="store_true", help="Separate custom centered mount and compact held arm")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["none"], enable_cameras=False)
args = parser.parse_args()
args.centered = True
launcher = AppLauncher(args)

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import wasman.tasks  # noqa: E402, F401
from wasman.assets.rexrov2_oberon7 import REXROV2_OBERON7_PREVIEW_CFG  # noqa: E402
from wasman.controllers.station_keeping import (  # noqa: E402
    BatchedStationKeepingController,
    StationKeepingGains,
    quaternion_angle_error_xyzw,
)
from wasman.physics.hydrodynamics import quat_apply_inverse_xyzw, quat_apply_xyzw  # noqa: E402
from wasman.physics.rexrov2 import ASSET, HeldArmRexHydrodynamics, RexThrusters, load_parameters  # noqa: E402


def main():
    if args.seconds < 15 or args.dt <= 0 or args.dt > 1 / 60:
        raise ValueError("Use >=15 seconds and dt<=1/60")
    if (args.output_dir / "report.json").exists():
        raise FileExistsError("Preserve the previous diagnostic; choose a new output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_script.py").write_bytes(Path(__file__).read_bytes())
    params = load_parameters()
    source_urdf = ASSET / "rexrov2_oberon7.urdf"
    prototype_cfg = REXROV2_OBERON7_PREVIEW_CFG
    if args.centered:
        from wasman.assets.rexrov2_centered import REXROV2_CENTERED_FLOATING_CFG
        from wasman.controllers.rexrov2_workspace import URDF
        from wasman.physics.rexrov2_centered import load_centered_parameters

        cpu = json.loads(Path(__file__).resolve().parents[1].joinpath("configs/studies/rex-centered-button-pose.json").read_text())
        held_q = np.array(cpu["targets"]["wall"]["q"])
        params = load_centered_parameters(held_q)
        source_urdf = URDF
        prototype_cfg = REXROV2_CENTERED_FLOATING_CFG.copy()
        prototype_cfg.init_state.joint_pos = dict(zip(cpu["joint_names"], held_q.tolist(), strict=True))
    cfg, _ = resolve_task_config("Wasman-Underwater-OpenHatch-Direct", "", overrides=("physics=isaacsim_physx",))
    cfg.sim.dt, cfg.sim.visualizer_cfgs = args.dt, []
    cfg.sim.physics.solver_type = args.solver_type
    sim = sim_utils.SimulationContext(cfg.sim)
    num_envs = len(args.seeds) + 1
    for index in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{index}", "Xform")
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
        "Wasman-Underwater-PressButton-T200-Direct", "", overrides=("physics=isaacsim_physx",)
    )
    fixture = button_cfg.scene.button.copy()
    fixture.prim_path = "/World/envs/env_[0-9]+/Button"
    button = Articulation(fixture)
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import Usd, UsdPhysics

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
                    "/World/envs/env_[0-9]+/Button/plunger",
                    "/World/envs/env_[0-9]+/Button/housing",
                ],
                track_contact_points=True,
                track_friction_forces=True,
                update_period=0,
                max_contact_data_count_per_prim=128,
            )
        )
        for name in params.names
    ]
    sim.reset()
    if robot.is_fixed_base:
        raise RuntimeError("Diagnostic must have a genuinely floating base")
    device = sim.device

    def tensor(value):
        return torch.tensor(value, dtype=torch.float32, device=device)

    origins = tensor([[0, index * 12, 0] for index in range(num_envs)])
    button_positions = origins + tensor([3, 0, 1.1])
    fixture_pose = torch.cat((button_positions, tensor([[0, 0, 0, 1]]).repeat(num_envs, 1)), -1)
    if args.control_mode=="joint-probe":fixture_pose[:,0]+=20 # no-contact diagnostic only
    button.write_root_pose_to_sim_index(root_pose=fixture_pose)
    button.write_joint_position_to_sim_index(position=torch.zeros(num_envs, 1, device=device))
    button.write_joint_velocity_to_sim_index(velocity=torch.zeros(num_envs, 1, device=device))
    button.reset()
    tcp_base = tensor(cpu["targets"]["wall"]["target_position_base_m"])
    targets = button_positions - tcp_base - tensor([0.12, 0, 0])
    initial_targets = targets.clone()
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
    # At the identity base orientation, public PhysX root columns are the same
    # world/body axes used by the nominal Pinocchio composite inertia.
    actual_base_mass=robot.data.mass_matrix.torch[:,:6,:6].cpu().numpy()
    import_audit["base_mass_matrix_max_error"]=float(np.abs(actual_base_mass-params.rigid_composite).max())
    import_audit["base_mass_matrix_matches"]=bool(np.allclose(actual_base_mass,params.rigid_composite,rtol=2e-5,atol=.03))
    import_audit["passed"] &= import_audit["base_mass_matrix_matches"]
    (args.output_dir / "import_audit.json").write_text(json.dumps(import_audit, indent=2) + "\n")
    if not import_audit["passed"]:
        raise RuntimeError("Imported mass/COM do not support the nominal added-inertia closure")
    if args.audit_only:
        print(json.dumps(import_audit), flush=True)
        return
    from wasman.physics.rexrov2_articulated import ArticulatedRexHydrodynamics
    hydro = (ArticulatedRexHydrodynamics(params, device,robot.joint_names)
             if args.closure=="articulated" else HeldArmRexHydrodynamics(params, device))
    initial_arm_targets=arm_targets.clone()
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
    tool_id = robot.find_bodies("oberon_end_effector")[0][0]
    plunger_id = button.find_bodies("plunger")[0][0]
    initial_tool_position = (
        (
            robot.data.body_link_pos_w.torch[:, tool_id]
            + quat_apply_xyzw(robot.data.body_link_quat_w.torch[:, tool_id], tensor([0.18, 0, 0]).expand(num_envs, 3))
            - origins
        )
        .cpu()
        .tolist()
    )
    from wasman.controllers.embodiment_ik import EmbodimentIK
    from wasman.controllers.embodiment_ik import ReferenceEmbodimentIK
    cls=ReferenceEmbodimentIK if args.ik_mode=="reference" else EmbodimentIK
    extra={"urdf":source_urdf,"offset_frame":args.offset_frame} if args.ik_mode=="reference" else {}
    ik=cls(robot,base_id,tool_id,list(range(6)),1/30,[.18,0,0],[.17,.17,.15,.25,.30,.15],targets,arm_targets[:,:6],**extra)
    for key in ["joint_target","joint_velocity","arm_excursion","base_acceleration_predicted","body_twist","added_reaction_bias","plunger_contact_force","normal_contact_by_body","friction_contact_by_body","contact_latched","planned_tool","base_reference"]:records[key]=[]
    started = time.monotonic()

    def total_at_base(force_world, torque_world, lever, base_quaternion):
        force = force_world.sum(1)
        torque = (torque_world + torch.cross(lever, force_world, dim=-1)).sum(1)
        return torch.cat(
            (quat_apply_inverse_xyzw(base_quaternion, force), quat_apply_inverse_xyzw(base_quaternion, torque)), dim=-1
        )

    for step in range(round(args.seconds / args.dt)):
        now = step * args.dt
        # Same fixed tool-space approach trajectory for all powered cases.
        fraction = min(1.0, max(0.0, (now - 5) / 8))
        if step % round(1 / 30 / args.dt) == 0:
            desired_tool = button_positions - tensor([0.12, 0, 0])
            desired_tool[:-1, 0] += args.stroke * fraction
        button.update(args.dt)
        for sensor in sensors:
            sensor.update(args.dt)
        contact_f = torch.stack(
            [
                (sensor.data.normal_force_matrix_w.torch + sensor.data.friction_force_matrix_w.torch)
                .reshape(num_envs, -1, 3)
                .sum(1)
                for sensor in sensors
            ],
            1,
        )
        contact_t = torch.zeros_like(contact_f)
        for j, sensor in enumerate(sensors):
            force = (sensor.data.normal_force_matrix_w.torch + sensor.data.friction_force_matrix_w.torch).reshape(
                num_envs, -1, 3
            )
            point = torch.nan_to_num(sensor.data.contact_pos_w.torch.reshape(num_envs, -1, 3))
            contact_t[:, j] = torch.cross(point - robot.data.body_link_pos_w.torch[:, 0, None, :], force, dim=-1).sum(1)
        position = robot.data.body_link_pos_w.torch[:, base_id]
        quaternion = robot.data.body_link_quat_w.torch[:, body_ids]
        base_quaternion = quaternion[:, 0]
        velocity = robot.data.body_com_lin_vel_w.torch[:, body_ids]
        angular = robot.data.body_com_ang_vel_w.torch[:, body_ids]
        centers = robot.data.body_com_pos_w.torch[:, body_ids]
        joints = robot.data.joint_pos.torch
        for state in (position, quaternion, velocity, angular, centers, joints):
            if not torch.isfinite(state).all():
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
        measured_tool = robot.data.body_link_pos_w.torch[:, tool_id] + quat_apply_xyzw(
            robot.data.body_link_quat_w.torch[:, tool_id], tensor([0.18, 0, 0]).expand(num_envs, 3)
        )
        if step % round(1 / 30 / args.dt) == 0:
            targets = initial_targets.clone()
            targets[:-1] = desired_tool[:-1] - (measured_tool - position)[:-1]
            if args.control_mode=="ik":
                touch=torch.stack([sensor.data.normal_force_matrix_w.torch[:,0,0] for sensor in sensors],1).sum(1).norm(dim=-1)>.5
                if args.hold_travel: touch=button.data.joint_pos.torch[:,0]>=args.hold_travel
                new_base,new_arm=ik.targets(desired_tool,target_quaternion,touch)
                targets[:-1]=new_base[:-1];arm_targets[:-1,:6]=new_arm[:-1]
            elif args.control_mode=="joint-probe":
                targets=initial_targets.clone()
                # Smooth bounded motion at native joint rates, no scene contact.
                phase=max(0.,now-5.)
                envelope=.5*(1-np.cos(2*np.pi*min(phase,10.)/10.))
                arm_targets[:-1,:6]=initial_arm_targets[:-1,:6]+envelope*tensor([.03,.08,-.10,0.,.08,0.])
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
        pulse = torch.zeros_like(realized)
        if False:
            pulse[2] = tensor([300, -180, 0, 40, -60, 80])
        non_added = total_at_base(force_world + gravity, torque_world, lever, base_quaternion) + realized + pulse
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
            robot.data.body_link_quat_w.torch[:, tool_id], tensor([0.18, 0, 0]).expand(num_envs, 3)
        )
        tool_axis = quat_apply_xyzw(
            robot.data.body_link_quat_w.torch[:, tool_id], tensor([1.0, 0, 0]).expand(num_envs, 3)
        )
        travel = button.data.joint_pos.torch[:, 0].clone()
        max_travel = torch.maximum(max_travel, travel)
        distance = (tool_position - button.data.body_link_pos_w.torch[:, plunger_id]).norm(dim=-1)
        pressed = (travel >= 0.004) & (distance < 0.13) & (tool_axis[:, 0] > 0.70)
        pressed &= (quaternion_angle_error_xyzw(base_quaternion, target_quaternion) < 0.25) & (
            angular[:, 0].norm(dim=-1) < 0.35
        )
        # Match the native PressButton contract: four qualifying 30 Hz samples.
        if step % round(1 / 30 / args.dt) == 0:
            success_counter = torch.where(pressed, success_counter + 1, 0)
            ever_success |= success_counter >= 4
        body_twist = torch.cat(
            (
                quat_apply_inverse_xyzw(base_quaternion, velocity[:, 0]),
                quat_apply_inverse_xyzw(base_quaternion, angular[:, 0]),
            ),
            dim=-1,
        )
        if args.closure=="articulated":
            hydro.update_articulation(joints,robot.data.joint_vel.torch,body_twist,args.dt)
        added, predicted_acceleration = hydro.close_added_mass(body_twist, quat_apply_inverse_xyzw(base_quaternion, currents), non_added)
        for wrench in (fluid, command, realized, non_added, added, thrusters.force):
            if not torch.isfinite(wrench).all():
                raise RuntimeError(f"Non-finite wrench at step {step}")
        if thrusters.force.abs().max() > params.max_thrust + 0.001:
            raise RuntimeError(f"Native motor force cap exceeded at step {step}")
        force_world[:, 0] += quat_apply_xyzw(base_quaternion, added[:, :3] + pulse[:, :3])
        torque_world[:, 0] += quat_apply_xyzw(base_quaternion, added[:, 3:] + pulse[:, 3:])
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
                "planned_tool":(ik.last_tool_reference if args.control_mode=="ik" and args.ik_mode=="reference" else desired_tool)-origins,
                "base_reference":targets-origins,
                "button_travel": travel,
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
                "normal_contact_by_body":torch.stack([sensor.data.normal_force_matrix_w.torch[:,0] for sensor in sensors],1),
                "friction_contact_by_body":torch.stack([sensor.data.friction_force_matrix_w.torch[:,0] for sensor in sensors],1),
                "contact_latched":ik.contact_latched,
                "joint_target":arm_targets,
                "joint_velocity":robot.data.joint_vel.torch,
                "arm_excursion":(joints-initial_arm_targets).abs().amax(-1),
                "base_acceleration_predicted":predicted_acceleration,
                "body_twist":body_twist,
                "added_reaction_bias":hydro.rigid_bias if args.closure=="articulated" else torch.zeros_like(body_twist),
                "plunger_contact_force":torch.stack([sensor.data.normal_force_matrix_w.torch[:,0,0] for sensor in sensors],1).sum(1),
            }
            for key, value in values.items():
                records[key].append(value.detach().cpu().numpy().copy())
        if step % round(5 / args.dt) == 0:
            print(
                f"t={now:.1f}s travel={travel.tolist()} tool={tool_position[0].tolist()} "
                f"drift={(joints - arm_targets).abs().max().item()}",
                flush=True,
            )
        robot.actuators.target_command.set_position_index(value=arm_targets)
        button.set_joint_position_target_index(target=torch.zeros(num_envs, 1, device=device), joint_ids=[0])
        button.write_data_to_sim()
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(args.dt)
    trace = {key: np.asarray(value) for key, value in records.items()}
    report = dict(
        kind="Rex articulated PressButton study",
        control_mode=args.control_mode,closure=args.closure,stroke_m=args.stroke,hold_travel_m=args.hold_travel,ik_mode=args.ik_mode,offset_frame=args.offset_frame,
        joint_names=robot.joint_names,contact_body_names=params.names,contact_targets=["plunger","housing"],
        limits="base-only added mass; lagged measured joint acceleration/contact; no hardware calibration",
        source_files={name:hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest() for name in ["src/wasman/physics/rexrov2_articulated.py","src/wasman/controllers/embodiment_ik.py"]},
        successes=ever_success.tolist(),
        max_travel_m=max_travel.cpu().tolist(),
        max_arm_drift_rad=trace["arm_drift"].max(0).tolist(),
        contact_closure="one-step measured contact force and point estimate; needs timestep convergence",
        dt=args.dt,
        solver_type=args.solver_type,
        seconds=args.seconds,
        fixed_base=False,
        gravity_enabled=True,
        runtime_pose_overwrites=0,
        zero_motor_control_index=num_envs - 1,
        seeds=args.seeds,
        initial_offsets_m=reset_offsets(args.seeds).tolist(),
        control_hz=30,success_sampling="four consecutive samples at 30 Hz",
        initial_tool_position_m=initial_tool_position,
        protocol="docs/studies/embodiment-v3-protocol.md" if args.offset_frame=="level" else "docs/studies/embodiment-v2-protocol.md",
        criterion=(
            "original short-start button: 4mm, distance<13cm, alignment>.7, "
            "attitude<.25rad, angular speed<.35rad/s, hold4/30s"
        ),
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
