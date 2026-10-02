"""Scripted BlueROV PressButton evaluation with the original success contract."""

# ruff: noqa: E402
import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--control-mode",choices=["held","ik"],default="held")
p.add_argument("--ik-mode",choices=["feedback","reference"],default="feedback")
p.add_argument("--offset-frame",choices=["world","level"],default="world")
p.add_argument("--stroke",type=float,default=.10)
p.add_argument("--hold-travel",type=float,default=0.,help="Latch load compensation after this button travel; zero selects the first-touch diagnostic")
p.add_argument("--dt",type=float,default=1/240)
p.add_argument("--solver-type",type=int,choices=[0,1],default=1,help="Opt-in numerical solver: 0 PGS, 1 native TGS")
p.add_argument("--hydro-variant",choices=["legacy","geometry-scaled"],default="legacy")
p.add_argument("--acceleration-cap",choices=["legacy","none"],default="legacy")
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--seeds", type=int, nargs="+", required=True)
p.add_argument("--action-replay",type=Path,help="Replay recorded normalized native base/joint commands, bypassing IK output")
p.add_argument("--physics-trace",action="store_true")
p.add_argument("--seconds", type=float, default=20)
AppLauncher.add_app_launcher_args(p)
p.set_defaults(visualizer=["none"], enable_cameras=False)
args = p.parse_args()
launcher = AppLauncher(args)
import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import resolve_task_config

import wasman.tasks  # noqa: F401
from wasman.assets.registered_bluerov import configure_registered_bluerov


def main():
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "run_script.py").write_bytes(Path(__file__).read_bytes())
    cfg, _ = resolve_task_config("Wasman-Underwater-PressButton-Direct", "", overrides=("physics=isaacsim_physx",))
    configure_registered_bluerov(cfg)
    if args.hydro_variant=="geometry-scaled":
        from wasman.assets.bluerov2_alpha import BLUEROV2_ALPHA_HYDRODYNAMICS
        from wasman.assets.grasp_hydrodynamics import geometry_scaled_arm_hydrodynamics
        cfg.link_hydrodynamics=geometry_scaled_arm_hydrodynamics(BLUEROV2_ALPHA_HYDRODYNAMICS)
    cfg.use_physical_thrusters = True
    n = len(args.seeds) + 1
    cfg.scene.num_envs = n
    cfg.scene.env_spacing = 12
    cfg.scene.panel = None
    cfg.sim.dt = args.dt
    cfg.sim.physics.solver_type = args.solver_type
    # Keep the nominal 2/240-second transport delay fixed during rate checks.
    cfg.thruster_command_delay_steps = round((2 / 240) / args.dt)
    cfg.acceleration_filter = 1 - (1 - .2) ** (args.dt * 240)
    cfg.decimation = round(1/30/args.dt)
    cfg.episode_length_s = args.seconds + 1
    cfg.max_base_distance = 5.0  # Fixture relocated to x=3m; original origin-centred2m box is inapplicable.
    cfg.sim.render_interval = 8
    cfg.base_target_position = (2.4, 0, 1.1)
    cfg.base_target_position_scale = (0.6, 0.6, 0.6)
    cfg.current_speed_range = cfg.current_vertical_range = (0.0, 0.0)
    cfg.turbulence_sigma = 0
    cfg.volume_scale_range = cfg.damping_scale_range = cfg.added_mass_scale_range = (1.0, 1.0)
    from isaaclab.sensors import ContactSensorCfg
    cfg.scene.plunger_contact=ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Button/plunger",update_period=0)
    cfg.scene.housing_contact=ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Button/housing",update_period=0)
    env = gym.make("Wasman-Underwater-PressButton-Direct", cfg=cfg)
    raw = env.unwrapped
    device = raw.device

    def tensor(x):
        return torch.tensor(x, device=device, dtype=torch.float32)

    env.reset(seed=7100)
    initial_pose = raw.robot.data.root_pose_w.torch.clone()
    initial_pose[:, 3:] = tensor([0, 0, 0, 1])
    raw.robot.write_root_pose_to_sim_index(root_pose=initial_pose)
    q = raw.robot.data.default_joint_pos.torch.clone()
    raw.robot.write_joint_position_to_sim_index(position=q)
    raw.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    raw.scene.update(0.0)
    raw.robot.reset()
    raw.robot.update(cfg.sim.dt)
    offset = (
        raw.robot.data.body_link_pos_w.torch[:, raw._tool_body_id]
        - raw.robot.data.body_link_pos_w.torch[:, raw._base_body_id]
    )
    fixture = raw.scene.env_origins + tensor([3, 0, 1.1])
    desired = fixture - tensor([0.12, 0, 0])
    pose = torch.cat((desired - offset, tensor([[0, 0, 0, 1]]).repeat(n, 1)), dim=-1)
    from wasman.controllers.held_arm_button_study import reset_offsets

    pose[:-1, :3] += tensor(reset_offsets(args.seeds))
    raw.robot.write_root_pose_to_sim_index(root_pose=pose)
    raw.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=device))
    raw.button.write_root_pose_to_sim_index(root_pose=torch.cat((fixture, tensor([[0, 0, 0, 1]]).repeat(n, 1)), dim=-1))
    raw.button.write_joint_position_to_sim_index(position=torch.zeros(n, 1, device=device))
    raw.button.write_joint_velocity_to_sim_index(velocity=torch.zeros(n, 1, device=device))
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    raw.scene.update(0.0)
    raw.robot.reset()
    raw.robot.update(cfg.sim.dt)
    raw.button.reset()
    raw.button.update(cfg.sim.dt)
    raw._hydrodynamics.reset(torch.arange(n, device=device))
    raw._station_keeper.reset(torch.arange(n, device=device))
    initial_tool_position = (
        (raw.robot.data.body_link_pos_w.torch[:, raw._tool_body_id] - raw.scene.env_origins).cpu().tolist()
    )
    # Observe every physics step without changing computed/applied forces.
    hydro=raw._hydrodynamics
    hydro_original=hydro.compute
    nominal_acceleration_limit=hydro.acceleration_limit.clone()
    legacy_threshold_counts=torch.zeros(n,device=device,dtype=torch.long)
    if args.acceleration_cap=="none":
        assert args.hydro_variant=="geometry-scaled", "Unclipped acceleration requires the geometry-scaled inertial model"
        hydro.acceleration_limit.fill_(float("inf"))
    hydro_counts=torch.zeros(n,3,device=device,dtype=torch.long)
    hydro_max=torch.zeros(n,3,device=device)
    def hydro_observer(twist,quaternion):
        accel=(twist-hydro.previous_relative_twist)/hydro.dt
        accel_ratio=(accel.abs()/hydro.acceleration_limit).amax(dim=(1,2))
        accel_ratio=torch.where(hydro.history_valid.reshape(n),accel_ratio,0.)
        legacy_ratio=(accel.abs()/nominal_acceleration_limit).amax(dim=(1,2))
        legacy_threshold_counts.add_(((legacy_ratio>1)&hydro.history_valid.reshape(n)).long())
        force,torque,filtered=hydro_original(twist,quaternion)
        if not (torch.isfinite(accel).all() & torch.isfinite(force).all() & torch.isfinite(torque).all() & torch.isfinite(filtered).all()):
            raise RuntimeError("Non-finite hydrodynamic calculation before native sanitization")
        ratios=torch.stack((accel_ratio,force.abs().amax(dim=(1,2))/cfg.max_hydrodynamic_force,torque.abs().amax(dim=(1,2))/cfg.max_hydrodynamic_torque),-1)
        hydro_counts.add_((ratios>1).long());hydro_max.copy_(torch.maximum(hydro_max,ratios))
        return force,torque,filtered
    hydro.compute=hydro_observer
    physics_rows=[]
    original_apply=raw._apply_action
    def observed_apply():
        original_apply()
        if args.physics_trace:
            physics_rows.append(dict(time=len(physics_rows)*cfg.sim.dt,base_quaternion=raw.robot.data.body_link_quat_w.torch[:,raw._base_body_id].cpu().numpy().copy(),angular_velocity=raw.robot.data.body_com_ang_vel_w.torch[:,raw._base_body_id].cpu().numpy().copy(),motor_force=raw._thrusters.force.cpu().numpy().copy(),button_travel=raw.button.data.joint_pos.torch[:,0].cpu().numpy().copy(),plunger_contact=raw.scene["plunger_contact"].data.net_normal_forces_w.torch.sum(1).cpu().numpy().copy()))
    raw._apply_action=observed_apply
    replay=None
    if args.action_replay:
        replay=np.load(args.action_replay,allow_pickle=False)['action']
        parent=json.loads(args.action_replay.with_name('report.json').read_text())
        assert parent['seeds']==args.seeds and replay.shape==(round(args.seconds*30),n,10)
    original = raw._thrusters.step

    def motors(force, torque):
        force = force.clone()
        torque = torque.clone()
        force[-1] = 0
        torque[-1] = 0
        return original(force, torque)

    raw._thrusters.step = motors
    from wasman.controllers.embodiment_ik import EmbodimentIK
    # Calibrate the gripper's intrinsic roll, rather than demanding an unreachable raw-link identity pose.
    nominal_q=raw.robot.data.body_link_quat_w.torch[0,raw._tool_body_id]
    x,y,z,w=nominal_q
    roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    tool_orientation_offset=[float(torch.sin(roll/2)),0.,0.,float(torch.cos(roll/2))]
    action_limits=torch.stack((raw._arm_nominal_targets-raw._arm_target_scale,raw._arm_nominal_targets+raw._arm_target_scale),-1)
    from wasman.controllers.embodiment_ik import ReferenceEmbodimentIK
    cls=ReferenceEmbodimentIK if args.ik_mode=="reference" else EmbodimentIK
    extra={"urdf":cfg.scene.robot.spawn.asset_path,"offset_frame":args.offset_frame} if args.ik_mode=="reference" else {}
    ik=cls(raw.robot,raw._base_body_id,raw._tool_body_id,raw._arm_joint_ids,1/30,[0,0,0],[.35]*4,pose[:,:3],q[:,raw._arm_joint_ids],tool_orientation_offset,action_limits,**extra)
    target_quaternion=tensor([[0,0,0,1]]).repeat(n,1)
    trace = []
    try:
        for step in range(round(args.seconds * 30)):
            now = step / 30
            goal = desired.clone()
            goal[:-1, 0] += args.stroke * min(1.0, max(0.0, (now - 5) / 8))
            base = raw.robot.data.body_link_pos_w.torch[:, raw._base_body_id]
            tool = raw.robot.data.body_link_pos_w.torch[:, raw._tool_body_id]
            target = goal - (tool - base)
            action = torch.zeros(n, 10, device=device)
            action[:, :3] = (
                target - raw.scene.env_origins - raw._base_target_nominal
            ) / raw._base_target_position_scale
            if args.control_mode=="ik":
                touch=raw.scene["plunger_contact"].data.net_normal_forces_w.torch.sum(1).norm(dim=-1)>.5
                if args.hold_travel: touch=raw._button_travel>=args.hold_travel
                base_ref,arm_ref=ik.targets(goal,target_quaternion,touch)
                action[:-1,:3]=(base_ref[:-1]-raw.scene.env_origins[:-1]-raw._base_target_nominal)/raw._base_target_position_scale
                action[:-1,6:10]=(arm_ref[:-1]-raw._arm_nominal_targets[:-1])/raw._arm_target_scale
            if replay is not None: action=tensor(replay[step])
            # All references pass through the original bounded actuator interface.
            _, _, done, timeout, info = env.step(action)
            if (done | timeout).any():
                raise RuntimeError("Unexpected reset in first episode")
            row = dict(
                time=now + 1 / 30,
                button_travel=raw._button_travel.cpu().numpy().copy(),
                tool_position=(raw.robot.data.body_link_pos_w.torch[:, raw._tool_body_id] - raw.scene.env_origins)
                .cpu()
                .numpy()
                .copy(),
                target_tool=(goal - raw.scene.env_origins).cpu().numpy().copy(),
                planned_tool=((ik.last_tool_reference if args.control_mode=="ik" and args.ik_mode=="reference" else goal)-raw.scene.env_origins).cpu().numpy().copy(),
                base_reference=(raw._base_target_pos_w-raw.scene.env_origins).cpu().numpy().copy(),
                tool_alignment=raw._alignment.cpu().numpy().copy(),
                distance=raw._distance.cpu().numpy().copy(),
                attitude=raw._base_attitude_error.cpu().numpy().copy(),
                angular_speed=raw._base_angular_speed.cpu().numpy().copy(),
                arm_drift=(raw.robot.data.joint_pos.torch - q).abs().amax(-1).cpu().numpy().copy(),
                success=info["wasman_success"].cpu().numpy().copy(),
                position_error=raw._station_position_error_w.cpu().numpy().copy(),
                motor_force=raw._thrusters.force.cpu().numpy().copy(),
                motor_scale=raw._thrusters.saturation_scale.squeeze(-1).cpu().numpy().copy(),
                joint_position=raw.robot.data.joint_pos.torch.cpu().numpy().copy(),
                joint_target=raw._arm_targets.cpu().numpy().copy(),
                base_quaternion=raw.robot.data.body_link_quat_w.torch[:,raw._base_body_id].cpu().numpy().copy(),
                joint_velocity=raw.robot.data.joint_vel.torch.cpu().numpy().copy(),
                plunger_contact_force=raw.scene["plunger_contact"].data.net_normal_forces_w.torch.sum(1).cpu().numpy().copy(),
                housing_contact_force=raw.scene["housing_contact"].data.net_normal_forces_w.torch.sum(1).cpu().numpy().copy(),
                base_position=(raw.robot.data.body_link_pos_w.torch[:,raw._base_body_id]-raw.scene.env_origins).cpu().numpy().copy(),
                hydro_clip_counts=hydro_counts.cpu().numpy().copy(),
                legacy_acceleration_threshold_exceedances=legacy_threshold_counts.cpu().numpy().copy(),
                hydro_clip_peak_ratio=hydro_max.cpu().numpy().copy(),
                filtered_acceleration=hydro.filtered_acceleration.cpu().numpy().copy(),
                contact_latched=ik.contact_latched.cpu().numpy().copy(),
                action=action.cpu().numpy().copy(),
            )
            trace.append(row)
            if step % 150 == 0:
                print("t", now, "travel", row["button_travel"], "tool", row["tool_position"][0], flush=True)
        arrays = {k: np.array([r[k] for r in trace]) for k in trace[0]}
        np.savez_compressed(args.output_dir / "trace.npz", **arrays)
        if physics_rows:
            np.savez_compressed(args.output_dir/'physics_trace.npz',**{k:np.array([r[k] for r in physics_rows]) for k in physics_rows[0]})
        result = dict(
            action_replay=str(args.action_replay) if args.action_replay else None,
            successes=arrays["success"].any(0).tolist(),
            max_travel_m=arrays["button_travel"].max(0).tolist(),
            max_arm_drift_rad=arrays["arm_drift"].max(0).tolist(),
            fixed_base=bool(raw.robot.is_fixed_base),
            dt=cfg.sim.dt,solver_type=args.solver_type,thruster_command_delay_s=cfg.thruster_command_delay_steps*cfg.sim.dt,
            acceleration_filter_per_step=cfg.acceleration_filter,acceleration_filter_tau_s=-(1/240)/__import__("math").log(.8),
            control_hz=30,
            initial_tool_position_m=initial_tool_position,
            seconds=args.seconds,
            zero_motor_control_index=n - 1,
            seeds=args.seeds,
            initial_offsets_m=reset_offsets(args.seeds).tolist(),
            protocol="docs/studies/embodiment-v3-protocol.md" if args.offset_frame=="level" else "docs/studies/embodiment-v2-protocol.md",
            criterion="original short-start PressButton contract; unchanged",
            kind="BlueROV articulated PressButton study",control_mode=args.control_mode,stroke_m=args.stroke,hydro_variant=args.hydro_variant,acceleration_cap=args.acceleration_cap,hold_travel_m=args.hold_travel,ik_mode=args.ik_mode,offset_frame=args.offset_frame,
            joint_names=raw.robot.joint_names,tool_orientation_offset_xyzw=tool_orientation_offset,
            hydro_audit_columns=["acceleration","force","torque"],
            contact_force_scope="normal force on plunger/housing; friction not included",
            source_ik_sha256=__import__('hashlib').sha256((Path(__file__).resolve().parents[1]/"src/wasman/controllers/embodiment_ik.py").read_bytes()).hexdigest(),
        )
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        env.close()


exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    launcher.app.close(exit_code=exit_code)
