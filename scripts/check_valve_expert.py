"""Finite valve expert diagnostic and evaluation, never mislabeled as PPO."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, help="Defaults to ten steps less than the task horizon")
parser.add_argument("--approach", action="store_true", help="Folded-arm swim-in before manipulation; engineered gate")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--randomized", action="store_true")
parser.add_argument("--record", action="store_true")
parser.add_argument("--camera-streams", action="store_true")
parser.add_argument(
    "--registered-geometry", action="store_true", help="Opt in to registered-v1 geometry and geometry-v2 cameras"
)
parser.add_argument(
    "--success-tail",
    type=float,
    help="For single-environment recordings, stop this many seconds after measured success",
)
parser.add_argument("--real-time", action="store_true", help="Pace live playback at no more than real time")
parser.add_argument("--checkpoint", type=Path, help="Evaluate a learned state actor without constructing an expert")
parser.add_argument(
    "--diagnostic-execution",
    choices=("checkpoint", "none", "rate"),
    default="checkpoint",
    help="Audit only: override checkpoint execution mapping; requires action trace and forbids recording",
)
parser.add_argument(
    "--action-trace", action="store_true", help="Record before-action observations and actuator targets for diagnosis"
)
parser.add_argument(
    "--diagnostic-freeze-base-x",
    action="store_true",
    help="Diagnostic ONLY: freeze base X after settled grasp; not a standalone learned-policy score",
)
parser.add_argument("--require-success", action="store_true")
parser.add_argument(
    "--success-contract",
    choices=("ambench-angle-170-v1", "valve-grasp-turn-release-v1"),
    default="ambench-angle-170-v1",
    help="Task scoring version; use the legacy version to reproduce old evaluations",
)
parser.add_argument("--expert-hold-angle", type=float, default=173.8)
parser.add_argument(
    "--axial-release", action="store_true", help="Expert-only simultaneous gentle opening and axial escape"
)
parser.add_argument("--strict-unload", action="store_true", help="Expert-only adaptive unloading before withdrawal")
parser.add_argument(
    "--compliant-unload", action="store_true", help="Expert-only contact-reaction admittance during unloading"
)
parser.add_argument("--output-dir", type=Path, default=Path("artifacts/valve_cartesian"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["none"])
args = parser.parse_args()
if args.diagnostic_execution != "checkpoint" and (not args.checkpoint or not args.action_trace or args.record):
    parser.error("execution override requires --checkpoint --action-trace and forbids recording")
if args.strict_unload and args.checkpoint:
    parser.error("Strict unloading is an expert collection option, never a learned-policy intervention")
if args.checkpoint and args.expert_hold_angle != 173.8:
    parser.error("Expert hold reference cannot modify checkpoint evaluation")
if args.axial_release and args.checkpoint:
    parser.error("Expert axial-release control cannot modify checkpoint evaluation")
if args.compliant_unload and not args.strict_unload:
    parser.error("Compliant unloading requires --strict-unload")
if args.diagnostic_freeze_base_x and (not args.checkpoint or not args.action_trace or args.record):
    parser.error("diagnostic freeze requires --checkpoint --action-trace and forbids recording")
if args.camera_streams and not args.record:
    parser.error("--camera-streams requires --record")
if args.success_tail is not None and (not args.record or args.num_envs != 1 or args.success_tail < 1):
    parser.error("success-tail requires --record --num-envs 1 and at least one second")
if args.record:
    args.visualizer = ["kit"]
    args.enable_cameras = True
launcher = AppLauncher(args)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import wasman.tasks  # noqa: E402, F401
from wasman.controllers.cinema import valve_camera  # noqa: E402
from wasman.controllers.valve import ValveExpert  # noqa: E402
from wasman.controllers.valve_success import valve_success_metadata  # noqa: E402
from wasman.robot_camera_profiles import robot_camera_profile  # noqa: E402


def main():
    task = f"Wasman-Underwater-RotateValve-{'Approach-' if args.approach else ''}T200-Direct"
    root = Path(__file__).resolve().parents[1]
    sources = [
        "src/wasman/controllers/valve.py",
        "src/wasman/controllers/valve_contract.py",
        "src/wasman/controllers/valve_success.py",
        "src/wasman/controllers/tool_pose.py",
        "src/wasman/tasks/underwater_panel/valve.py",
        "src/wasman/tasks/underwater_panel/env_cfg.py",
        "src/wasman/assets/grasp_geometry.py",
        "src/wasman/assets/grasp_hydrodynamics.py",
        "src/wasman/controllers/valve_policy.py",
        "src/wasman/controllers/valve_execution.py",
        "src/wasman/controllers/valve_local_policy.py",
        "src/wasman/controllers/valve_recurrent_policy.py",
        "src/wasman/controllers/valve_mixture_policy.py",
        "src/wasman/controllers/valve_policy_batch.py",
        "src/wasman/controllers/valve_mode_router.py",
        "src/wasman/controllers/valve_sequential_router.py",
        "src/wasman/tasks/underwater_panel/approach.py",
        "src/wasman/tasks/underwater_press_button/config/bluerov2_alpha/env.py",
        "src/wasman/assets/panels.py",
        "src/wasman/controllers/approach.py",
        "src/wasman/controllers/cinema.py",
        "scripts/check_valve_expert.py",
        "src/wasman/camera_recording.py",
        "src/wasman/assets/robot_cameras.py",
        "src/wasman/robot_camera_profiles.py",
        "src/wasman/assets/registered_bluerov.py",
        "src/wasman/media.py",
    ]
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    cfg, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=("physics=isaacsim_physx",))
    cfg.scene.num_envs = args.num_envs
    cfg.valve_success_contract = args.success_contract
    if args.registered_geometry:
        from wasman.assets.registered_bluerov import configure_registered_bluerov

        configure_registered_bluerov(cfg)
    cfg.seed = args.seed
    if args.steps is None:
        args.steps = round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)) - 10
    if not args.randomized:
        cfg.current_speed_range = (0.0, 0.0)
        cfg.current_vertical_range = (0.0, 0.0)
        cfg.turbulence_sigma = 0.0
        cfg.button_y_range = (-0.10, -0.10)
        cfg.button_z_range = (0.78, 0.78)
    if not 0 < args.steps < round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)):
        raise ValueError("Diagnostic must end before automatic timeout")
    if args.record:
        from wasman.media import enable_current_moviepy_recorder

        enable_current_moviepy_recorder()
        if args.num_envs != 1:
            raise ValueError("Record only a single environment")
        cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:kit",
                output_dir=str(args.output_dir),
                video_length=args.steps,
                output_filename_prefix="valve_policy" if args.checkpoint else "valve_expert",
            )
        ]
        if args.camera_streams:
            from wasman.camera_recording import configure_robot_streams

            configure_robot_streams(cfg, args.steps)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    camera_recording = None
    geometry_audit = None
    try:
        env.reset(seed=args.seed)
        if args.registered_geometry:
            from wasman.assets.registered_bluerov import audit_registered_stage

            geometry_audit = audit_registered_stage(raw)
        policy, policy_info, local_policy = None, None, False
        recurrent_policy = False
        mixture_policy = False
        batch_policy = False
        inference_parameters = None
        if args.checkpoint:
            loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            inference_parameters = loaded.get("parameters")
            local_policy = loaded.get("format") == "wasman-valve-local-v1"
            recurrent_policy = loaded.get("format") == "wasman-valve-recurrent-v1"
            mixture_policy = loaded.get("format") == "wasman-valve-mixture-v1"
            batch_policy = loaded.get("format") == "wasman-valve-batch-v1"
            if local_policy:
                from wasman.controllers.valve_local_policy import ValveLocalPolicy

                policy = ValveLocalPolicy.from_checkpoint(loaded, raw.device)
                policy_info = loaded["infos"]
            elif recurrent_policy:
                from wasman.controllers.valve_recurrent_policy import ValveRecurrentPolicy

                policy = ValveRecurrentPolicy.from_checkpoint(loaded, raw.device)
                policy_info = loaded["infos"]
            elif mixture_policy:
                from wasman.controllers.valve_mixture_policy import ValveMixturePolicy

                policy = ValveMixturePolicy.from_checkpoint(loaded, raw.device)
                policy_info = loaded["infos"]
            elif batch_policy:
                from wasman.controllers.valve_policy_batch import ValvePolicyBatch

                policy = ValvePolicyBatch.from_checkpoint(loaded, raw.device)
                if policy.num_envs != raw.num_envs:
                    raise ValueError("Use the bundle's exact environment count")
                policy_info = loaded["infos"]
            else:
                from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

                agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version("rsl-rl-lib"))
                runner = create_rsl_rl_runner(RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions), agent)
                policy_info = runner.load(str(args.checkpoint))
                policy = runner.get_inference_policy(device=raw.device)
            del loaded
            ctrl = None
        else:
            ctrl = ValveExpert(
                raw,
                strict_unload=args.strict_unload,
                compliant_unload=args.compliant_unload,
                hold_angle_deg=args.expert_hold_angle,
                axial_release=args.axial_release,
            )
        # Wrappers may reset during construction; freeze the actual trial's RNG
        # after loading so expert and learned runs share the same initial seed.
        obs, _ = env.reset(seed=args.seed)
        if args.camera_streams:
            from wasman.camera_recording import RobotCameraRecording

            camera_recording = RobotCameraRecording(env, launcher.app, args.output_dir, "valve")
            obs = camera_recording.observation
            if policy is None:
                ctrl = ValveExpert(
                    raw,
                    strict_unload=args.strict_unload,
                    compliant_unload=args.compliant_unload,
                    hold_angle_deg=args.expert_hold_angle,
                    axial_release=args.axial_release,
                )
            elif local_policy or recurrent_policy or mixture_policy:
                policy.reset()
        trace = []
        success = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
        failed = success.clone()
        phase = torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device)
        max_phase = phase.clone()
        bilateral_steps = torch.zeros_like(max_phase)
        diagnostic_latched = torch.zeros_like(success)
        diagnostic_base_x = torch.zeros(raw.num_envs, device=raw.device)
        first_success_step = None
        with torch.inference_mode():
            for step in range(args.steps):
                started = time.perf_counter()
                if args.record or "kit" in args.visualizer:
                    raw.sim.set_camera_view(*valve_camera(step * raw.step_dt))
                if ctrl is not None and hasattr(raw, "manipulation_ready"):
                    ctrl.reset((~raw.manipulation_ready).nonzero().flatten())
                before_action = obs["policy"].clone() if args.action_trace else None
                actions = policy(obs) if policy is not None else ctrl.actions()
                learned_mode = None
                if local_policy and policy.mode_router is not None:
                    learned_mode = policy.mode_router.mode.tolist()
                elif batch_policy:
                    learned_mode = [
                        value
                        for candidate in policy.policies
                        for value in (
                            candidate.mode_router.mode.tolist()
                            if candidate.mode_router is not None
                            else [None] * policy.environments_per_policy
                        )
                    ]
                actor_output = actions.clone() if args.action_trace else None
                if args.diagnostic_execution == "rate" or (
                    args.diagnostic_execution == "checkpoint"
                    and (policy_info or {}).get("execution") == "valve-rate-level-v1"
                ):
                    from wasman.controllers.valve_execution import execute_valve_action

                    actions = execute_valve_action(actions, obs["policy"], raw.step_dt)
                if args.diagnostic_freeze_base_x:
                    latch = ~diagnostic_latched & (raw.valve_contract.grasp_steps >= 15) & (raw.actions[:, 10] <= -0.99)
                    diagnostic_base_x[latch] = raw.actions[latch, 0]
                    diagnostic_latched |= latch
                    actions = actions.clone()
                    actions[diagnostic_latched, 0] = diagnostic_base_x[diagnostic_latched]
                obs, reward, terminated, truncated, info = env.step(actions)
                if camera_recording:
                    camera_recording.capture(step)
                if not torch.isfinite(obs["policy"]).all():
                    raise RuntimeError("Non-finite observation")
                success |= info["wasman_success"] & ~failed & ~(terminated | truncated)
                failed |= terminated | truncated
                if args.record and failed.any():
                    raise RuntimeError("Refusing to record a rollout containing an automatic reset")
                phase = (
                    ctrl.phase
                    if ctrl is not None
                    else torch.where(
                        info["wasman_strict_valve_success"],
                        7,
                        torch.where(info["wasman_valve_held"], 5, torch.where(info["wasman_grasped"], 3, 0)),
                    )
                )
                max_phase = torch.maximum(max_phase, phase)
                bilateral_steps += info["wasman_grasped"].long()
                if ctrl is not None and (terminated | truncated).any():
                    ctrl.reset((terminated | truncated).nonzero().flatten())
                if (recurrent_policy or local_policy or mixture_policy or batch_policy) and (
                    terminated | truncated
                ).any():
                    policy.reset((terminated | truncated).nonzero().flatten())
                trace.append(
                    {
                        "t": (step + 1) * raw.step_dt,
                        "phase": phase.tolist(),
                        "learned_router_mode_at_command": learned_mode,
                        "navigation": [
                            "manipulation" if ready else "deployment" if deploy else "transit"
                            for ready, deploy in zip(
                                raw.manipulation_ready.tolist(), raw.navigation.enabled.tolist(), strict=True
                            )
                        ]
                        if hasattr(raw, "manipulation_ready")
                        else None,
                        "angle_rad": info["wasman_valve_angle"].tolist(),
                        "wheel_speed_rad_s": info["wasman_valve_angular_speed"].tolist(),
                        "reference_angle_rad": ctrl.reference_angle.tolist() if ctrl is not None else None,
                        "grasp_turn_rad": info["wasman_grasp_turn"].tolist(),
                        "ungrasped_motion_rad": info["wasman_ungrasped_motion"].tolist(),
                        "opposing_contacts": info["wasman_contact_opposition"].tolist(),
                        "finger_forces_n": info["wasman_finger_forces"].tolist(),
                        "held": info["wasman_valve_held"].tolist(),
                        "success": info["wasman_success"].tolist(),
                        "strict_valve_success": info["wasman_strict_valve_success"].tolist(),
                        "success_criteria": {k: v.tolist() for k, v in info["success_criteria"].items()},
                        "tool_error_m": (raw.robot.data.body_link_pos_w.torch[:, raw._tool_body_id] - ctrl.target)
                        .norm(dim=-1)
                        .tolist()
                        if ctrl is not None
                        else None,
                        "tool_position_m": raw.robot.data.body_link_pos_w.torch[:, raw._tool_body_id].tolist(),
                        "base_position_m": raw.robot.data.root_pos_w.torch.tolist(),
                        "base_attitude_rad": info["wasman_base_attitude_error"].tolist(),
                        "base_angular_speed_rad_s": raw._base_angular_speed.tolist(),
                        "tool_axis_alignment": raw._alignment.tolist(),
                        "tool_speed_m_s": info["wasman_tool_speed"].tolist(),
                        "hold_steps": raw.valve_contract.hold_steps.tolist(),
                        "wheel_clearance_m": raw.wheel_clearance().tolist(),
                        "joint_positions": raw.robot.data.joint_pos.torch.tolist(),
                    }
                )
                if args.action_trace:
                    grip_limits = raw.robot.data.soft_joint_pos_limits.torch[:, raw._gripper_joint_ids]
                    grip_target = grip_limits[..., 0] + 0.5 * (raw.actions[:, 10:11] + 1) * (
                        grip_limits[..., 1] - grip_limits[..., 0]
                    )
                    trace[-1]["action_diagnostic"] = {
                        "observation_before_action": before_action.tolist(),
                        "actor_output_unclipped": actor_output.tolist(),
                        "applied_action": raw.actions.tolist(),
                        "previous_applied_action": raw.previous_actions.tolist(),
                        "arm_target_rad": raw._arm_targets.tolist(),
                        "arm_position_rad": raw.robot.data.joint_pos.torch[:, raw._arm_joint_ids].tolist(),
                        "gripper_target_m": grip_target.tolist(),
                        "gripper_position_m": raw.robot.data.joint_pos.torch[:, raw._gripper_joint_ids].tolist(),
                        "tool_quaternion_xyzw": raw.robot.data.body_link_quat_w.torch[:, raw._tool_body_id].tolist(),
                        "wheel_center_m": raw.button.data.body_link_pos_w.torch[:, raw._button_body_id].tolist(),
                        "bilateral_contact": info["wasman_grasped"].tolist(),
                        "diagnostic_base_x_latched": diagnostic_latched.tolist(),
                    }
                if step % 150 == 0:
                    print(
                        f"step={step} phases={torch.bincount(phase, minlength=8).tolist()} "
                        f"angle={trace[-1]['angle_rad'][0]:.3f} forces={trace[-1]['finger_forces_n'][0]} "
                        f"grasp_turn={trace[-1]['grasp_turn_rad'][0]:.3f} success={int(success.sum())} "
                        f"navigation={trace[-1]['navigation'][0] if trace[-1]['navigation'] else 'short-start'}",
                        flush=True,
                    )
                if args.real_time:
                    time.sleep(max(0.0, raw.step_dt - (time.perf_counter() - started)))
                if args.success_tail is not None and bool(success.all()):
                    if first_success_step is None:
                        first_success_step = step
                    if step - first_success_step >= round(args.success_tail / raw.step_dt):
                        break
        summary = {
            "robot_geometry_version": cfg.robot_geometry_version,
            "robot_camera_profile": cfg.robot_camera_profile,
            "robot_camera_parameters": robot_camera_profile(cfg.robot_camera_profile).metadata(),
            "registered_geometry_audit": geometry_audit,
            "task": task,
            "controller": "DIAGNOSTIC INTERVENTION: learned actor with longitudinal base command frozen after grasp"
            if args.diagnostic_freeze_base_x
            else (
                "state-based local-linear imitation policy; optional learned history partition; "
                "not PPO or a neural actor; no expert at inference"
            )
            if local_policy
            else "recurrent state-only neural imitation policy; no expert at inference"
            if recurrent_policy
            else "learned local/recurrent imitation mixture; training-fitted routing; no expert at inference"
            if mixture_policy
            else "development batch of independent learned candidates; not a single-policy score"
            if batch_policy
            else "learned state-only actor; no expert at inference"
            if policy is not None
            else "Cartesian state-feedback expert; not PPO",
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "action_trace_enabled": args.action_trace,
            "diagnostic_execution_override": args.diagnostic_execution,
            "diagnostic_freeze_base_x": args.diagnostic_freeze_base_x,
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
            "training_metadata": policy_info,
            "success_contract": valve_success_metadata(cfg.valve_success_contract),
            "expert_options": {
                "strict_unload": args.strict_unload,
                "compliant_unload": args.compliant_unload,
                "hold_angle_deg": args.expert_hold_angle,
                "axial_release": args.axial_release,
            }
            if ctrl is not None
            else None,
            "inference_parameters": inference_parameters,
            "torch_numerics": {
                "version": torch.__version__,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            "seed": args.seed,
            "moviepy_version": importlib.metadata.version("moviepy") if args.record else None,
            "source_sha256_at_start": source_hashes,
            "randomized": args.randomized,
            "num_envs": args.num_envs,
            "steps": len(trace),
            "requested_steps": args.steps,
            "duration_s": len(trace) * raw.step_dt,
            "first_success_time_s": [
                next((frame["t"] for frame in trace if frame["success"][i]), None) for i in range(raw.num_envs)
            ],
            "episodes": args.num_envs,
            "successes": int(success.sum()),
            "failed_resets": int(failed.sum()),
            "task_success_observed": bool(success.all()),
            "success_per_env": success.tolist(),
            "max_phase_per_env": max_phase.tolist(),
            "bilateral_contact_steps": bilateral_steps.tolist(),
            "phase_names": ValveExpert.phase_names,
            "phase_source": "expert state" if ctrl is not None else "measured-state summary, not policy input",
            "protocol": {
                **valve_success_metadata(cfg.valve_success_contract),
                "physics_dt_s": cfg.sim.dt,
                "policy_dt_s": raw.step_dt,
                "horizon_s": cfg.episode_length_s,
                "current_speed_range_m_s": cfg.current_speed_range,
                "volume_scale_range": cfg.volume_scale_range,
                "seal_static_friction_nm": cfg.scene.button.actuators["passive"].friction,
                "seal_dynamic_friction_nm": cfg.scene.button.actuators["passive"].dynamic_friction,
                "seal_viscous_damping_nm_s_rad": cfg.scene.button.actuators["passive"].damping,
                "observation_dim": cfg.observation_space,
                "action_dim": cfg.action_space,
            },
            "final": trace[-1],
            "camera_streams": camera_recording.metadata() if camera_recording else {"enabled": False},
        }
        if batch_policy:
            summary["evaluation_role"] = "Development sweep; aggregate successes are not a single-policy score"
            summary["per_policy"] = [
                {
                    **candidate,
                    "episodes": len(candidate["environment_ids"]),
                    "successes": sum(summary["success_per_env"][i] for i in candidate["environment_ids"]),
                    "failed_resets": int(failed[candidate["environment_ids"]].sum()),
                }
                for candidate in policy_info["candidates"]
            ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "trace.json").write_text(json.dumps({**summary, "trace": trace}, indent=2) + "\n")
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if args.require_success and not success.all():
            raise RuntimeError(f"Some valve attempts failed the selected criterion: {cfg.valve_success_contract}")
    finally:
        if camera_recording:
            camera_recording.close()
        env.close()


exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    launcher.app.close(exit_code=exit_code)
