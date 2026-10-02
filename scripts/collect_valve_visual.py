# ruff: noqa: E402
"""Record first-episode, pre-action wrist RGB and absolute EE expert targets.

Physics/control remain 240/30 Hz. Raw samples are 30 Hz (not AM-Bench's
120 Hz); a training dataset may resample these to a logical 20 Hz timeline.
Each environment is censored independently at its first success or reset.
"""

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
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--seeds", type=int, nargs="+", required=True)
parser.add_argument("--steps", type=int, default=2240)
parser.add_argument("--size", type=int, default=384)
parser.add_argument("--replay-dataset", type=Path)
parser.add_argument("--replay-policy-hz", type=int, choices=[20, 30])
parser.add_argument("--expert-approach-offset", type=float)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["none"], enable_cameras=True)
args = parser.parse_args()
if args.replay_policy_hz and not args.replay_dataset:
    parser.error("Policy-clock playback requires recorded commands")
if args.expert_approach_offset is None:
    args.expert_approach_offset = -0.020 if os.environ.get("WASMAN_ASSET_PROFILE") == "open-procedural-v1" else 0.0
if args.output_dir.exists() or len(set(args.seeds)) != len(args.seeds):
    parser.error("Choose a fresh output directory and distinct seeds")
launcher = AppLauncher(args)

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_tasks.utils import resolve_task_config
from PIL import Image

import wasman.tasks  # noqa: F401
from wasman.assets.registered_bluerov import configure_registered_bluerov
from wasman.assets.robot_cameras import add_robot_cameras
from wasman.controllers.ee_target import ABSOLUTE_EE, EETargetInterface, encode_ee_target
from wasman.controllers.valve import ValveExpert
from wasman.controllers.valve_success import valve_success_metadata


class CaptureTargets:
    def __init__(self, controller):
        self.controller = controller

    def actions(self, position, quaternion, jaw):
        self.targets = position.clone(), quaternion.clone(), jaw.clone()
        return self.controller.actions(position, quaternion, jaw)


def main():
    task = "Wasman-Underwater-RotateValve-T200-Direct"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=("physics=isaacsim_physx",))
    configure_registered_bluerov(cfg)
    cfg.scene.num_envs, cfg.seed = len(args.seeds), args.seeds[0]
    cfg.current_speed_range = cfg.current_vertical_range = (0.0, 0.0)
    cfg.turbulence_sigma = 0.0
    cfg.button_y_range, cfg.button_z_range = (-0.10, -0.10), (0.78, 0.78)
    cfg.sim.render_interval = cfg.decimation
    add_robot_cameras(cfg, width=args.size, height=args.size, profile="geometry-v2")
    cfg.scene.base_camera = None
    cfg.scene.gripper_camera.update_period = 0.0
    cfg.scene.gripper_camera.renderer_cfg = IsaacRtxRendererCfg()
    if not 0 < args.steps < round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)) - 1:
        raise ValueError("Stop before timeout including one warmup step")
    args.output_dir.mkdir(parents=True)
    from wasman.learning.recorded_commands import RecordedCommands

    replay = (
        RecordedCommands(
            args.replay_dataset,
            args.seeds,
            profile=os.environ.get("WASMAN_ASSET_PROFILE", "historical-cad-v1"),
            policy_hz=args.replay_policy_hz,
            quaternion=True,
        )
        if args.replay_dataset
        else None
    )
    sources = [
        Path(__file__),
        *[
            Path("src/wasman") / p
            for p in (
                "controllers/valve.py",
                "controllers/tool_pose.py",
                "controllers/ee_target.py",
                "assets/robot_cameras.py",
                "robot_camera_profiles.py",
                "tasks/underwater_panel/valve.py",
                "tasks/underwater_panel/env_cfg.py",
            )
        ],
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for i, p in enumerate(sources):
        (args.output_dir / f"source_{i}_{p.name}").write_bytes(p.read_bytes())
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    records = [[] for _ in args.seeds]
    images = []
    paths = []
    for seed in args.seeds:
        path = args.output_dir / f"seed_{seed}"
        path.mkdir()
        paths.append(path)
        images.append(
            np.memmap(path / "wrist.rgb", mode="w+", dtype=np.uint8, shape=(args.steps, args.size, args.size, 3))
        )
    finished = np.zeros(len(args.seeds), dtype=bool)
    outcomes = [None] * len(args.seeds)
    start = time.perf_counter()
    try:
        env.reset(seed=args.seeds[0])
        for i, seed in enumerate(args.seeds):
            raw.seed(seed)
            raw._reset_idx(torch.tensor([i], device=raw.device, dtype=torch.int32))
        raw.scene.write_data_to_sim()
        raw.sim.forward()
        warmup = torch.zeros((len(args.seeds), 11), device=raw.device)
        warmup[:, :3] = (
            raw.robot.data.body_link_pos_w.torch[:, raw._base_body_id]
            - raw.scene.env_origins
            - raw._base_target_nominal
        ) / raw._base_target_position_scale
        _, _, done, timeout, info = env.step(warmup)
        if (done | timeout).any() or info["wasman_valve_angle"].abs().max() > 0.001:
            raise RuntimeError("Camera warmup changed the task or terminated")
        camera = raw.scene["gripper_camera"]
        _ = camera.data
        for _ in range(8):
            raw.sim.forward()
            raw.scene.update(0.0)
            launcher.app.update()
            camera.update(0.0, force_recompute=True)
        expert = None
        if replay is None:
            expert = ValveExpert(raw, hold_angle_deg=172.5, approach_offset_x=args.expert_approach_offset)
            capture = CaptureTargets(expert.ik)
            expert.ik = capture
        interface = EETargetInterface(raw)
        with torch.inference_mode():
            for step in range(args.steps):
                anchor = interface.capture_anchor()
                limits = raw.robot.data.soft_joint_pos_limits.torch[:, raw._gripper_joint_ids[0]]
                grip = raw.robot.data.joint_pos.torch[:, raw._gripper_joint_ids[0]]
                normalized_grip = 2 * (grip - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1
                measured = encode_ee_target(
                    anchor.position_w,
                    anchor.quaternion_w,
                    normalized_grip,
                    origins=raw.scene.env_origins,
                    frame=ABSOLUTE_EE,
                )
                rgb = camera.data.output["rgb"].torch[..., :3].cpu().numpy().copy()
                timestamps = wp.to_torch(camera._timestamp_last_update).cpu().numpy().copy()
                frames = camera.frame.torch.cpu().numpy().copy()
                if step == 0 and (rgb.reshape(len(args.seeds), -1).std(-1) < 1).any():
                    raise RuntimeError("Blank camera after warmup")
                if replay is None:
                    action = expert.actions()
                    command = encode_ee_target(*capture.targets, origins=raw.scene.env_origins, frame=ABSOLUTE_EE)
                else:
                    command = torch.from_numpy(replay.at(step)).to(raw.device)
                    action = interface.actions(command)
                measured, command, actuator = (v.cpu().numpy().copy() for v in (measured, command, action))
                _, reward, done, timeout, info = env.step(action)
                if not torch.isfinite(reward).all():
                    raise RuntimeError("Nonfinite simulation")
                terminal = (done | timeout).cpu().numpy()
                success = info["wasman_success"].cpu().numpy() & ~terminal
                angle = info["wasman_valve_angle"].cpu().numpy()
                for i, path in enumerate(paths):
                    if finished[i]:
                        continue
                    images[i][len(records[i])] = rgb[i]
                    records[i].append(
                        dict(
                            measured=measured[i],
                            command=command[i],
                            actuator=actuator[i],
                            time=step * raw.step_dt,
                            camera_time=timestamps[i],
                            camera_frame=frames[i],
                            angle_after=angle[i],
                            terminal=terminal[i],
                            success_after=success[i],
                        )
                    )
                    if step in (0, 300, 600, 900):
                        Image.fromarray(rgb[i]).save(path / f"wrist_{step}.png")
                    if success[i] or terminal[i]:
                        finished[i] = True
                        outcomes[i] = "success" if success[i] else "reset"
                if terminal.any() and expert is not None:
                    expert.reset(torch.as_tensor(np.flatnonzero(terminal), device=raw.device))
                if step % 150 == 0:
                    print(
                        f"step={step} finished={int(finished.sum())}/{len(args.seeds)} "
                        f"success={outcomes.count('success')} angle={np.rad2deg(angle).round(1).tolist()} "
                        f"wall_s={time.perf_counter() - start:.1f}",
                        flush=True,
                    )
                if finished.all():
                    break
        for i, path in enumerate(paths):
            images[i].flush()
            images[i]._mmap.close()
            length = len(records[i])
            with (path / "wrist.rgb").open("r+b") as f:
                f.truncate(length * args.size * args.size * 3)
            arrays = {k: np.stack([r[k] for r in records[i]]) for k in records[i][0]}
            np.savez_compressed(path / "trajectory.npz", **arrays)
            outcome = outcomes[i] or "horizon"
            (path / "metadata.json").write_text(
                json.dumps(
                    dict(
                        asset_profile=os.environ.get("WASMAN_ASSET_PROFILE", "historical-cad-v1"),
                        command_source="recorded playback" if replay else "feedback expert",
                        expert_approach_offset_m=args.expert_approach_offset,
                        schema="wasman-valve-rgb-ee-v1",
                        seed=args.seeds[i],
                        outcome=outcome,
                        length=length,
                        image_shape=[args.size, args.size, 3],
                        image_dtype="uint8",
                        raw_hz=1 / raw.step_dt,
                        physics_dt=cfg.sim.dt,
                        action_frame=ABSOLUTE_EE,
                        camera_profile="geometry-v2",
                        proprio="measured EE xyz, quaternion XYZW, normalized measured jaw",
                        success_contract=valve_success_metadata(cfg.valve_success_contract),
                        warmup_steps=1,
                        command_timing="image and measured pose before command; angle/success after",
                        max_angle_deg=float(np.rad2deg(arrays["angle_after"]).max()),
                    ),
                    indent=2,
                )
                + "\n"
            )
        report = dict(
            command_source="recorded playback" if replay else "feedback expert",
            replay_policy_hz=args.replay_policy_hz,
            replay_sources=replay.sources if replay else None,
            expert_approach_offset_m=args.expert_approach_offset,
            seeds=args.seeds,
            outcomes=[x or "horizon" for x in outcomes],
            lengths=[len(r) for r in records],
            source_sha256=hashes,
            elapsed_s=time.perf_counter() - start,
            task=task,
            deviations_from_ambench=[
                "raw data 30 Hz instead of 120 Hz",
                "75 second task horizon",
                "different robot, physics, scene and virtual camera",
            ],
        )
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
