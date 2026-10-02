# ruff: noqa: E402
"""Standalone ACT/DP rollout: wrist image + measured robot state -> EE chunk -> IK.

No expert is constructed. First-episode success/reset are censored separately.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from collections import deque
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--seeds", type=int, nargs="+", required=True)
p.add_argument("--purpose", choices=["development", "validation", "test", "research"], default="development")
p.add_argument("--steps", type=int, default=2240)
p.add_argument("--integral-multiplier", type=float, choices=[0.0, 0.5, 1.0], default=1.0)
p.add_argument(
    "--hydro-condition",
    choices=["nominal", "published-damping", "published-added-mass", "published-both"],
    default="nominal",
)
p.add_argument("--record-video", action="store_true", help="Record post-physics wrist RGB for a single episode")
AppLauncher.add_app_launcher_args(p)
p.set_defaults(visualizer=["none"], enable_cameras=True)
args = p.parse_args()
if args.output_dir.exists() or len(set(args.seeds)) != len(args.seeds):
    p.error("Fresh output directory and distinct seeds required")
if args.record_video and len(args.seeds) != 1:
    p.error("Video mode requires one seed")
launcher = AppLauncher(args)

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as TF
from isaaclab.utils.math import quat_from_matrix
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_tasks.utils import resolve_task_config
from PIL import Image
from train_valve_act import make_actor

import wasman.tasks  # noqa: F401
from wasman.assets.registered_bluerov import configure_registered_bluerov
from wasman.assets.robot_cameras import add_robot_cameras
from wasman.controllers.ee_target import RELATIVE_EE, EETargetInterface
from wasman.controllers.valve_success import valve_success_metadata
from wasman.learning.valve_visual_dataset import normalize_act_batch


def main():
    torch.set_num_threads(8)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    from wasman.learning.revision_protocol import validate_evaluation

    validate_evaluation(config, "RotateValve", args.seeds, args.purpose, steps=args.steps)
    if set(args.seeds) & set(config["train_seeds"] + config.get("validation_seeds", [])):
        raise ValueError("Evaluation seeds overlap model development episodes")
    if config["model"] == "BC":
        from wasman.learning.chunk_bc import make_bc

        actor = make_bc("RotateValve").cuda().eval()
    elif config["model"] == "ACT":
        actor = make_actor(config["image_size"], pretrained=False).cuda().eval()
    elif config["model"] == "DP":
        from wasman.learning.valve_dp import dp_lowdim, make_dp

        actor = make_dp(pretrained=False).cuda().eval()
    else:
        raise ValueError("Unsupported policy")
    actor.load_state_dict(checkpoint["state_dict"])
    from wasman.learning.revision_protocol import configure_inference

    configure_inference(actor, config)
    actor.cuda()  # DP reconstructs normalization parameters on CPU during load_state_dict.
    stats = checkpoint.get("statistics")
    del checkpoint
    task = "Wasman-Underwater-RotateValve-T200-Direct"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=("physics=isaacsim_physx",))
    configure_registered_bluerov(cfg)
    from wasman.learning.revision_diagnostics import (
        apply_condition,
        reset_state,
        trace_state,
        validate_condition_cohort,
    )

    validate_condition_cohort(config, args.seeds, args.purpose, args.integral_multiplier, args.hydro_condition)
    condition = apply_condition(cfg, integral_multiplier=args.integral_multiplier, hydro_condition=args.hydro_condition)
    cfg.scene.num_envs, cfg.seed = len(args.seeds), args.seeds[0]
    cfg.current_speed_range = cfg.current_vertical_range = (0.0, 0.0)
    cfg.turbulence_sigma = 0.0
    cfg.button_y_range, cfg.button_z_range = (-0.10, -0.10), (0.78, 0.78)
    cfg.sim.render_interval = cfg.decimation
    add_robot_cameras(cfg, width=config["image_size"], height=config["image_size"], profile="geometry-v2")
    cfg.scene.base_camera = None
    cfg.scene.gripper_camera.update_period = 0.0
    cfg.scene.gripper_camera.renderer_cfg = IsaacRtxRendererCfg()
    if not 0 < args.steps < round(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)) - 1:
        raise ValueError("Evaluation must end before reset timeout including warmup")
    args.output_dir.mkdir(parents=True)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    n = len(args.seeds)
    active = torch.ones(n, dtype=torch.bool, device=raw.device)
    succeeded = torch.zeros_like(active)
    first_success = torch.full((n,), -1, device=raw.device, dtype=torch.long)
    first_reset = torch.full_like(first_success, -1)
    trace, predictions = [], []
    policy_hz = config.get("policy_hz", 20)
    history = deque(maxlen=3 if policy_hz == 20 else 2)
    replan_steps = round(config.get("n_action_steps", 16 if config["model"] == "ACT" else 8) * 30 / policy_hz)
    start = time.perf_counter()
    writer = None
    try:
        env.reset(seed=args.seeds[0])
        for i, seed in enumerate(args.seeds):
            raw.seed(seed)
            raw._reset_idx(torch.tensor([i], device=raw.device, dtype=torch.int32))
        raw.scene.write_data_to_sim()
        raw.sim.forward()
        initial_condition = reset_state(raw)
        warmup = torch.zeros((n, 11), device=raw.device)
        warmup[:, :3] = (
            raw.robot.data.body_link_pos_w.torch[:, raw._base_body_id]
            - raw.scene.env_origins
            - raw._base_target_nominal
        ) / raw._base_target_position_scale
        _, _, done, timeout, info = env.step(warmup)
        if (done | timeout).any() or info["wasman_valve_angle"].abs().max() > 0.001:
            raise RuntimeError("Camera warmup changed task")
        camera = raw.scene["gripper_camera"]
        _ = camera.data
        for _ in range(8):
            raw.sim.forward()
            raw.scene.update(0.0)
            launcher.app.update()
            camera.update(0.0, force_recompute=True)
        interface = EETargetInterface(raw, frame=RELATIVE_EE)
        torch.manual_seed(config.get("rollout_seed", config["seed"]))
        if args.record_video:
            import imageio_ffmpeg

            writer = imageio_ffmpeg.write_frames(
                str(args.output_dir / "wrist.mp4"),
                (config["image_size"], config["image_size"]),
                fps=1 / raw.step_dt,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                quality=8,
                output_params=["-movflags", "+faststart"],
            )
            writer.send(None)
        with torch.inference_mode():
            for step in range(args.steps):
                measured = interface.capture_anchor()
                limits = raw.robot.data.soft_joint_pos_limits.torch[:, raw._gripper_joint_ids[0]]
                grip = raw.robot.data.joint_pos.torch[:, raw._gripper_joint_ids[0]]
                grip = 2 * (grip - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1
                state = torch.cat(
                    (measured.position_w - raw.scene.env_origins, measured.quaternion_w, grip[:, None]), -1
                )
                if config["model"] == "DP":
                    rgb = camera.data.output["rgb"].torch[..., :3].permute(0, 3, 1, 2).clone()
                    history.append((rgb, state.cpu().numpy()))
                    if step == 0:
                        start_state = state.cpu().numpy().copy()
                if step % replan_steps == 0:
                    anchor = measured
                    if config["model"] in ("ACT", "BC"):
                        if config["state_mode"] == "local":
                            state[:, :7] = 0
                            state[:, 6] = 1
                        rgb = camera.data.output["rgb"].torch[..., :3].permute(0, 3, 1, 2).clone()
                        batch = normalize_act_batch(
                            {"observation.images.wrist": rgb, "observation.state": state}, stats, raw.device
                        )
                        with torch.autocast(
                            "cuda", dtype=torch.bfloat16, enabled=config.get("inference_precision") != "fp32"
                        ):
                            chunk = actor.predict_action_chunk(batch).float()
                        chunk = chunk * stats["action_std"].to(raw.device) + stats["action_mean"].to(raw.device)
                    else:
                        lowdim = [
                            dp_lowdim(np.stack((history[0][1][i], history[-1][1][i])), start_state[i]) for i in range(n)
                        ]
                        batch = {
                            k: torch.from_numpy(np.stack([x[k] for x in lowdim])).to(raw.device) for k in lowdim[0]
                        }
                        frames = torch.stack((history[0][0], history[-1][0]), dim=1).float() / 255
                        batch["camera0_rgb"] = TF.resize(frames.flatten(0, 1), [224, 224], antialias=True).reshape(
                            n, 2, 3, 224, 224
                        )
                        with torch.autocast(
                            "cuda", dtype=torch.bfloat16, enabled=config.get("inference_precision") != "fp32"
                        ):
                            output = actor.predict_action(batch)["action"].float()
                        d6 = output[..., 3:9]
                        axis1 = torch.nn.functional.normalize(d6[..., :3], dim=-1)
                        axis2 = torch.nn.functional.normalize(
                            d6[..., 3:] - (axis1 * d6[..., 3:]).sum(-1, keepdim=True) * axis1, dim=-1
                        )
                        matrix = torch.stack((axis1, axis2, torch.cross(axis1, axis2, dim=-1)), dim=-2)
                        quat = quat_from_matrix(matrix)
                        chunk = torch.cat((output[..., :3], quat, output[..., 9:]), -1)
                    predictions.append(
                        dict(
                            step=step,
                            chunk=chunk.cpu(),
                            input_state=state.cpu(),
                            anchor=torch.cat(
                                (anchor.position_w - raw.scene.env_origins, anchor.quaternion_w), -1
                            ).cpu(),
                        )
                    )
                    if step in (0, 288, 600, 912):
                        Image.fromarray(rgb[0].permute(1, 2, 0).cpu().numpy()).save(
                            args.output_dir / f"wrist_{step}.png"
                        )
                # Declared policy trajectory, sampled by the unchanged 30 Hz IK.
                offset = (step % replan_steps) * policy_hz / 30
                lo, hi = min(int(offset), 15), min(int(offset) + 1, 15)
                a, b = chunk[:, lo], chunk[:, hi].clone()
                b[:, 3:7] *= torch.where((a[:, 3:7] * b[:, 3:7]).sum(-1, keepdim=True) < 0, -1, 1)
                target = a.lerp(b, offset - int(offset))
                action = interface.actions(target, anchor=anchor)
                _, _, done, timeout, info = env.step(action)
                if writer is not None:
                    writer.send(camera.data.output["rgb"].torch[0, ..., :3].contiguous().cpu().numpy())
                terminal = done | timeout
                success = info["wasman_success"] & active & ~terminal
                first_success[success] = step + 1
                first_reset[terminal & active] = step + 1
                succeeded |= success
                trace.append(
                    dict(
                        step=step,
                        active=active.cpu().clone(),
                        target=target.cpu(),
                        action=action.cpu(),
                        measured=torch.cat(
                            (measured.position_w - raw.scene.env_origins, measured.quaternion_w), -1
                        ).cpu(),
                        angle=info["wasman_valve_angle"].cpu().clone(),
                        reset=terminal.cpu().clone(),
                        grasped=info["wasman_grasped"].cpu().clone(),
                        success=success.cpu(),
                        **trace_state(raw),
                    )
                )
                active &= ~(success | terminal)
                if step % 150 == 0:
                    print(
                        f"step={step} success={int(succeeded.sum())}/{n} active={int(active.sum())} "
                        f"angle={torch.rad2deg(info['wasman_valve_angle']).tolist()}",
                        flush=True,
                    )
                if not active.any():
                    break
        if writer is not None:
            writer.close()
            writer = None
        valid = torch.stack([x["active"] & ~x["reset"] for x in trace])
        angles = torch.stack([x["angle"] for x in trace])
        report = dict(
            model=config["model"],
            purpose=args.purpose,
            diagnostic_condition=condition,
            initial_reset=initial_condition,
            checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            seeds=args.seeds,
            successes=int(succeeded.sum()),
            episodes=n,
            success_per_seed=succeeded.tolist(),
            first_success_step=first_success.tolist(),
            first_reset_step=first_reset.tolist(),
            max_angle_deg=torch.rad2deg(torch.where(valid, angles, -torch.inf).amax(0)).tolist(),
            ever_grasped=(torch.stack([x["grasped"] for x in trace]) & valid).any(0).tolist(),
            elapsed_s=time.perf_counter() - start,
            success_contract=valve_success_metadata(cfg.valve_success_contract),
            task_horizon_s=cfg.episode_length_s,
            evaluated_steps=args.steps,
            step_dt=raw.step_dt,
            inference_inputs=["wrist RGB", "measured robot EE pose and jaw"],
            expert_at_inference=False,
            policy_random_seed=config.get("rollout_seed", config["seed"]),
            inference_precision=config.get("inference_precision", "bfloat16"),
            asset_profile=config.get("asset_profile", "historical-cad-v1"),
            video_frames=len(trace) if args.record_video else 0,
            video_sampling="post-physics at 30 Hz" if args.record_video else None,
            replan_every_control_steps=replan_steps,
            policy_hz=policy_hz,
            target_interpolation="xyz/jaw linear, quaternion normalized shortest arc",
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        torch.save(dict(trace=trace, predictions=predictions, summary=report), args.output_dir / "rollout.pt")
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
