# ruff: noqa: E402
"""Standalone ACT/DP rollout: wrist image + measured robot state -> absolute actuator chunk -> controller.

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
p.add_argument("--steps", type=int, default=1790)
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
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_tasks.utils import resolve_task_config
from PIL import Image
from train_hatch_act import make_actor

import wasman.tasks  # noqa: F401
from wasman.assets.registered_bluerov import configure_registered_bluerov
from wasman.assets.robot_cameras import add_robot_cameras
from wasman.controllers.hatch_actuator import measured_state, unpack_action
from wasman.learning.hatch_visual_dataset import normalize_act_batch


def main():
    torch.set_num_threads(8)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    from wasman.learning.revision_protocol import validate_evaluation

    validate_evaluation(config, "OpenHatch", args.seeds, args.purpose, steps=args.steps)
    if set(args.seeds) & set(config["train_seeds"] + config.get("validation_seeds", [])):
        raise ValueError("Evaluation seeds overlap model development episodes")
    if config["model"] == "BC":
        from wasman.learning.chunk_bc import make_bc

        actor = make_bc("OpenHatch").cuda().eval()
    elif config["model"] == "ACT":
        actor = make_actor(config["image_size"], pretrained=False).cuda().eval()
    elif config["model"] == "DP":
        from wasman.learning.hatch_dp import make_dp

        actor = make_dp(pretrained=False).cuda().eval()
    else:
        raise ValueError("Unsupported policy")
    actor.load_state_dict(checkpoint["state_dict"])
    from wasman.learning.revision_protocol import configure_inference

    configure_inference(actor, config)
    actor.cuda()  # DP reconstructs normalization parameters on CPU during load_state_dict.
    stats = checkpoint.get("statistics")
    del checkpoint
    task = "Wasman-Underwater-OpenHatch-Direct"
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
    history = deque(maxlen=3)
    replan_steps = round(config.get("n_action_steps", 16 if config["model"] == "ACT" else 8) * 30 / 20)
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
        if (done | timeout).any() or info["wasman_hatch_angle_rad"].abs().max() > 0.001:
            raise RuntimeError("Camera warmup changed task")
        camera = raw.scene["gripper_camera"]
        _ = camera.data
        for _ in range(8):
            raw.sim.forward()
            raw.scene.update(0.0)
            launcher.app.update()
            camera.update(0.0, force_recompute=True)
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
                state = measured_state(raw)
                if config["model"] == "DP":
                    rgb = camera.data.output["rgb"].torch[..., :3].permute(0, 3, 1, 2).clone()
                    history.append((rgb, state.cpu().numpy()))
                if step % replan_steps == 0:
                    if config["model"] in ("ACT", "BC"):
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
                        batch = {
                            "robot_state": torch.from_numpy(np.stack((history[0][1], history[-1][1]), axis=1)).to(
                                raw.device
                            )
                        }
                        frames = torch.stack((history[0][0], history[-1][0]), dim=1).float() / 255
                        batch["camera0_rgb"] = TF.resize(frames.flatten(0, 1), [224, 224], antialias=True).reshape(
                            n, 2, 3, 224, 224
                        )
                        with torch.autocast(
                            "cuda", dtype=torch.bfloat16, enabled=config.get("inference_precision") != "fp32"
                        ):
                            output = actor.predict_action(batch)["action"].float()
                        chunk = output
                    predictions.append(
                        dict(
                            step=step,
                            chunk=chunk.cpu(),
                            input_state=state.cpu(),
                        )
                    )
                    if step in (0, 288, 600, 912):
                        Image.fromarray(rgb[0].permute(1, 2, 0).cpu().numpy()).save(
                            args.output_dir / f"wrist_{step}.png"
                        )
                # 20 Hz absolute actuator targets, linearly sampled at 30 Hz.
                offset = (step % replan_steps) * 20 / 30
                lo, hi = min(int(offset), 15), min(int(offset) + 1, 15)
                a, b = chunk[:, lo], chunk[:, hi].clone()
                target = a.lerp(b, offset - int(offset))
                action = unpack_action(target)
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
                        measured=state.cpu().clone(),
                        angle=info["wasman_hatch_angle_rad"].cpu().clone(),
                        reset=terminal.cpu().clone(),
                        grasped=info["wasman_hatch_opposing_grasp"].cpu().clone(),
                        speed=info["wasman_hatch_speed_rad_s"].cpu().clone(),
                        grasp_motion=info["wasman_hatch_grasp_motion"].cpu().clone(),
                        ungrasped_motion=info["wasman_hatch_ungrasped_motion"].cpu().clone(),
                        forces=raw.finger_force_vectors().cpu().clone(),
                        near_handle=(raw._distance < cfg.tool_contact_distance).cpu().clone(),
                        tool_speed=info["wasman_tool_speed"].cpu().clone(),
                        stable=(
                            (raw._base_attitude_error < cfg.success_max_attitude_error)
                            & (raw._base_angular_speed < cfg.success_max_angular_speed)
                        )
                        .cpu()
                        .clone(),
                        success=success.cpu(),
                        **trace_state(raw),
                    )
                )
                active &= ~(success | terminal)
                if step % 150 == 0:
                    print(
                        f"step={step} success={int(succeeded.sum())}/{n} active={int(active.sum())} "
                        f"angle={torch.rad2deg(info['wasman_hatch_angle_rad']).tolist()}",
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
            success_contract=(
                "unchanged HatchContract; >=80deg + >=75deg grasped motion + "
                "<=5deg ungrasped motion + opposing grasp + stable one-second hold"
            ),
            task_horizon_s=cfg.episode_length_s,
            evaluated_steps=args.steps,
            step_dt=raw.step_dt,
            inference_inputs=["wrist RGB", "measured normalized base XYZ, four arm joints and jaw"],
            expert_at_inference=False,
            policy_random_seed=config.get("rollout_seed", config["seed"]),
            inference_precision=config.get("inference_precision", "bfloat16"),
            asset_profile=config.get("asset_profile", "historical-cad-v1"),
            video_frames=len(trace) if args.record_video else 0,
            video_sampling="post-physics at 30 Hz" if args.record_video else None,
            replan_every_control_steps=replan_steps,
            policy_hz=20,
            target_interpolation="linear absolute actuator targets; level attitude",
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
