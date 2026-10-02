# ruff: noqa: E402
"""Versioned first-episode collection, replay and standalone policy evaluation."""

import argparse
import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--task", choices=["PressButton"], default="PressButton")
p.add_argument("--mode", choices=["collect", "expert", "zero", "replay", "policy"], required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--seeds", type=int, nargs="+", required=True)
p.add_argument("--steps", type=int, default=470)
p.add_argument("--checkpoint", type=Path)
p.add_argument("--replay-episode", type=Path)
p.add_argument("--purpose", choices=["development", "validation", "test", "single42"], default="development")
p.add_argument("--observer-video", action="store_true")
AppLauncher.add_app_launcher_args(p)
p.set_defaults(visualizer=["none"])
a = p.parse_args()
if a.output_dir.exists() or len(set(a.seeds)) != len(a.seeds):
    p.error("Fresh directory and distinct seeds required")
if a.mode == "policy" and not a.checkpoint:
    p.error("Policy requires checkpoint")
if a.mode == "replay" and (not a.replay_episode or len(a.seeds) != 1):
    p.error("Replay requires one recorded episode")
if a.observer_video and len(a.seeds) != 1:
    p.error("Video requires one environment")
a.enable_cameras = a.mode in ["collect", "policy"] or a.observer_video
launcher = AppLauncher(a)
import gymnasium as gym
import numpy as np
import torch
from button_visual_support import ButtonTeacher, capture, contract, measured_state, pack_action, unpack_action, verify
from isaaclab_tasks.utils import resolve_task_config

import wasman.tasks  # noqa: F401


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(16 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    task = "Wasman-Underwater-PressButton-T200-Direct"
    cfg, _ = resolve_task_config(task, "", overrides=("physics=isaacsim_physx",))
    cfg.scene.num_envs, cfg.seed = len(a.seeds), a.seeds[0]
    if not 0 < a.steps < 479:
        raise ValueError("End before reset horizon")
    if a.mode in ["collect", "policy"]:
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        from wasman.assets.robot_cameras import add_robot_cameras

        add_robot_cameras(cfg, width=384, height=384, profile="legacy-v1")
        cfg.scene.base_camera = None
        cfg.scene.gripper_camera.update_period = 0.0
        cfg.scene.gripper_camera.renderer_cfg = IsaacRtxRendererCfg()
    if a.observer_video:
        import isaaclab.sim as sim
        from isaaclab.sensors import CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        cfg.scene.observer = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Observer",
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim.PinholeCameraCfg(focal_length=24, horizontal_aperture=36, clipping_range=(0.03, 12)),
            renderer_cfg=IsaacRtxRendererCfg(),
        )
    actor, stats, config = None, None, {}
    if a.mode == "policy":
        checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        from wasman.learning.revision_protocol import PROTOCOL, validate_evaluation

        validate_evaluation(config, a.task, a.seeds, a.purpose, steps=a.steps)
        if config.get("protocol") != PROTOCOL:
            validation_start, test_start = (9100, 9110)
            if a.purpose == "validation" and set(a.seeds) != set(range(validation_start, validation_start + 8)):
                raise ValueError("Validation split mismatch")
            if a.purpose == "test" and set(a.seeds) != set(range(test_start, test_start + 30)) | {42}:
                raise ValueError("Final split mismatch")
            if a.purpose == "single42" and a.seeds != [42]:
                raise ValueError("Standalone42 only")
        if config.get("task") != a.task or config["policy_hz"] != 30:
            raise ValueError("Checkpoint task/interface mismatch")
        if set(a.seeds) & set(config["train_seeds"] + config.get("validation_seeds", [])):
            raise ValueError("Train/evaluation leakage")
        if config["model"] == "BC":
            from wasman.learning.chunk_bc import make_bc

            actor = make_bc(a.task)
        elif config["model"] == "ACT":
            from train_button_act import make_actor

            actor = make_actor(pretrained=False)
        else:
            from button_dp import make_dp

            actor = make_dp(pretrained=False)
        actor.load_state_dict(checkpoint["state_dict"])
        from wasman.learning.revision_protocol import configure_inference

        configure_inference(actor, config)
        actor.cuda().eval()
        stats = checkpoint.get("statistics")
        del checkpoint
    a.output_dir.mkdir(parents=True)
    paths = [
        Path(__file__),
        Path("scripts/button_visual_support.py"),
        Path("checkpoints/wasman_press_button_smooth_seed42.pt"),
        Path("src/wasman/tasks/underwater_press_button/config/bluerov2_alpha/smooth.py"),
        Path("src/wasman/controllers/marine_mechanism.py"),
        Path("src/wasman/controllers/marine_evidence.py"),
        Path("src/wasman/controllers/tool_pose.py"),
        Path("src/wasman/controllers/hatch_actuator.py"),
        Path("src/wasman/tasks/underwater_panel/linear_cfg.py"),
        Path("src/wasman/tasks/underwater_panel/env_cfg.py"),
        Path("src/wasman/tasks/underwater_press_button/config/bluerov2_alpha/env.py"),
        Path("src/wasman/tasks/underwater_press_button/config/bluerov2_alpha/env_cfg.py"),
    ]
    paths += list(Path("src/wasman/assets/data/objects/marine_mechanisms_v1").glob("*.usda"))
    hashes = {str(p): sha(p) for p in paths}
    (a.output_dir / "source_manifest.json").write_text(json.dumps(hashes, indent=2))
    (a.output_dir / "contract.json").write_text(json.dumps(contract(cfg), indent=2))
    env = gym.make(task, cfg=cfg)
    r = env.unwrapped
    writer = None
    buffers = []
    rows = []
    records = [[] for _ in a.seeds]
    start = time.perf_counter()
    try:
        env.reset(seed=a.seeds[0])
        for i, seed in enumerate(a.seeds):
            r.seed(seed)
            r._reset_idx(torch.tensor([i], device=r.device, dtype=torch.int32))
        r.scene.write_data_to_sim()
        r.sim.forward()
        warmup = torch.zeros_like(r.actions)
        warmup[:, :3] = (
            r.robot.data.body_link_pos_w.torch[:, r._base_body_id] - r.scene.env_origins - r._base_target_nominal
        ) / r._base_target_position_scale
        _, _, done, timeout, info = env.step(warmup)
        if (done | timeout | info["wasman_success"]).any():
            raise ValueError("Warmup invalid")
        camera = r.scene["gripper_camera"] if a.mode in ["collect", "policy"] else None
        if camera is not None:
            _ = camera.data
            for _ in range(8):
                r.sim.forward()
                r.scene.update(0)
                launcher.app.update()
                camera.update(0, force_recompute=True)
        if a.observer_video:
            import imageio_ffmpeg

            cam = r.scene["observer"]
            origins = r.scene.env_origins
            cam.set_world_poses_from_view(
                eyes=origins + origins.new_tensor([-0.35, -1.1, 1.05]),
                targets=origins + origins.new_tensor([0.79, -0.03, 0.67]),
            )
            writer = imageio_ffmpeg.write_frames(
                str(a.output_dir / "observer.mp4"),
                (1280, 720),
                fps=30,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                quality=8,
                output_params=["-movflags", "+faststart"],
            )
            writer.send(None)
        expert = None
        if a.mode in ["collect", "expert"]:
            expert = ButtonTeacher(r)
        replay = None
        if a.mode == "replay":
            replay = np.load(a.replay_episode / "trajectory.npz")["command"]
            meta = json.loads((a.replay_episode / "metadata.json").read_text())
            if meta["seed"] != a.seeds[0] or meta["task"] != a.task:
                raise ValueError("Replay identity mismatch")
        if a.mode == "collect":
            for seed in a.seeds:
                folder = a.output_dir / f"seed_{seed}"
                folder.mkdir()
                buffers.append(np.memmap(folder / "wrist.rgb", mode="w+", dtype=np.uint8, shape=(a.steps, 384, 384, 3)))
        torch.manual_seed(config.get("rollout_seed", config.get("seed", 42)))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        active = torch.ones(r.num_envs, dtype=torch.bool, device=r.device)
        history = deque(maxlen=2)
        predictions = []
        chunk = None
        with torch.inference_mode():
            for step in range(a.steps):
                measured = measured_state(r).clone()
                rgb = camera.data.output["rgb"].torch[..., :3].clone() if camera is not None else None
                if step == 0 and rgb is not None:
                    np.savez_compressed(a.output_dir / "initial_rgb.npz", rgb=rgb.cpu().numpy())
                if a.mode == "collect":
                    import warp as wp

                    timestamps = wp.to_torch(camera._timestamp_last_update).cpu().numpy().copy()
                    frames = camera.frame.torch.cpu().numpy().copy()
                if a.mode == "policy":
                    history.append((rgb.permute(0, 3, 1, 2), measured))
                    replan = config.get("n_action_steps", 16 if config["model"] == "ACT" else 8)
                    if step % replan == 0:
                        if config["model"] in ("ACT", "BC"):
                            from wasman.learning.marine_visual_dataset import normalize_act_batch

                            batch = normalize_act_batch(
                                {"observation.images.wrist": history[-1][0], "observation.state": measured},
                                stats,
                                r.device,
                            )
                            chunk = actor.predict_action_chunk(batch).float() * stats["action_std"].to(
                                r.device
                            ) + stats["action_mean"].to(r.device)
                        else:
                            import torchvision.transforms.functional as TF

                            im = torch.stack([history[0][0], history[-1][0]], 1).float() / 255
                            obs = {
                                "camera0_rgb": TF.resize(im.flatten(0, 1), [224, 224], antialias=True).reshape(
                                    r.num_envs, 2, 3, 224, 224
                                ),
                                "robot_state": torch.stack([history[0][1], history[-1][1]], 1),
                            }
                            chunk = actor.predict_action(obs)["action"].float()
                        if not torch.isfinite(chunk).all():
                            raise ValueError("Nonfinite network output")
                        predictions.append({"step": step, "input_state": measured.cpu(), "chunk": chunk.cpu()})
                    action = unpack_action(chunk[:, step % replan])
                elif expert is not None:
                    action = expert.actions()
                elif replay is not None:
                    action = unpack_action(torch.from_numpy(replay[min(step, len(replay) - 1)]).to(r.device)[None])
                else:
                    action = torch.zeros_like(r.actions)
                command = pack_action(action)
                _, reward, done, timeout, info = env.step(action)
                if not torch.isfinite(reward).all():
                    raise ValueError("Nonfinite reward")
                evidence = {k: v.cpu().numpy().copy() for k, v in capture(r).items()}
                evidence.update(
                    active=active.cpu().numpy().copy(),
                    terminal=(done | timeout).cpu().numpy().copy(),
                    success=info["wasman_success"].cpu().numpy().copy(),
                    measured=measured.cpu().numpy().copy(),
                    command=command.cpu().numpy().copy(),
                    actuator=action.cpu().numpy().copy(),
                )
                rows.append(evidence)
                if a.mode == "collect":
                    for i in range(r.num_envs):
                        if not evidence["active"][i]:
                            continue
                        buffers[i][len(records[i])] = rgb[i].cpu().numpy()
                        record = {k: v[i] for k, v in evidence.items()}
                        record.update(
                            time=step * r.step_dt,
                            camera_time=timestamps[i],
                            camera_frame=frames[i],
                            success_after=evidence["success"][i],
                        )
                        records[i].append(record)
                        if step in [0, 300, 600, 900]:
                            from PIL import Image

                            Image.fromarray(rgb[i].cpu().numpy()).save(
                                a.output_dir / f"seed_{a.seeds[i]}" / f"wrist_{step}.png"
                            )
                if writer is not None:
                    blend = min(max((step * r.step_dt - 3) / 9, 0), 1)
                    blend = blend * blend * (3 - 2 * blend)
                    origin = r.scene.env_origins
                    eye = origin.new_tensor([-0.35, -1.1, 1.05]).lerp(origin.new_tensor([0.25, -0.65, 0.98]), blend)
                    center = r.button.data.root_pos_w.torch.clone()
                    center[:, 0] = origin[:, 0] + 0.77
                    cam.set_world_poses_from_view(eyes=origin + eye, targets=center)
                    r.sim.render()
                    cam.update(0, force_recompute=True)
                    frame = cam.data.output["rgb"].torch[0, :, :, :3].contiguous().cpu().numpy()
                    writer.send(frame)
                    if step in [30, 450, 900]:
                        from PIL import Image

                        Image.fromarray(frame).save(a.output_dir / f"observer_{step}.png")
                active &= ~(done | timeout | info["wasman_success"])
                if step % 150 == 0:
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "active": int(active.sum()),
                                "progress": ((evidence["q"] - cfg.mechanism_initial_position) * cfg.mechanism_direction)
                                .round(4)
                                .tolist(),
                                "elapsed_s": round(time.perf_counter() - start, 1),
                            }
                        ),
                        flush=True,
                    )
                if not active.any():
                    break
        arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
        np.savez_compressed(a.output_dir / "trace.npz", **arrays)
        verified = verify(arrays, contract(cfg))
        if a.mode == "collect":
            for i, seed in enumerate(a.seeds):
                buf = buffers[i]
                buf.flush()
                buf._mmap.close()
                folder = a.output_dir / f"seed_{seed}"
                n = len(records[i])
                with (folder / "wrist.rgb").open("r+b") as f:
                    f.truncate(n * 384 * 384 * 3)
                trajectory = {k: np.stack([x[k] for x in records[i]]) for k in records[i][0]}
                np.savez_compressed(folder / "trajectory.npz", **trajectory)
                meta = dict(
                    asset_profile=os.environ.get("WASMAN_ASSET_PROFILE", "historical-cad-v1"),
                    schema="wasman-marine-rgb-actuator-v1",
                    teacher="frozen privileged PPO, demonstration generation only",
                    task=a.task,
                    seed=seed,
                    outcome="success" if verified["success_per_seed"][i] else "failure",
                    length=n,
                    image_shape=[384, 384, 3],
                    image_dtype="uint8",
                    raw_hz=30,
                    physics_dt=cfg.sim.dt,
                    success_contract=contract(cfg),
                    warmup_steps=1,
                    command_timing="image and measured state before command; physical evidence after",
                    camera_profile="legacy-v1",
                )
                (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
        if writer is not None:
            writer.close()
            writer = None
        report = dict(
            task=a.task,
            mode=a.mode,
            purpose=a.purpose,
            seeds=a.seeds,
            steps=len(rows),
            dt=r.step_dt,
            elapsed_s=time.perf_counter() - start,
            **verified,
            source_sha256=hashes,
            expert_at_inference=expert is not None,
            checkpoint_sha256=sha(a.checkpoint) if a.checkpoint else None,
            inference_precision="fp32",
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            tf32=False,
            video_frames=len(rows) if a.observer_video else 0,
        )
        (a.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        if predictions:
            torch.save(predictions, a.output_dir / "predictions.pt")
        print(json.dumps(report), flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


try:
    main()
except BaseException as error:
    import traceback

    traceback.print_exc()
    if a.output_dir.exists():
        (a.output_dir / "error.json").write_text(json.dumps({"error": repr(error)}))
    raise
finally:
    launcher.app.close()
