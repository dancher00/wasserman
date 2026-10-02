"""Check the passive hatch and both moving RGB sensors; not a robot success demonstration."""

import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hatch_check"))
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"], enable_cameras=True)
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from PIL import Image  # noqa: E402

import wasman.tasks  # noqa: E402, F401


def main():
    task = "Wasman-Underwater-OpenHatch-Cameras-Direct"
    cfg, _ = resolve_task_config(task, "", overrides=("physics=isaacsim_physx",))
    cfg.scene.num_envs = 1
    cfg.current_speed_range = (0.0, 0.0)
    cfg.current_vertical_range = (0.0, 0.0)
    cfg.turbulence_sigma = 0.0
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"task": task, "seed": 42, "kind": "sensor and passive-mechanism smoke check, not a policy evaluation"}
    try:
        env.reset(seed=42)
        action = torch.zeros(env.action_space.shape, device=raw.device)
        action[:, 10] = -1  # Closed fingers; this sensor diagnostic stays clear of the handle.
        snapshots = {}
        for step in range(100):
            if step > 60:
                action[:, 6] = 0.12  # A real joint command to verify wrist camera motion.
            obs, reward, done, timeout, extra = env.step(action)
            for name in ("base_camera", "gripper_camera"):
                _ = raw.scene[name].data.output  # Read each step; sensors use lazy buffer updates.
            assert not done.any() and not timeout.any(), "Unexpected reset"
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            assert obs["policy"].shape == (1, 38)
            assert not extra["wasman_success"].any(), "A closed, untouched hatch cannot count as success"
            if step in (50, 99):
                snapshots[step] = {}
                for name in ("base_camera", "gripper_camera"):
                    camera = raw.scene[name]
                    rgb = camera.data.output["rgb"]
                    rgb = rgb.torch if hasattr(rgb, "torch") else rgb
                    pixels = rgb[0, ..., :3].cpu().numpy()
                    assert pixels.shape == (384, 384, 3)
                    Image.fromarray(pixels).save(args.output_dir / f"{name}_{step}.png")
                    snapshots[step][name] = {
                        "position_w": camera.data.pos_w.torch[0].tolist(),
                        "quaternion_xyzw_world": camera.data.quat_w_world.torch[0].tolist(),
                        "rgb_std": float(np.std(pixels)),
                    }
                    print(name, snapshots[step][name], flush=True)
                assert all(s["rgb_std"] > 5 for s in snapshots[step].values()), "Blank camera image"
        report["cameras"] = snapshots
        report["closed_angle_rad"] = raw.button.data.joint_pos.torch[0, raw._button_joint_id].item()
        assert abs(report["closed_angle_rad"]) < 0.03
        for name in ("base_camera", "gripper_camera"):
            p0 = np.array(snapshots[50][name]["position_w"])
            p1 = np.array(snapshots[99][name]["position_w"])
            report[name + "_motion_m"] = float(np.linalg.norm(p1 - p0))
            assert report[name + "_motion_m"] > 1e-4, "Sensor transform did not follow the robot"
        # A known external hinge effort tests the mechanism independently of the robot.
        # No joint-position teleport or drive target is used to open the lid.
        original_apply = raw._apply_action

        def probe():
            original_apply()
            raw.button.actuators.target_command.set_effort_index(
                value=torch.full((1, 1), 1.5, device=raw.device), joint_ids=raw._button_joint_ids
            )

        raw._apply_action = probe
        angles = []
        for _ in range(90):
            _, _, done, timeout, extra = env.step(action)
            assert not done.any() and not timeout.any()
            assert not extra["wasman_success"].any(), "External effort without grasp is not task success"
            angles.append(raw.button.data.joint_pos.torch[0, raw._button_joint_id].item())
        report["external_effort_probe_nm"] = 1.5
        report["probe_angle_rad"] = angles
        assert max(angles) > 0.2, "Hinge did not open under a physical test effort"
        assert max(angles) < np.radians(108), "Hinge exceeded its stop"
        # Separate sensor framing check: reset closed, physically deploy the arm
        # above the handle using the existing rate-limited Cartesian controller.
        # This is a camera preview, not an attempt to open the hatch.
        from wasman.controllers.tool_pose import ToolPoseController

        raw._apply_action = original_apply
        env.reset(seed=42)
        raw.button.actuators.target_command.set_effort_index(
            value=torch.zeros((1, 1), device=raw.device), joint_ids=raw._button_joint_ids
        )
        controller = ToolPoseController(raw)
        target = torch.tensor([[0.60, 0.0, 0.42]], device=raw.device)
        target_q = torch.tensor([[0.0, 2**-0.5, 0.0, 2**-0.5]], device=raw.device)
        for _ in range(420):
            _, _, done, timeout, extra = env.step(controller.actions(target, target_q, -1))
            assert not done.any() and not timeout.any()
            assert not extra["wasman_success"].any()
            for name in ("base_camera", "gripper_camera"):
                _ = raw.scene[name].data.output
        report["camera_preview_tool_position_w"] = raw.robot.data.body_link_pos_w.torch[0, raw._tool_body_id].tolist()
        report["camera_preview_target_error_m"] = float(
            (raw.robot.data.body_link_pos_w.torch[:, raw._tool_body_id] - target).norm()
        )
        for name in ("base_camera", "gripper_camera"):
            rgb = raw.scene[name].data.output["rgb"]
            rgb = rgb.torch if hasattr(rgb, "torch") else rgb
            Image.fromarray(rgb[0, ..., :3].cpu().numpy()).save(args.output_dir / f"{name}_deployed.png")
        report["passed"] = True
    finally:
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        env.close()
    print(json.dumps(report, indent=2))


exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    launcher.app.close(exit_code=exit_code)
