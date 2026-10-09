#!/usr/bin/env python3
"""Swim with a keyboard in a selected BlueROV2 scene, with a locked arm."""

# ruff: noqa: E402 -- Simulator imports must follow AppLauncher.
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("WASMAN_ASSET_PROFILE", "open-procedural-v1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")

from wasman.teleoperation import CONTROLS, ENVIRONMENTS, KeyboardState, PoseReference, rotate_vector


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--task", choices=ENVIRONMENTS)
    selection.add_argument("--env", help="Registered BlueROV2 environment ID")
    parser.add_argument("--list-envs", action="store_true", help="List scenes without starting the simulator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=0.20, help="Cruise speed in m/s")
    parser.add_argument("--turn-speed", type=float, default=0.45, help="Angular speed in rad/s")
    parser.add_argument("--steps", type=int, default=0, help="Stop after N control steps (0: unlimited)")
    parser.add_argument("--smoke-test", action="store_true", help="Exercise thrust, locked joints and reset")
    parser.add_argument("--water-test", action="store_true", help="Verify live water controls and camera with physics")
    parser.add_argument("--report", type=Path, help="Write small JSON telemetry for verification")
    parser.add_argument("--no-realtime", action="store_true", help="Run a bounded check as fast as possible")
    if "--list-envs" in sys.argv:
        for name, identifier in ENVIRONMENTS.items():
            print(f"{name:14} {identifier}")
        raise SystemExit(0)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--headless", action="store_true", help="Run a bounded check without a window")
    args = parser.parse_args()
    if args.speed <= 0 or args.turn_speed <= 0 or args.steps < 0:
        parser.error("Speeds must be positive and steps nonnegative")
    if args.headless and not (args.smoke_test or args.water_test or args.steps):
        parser.error("Headless mode requires --smoke-test, --water-test or --steps")
    if args.water_test:
        args.smoke_test = True
    args.enable_cameras = not args.headless or args.water_test
    if args.visualizer is not None and args.visualizer != (["none"] if args.headless else ["kit"]):
        parser.error("Keyboard teleop uses --viz kit; headless checks use --viz none")
    args.visualizer = ["none"] if args.headless else ["kit"]
    if not args.task and not args.env:
        if args.headless:
            args.task = "Pool"
        elif sys.stdin.isatty():
            names = list(ENVIRONMENTS)
            print("\nSelect an environment:")
            for index, name in enumerate(names, 1):
                print(f"  {index}. {name}")
            while True:
                choice = input("Environment [1 / Pool]: ").strip() or "1"
                if choice in names:
                    args.task = choice
                    break
                if choice.isdigit() and 1 <= int(choice) <= len(names):
                    args.task = names[int(choice) - 1]
                    break
                print("Enter a scene name or its number.")
        else:
            args.task = "Pool"
    args.env = args.env or ENVIRONMENTS[args.task]
    return args, AppLauncher


class Keyboard:
    def __init__(self, state):
        import carb.input
        import omni.appwindow

        self.state = state
        self.window = omni.appwindow.get_default_app_window()
        self.input = carb.input.acquire_input_interface()
        self.keyboard = self.window.get_keyboard()
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self.on_event)
        self.events = 0

    def on_event(self, event, *_):
        import carb.input

        if event.type not in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_RELEASE):
            return True
        handled = self.state.event(event.input.name, event.type == carb.input.KeyboardEventType.KEY_PRESS)
        self.events += int(handled)
        return not handled

    def close(self):
        self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)


class Pilot:
    """Use native forces without benchmark horizons or automatic resets."""

    def __init__(self, env, keys):
        self.env, self.keys = env.unwrapped, keys
        self.robot = self.env.robot
        self.original_limits = self.robot.data.joint_pos_limits.torch.clone()
        self.original_defaults = self.robot.data.default_joint_pos.torch.clone()
        self.max_arm_drift = 0.0
        self.resets = 0
        self.reset()

    def pose(self):
        data, body = self.robot.data, self.env._base_body_id
        return (
            data.body_link_pos_w.torch[0, body].detach().cpu().numpy().copy(),
            data.body_link_quat_w.torch[0, body].detach().cpu().numpy().copy(),
        )

    def reset(self):
        self.robot.write_joint_position_limit_to_sim_index(limits=self.original_limits, warn_limit_violation=False)
        self.robot.data.default_joint_pos.torch.copy_(self.original_defaults)
        self.env.reset()
        self.fixed_joints = self.robot.data.joint_pos.torch.clone()
        # A PhysX constraint locks all arm, gripper and finger DOFs.
        limits = self.fixed_joints.unsqueeze(-1).repeat(1, 1, 2)
        self.robot.write_joint_position_limit_to_sim_index(limits=limits, warn_limit_violation=False)
        self.reference = PoseReference(*self.pose())
        self.keys.clear()
        self.env._station_keeper.reset()
        self.resets += 1
        if hasattr(self, "water"):
            self.water.apply(force=True)

    def hold(self):
        self.keys.clear()
        self.reference.hold(*self.pose())
        self.env._station_keeper.reset()

    def step(self, speed, turn_speed):
        import torch

        env = self.env
        pos, quat = self.pose()
        if not np.isfinite(np.r_[pos, quat]).all():
            raise RuntimeError("Vehicle state became nonfinite; stop and reset the scene")
        linear, angular = self.keys.velocities(speed, turn_speed)
        target_pos, target_quat = self.reference.advance(linear, angular, pos, quat, env.step_dt)
        env._pre_physics_step(torch.zeros_like(env.actions))
        env._base_target_pos_w[:] = torch.as_tensor(target_pos, device=env.device, dtype=torch.float32)
        env._base_target_quat_w[:] = torch.as_tensor(target_quat, device=env.device, dtype=torch.float32)
        env._arm_targets.copy_(self.fixed_joints[:, env._arm_joint_ids])
        for _ in range(env.cfg.decimation):
            env._sim_step_counter += 1
            env._apply_action()
            self.robot.set_joint_position_target_index(target=self.fixed_joints)
            env.scene.write_data_to_sim()
            env.sim.step(render=env._sim_step_counter % env.cfg.sim.render_interval == 0)
            env.scene.update(env.physics_dt)
        drift = float((self.robot.data.joint_pos.torch - self.fixed_joints).abs().max())
        self.max_arm_drift = max(self.max_arm_drift, drift)


def smoke_keys(t, keys):
    # Retreat from the mechanism first; then ascend and turn in open water.
    key = "S" if 1 <= t < 3 else "R" if 4 <= t < 5.5 else "Q" if 6.5 <= t < 8 else None
    keys.held = {key} if key else set()


def run(args, launcher):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import resolve_task_config

    import wasman.tasks  # noqa: F401
    from wasman.teleop_environment import BaseCameraFeed, CameraPanel, WaterControls, WaterPanel

    torch.set_num_threads(4)
    cfg, _ = resolve_task_config(args.env, "", overrides=("physics=isaacsim_physx",))
    cfg.scene.num_envs, cfg.seed = 1, args.seed
    cfg.ui_window_class_type = None
    if args.enable_cameras:
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        from wasman.assets.robot_cameras import add_robot_cameras

        add_robot_cameras(cfg, width=640, height=360)
        cfg.scene.gripper_camera = None
        cfg.scene.base_camera.offset.rot = (0.0, 0.0, 0.0, 1.0)
        cfg.scene.base_camera.update_period = 1 / 30
        cfg.scene.base_camera.data_types = ["rgb", "distance_to_camera"]
        cfg.scene.base_camera.renderer_cfg = IsaacRtxRendererCfg()
    env = gym.make(args.env, cfg=cfg)
    keys, keyboard, window = KeyboardState(), None, None
    water_panel = camera_panel = None
    samples = []
    try:
        if not all(hasattr(env.unwrapped, name) for name in ("robot", "_station_keeper", "_base_target_pos_w")):
            raise ValueError("This environment does not expose the BlueROV2 swimming controller")
        pilot = Pilot(env, keys)
        water = pilot.water = WaterControls(env.unwrapped)
        feed = BaseCameraFeed(env.unwrapped, water) if args.enable_cameras else None
        initial_pos, initial_quat = pilot.pose()
        follow, quit_requested = True, False
        if not args.headless:
            import omni.ui as ui

            keyboard = Keyboard(keys)
            water_panel, camera_panel = WaterPanel(water), CameraPanel()
            window = ui.Window("WasserMan - keyboard pilot", width=460, height=290)
            with window.frame, ui.VStack(spacing=8):
                ui.Label(args.task or args.env, height=24)
                ui.Label(CONTROLS, word_wrap=True, height=125)
                status = ui.Label("Arm locked | Hold position", height=24)
                with ui.HStack(height=28):
                    ui.Button("Hold", clicked_fn=lambda: keys.commands.append("hold"))
                    ui.Button("Reset", clicked_fn=lambda: keys.commands.append("reset"))
            window.setPosition(15, 40)
        print(f"\nWasserMan keyboard pilot: {args.env}\n{CONTROLS}\nTELEOP_READY", flush=True)
        steps = args.steps or (int(10.5 / cfg.sim.dt / cfg.decimation) if args.smoke_test else 0)
        start = time.monotonic()
        index = 0
        smoke_reset = False
        was_focused = True
        water_stages = set()
        turbidity_difference = None
        while launcher.app.is_running() and not quit_requested and (not steps or index < steps):
            focused = keyboard is None or keyboard.window.is_focused()
            if was_focused and not focused:
                pilot.hold()
            was_focused = focused
            commands, keys.commands = keys.commands, []
            for command in commands:
                if command == "quit":
                    quit_requested = True
                elif command == "reset":
                    pilot.reset()
                elif command == "hold":
                    pilot.hold()
                elif command == "camera":
                    follow = not follow
            if quit_requested:
                break
            if not env.unwrapped.sim.is_playing():
                pilot.hold()
                launcher.app.update()
                time.sleep(0.01)
                start = time.monotonic() - index * env.unwrapped.step_dt
                continue
            t = index * env.unwrapped.step_dt
            if args.water_test:
                if t >= 1.0 and 1 not in water_stages:
                    water.set(current=(0.25, 0.10, 0.0), density=1050, viscosity=1.5, resistance=1.25)
                    water_stages.add(1)
                if t >= 2.0 and 2 not in water_stages:
                    water.set(turbidity=0.8, radius=3)
                    water_stages.add(2)
                if t >= 6.0 and 3 not in water_stages:
                    water.rotate_current()
                    water.reverse_current()
                    water_stages.add(3)
            water.apply()
            if args.smoke_test:
                smoke_keys(t, keys)
                if t >= 9.5 and not smoke_reset:
                    pilot.reset()
                    smoke_reset = True
            pilot.step(args.speed, args.turn_speed)
            pos, quat = pilot.pose()
            if feed:
                image = feed.image()
                if camera_panel:
                    camera_panel.update(image)
                if args.water_test and t >= 3.0 and turbidity_difference is None:
                    turbidity_difference = float((image.float() - feed.last_raw.float()).abs().mean())
            if follow and not args.headless:
                env.unwrapped.sim.set_camera_view(pos + np.array([-1.5, -2.0, 1.1]), pos + [0.2, 0, 0])
            if not args.headless and index % 10 == 0:
                status.text = f"Arm locked | z {pos[2]:.2f} m | {' '.join(sorted(keys.held)) or 'Hold position'}"
            if args.report or args.smoke_test:
                samples.append(
                    {
                        "time": t,
                        "position": pos.tolist(),
                        "quaternion_xyzw": quat.tolist(),
                        "keys": sorted(keys.held),
                        "resets": pilot.resets,
                        "water": water.metadata(),
                    }
                )
            index += 1
            if not args.no_realtime:
                time.sleep(max(0.0, start + index * env.unwrapped.step_dt - time.monotonic()))
        report = {
            "env": args.env,
            "steps": index,
            "max_joint_drift_rad": pilot.max_arm_drift,
            "resets": pilot.resets,
            "keyboard_events": keyboard.events if keyboard else 0,
            "samples": samples,
            "water": water.metadata(),
            "water_changes": water.changes,
            "camera": feed.metadata() if feed else None,
        }
        if args.smoke_test:

            def at(t):
                return min(samples, key=lambda sample: abs(sample["time"] - t))

            retreat = (np.array(at(3.9)["position"]) - initial_pos) @ rotate_vector(
                initial_quat, np.array([-1.0, 0, 0])
            )
            ascent = at(6.4)["position"][2] - at(3.9)["position"][2]
            q0, q1 = np.array(at(6.4)["quaternion_xyzw"]), np.array(at(9.4)["quaternion_xyzw"])
            turn = 2 * np.arccos(np.clip(abs(q0 @ q1), 0, 1))
            report["checks"] = {"retreat_m": float(retreat), "ascent_m": float(ascent), "turn_rad": float(turn)}
            report["passed"] = bool(
                retreat > 0.03 and ascent > 0.03 and turn > 0.03 and pilot.max_arm_drift < 0.005 and smoke_reset
            )
        if args.water_test:
            report["turbidity_difference_rgb"] = turbidity_difference
            report["passed"] = bool(
                report["passed"]
                and feed.frames > 50
                and feed.metadata()["raw_std"] > 1
                and turbidity_difference > 2
                and water.changes >= 5
                and env.unwrapped._hydrodynamics.water_density == water.settings.density
            )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
        if args.smoke_test and not report["passed"]:
            raise RuntimeError("Teleoperation physics check failed; inspect the report")
    finally:
        if keyboard:
            keyboard.close()
        if window:
            window.destroy()
        if water_panel:
            water_panel.close()
        if camera_panel:
            camera_panel.close()
        env.close()


if __name__ == "__main__":
    args, app_launcher = parse_args()
    launcher = app_launcher(vars(args).copy())
    import numpy as np

    try:
        run(args, launcher)
    except Exception:
        # Kit's windowed fast shutdown can exit the process before Python unwinds.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        launcher.app.close(exit_code=1)
        raise
    else:
        launcher.app.close(exit_code=0)
