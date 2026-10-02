#!/usr/bin/env python3
"""Matched ON/OFF clearance sweep of the uncalibrated boundary-loss surrogate.

Real PhysX, native pool and robot collisions; unchanged motor/controller/hydro
coefficients. Rejected cases are retained. No training or website defaults change.
"""

import argparse
import hashlib
import itertools
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--floor-heights", type=float, nargs="*", default=[0.60, 0.30, 0.20, 0.16])
parser.add_argument("--wall-distances", type=float, nargs="*", default=[0.70, 0.45, 0.35, 0.30])
parser.add_argument("--seeds", type=int, nargs="+", default=[2058])
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--protocols", nargs="+", choices=["hold", "away_step"], default=["hold", "away_step"])
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["none"])
args = parser.parse_args()
launcher = AppLauncher(args)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.sim import get_current_stage  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import wasman.tasks  # noqa: E402, F401
from wasman.assets.pool_geometry import POOL_GEOMETRY, POOL_USD_PATH, configure_finite_pool  # noqa: E402
from wasman.controller_diagnostic import ControllerDiagnosticEnv  # noqa: E402
from wasman.physics.boundary_effects import BlueROVBoundaryEffect, BoundaryEffectCfg  # noqa: E402
from wasman.physics.hydrodynamics import quat_apply_xyzw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WALL_X = 1.1
POOL_CENTER = (WALL_X - 12.5, 9.5, 0.0)
SAFE_GAP_M = 0.02  # Conservative collider AABB separation, in addition to contact sensing.


def collision_corners(env):
    """Enclose actual imported USD colliders in their owning rigid-link frames.

    USD is used only for static collider geometry. Moving world poses always
    come from live articulation buffers (Fabric does not update USD poses).
    """
    stage = get_current_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False)
    indices, corners, audit = [], [], []
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot"), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        body = prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body or body.GetName() not in env.robot.body_names:
            raise RuntimeError(f"Cannot associate collider with articulation link: {prim.GetPath()}")
        bounds = cache.ComputeRelativeBound(prim, body).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise RuntimeError(f"Empty collider bounds: {prim.GetPath()}")
        low, high = list(bounds.GetMin()), list(bounds.GetMax())
        points = list(itertools.product(*zip(low, high, strict=True)))
        indices.extend([env.robot.body_names.index(body.GetName())] * 8)
        corners.extend(points)
        audit.append({"prim": str(prim.GetPath()), "body": body.GetName(), "local_min": low, "local_max": high})
    if not audit:
        raise RuntimeError("No collision geometry found")
    return torch.tensor(indices, device=env.device), torch.tensor(corners, device=env.device), audit


class ClearanceDiagnosticEnv(ControllerDiagnosticEnv):
    watching = False

    def sample_geometry(self):
        ids, corners = self.clearance_ids, self.clearance_corners
        points = self.robot.data.body_link_pos_w.torch[0, ids] + quat_apply_xyzw(
            self.robot.data.body_link_quat_w.torch[0, ids], corners
        )
        gaps = torch.stack(
            (
                points[:, 2].min(),
                (WALL_X - points[:, 0]).min(),
                (points[:, 0] - (WALL_X - 25)).min(),
                (points[:, 1] + 3).min(),
                (22 - points[:, 1]).min(),
                (2.5 - points[:, 2]).min(),
            )
        )
        self.minimum_gaps = torch.minimum(self.minimum_gaps, gaps)
        return gaps

    def _apply_action(self):
        if self.watching:
            self.sample_geometry()
        super()._apply_action()
        if self.watching:
            delta = self._boundary_applied_wrench - self._thrusters.realized_wrench
            self.peak_force_delta = torch.maximum(self.peak_force_delta, delta[:, :3].norm(dim=-1).max())
            self.peak_torque_delta = torch.maximum(self.peak_torque_delta, delta[:, 3:].norm(dim=-1).max())
            self.minimum_gain = torch.minimum(self.minimum_gain, self._boundary_gain.min())


def run_case(env, mode, distance, protocol, seed, enabled):
    env.watching = False
    # Only reset-state placement and nominal target vary between distances.
    position = (0, 0, distance) if mode == "floor" else (WALL_X - distance, 0, 0.85)
    env.robot.data.default_root_pose.torch[:, :3] = env.actions.new_tensor(position)
    env._base_target_nominal[:] = env.actions.new_tensor(position)
    env._boundary_effect = BlueROVBoundaryEffect(
        env.device,
        BoundaryEffectCfg(
            seabed_loss=0.2 if enabled and mode == "floor" else 0, wall_loss=0.2 if enabled and mode == "wall" else 0
        ),
        pool_geometry=POOL_GEOMETRY,
        pool_center=POOL_CENTER,
        env_origins=env.scene.env_origins,
    )
    env.reset(seed=seed)
    env.minimum_gaps = env.actions.new_full((6,), float("inf"))
    env.peak_force_delta = env.actions.new_tensor(0.0)
    env.peak_torque_delta = env.actions.new_tensor(0.0)
    env.minimum_gain = env.actions.new_tensor(1.0)
    env.watching = True
    initial = {
        "root_pose_xyzw": env.robot.data.root_pose_w.torch[0].tolist(),
        "root_velocity": env.robot.data.root_vel_w.torch[0].tolist(),
        "joint_position": env.robot.data.joint_pos.torch[0].tolist(),
        "joint_velocity": env.robot.data.joint_vel.torch[0].tolist(),
    }
    start_gaps = env.sample_geometry().tolist()
    reason = "initial_collider_clearance" if min(start_gaps) < SAFE_GAP_M else None
    frames, peak_contact, reset = [], 0.0, False
    actions = torch.zeros((1, env.cfg.action_space), device=env.device)
    axis, sign = (2, 1) if mode == "floor" else (0, -1)
    if reason is None:
        for step in range(args.steps):
            if protocol == "away_step" and step * env.step_dt >= 3.0:
                actions[0, axis] = sign * 0.12 / env.cfg.base_target_position_scale[axis]
            _, _, terminated, truncated, _ = env.step(actions)
            # History is updated on every physics step, not merely at control rate.
            contact = env.scene["clearance_contact"].data.net_normal_forces_w_history.torch
            if not torch.isfinite(contact).all():
                raise RuntimeError("Nonfinite contact sensor output")
            peak_contact = max(peak_contact, float(contact.norm(dim=-1).max()))
            reset = bool((terminated | truncated).any())
            if reset:
                reason = "environment_reset"
                break
            env.sample_geometry()
            body = env._base_body_id
            frames.append(
                {
                    "t_s": (step + 1) * env.step_dt,
                    "position_w_m": env.robot.data.body_link_pos_w.torch[0, body].tolist(),
                    "quaternion_xyzw": env.robot.data.body_link_quat_w.torch[0, body].tolist(),
                    "target_w_m": env._base_target_pos_w[0].tolist(),
                    "velocity_w_m_s": env.robot.data.body_com_lin_vel_w.torch[0, body].tolist(),
                    "motor_force_before_boundary_n": env._thrusters.force[0].tolist(),
                    "boundary_gain": env._boundary_gain[0].tolist(),
                }
            )
            if peak_contact > 1e-5:
                reason = "contact"
                break
            if float(env.minimum_gaps.min()) < SAFE_GAP_M:
                reason = "collider_clearance_guard"
                break
    env.watching = False
    key = f"{mode}-{distance:.3f}-{protocol}-{seed}-{'on' if enabled else 'off'}"
    trace_path = args.output_dir / f"{key}.json"
    trace_path.write_text(json.dumps(frames, allow_nan=False) + "\n")
    return {
        "id": key,
        "mode": mode,
        "base_to_surface_m": distance,
        "protocol": protocol,
        "seed": seed,
        "enabled": enabled,
        "initial_state": initial,
        "initial_conservative_gaps_m": start_gaps,
        "min_conservative_gaps_m": env.minimum_gaps.tolist(),
        "peak_normal_contact_n": peak_contact,
        "max_boundary_force_delta_n": float(env.peak_force_delta),
        "max_boundary_torque_delta_nm": float(env.peak_torque_delta),
        "max_per_motor_loss_percent": 100 * (1 - float(env.minimum_gain)),
        "passed": reason is None and len(frames) == args.steps,
        "rejection_reason": reason,
        "reset": reset,
        "frames": len(frames),
        "trace": trace_path.name,
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
    }, frames


def compare_pair(off, on, off_frames, on_frames):
    initial_match = off["initial_state"] == on["initial_state"]
    result = {
        "off": off["id"],
        "on": on["id"],
        "initial_state_exact_match": initial_match,
        "valid": initial_match and off["passed"] and on["passed"],
    }
    if result["valid"]:
        p0, p1 = [np.array([r["position_w_m"] for r in rows]) for rows in (off_frames, on_frames)]
        target = np.array([r["target_w_m"] for r in off_frames])
        if not np.array_equal(target, [r["target_w_m"] for r in on_frames]):
            raise RuntimeError("Paired target mismatch")
        q0, q1 = [np.array([r["quaternion_xyzw"] for r in rows], dtype=float) for rows in (off_frames, on_frames)]
        q0 /= np.linalg.norm(q0, axis=-1, keepdims=True)
        q1 /= np.linalg.norm(q1, axis=-1, keepdims=True)
        result.update(
            {
                "max_on_off_position_delta_mm": float(1000 * np.linalg.norm(p1 - p0, axis=-1).max()),
                "max_on_off_attitude_delta_deg": float(
                    np.rad2deg(2 * np.arccos(np.clip(np.abs((q0 * q1).sum(-1)), 0, 1))).max()
                ),
                "last_second_on_off_position_delta_mm": float(
                    1000 * np.linalg.norm(p1[-30:] - p0[-30:], axis=-1).mean()
                ),
                "tracking_rmse_off_mm": float(1000 * np.sqrt(np.mean(np.sum((p0 - target) ** 2, axis=-1)))),
                "tracking_rmse_on_mm": float(1000 * np.sqrt(np.mean(np.sum((p1 - target) ** 2, axis=-1)))),
            }
        )
    return result


def main():
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory; preserve sweep evidence")
    if args.steps < 120 or not all(np.isfinite(d) and d > 0 for d in args.floor_heights + args.wall_distances):
        raise ValueError("Require >=120 steps and positive finite distances")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "run_script.py").write_bytes(Path(__file__).read_bytes())
    cfg, _ = resolve_task_config("Wasman-Underwater-PressButton-T200-Direct", "", overrides=("physics=isaacsim_physx",))
    cfg.scene.num_envs, cfg.seed = 1, args.seeds[0]
    cfg.episode_length_s = args.steps / 30 + 5
    cfg.scene.robot.init_state.pos = cfg.base_target_position = (0, 0, 0.85)
    cfg.scene.robot.init_state.joint_pos["alpha_axis_c"] = 0.30
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.clearance_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0,
        history_length=cfg.decimation,
        max_contact_data_count_per_prim=64,
    )
    cfg.scene.panel = cfg.scene.button = None
    configure_finite_pool(cfg, center=POOL_CENTER)
    cfg.current_speed_range = cfg.current_vertical_range = (0, 0)
    cfg.turbulence_sigma = 0
    cfg.volume_scale_range = cfg.damping_scale_range = cfg.added_mass_scale_range = (1, 1)
    cfg.enable_boundary_effects = True
    cfg.thruster_time_constant, cfg.thruster_command_delay_steps = 0.08, 2
    # Diagnostic-only: collider bounds + per-substep contact checks replace the
    # training task's coarse .25 m root-height guard. Training cfg stays intact.
    cfg.min_base_height = 0.05
    cfg.sim.visualizer_cfgs, cfg.video_recorders = [], []
    env = ClearanceDiagnosticEnv(cfg)
    sources = [
        "scripts/sweep_boundary_clearance.py",
        "src/wasman/controller_diagnostic.py",
        "src/wasman/tasks/underwater_press_button/config/bluerov2_alpha/env.py",
        "src/wasman/tasks/underwater_press_button/config/bluerov2_alpha/env_cfg.py",
        "src/wasman/physics/boundary_effects.py",
        "src/wasman/physics/thrusters.py",
        "src/wasman/physics/hydrodynamics.py",
        "src/wasman/controllers/station_keeping.py",
        "src/wasman/assets/data/robots/bluerov2_alpha/bluerov2_alpha.urdf",
        str(POOL_USD_PATH.relative_to(ROOT)),
    ]
    report = {
        "kind": "Matched real-PhysX clearance sensitivity test; NOT calibrated boundary hydrodynamics",
        "steps": args.steps,
        "control_hz": 30,
        "physics_hz": 120,
        "seeds": args.seeds,
        "loss_coefficient": 0.2,
        "range_diameters": 10,
        "diameter_m": 0.0762,
        "away_step_onset_s": 3,
        "away_step_distance_m": 0.12,
        "minimum_safe_collider_gap_m": SAFE_GAP_M,
        "water_current_m_s": [0, 0, 0],
        "gap_order": ["floor", "east_wall", "west_wall", "south_wall", "north_wall", "waterline"],
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "cases": [],
        "pairs": [],
        "completed": False,
    }
    try:
        env.clearance_ids, env.clearance_corners, report["usd_collision_bounds"] = collision_corners(env)
        report["contact_sensor_bodies"] = env.scene["clearance_contact"].body_names
        for mode, distances in (("floor", args.floor_heights), ("wall", args.wall_distances)):
            for distance, protocol, seed in itertools.product(distances, args.protocols, args.seeds):
                off, off_frames = run_case(env, mode, distance, protocol, seed, False)
                on, on_frames = run_case(env, mode, distance, protocol, seed, True)
                report["cases"].extend((off, on))
                pair = compare_pair(off, on, off_frames, on_frames)
                report["pairs"].append(pair)
                print(
                    json.dumps(
                        {
                            **pair,
                            "min_gaps": on["min_conservative_gaps_m"][:2],
                            "loss_percent": on["max_per_motor_loss_percent"],
                            "delta_force_n": on["max_boundary_force_delta_n"],
                            "rejected": [off["rejection_reason"], on["rejection_reason"]],
                        }
                    ),
                    flush=True,
                )
                (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["completed"] = True
    finally:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        env.close()


exit_code = 0
try:
    main()
except Exception:
    traceback.print_exc()
    exit_code = 1
finally:
    launcher.app.close(exit_code=exit_code)
