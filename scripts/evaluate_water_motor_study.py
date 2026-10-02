"""Separate declared sensitivity conditions around frozen Valve/Hatch evaluators."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import MethodType

import gymnasium as gym

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--task", choices=["RotateValve", "OpenHatch"], required=True)
p.add_argument(
    "--condition",
    choices=["reference", "current_y_010", "current_y_020", "current_y_m020", "motor_080", "motor_060"],
    required=True,
)
a, remaining = p.parse_known_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "scripts"))
out = Path(remaining[remaining.index("--output-dir") + 1])
original_make = gym.make
records = []
provenance = dict(
    task=a.task,
    condition=a.condition,
    success_criteria_changed=False,
    expert_at_inference=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)


def make(task, *, cfg, **kwargs):
    import torch

    cfg.current_speed_range = cfg.current_vertical_range = (0.0, 0.0)
    cfg.turbulence_sigma = 0.0
    env = original_make(task, cfg=cfg, **kwargs)
    e = env.unwrapped
    if e._thrusters is None:
        raise ValueError("Physical motors required")
    fraction = {"motor_080": 0.8, "motor_060": 0.6}.get(a.condition, 1.0)
    current = {"current_y_010": 0.10, "current_y_020": 0.20, "current_y_m020": -0.20}.get(a.condition, 0.0)
    motors = e._thrusters
    limits = [float(motors.force_knots[0]) * fraction, float(motors.force_knots[-1]) * fraction]

    def allocate(self, wrench):
        from research_conditions import capped_allocation

        return capped_allocation(self, wrench, fraction)

    motors.allocate = MethodType(allocate, motors)
    provenance.update(
        current_w_m_s=[0, current, 0],
        capacity_fraction=fraction,
        limits_n=limits,
        motor_change="allocation capacity before unchanged PWM/RPM lag and static force map",
        physics_dt=cfg.sim.dt,
        control_dt=e.step_dt,
    )
    original_step = env.step
    calls = 0

    def step(action):
        nonlocal calls
        if calls == 0:
            initial = dict(
                robot_root=e.robot.data.root_state_w.torch.cpu().clone(),
                robot_joints=e.robot.data.joint_pos.torch.cpu().clone(),
                robot_joint_velocities=e.robot.data.joint_vel.torch.cpu().clone(),
                mechanism_root=e.button.data.root_state_w.torch.cpu().clone(),
                mechanism_joints=e.button.data.joint_pos.torch.cpu().clone(),
                mechanism_joint_velocities=e.button.data.joint_vel.torch.cpu().clone(),
                origins=e.scene.env_origins.cpu().clone(),
            )
            torch.save(initial, out / "paired_initial.pt")
        e._mean_current_w[:] = e._mean_current_w.new_tensor([0, current, 0])
        e._turbulent_current_w.zero_()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        result = original_step(action)
        records.append(
            {
                k: v.detach().cpu().clone()
                for k, v in dict(
                    motor_force=motors.force,
                    motor_request=motors.requested_force,
                    saturation_scale=motors.saturation_scale,
                    attitude=e._base_attitude_error,
                    angular_speed=e._base_angular_speed,
                    tracking_error=e._station_position_error_w,
                    current=e.current_w,
                    tool=e.robot.data.body_link_pos_w.torch[:, e._tool_body_id] - e.scene.env_origins,
                ).items()
            }
        )
        calls += 1
        return result

    env.step = step
    original_close = env.close

    def close():
        torch.save(records, out / "sensitivity_telemetry.pt")
        (out / "condition.json").write_text(json.dumps(provenance, indent=2) + "\n")
        original_close()

    env.close = close
    return env


gym.make = make
name = "valve" if a.task == "RotateValve" else "hatch"
sys.argv = [str(root / f"scripts/evaluate_{name}_visual.py"), *remaining]
runpy.run_path(sys.argv[0], run_name="__main__")
