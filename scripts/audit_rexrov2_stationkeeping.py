"""Independently replay a completed Rex diagnostic; never rewrite its report."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wasman.physics.rexrov2 import load_parameters


def audit(folder: Path):
    report = json.loads((folder / "report.json").read_text())
    with np.load(folder / "trace.npz", allow_pickle=False) as saved:
        trace = dict(saved)
    finite = {name: bool(np.isfinite(value).all()) for name, value in trace.items()}
    window = trace["time"] >= report["seconds"] - 10
    position = np.sqrt(np.mean(trace["position_error"][window] ** 2, axis=0))
    attitude = np.sqrt(np.mean(trace["attitude_error"][window] ** 2, axis=0))
    speed = np.sqrt(np.mean(trace["speed"][window] ** 2, axis=0))
    drift = trace["arm_drift"].max(0)
    cap = load_parameters().max_thrust
    checks = {
        "all_recorded_states_and_wrenches_finite": all(finite.values()),
        "native_force_cap": bool(np.abs(trace["motor_force"]).max() <= cap + 0.001),
        "motors_off_exactly_zero": bool(np.count_nonzero(trace["motor_force"][:, 3]) == 0),
        "motors_off_drift": bool(trace["position_error"][-1, 3] > 0.5),
        "powered_position_rms": bool((position[:3] < 0.10).all()),
        "powered_attitude_rms": bool((attitude[:3] < np.deg2rad(3)).all()),
        "powered_speed_rms": bool((speed[:3] < 0.05).all()),
        "powered_recorded_arm_drift": bool((drift[:3] < 0.10).all()),
        "recorded_metrics_match_report": bool(
            np.allclose(position, report["last_10s_position_rms_m"])
            and np.allclose(np.rad2deg(attitude), report["last_10s_attitude_rms_deg"])
            and np.allclose(speed, report["last_10s_speed_rms_m_s"])
            and np.allclose(drift, report["max_arm_drift_rad"])
        ),
        "complete_60_second_recording": bool(
            report["seconds"] >= 60
            and len(trace["time"]) == round(report["seconds"] / report["dt"] / 12)
            and np.allclose(np.diff(trace["time"]), report["dt"] * 12)
        ),
    }
    sampled_collisions = []
    if report.get("variant") == "WASMAN centered compact-held-arm":
        from wasman.controllers.rexrov2_workspace import RexWorkspace

        workspace = RexWorkspace()
        for index, poses in enumerate(trace["joint_position"]):
            for environment, q in enumerate(poses):
                if collisions := workspace.collisions(q):
                    sampled_collisions.append(
                        {"time_s": float(trace["time"][index]), "environment": environment, "pairs": collisions}
                    )
        checks["recorded_centered_joint_poses_clear"] = not sampled_collisions
    modern_checks = report.get("finite_states_and_wrenches_checked_every_physics_step", False)
    return {
        "kind": "independent CPU replay audit; original report left unchanged",
        "passed": all(checks.values()),
        "checks": checks,
        "finite_by_recorded_field": finite,
        "trace_samples": len(trace["time"]),
        "trace_period_s": report["dt"] * 12,
        "max_recorded_motor_force_N": float(np.abs(trace["motor_force"]).max()),
        "native_motor_force_cap_N": cap,
        "report_sha256": hashlib.sha256((folder / "report.json").read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256((folder / "trace.npz").read_bytes()).hexdigest(),
        "run_script_snapshot_sha256": hashlib.sha256((folder / "run_script.py").read_bytes()).hexdigest(),
        **(
            {"sampled_CAD_self_collisions": sampled_collisions}
            if report.get("variant") == "WASMAN centered compact-held-arm"
            else {}
        ),
        "original_run_limitations": [
            "10 Hz replay; executable additionally checked finite state/wrench and force cap every physics step.",
            "Imported PhysX link masses/COM checked in the same process before integration.",
            "run_script.py snapshotted before integration; source hash is also in original report.",
            "Arm drift and CAD self-collision replay are sampled at 10 Hz, not a continuous-path proof.",
            "Floating run does not record filtered self-contact forces; dedicated reach smoke does.",
        ]
        if modern_checks
        else [
            "State finiteness checked every physics step; wrench finiteness/cap audited at recorded 10 Hz only.",
            "Arm-drift maxima are observed 10 Hz samples, not every-step maxima.",
            "Import-mass/COM audit was run separately after the trajectory, not inside its original process.",
            "run_script.py copied after execution from the still-unmodified script, before new checks were added.",
            "Current executable adds stricter wrench/cap/import assertions; this record does not claim those ran.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    output = audit(args.folder)
    (args.folder / "replay_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
