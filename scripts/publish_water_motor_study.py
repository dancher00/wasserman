"""Paired-state audit, original success replay and contact/stability sensitivity metrics."""

from pathlib import Path

import numpy as np
import torch
from research_campaign_runtime import BASE, ROOT, read, sha, write
from scipy.stats import binomtest
from verify_valve_visual_results import verify as verify_valve

from wasman.controllers.hatch_contract import HatchContract


def main():
    registry = read(BASE / "water_motor_records.json")
    if not registry["complete"] or len(registry["records"]) != 12:
        raise ValueError("Conditions incomplete")
    rows = []
    initials = {}
    for record in registry["records"]:
        path = Path(record["path"])
        r = read(path / "summary.json")
        condition = read(path / "condition.json")
        if (
            r["seeds"] != list(range(10000, 10030))
            or r["checkpoint_sha256"] != record["checkpoint_sha256"]
            or r["expert_at_inference"]
        ):
            raise ValueError("Frozen policy or split differs")
        initial = torch.load(path / "paired_initial.pt", weights_only=True, map_location="cpu")
        task = record["task"]
        if task in initials:
            if any(not torch.equal(v, initials[task][k]) for k, v in initial.items()):
                raise ValueError("Initial physical states not paired")
        else:
            initials[task] = initial
        data = torch.load(path / "rollout.pt", weights_only=True, map_location="cpu")
        trace = data["trace"]
        if data["summary"] != r:
            raise ValueError("Summary mismatch")
        if task == "RotateValve":
            verify_valve(path)
        else:
            contract = HatchContract(30, "cpu", r["step_dt"])
            first = torch.full((30,), -1, dtype=torch.long)
            valid = torch.ones(30, dtype=torch.bool)
            for step, row in enumerate(trace):
                valid &= ~row["reset"]
                success = contract.update(
                    row["angle"], row["speed"], row["forces"], row["near_handle"], row["tool_speed"], row["stable"]
                )
                new = success & valid & (first < 0)
                if not torch.equal(new, row["success"]):
                    raise ValueError("Hatch physical contract mismatch")
                first[new] = step + 1
            if first.tolist() != r["first_success_step"]:
                raise ValueError("Hatch completion mismatch")
        telemetry = torch.load(path / "sensitivity_telemetry.pt", weights_only=True, map_location="cpu")[1:]
        if len(telemetry) != len(trace):
            raise ValueError("Telemetry clock mismatch")
        metrics = []
        for i in range(30):
            indices = [j for j, row in enumerate(trace) if row["active"][i] and not row["reset"][i]]
            if not indices:
                raise ValueError("Empty scored episode")
            grasp = np.asarray([bool(trace[j]["grasped"][i]) for j in indices])
            metrics.append(
                dict(
                    contact_losses=int(np.count_nonzero(grasp[:-1] & ~grasp[1:])),
                    ever_grasped=bool(grasp.any()),
                    scored_duration_s=len(indices) * r["step_dt"],
                    initial_2s_tracking_rms_m=float(
                        np.sqrt(
                            np.mean(
                                [
                                    float(telemetry[j]["tracking_error"][i].square().sum())
                                    for j in indices
                                    if j * r["step_dt"] < 2.0
                                ]
                            )
                        )
                    ),
                    initial_2s_window_complete=all(j in indices for j in range(round(2.0 / r["step_dt"]))),
                    peak_attitude_rad=max(float(telemetry[j]["attitude"][i]) for j in indices),
                    peak_angular_speed_rad_s=max(float(telemetry[j]["angular_speed"][i]) for j in indices),
                    saturation_fraction=float(
                        np.mean([float(telemetry[j]["saturation_scale"][i]) < 0.99999 for j in indices])
                    ),
                )
            )
        expected = torch.tensor(condition["current_w_m_s"])
        for step, row in enumerate(telemetry):
            valid = trace[step]["active"] & ~trace[step]["reset"]
            # Automatic resets reinitialize current; these frames are outside the scored episode.
            if not torch.allclose(row["current"][valid], expected.expand_as(row["current"])[valid]):
                raise ValueError("Actual current differs")
            if not torch.isfinite(row["motor_force"]).all():
                raise ValueError("Nonfinite motor force")
            low, high = condition["limits_n"]
            if row["motor_force"].min() < low - 1e-3 or row["motor_force"].max() > high + 1e-3:
                raise ValueError("Motor limit exceeded")
        k = sum(r["success_per_seed"])
        ci = binomtest(k, 30).proportion_ci(method="exact")
        rows.append(
            dict(
                **record,
                successes=k,
                test_episodes=30,
                success_percent=k / 30 * 100,
                ci95=[100 * ci.low, 100 * ci.high],
                condition_parameters=condition,
                episode_diagnostics=metrics,
                trace_sha256=sha(path / "rollout.pt"),
            )
        )
    value = dict(
        complete=True,
        records=rows,
        scope="Sensitivity in simulation; one frozen DP per task;30 paired resets. No hardware calibration claim.",
    )
    write(ROOT / "website/public/static/water-motor-study.json", value)
    write(BASE / "water_motor_published.json", value)


if __name__ == "__main__":
    main()
