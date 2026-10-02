"""Build the three-condition report from complete, independently replayed runs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs", type=Path, nargs=3, required=True)
parser.add_argument("--split", choices=["development", "final"], required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
labels = {"free": "Left arm parked", "support": "Left arm on support", "two-hands": "Both hands on wheel"}
modes = []
paired_seeds = None
paired_sources = None
paired_configuration = None
for run in args.runs:
    report = json.loads((run / "report.json").read_text())
    replay = json.loads((run / "independent-replay.json").read_text())
    geometry = json.loads((run / "geometry-audit.json").read_text())
    assert replay["complete"] and replay["finite"] and replay["recorded_success_agrees"]
    assert not replay["episodes"][-1]["success"], "Motor-off control unexpectedly turns valve"
    assert report["fixture"] == "large" and not report["fixed_base"] and report["runtime_pose_overwrites"] == 0
    if paired_seeds is None:
        paired_seeds = report["seeds"]
    assert paired_seeds == report["seeds"]
    if args.split == "final":
        assert paired_seeds == list(range(92000, 92030))
    assert report["seconds"] == 180 and np.isclose(report["dt"], 1 / 240)
    assert report.get("fixtures_spawned_at_reset_pose"), "Uncorrected fixture initialization"
    assert not report.get("hold_reset_commands", False)
    assert not report.get("dry_diagnostic", False) and not report.get("isolated_fixture_diagnostic", False)
    assert len(geometry["episodes"]) == len(paired_seeds)
    for row in geometry["episodes"]:
        assert not row["cad_collisions"] and row["max_joint_limit_excess_rad"] <= 0.002
        assert row["max_motor_force_N"] <= 1540.001
        assert max(row["tcp_fk_max_error_m"].values()) < 1e-4
    configuration = {key: report[key] for key in (
        "feedback_frame", "grasp_effort_command_Nm", "robot_solver_iterations",
        "initial_load_balanced", "base_reference_m", "dt", "seconds", "solver_type",
    )}
    if paired_sources is None:
        paired_sources, paired_configuration = report["source_files"], configuration
    assert paired_sources == report["source_files"], "Compared modes use different source revisions"
    assert paired_configuration == configuration, "Compared modes use different control/physics configurations"
    with np.load(run / "trace.npz") as source:
        d = {
            key: source[key]
            for key in [
                "time",
                "valve_angle",
                "normal_contact_by_body",
                "attitude_error",
                "quaternion",
                "position_error",
                "motor_force",
            ]
        }
    n = len(paired_seeds)
    orientation_error = Rotation.from_quat(d["quaternion"].astype(np.float64).reshape(-1, 4)).magnitude().reshape(d["attitude_error"].shape)
    force = d["normal_contact_by_body"][:, :n]
    contact = {}
    for side, groups in [("left", ([8, 9], [10, 11])), ("right", ([19, 20], [21, 22]))]:
        target = 2 if side == "left" and report["mode"] == "support" else 0
        a = np.linalg.norm(force[:, :, groups[0], target].sum(2), axis=-1)
        b = np.linalg.norm(force[:, :, groups[1], target].sum(2), axis=-1)
        contact[side] = (a > 0.5) & (b > 0.5)
    series = []
    for i in sorted(set(range(0, len(d["time"]), 30)) | {len(d["time"]) - 1}):
        angle = np.rad2deg(d["valve_angle"][i, :n])
        series.append(
            dict(
                time_s=float(d["time"][i]),
                angle_mean_deg=float(angle.mean()),
                angle_low_deg=float(np.quantile(angle, 0.1)),
                angle_high_deg=float(np.quantile(angle, 0.9)),
                base_attitude_mean_deg=float(np.rad2deg(orientation_error[i, :n]).mean()),
                base_displacement_mean_mm=float(d["position_error"][i, :n].mean() * 1000),
                right_contact_fraction=float(contact["right"][i].mean()),
                left_contact_fraction=float(contact["left"][i].mean()),
                motor_utilization_mean=float(np.max(abs(d["motor_force"][i, :n]), axis=-1).mean() / 1540),
            )
        )
    successes = [r["success"] for r in replay["episodes"][:-1]]
    modes.append(
        dict(
            mode=report["mode"],
            label=labels[report["mode"]],
            episodes=n,
            successes=sum(successes),
            per_seed_success=successes,
            per_seed=replay["episodes"][:-1],
            series=series,
            source_run=str(run),
            seconds=report["seconds"],
            dt=report["dt"],
        )
    )
assert {r["mode"] for r in modes} == set(labels)
result = dict(
    complete=True,
    evaluation_split=args.split,
    study="bimanual-valve-v1",
    seeds=paired_seeds,
    source_sha256=paired_sources,
    configuration=paired_configuration,
    modes=sorted(modes, key=lambda m: list(labels).index(m["mode"])),
    criterion="signed valve rotation >=170 degrees",
    orientation_measurement="float64 rotation magnitude from recorded XYZW quaternion relative to fixed identity target",
    note="Scripted physical-feedback experts on a custom twin-Oberon Rex; not learned-policy transfer or hardware evidence. A separate geometrically scaled 506 mm valve is common to all modes. The free/support contrast keeps identical scene geometry; the unused rail is moved aside for the two-hand extension.",
    contact_note="Opposing-finger normal forces both exceed 0.5 N. Contact is diagnostic; commanded release during regrasp is expected. Lines are paired-reset means; angle bands are 10th–90th percentiles.",
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print([(m["mode"], m["successes"], m["episodes"]) for m in result["modes"]])
