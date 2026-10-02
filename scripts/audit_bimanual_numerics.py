"""Compare paired 240/480 Hz bimanual traces without changing acceptance gates."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("coarse", type=Path)
parser.add_argument("fine", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--until", type=float, help="Explicit prefix diagnostic; never a full-horizon gate")
args = parser.parse_args()
with np.load(args.coarse / "trace.npz") as source:
    a = dict(source)
with np.load(args.fine / "trace.npz") as source:
    b = dict(source)
ra = json.loads((args.coarse / "report.json").read_text())
rb = json.loads((args.fine / "report.json").read_text())
assert ra["seeds"] == rb["seeds"] and ra["mode"] == rb["mode"] and ra["fixture"] == rb["fixture"]
assert np.isclose(ra["dt"], 1 / 240) and np.isclose(rb["dt"], 1 / 480)
if args.until is not None:
    assert 0 < args.until <= min(ra["seconds"], rb["seconds"])
    a = {k: v[a["time"] < args.until] for k, v in a.items()}
    b = {k: v[b["time"] < args.until] for k, v in b.items()}
np.testing.assert_allclose(a["time"], b["time"], atol=1e-6)
rows = []
for n, seed in enumerate(ra["seeds"]):
    tcp = max(
        float(np.sqrt(np.mean(np.sum((a[k][:, n] - b[k][:, n]) ** 2, axis=-1))))
        for k in ["tool_position", "left_tool_position"]
    )
    angle = float(np.rad2deg(np.max(abs(a["valve_angle"][:, n] - b["valve_angle"][:, n]))))
    attitude = float(
        np.rad2deg(
            (Rotation.from_quat(a["quaternion"][:, n]).inv() * Rotation.from_quat(b["quaternion"][:, n]))
            .magnitude()
            .max()
        )
    )
    joints = float(abs(a["joint_position"][:, n] - b["joint_position"][:, n]).max())
    success_a = bool((a["valve_angle"][:, n] >= np.deg2rad(170)).any())
    success_b = bool((b["valve_angle"][:, n] >= np.deg2rad(170)).any())
    checks = dict(
        tcp_rms=tcp <= 0.01,
        base_orientation_max=attitude <= 0.5,
        joint_position_max=joints <= 0.02,
        shaft_angle_max=angle <= 5,
        success=success_a == success_b,
    )
    rows.append(
        dict(
            seed=seed,
            tcp_rms_m=tcp,
            shaft_angle_max_deg=angle,
            base_orientation_max_deg=attitude,
            joint_position_max_rad=joints,
            successes=[success_a, success_b],
            checks=checks,
            passed=all(checks.values()),
        )
    )
result = dict(
    passed=all(r["passed"] for r in rows),
    scope="full horizon" if args.until is None else "prefix diagnostic only",
    window_end_exclusive_s=args.until,
    alignment="same wall-clock 30 Hz samples",
    coarse=str(args.coarse),
    fine=str(args.fine),
    episodes=rows,
)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
