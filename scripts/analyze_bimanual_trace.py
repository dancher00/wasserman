"""Independent shaft-angle replay and phase-specific contact diagnostics."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("--progress", action="store_true")
args = parser.parse_args()
path = args.run / ("progress.npz" if args.progress else "trace.npz")
with np.load(path) as source:
    d = dict(source)
# The fixed reference is identity. Float32 acos(dot(q, identity)) rounds small
# nonzero errors to zero; derive diagnostics from the recorded quaternion in
# float64. This does not change controls, recorded states or angle success.
orientation_error = Rotation.from_quat(d["quaternion"].astype(np.float64).reshape(-1, 4)).magnitude().reshape(d["attitude_error"].shape)
spec = importlib.util.spec_from_file_location("recorded_plan", args.run / "bimanual_turn_plan.py")
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)
report = json.loads((args.run / "report.json").read_text()) if (args.run / "report.json").exists() else {}
names = report.get("contact_body_names")
finger_indices = {"left": ([8, 9], [10, 11]), "right": ([19, 20], [21, 22])}
if names:
    finger_indices = {
        s: (
            [names.index(s + "_oberon_finger_left"), names.index(s + "_oberon_finger_tip_left")],
            [names.index(s + "_oberon_finger_right"), names.index(s + "_oberon_finger_tip_right")],
        )
        for s in finger_indices
    }
force = d["normal_contact_by_body"]
bilateral = {}
for side, fingers in finger_indices.items():
    for target, index in [("wheel", 0), ("rail", 2)]:
        a = np.linalg.norm(force[:, :, fingers[0], index].sum(2), axis=-1)
        b = np.linalg.norm(force[:, :, fingers[1], index].sum(2), axis=-1)
        bilateral[side + "_" + target] = (a > 0.5) & (b > 0.5)
rows = []
for env in range(d["valve_angle"].shape[1]):
    times = d["plan_time"][:, env] if "plan_time" in d else d["time"]
    phases = np.array([plan.two_hand_plan(float(t))[3] for t in times])
    row = dict(
        environment=env,
        seed=(report.get("seeds", []) + [None] * (env + 1))[env],
        zero_motor_control=env == d["valve_angle"].shape[1] - 1,
        success=bool(np.any(d["valve_angle"][:, env] >= np.deg2rad(170))),
        max_angle_deg=float(np.rad2deg(d["valve_angle"][:, env].max())),
        final_angle_deg=float(np.rad2deg(d["valve_angle"][-1, env])),
        phases={},
    )
    row["final_plan_time_s"] = float(times[-1])
    if "plan_waiting" in d:
        row["waiting_time_s"] = float(d["plan_waiting"][:, env].sum() / 30)
    for phase in dict.fromkeys(phases):
        mask = phases == phase
        row["phases"][phase] = dict(
            samples=int(mask.sum()),
            bilateral_contact_fraction={k: float(v[mask, env].mean()) for k, v in bilateral.items()},
            base_attitude_rms_deg=float(np.rad2deg(np.sqrt(np.mean(orientation_error[mask, env] ** 2)))),
            base_translation_rms_m=float(np.sqrt(np.mean(d["position_error"][mask, env] ** 2))),
            right_tcp_rms_m=float(
                np.sqrt(np.mean(np.sum((d["tool_position"][mask, env] - d["target_tool"][mask, env]) ** 2, axis=-1)))
            ),
            max_motor_utilization=float(np.abs(d["motor_force"][mask, env]).max() / 1540),
        )
        if "wheel_torque_by_hand_Nm" in d:
            torque = d["wheel_torque_by_hand_Nm"][mask, env, :, 0]
            power = torque * d["valve_angular_speed"][mask, env, None]
            row["phases"][phase]["hand_wheel_interaction_estimate"] = {
                side: {
                    "mean_shaft_torque_Nm": float(torque[:, hand].mean()),
                    "signed_work_J": float(power[:, hand].sum() / 30),
                    "positive_work_J": float(np.maximum(power[:, hand], 0).sum() / 30),
                }
                for hand, side in enumerate(["left", "right"])
            }
    rows.append(row)
result = dict(
    complete=not args.progress,
    last_time_s=float(d["time"][-1]),
    criterion="sampled signed shaft angle >=170 degrees",
    contact_diagnostic="both opposing finger groups >0.5 N normal force; not a success criterion",
    hand_work_note=report.get("hand_torque_note", "Moment estimator unspecified") + " Work uses 30 Hz rectangular integration; not actuator energy.",
    finite=all(np.isfinite(v).all() for v in d.values()),
    orientation_measurement="float64 rotation magnitude from recorded XYZW quaternion relative to fixed identity target",
    analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    episodes=rows,
)
if report:
    result["recorded_success_agrees"] = [r["success"] for r in rows] == report["successes"]
if not args.progress:
    (args.run / "independent-replay.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
