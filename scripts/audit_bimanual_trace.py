"""Independent CPU CAD, native-limit and both-tool FK audit of recorded states."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from wasman.controllers.bimanual_kinematics import BimanualKinematics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
args = parser.parse_args()
with np.load(args.run / "trace.npz") as source:
    # NPZ access decompresses a whole array each time. Load the small geometry
    # fields once before the per-pose loop (especially important for 30 resets).
    d = {
        key: source[key]
        for key in [
            "time",
            "joint_position",
            "quaternion",
            "position",
            "left_tool_position",
            "tool_position",
            "motor_force",
        ]
    }
report = json.loads((args.run / "report.json").read_text())
w = BimanualKinematics()
ids = [w.model.joints[w.model.getJointId(n)].idx_q for n in report["joint_names"]]
rows = []
for env, seed in enumerate(report["seeds"]):
    collisions = []
    errors = {"left": [], "right": []}
    limit_excess = 0.0
    for i, t in enumerate(d["time"]):
        q = np.zeros(w.model.nq)
        q[ids] = d["joint_position"][i, env]
        limit_excess = max(limit_excess, float(np.max(w.low - q)), float(np.max(q - w.high)))
        contact = w.collisions(q)
        if contact:
            collisions.append(dict(time_s=float(t), pairs=contact))
        rotation = Rotation.from_quat(d["quaternion"][i, env]).as_matrix()
        for side, (position, _) in w.poses(q).items():
            expected = d["position"][i, env] + rotation @ position
            measured = d["left_tool_position" if side == "left" else "tool_position"][i, env]
            errors[side].append(float(np.linalg.norm(expected - measured)))
    rows.append(
        dict(
            seed=seed,
            samples=len(d["time"]),
            cad_collisions=collisions,
            max_joint_limit_excess_rad=limit_excess,
            max_motor_force_N=float(np.abs(d["motor_force"][:, env]).max()),
            tcp_fk_max_error_m={s: max(v) for s, v in errors.items()},
        )
    )
result = dict(
    method="native CAD convex hulls; same-link/directly-connected pairs excluded; all powered 30 Hz samples",
    episodes=rows,
)
(args.run / "geometry-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print([{**r, "cad_collisions": len(r["cad_collisions"])} for r in rows])
