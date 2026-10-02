"""Audit native-limit IK and inter-link CAD collisions along all three plans."""

import argparse
import json
from pathlib import Path

import numpy as np

from wasman.controllers.bimanual_kinematics import BimanualKinematics
from wasman.controllers.bimanual_turn_plan import mode_plan, BASE_POSITION

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
w = BimanualKinematics()
rows = {}
initial = json.loads(Path("configs/studies/rex-centered-button-pose.json").read_text())["targets"]["wall"]["q"]
for mode in ["free", "support", "two-hands"]:
    q = np.zeros(w.model.nq)
    for side in w.indices:
        q[w.indices[side]] = initial
    samples = []
    for t in np.arange(0, 180.01, 0.5):
        positions, rotations, grip, phase = mode_plan(t, mode)
        errors = {}
        for side in w.indices:
            q, errors[side] = w.solve_arm(side, positions[side] - BASE_POSITION, rotations[side], q)
            q[w.indices[side][-2:]] = grip[side]
        samples.append(dict(time_s=float(t), phase=phase, error=errors, collisions=w.collisions(q), q=q.tolist()))
    arm_ids = [i for side in w.indices.values() for i in side[:6]]
    velocity = abs(np.diff(np.array([s["q"] for s in samples]), axis=0)) / 0.5
    rows[mode] = dict(
        max_pose_residual=max(max(s["error"].values()) for s in samples),
        max_nominal_joint_velocity_fraction=float((velocity[:, arm_ids] / w.model.velocityLimit[arm_ids]).max()),
        colliding_samples=sum(bool(s["collisions"]) for s in samples),
        samples=samples,
    )
args.output.write_text(json.dumps(rows, indent=2) + "\n")
print({m: {k: v for k, v in r.items() if k != "samples"} for m, r in rows.items()})
