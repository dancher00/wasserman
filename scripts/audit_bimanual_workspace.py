"""Bounded dual-arm workspace check before dynamic experiments."""

from pathlib import Path
import json, numpy as np
from scipy.spatial.transform import Rotation
from wasman.controllers.bimanual_kinematics import BimanualKinematics
from wasman.physics.rexrov2_bimanual import load_bimanual_parameters

out = Path("artifacts/bimanual_valve_v1_20260927")
out.mkdir(exist_ok=True)
w = BimanualKinematics()
q = np.zeros(w.model.nq)
a = json.load(open("configs/studies/rex-centered-button-pose.json"))["targets"]["wall"]["q"]
for s in w.indices:
    q[w.indices[s]] = a
results = {}
for mode in ["support", "two-hands"]:
    p = q.copy()
    rows = []
    for angle in np.linspace(0, np.deg2rad(175), 71):
        rot = Rotation.from_rotvec([angle, 0, 0]).as_matrix()
        # Native fixture for support study; separate 0.5 m wheel for twin contact.
        if mode == "support":
            targets = {
                "left": (np.array([2.60, 0.50, -0.4]), np.eye(3)),
                "right": (np.array([2.60, -0.50, -0.4]) + rot @ np.array([0, -0.059, 0]), rot),
            }
        else:
            targets = {
                s: (np.array([2.50, 0, -0.4]) + rot @ np.array([0, sign * 0.25, 0]), rot)
                for s, sign in [("left", 1), ("right", -1)]
            }
        errors = {}
        for s, (pos, R) in targets.items():
            p, errors[s] = w.solve_arm(s, pos, R, p)
        collisions = w.collisions(p)
        rows.append({"angle_rad": float(angle), "q": p.tolist(), "errors": errors, "collisions": collisions})
    results[mode] = {
        "max_error": max(max(r["errors"].values()) for r in rows),
        "colliding_poses": sum(bool(r["collisions"]) for r in rows),
        "samples": rows,
    }
p = load_bimanual_parameters({w.model.names[i]: q[w.model.joints[i].idx_q] for i in range(1, w.model.njoints)})
results["platform"] = {
    "mass_kg": float(p.mass.sum()),
    "volume_m3": float(p.volume.sum()),
    "net_weight_N": float((p.mass.sum() - p.water_density * p.volume.sum()) * p.gravity),
    "rank": int(np.linalg.matrix_rank(p.allocation)),
}
(out / "workspace-initial.json").write_text(json.dumps(results, indent=2) + "\n")
print({k: {x: v for x, v in val.items() if x != "samples"} for k, val in results.items()})
