"""Package a separate expert film with contact intervals from its physical trace."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path)
p.add_argument("--poster", type=Path, required=True)
p.add_argument("--destination", type=Path, required=True)
p.add_argument("--observer-dir", type=Path, help="Pool state-replay renderer output for this same run")
p.add_argument("--url-prefix", default="./static/bimanual-valve")
a = p.parse_args()
r = json.loads((a.run / "report.json").read_text())
observer_dir = a.observer_dir or a.run
o = json.loads((observer_dir / "observer.json").read_text())
assert r["seeds"] == [91003], "Demo must use its preselected separate development reset"
assert o["fps"] == 30 and not o["overlays"]
if a.observer_dir:
    assert o["complete"] and o["seed"] == 91003 and o["mode"] == r["mode"]
    assert o["presentation"] == "standard-pool-ship-green-v1"
    assert o["physics_steps"] == 0 and o["max_rendered_tcp_error_m"] < 1e-4
    assert o["source_trace_sha256"] == hashlib.sha256((a.run / "trace.npz").read_bytes()).hexdigest()
    assert min(o["robot_pool_clearance"].values()) > 0
    assert o["frames"] == 5400
d = np.load(a.run / "trace.npz")
force = d["normal_contact_by_body"][:, 0]
intervals = []
for side, groups in [("left", ([8, 9], [10, 11])), ("right", ([19, 20], [21, 22]))]:
    target = 2 if side == "left" and r["mode"] == "support" else 0
    aa = np.linalg.norm(force[:, groups[0], target].sum(1), axis=-1)
    bb = np.linalg.norm(force[:, groups[1], target].sum(1), axis=-1)
    mask = (aa > 0.5) & (bb > 0.5)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        intervals.append(dict(side=side, start_s=float(start / 30), end_s=float(end / 30)))
a.destination.mkdir(parents=True, exist_ok=True)
paths = {}
for kind, source, suffix in [("video", observer_dir / "observer.mp4", ".mp4"), ("poster", a.poster, a.poster.suffix)]:
    dest = a.destination / (r["mode"] + suffix)
    if dest.exists():
        raise FileExistsError(dest)
    shutil.copy2(source, dest)
    paths[kind] = a.url_prefix.rstrip("/") + "/" + dest.name
    paths[kind + "_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
result = dict(
    **paths,
    duration_s=o["frames"] / 30,
    contact_intervals=intervals,
    contact_sampling=("30 Hz recorded physical states and contacts at the same timestamps" if a.observer_dir else "30 Hz pre-control telemetry; camera samples after the corresponding physical interval (one-frame offset)"),
    demo_seed=91003,
    mode=r["mode"],
    success=bool(r["successes"][0]),
    rotor_presentation=o["rotor_presentation"],
    presentation=o.get("presentation"),
    presentation_only=bool(a.observer_dir),
    source_trace_sha256=hashlib.sha256((a.run / "trace.npz").read_bytes()).hexdigest(),
)
(a.destination / (r["mode"] + "-media.json")).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "contact_intervals"}, indent=2))
