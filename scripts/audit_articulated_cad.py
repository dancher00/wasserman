"""Replay every powered Rex pose against nonadjacent native CAD collision hulls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from wasman.controllers.rexrov2_workspace import RexWorkspace

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
r = json.loads((a.run/'report.json').read_text())
t = np.load(a.run/'trace.npz', allow_pickle=False)
w = RexWorkspace()
indices = [w.model.joints[w.model.getJointId(n)].idx_q for n in r['joint_names']]
bad = []
poses = 0
for frame, states in enumerate(t['joint_position']):
    for episode, state in enumerate(states[:len(r['seeds'])]):
        q = np.zeros(w.model.nq)
        q[indices] = state
        collisions = w.collisions(q)
        if collisions:
            bad.append(dict(frame=frame, episode=episode, pairs=collisions))
        poses += 1
result = dict(passed=not bad, poses=poses, violations=bad, scope='Native nonadjacent CAD hull self-collision only; directly adjacent link geometry excluded.', trace_sha256=hashlib.sha256((a.run/'trace.npz').read_bytes()).hexdigest())
with a.output.open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result))
