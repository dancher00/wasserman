import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]


def test_published_boundary_replay_matches_recorded_evidence():
    public = ROOT / "website/public"
    manifest = json.loads((public / "static/effects/boundary_replay_v1/manifest.json").read_text())
    assert manifest["audit"]["passed"]
    assert manifest["calibrated"] is False
    assert manifest["feedback_compensation"] is False
    assert {mode["id"] for mode in manifest["modes"]} == {"seabed", "wall"}
    for mode in manifest["modes"]:
        traces = []
        for variant in mode["variants"]:
            for key, hash_key in (("video", "sha256"), ("trace", "trace_sha256")):
                path = public / variant[key].removeprefix("./")
                assert hashlib.sha256(path.read_bytes()).hexdigest() == variant[hash_key]
            trace = json.loads((public / variant["trace"].removeprefix("./")).read_text())["frames"]
            assert len(trace) == 240
            source_mode = "floor" if mode["id"] == "seabed" else "wall"
            source = ROOT / "artifacts/boundary_motor_replay_v1" / source_mode / variant["id"] / "trace.json"
            assert trace == json.loads(source.read_text())["frames"]
            traces.append(np.asarray([row["position_w_m"] for row in trace]))
        assert abs(np.linalg.norm(traces[1] - traces[0], axis=1).max() - mode["maximum_position_difference_m"]) < 1e-12
