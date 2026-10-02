import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]


def test_main_proximity_delta_uses_measured_matched_reference():
    public = ROOT / "website/public"
    manifest = json.loads((public / "static/effects/manifest.json").read_text())
    for mode in manifest["modes"]:
        if mode["id"] not in ("seabed", "wall"):
            continue
        maxima = {}
        for variant in mode["variants"]:
            assert variant["hemisphere_radius_m"] == 0.018
            assert variant["feedback_compensation"] is False
            trace_file = public / variant["trace"].removeprefix("./")
            assert hashlib.sha256(trace_file.read_bytes()).hexdigest() == variant["trace_sha256"]
            frames = json.loads(trace_file.read_text())["frames"]
            on = json.loads((ROOT / variant["source_report"]).with_name("trace.json").read_text())["frames"]
            off = json.loads((ROOT / variant["reference_report"]).with_name("trace.json").read_text())["frames"]
            assert len(frames) == len(on) == len(off) == 240
            for frame, actual, reference in zip(frames, on, off, strict=True):
                assert frame["t_s"] == actual["t_s"] == reference["t_s"]
                assert frame["base_position_w_m"] == actual["position_w_m"]
                assert frame["reference_off_position_w_m"] == reference["position_w_m"]
                difference = np.linalg.norm(np.asarray(actual["position_w_m"]) - reference["position_w_m"])
                assert abs(frame["boundary_displacement_m"] - difference) < 1e-12
                for ray in frame["annotations"]["rotor_rays"]:
                    cap = ray["hemisphere"]
                    assert cap["radius_m"] == 0.018
                    assert np.allclose(cap["center_w_m"], ray["end_w_m"])
            maxima[variant["id"].split("-")[-1]] = max(frame["boundary_displacement_m"] for frame in frames)
        assert maxima["far"] < 0.001
        assert maxima["near"] > 0.01
