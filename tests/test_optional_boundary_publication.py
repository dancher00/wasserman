import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]


def test_optional_demo_defaults_disabled_and_uses_actual_recorded_target():
    public = ROOT / "website/public"
    manifest = json.loads((public / "static/effects/manifest.json").read_text())
    assert manifest["version"] == "v5-optional"
    for mode in manifest["modes"]:
        if mode["id"] not in ("seabed", "wall"):
            continue
        for variant in mode["variants"]:
            assert variant["default_enabled"] is False
            assert variant["delta_semantics"] == "commanded_position_error"
            for take, asset in (("on", variant), ("off", variant["disabled"])):
                for field, hash_key in (("trace", "trace_sha256"), ("video", "sha256")):
                    path = public / asset[field].removeprefix("./")
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == asset[hash_key]
                frames = json.loads((public / asset["trace"].removeprefix("./")).read_text())["frames"]
                source = (ROOT / variant["source_report"]).parents[1] / take / "trace.json"
                original = json.loads(source.read_text())["frames"]
                for frame, row in zip(frames, original, strict=True):
                    assert frame["base_position_w_m"] == row["position_w_m"]
                    assert frame["target_position_w_m"] == row["target_w_m"]
                    target = row["target_w_m"][2] if mode["id"] == "seabed" else 1.1 - row["target_w_m"][0]
                    assert np.isclose(frame["annotations"]["commanded_distance"]["value_m"], target)
                    if take == "off":
                        assert frame["boundary_gain"] == [1] * 8
