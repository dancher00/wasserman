import hashlib
import json
from pathlib import Path

import pytest

from wasman.controllers.hatch_recording import replay_hatch_trace
from wasman.controllers.valve_recording import verified_recording
from wasman.robot_camera_profiles import robot_camera_profile

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("task,prefix", [("rotate-valve", "rotate_valve"), ("open-hatch", "hatch")])
def test_versioned_manipulation_film_has_correct_cameras_and_real_completion(task, prefix):
    report = json.loads((ROOT / f"artifacts/{prefix}_registered-v1_demo.json").read_text())
    assert report["robot_geometry_version"] == "registered-v1"
    assert report["robot_camera_profile"] == "geometry-v2"
    assert report["robot_camera_parameters"] == json.loads(json.dumps(robot_camera_profile("geometry-v2").metadata()))
    assert report["registered_geometry_audit"]["passed"]
    assert report["automatic_resets"] == 0
    assert report["successes"] == 1
    public = ROOT / "website/public"
    for url, digest in [(report["video_url"], report["video_sha256"]), (report["trace_url"], report["trace_sha256"])]:
        assert f"recordings/{task}/registered-v1/" in url
        assert hashlib.sha256((public / url.removeprefix("./")).read_bytes()).hexdigest() == digest
    for camera in report["camera_streams"]:
        assert f"recordings/{task}/registered-v1/" in camera["src"]
        assert hashlib.sha256((public / camera["src"].removeprefix("./")).read_bytes()).hexdigest() == camera["sha256"]
        assert abs(camera["duration_s"] - report["duration_s"]) < 0.001
    trace = json.loads((public / report["trace_url"].removeprefix("./")).read_text())
    encoding = report["observer_encoding"]
    assert encoding["full_decode_verified"] and encoding["fps"] == 30
    assert encoding["frames"] == report["camera_streams"][0]["frames"]
    if task == "rotate-valve":
        assert verified_recording(trace)["telemetry_contract_replayed"]
    else:
        assert replay_hatch_trace(trace)["successes"] == 1
