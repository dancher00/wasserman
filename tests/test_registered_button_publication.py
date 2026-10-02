"""Corrected-geometry movie is separately versioned and physically replayable."""

import hashlib
import json
from pathlib import Path

import numpy as np

from wasman.bluerov2_registered import NATIVE_POSITIONS, allocation_matrix
from wasman.controllers.button_recording import verified_button_recording
from wasman.robot_camera_profiles import robot_camera_profile

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "website/public/static"


def test_registered_film_has_native_motor_geometry_and_complete_press():
    report = json.loads((ROOT / "artifacts/press_button_registered_v1_demo.json").read_text())
    trace_path = ROOT / "website/public" / report["trace_url"].removeprefix("./")
    trace = json.loads(trace_path.read_text())
    replay = verified_button_recording(trace)
    assert replay["successes"] == 1 and replay["automatic_resets"] == 0
    assert replay["duration_s"] == 36 and len(trace["trace"]) == 1080
    assert report["robot_geometry_version"] == "registered-v1"
    assert report["robot_camera_profile"] == "geometry-v2"
    assert json.loads(json.dumps(robot_camera_profile("geometry-v2").metadata())) == report["robot_camera_parameters"]
    assert report["converted_geometry_audit"]["passed"]
    assert np.allclose(report["thruster_positions_m"], NATIVE_POSITIONS)
    assert np.allclose([m["origin_m"] for m in report["converted_geometry_audit"]["motors"]], NATIVE_POSITIONS)
    forces = np.array([f["allocated_thrust_n"] for f in trace["trace"]])
    actual = np.array([f["realized_wrench_body"] for f in trace["trace"]])
    assert np.allclose(forces @ allocation_matrix().T, actual, atol=2e-5)
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == report["trace_sha256"]
    for url, digest in [(report["video_url"], report["video_sha256"])] + [
        (s["src"], s["sha256"]) for s in report["camera_streams"]
    ]:
        assert "/registered-v1/" in url
        assert hashlib.sha256((ROOT / "website/public" / url.removeprefix("./")).read_bytes()).hexdigest() == digest
    assert report["camera_alignment"]["same_episode"]
    assert report["camera_alignment"]["max_pair_timestamp_difference_s"] < 1e-5
    assert all(s["frames"] == 1080 for s in report["camera_streams"])


def test_historical_film_and_paired_development_evaluations_are_retained():
    old = json.loads((ROOT / "artifacts/press_button_camera_demo.json").read_text())
    assert hashlib.sha256((STATIC / "press-button.mp4").read_bytes()).hexdigest() == old["video_sha256"]
    legacy = json.loads((ROOT / "artifacts/bluerov_registration_legacy_2071.json").read_text())
    registered = json.loads((ROOT / "artifacts/bluerov_registration_v1_2071.json").read_text())
    assert legacy["seed"] == registered["seed"] == 2071
    assert legacy["checkpoint"] == registered["checkpoint"]
    assert legacy["completed_per_env"] == registered["completed_per_env"] == [1] * 64
    assert legacy["robot_geometry_version"] == "legacy-v1"
    assert registered["robot_geometry_version"] == "registered-v1"
    assert legacy["successes"] == 55 and registered["successes"] == 54
    assert legacy["press_events_max"] == registered["press_events_max"] == 1


def test_active_publication_repairs_cfr_without_changing_physical_take():
    old = json.loads((ROOT / "artifacts/press_button_registered_v1_demo.json").read_text())
    report = json.loads((ROOT / "artifacts/press_button_registered_v2_demo.json").read_text())
    assert report["trace_sha256"] == old["trace_sha256"]
    assert report["robot_geometry_version"] == "registered-v1"
    assert report["robot_camera_profile"] == "geometry-v2"
    assert report["observer_encoding"] == {
        "frames": 1080, "fps": 30.0, "duration_s": 36.0, "full_decode_verified": True,
    }
    for url, digest in [(report["video_url"], report["video_sha256"])] + [
        (s["src"], s["sha256"]) for s in report["camera_streams"]
    ]:
        assert "/registered-v2/" in url
        assert hashlib.sha256((ROOT / "website/public" / url.removeprefix("./")).read_bytes()).hexdigest() == digest
