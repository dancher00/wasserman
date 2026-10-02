import copy
import json
import runpy
import sys
import types
from pathlib import Path

import pytest

from wasman.recording_publication import geometry_metadata, publication_paths
from wasman.robot_camera_profiles import robot_camera_profile

ROOT = Path(__file__).resolve().parents[1]


def geometry():
    return {
        "robot_geometry_version": "registered-v1",
        "robot_camera_profile": "geometry-v2",
        "robot_camera_parameters": json.loads(json.dumps(robot_camera_profile("geometry-v2").metadata())),
        "registered_geometry_audit": {"passed": True, "source_sha256": "recorded-source"},
        "camera_streams": {"enabled": True},
    }


@pytest.mark.parametrize("task,prefix,trace", [
    ("rotate-valve", "rotate_valve", "rotate_valve_trace.json"),
    ("open-hatch", "hatch", "hatch_expert_trace.json"),
])
def test_paths_keep_legacy_and_isolate_version(tmp_path, task, prefix, trace):
    legacy = publication_paths(tmp_path, task)
    assert legacy.destination == tmp_path / "website/public/static"
    assert legacy.artifact.name == f"{prefix}_site_demo.json"
    assert legacy.trace_name == trace
    version = publication_paths(tmp_path, task, "registered-v1")
    assert version.destination == legacy.destination / "recordings" / task / "registered-v1"
    assert version.artifact.name == f"{prefix}_registered-v1_demo.json"
    assert version.url_prefix == f"./static/recordings/{task}/registered-v1"
    assert (version.report_name, version.trace_name) == ("report.json", "trace.json")
    version.prepare()
    version.write_report({"successes": 1})
    assert json.loads((version.destination / "report.json").read_text()) == {"successes": 1}
    with pytest.raises(FileExistsError):
        publication_paths(tmp_path, task, "registered-v1")
    with pytest.raises(FileExistsError):
        version.prepare()
    with pytest.raises(FileExistsError):
        version.write_report({"successes": 0})
    assert json.loads(version.artifact.read_text())["successes"] == 1


@pytest.mark.parametrize("version", ["", "..", "../elsewhere", "a/b", "A", "a b", "a." , "a" * 65])
def test_unsafe_versions_rejected(tmp_path, version):
    with pytest.raises(ValueError):
        publication_paths(tmp_path, "open-hatch", version)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("target,symlink", [("artifact", False), ("destination", True), ("artifact", True)])
def test_existing_artifact_or_broken_symlink_is_not_overwritten(tmp_path, target, symlink):
    paths = publication_paths(tmp_path, "open-hatch", "test")
    path = getattr(paths, target)
    path.parent.mkdir(parents=True)
    if symlink:
        path.symlink_to(tmp_path / "absent")
    else:
        path.write_text("preserve")
    with pytest.raises(FileExistsError):
        publication_paths(tmp_path, "open-hatch", "test")
    assert path.is_symlink() if symlink else path.read_text() == "preserve"


@pytest.mark.parametrize("field,value", [
    ("robot_geometry_version", "legacy"),
    ("robot_camera_profile", "legacy-v1"),
    ("robot_camera_parameters", {}),
    ("registered_geometry_audit", {"passed": False}),
    ("registered_geometry_audit", {"passed": 1}),
    ("registered_geometry_audit", None),
    ("camera_streams", {"enabled": False}),
])
def test_registered_requires_actual_verified_geometry_and_cameras(field, value):
    record = geometry()
    record[field] = value
    with pytest.raises(ValueError):
        geometry_metadata(record, "registered-v1")


def test_metadata_is_recorded_not_relabelled():
    record = geometry()
    snapshot = copy.deepcopy(record)
    result = geometry_metadata(record, "registered-v1")
    assert result["registered_geometry_audit"] == snapshot["registered_geometry_audit"]
    assert result["recording_version"] == "registered-v1"
    assert record == snapshot
    with pytest.raises(ValueError):
        geometry_metadata({}, "registered-v1")
    assert geometry_metadata({}, None) == {}
    assert geometry_metadata({"robot_camera_profile": "legacy-v1"}, "archive") == {
        "robot_camera_profile": "legacy-v1", "recording_version": "archive",
    }


def source_fixture(source, kind):
    if kind in ("valve", "button"):
        record = runpy.run_path(str(ROOT / f"tests/test_{kind}_recording.py"))["recording"]()
        rows = record["trace"]
    else:
        rows = runpy.run_path(str(ROOT / "tests/test_hatch_recording.py"))["trace"]()
        for row in rows:
            row["phase"] = [4]
        record = {"num_envs": 1, "failed": [False], "seed": 42, "conditions": {}, "source_sha256": {}}
    record.update(geometry())
    if kind == "button":
        record.update(converted_geometry_audit={"passed": True}, robot_urdf_sha256="test", thruster_positions_m=[])
    record["camera_streams"].update({
        "frames": len(rows), "observer_initial_frames_to_trim": 1, "initial_hold_s": 1 / 30,
        "timestamps_s": [[row["t"], row["t"]] for row in rows],
    })
    source.mkdir()
    (source / "trace.json").write_text(json.dumps(record if kind != "hatch" else rows))
    if kind == "hatch":
        (source / "summary.json").write_text(json.dumps(record))
    for name in ("expert_0000", "base", "gripper"):
        filename = "policy_orbit_0000.mp4" if kind == "button" and name == "expert_0000" else f"{kind}_{name}.mp4"
        (source / filename).write_bytes(name.encode())
    return rows


@pytest.mark.parametrize("kind,task", [("valve", "rotate-valve"), ("hatch", "open-hatch"), ("button", "press-button")])
@pytest.mark.parametrize("fault", [None, "contact", "camera", "encoded_25fps", "missing_encoded_frame"])
def test_publish_version_routes_real_contract_replay_without_legacy_changes(tmp_path, monkeypatch, kind, task, fault):
    source = tmp_path / "source"
    rows = source_fixture(source, kind)
    if fault == "contact":
        data = json.loads((source / "trace.json").read_text())
        (data["trace"] if kind != "hatch" else data)[0]["success"] = [True]
        (source / "trace.json").write_text(json.dumps(data))
    legacy = tmp_path / "website/public/static"
    legacy.mkdir(parents=True)
    sentinel = legacy / "old-film.mp4"
    sentinel.write_bytes(b"untouched")
    duration = len(rows) / 30

    def read_frames(path):
        observer = "expert_0000" in path or "policy_orbit_0000" in path
        encoded = "/recordings/" in path
        yield {"fps": 25 if encoded and fault == "encoded_25fps" else 30,
               "duration": duration + (1 / 30 if observer else 0) + (
            1 if fault == "camera" and "gripper" in path else 0
        )}
        for _ in range(len(rows) + int(observer) - (1 if encoded and fault == "missing_encoded_frame" else 0)):
            yield b"decoded-frame"

    # Publication tests exercise real contract replay; only multimedia I/O is stubbed.
    ffmpeg = types.ModuleType("imageio_ffmpeg")
    ffmpeg.get_ffmpeg_exe = lambda: "ffmpeg"
    ffmpeg.read_frames = read_frames
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", ffmpeg)
    script = "publish_button_cameras.py" if kind == "button" else f"publish_{kind}_demo.py"
    module = runpy.run_path(str(ROOT / "scripts" / script))
    calls = []

    def encode(command, *, check):
        calls.append(command)
        if command[-1].endswith(".mp4"):
            assert command[command.index("-r") + 1] == "30"
            assert command[command.index("-fps_mode") + 1] == "cfr"
        Path(command[-1]).write_bytes(b"encoded test fixture")

    monkeypatch.setattr(module["subprocess"], "run", encode)
    version = "registered-v2" if kind == "button" else "registered-v1"
    args = [str(source), "--version", version]
    destination = legacy / "recordings" / task / version
    if fault:
        with pytest.raises(ValueError):
            module["main"](args, root=tmp_path)
        assert not (destination / "report.json").exists()
        if fault in ("contact", "camera"):
            assert not destination.exists()
            assert calls == []
    else:
        module["main"](args, root=tmp_path)
        report = json.loads((destination / "report.json").read_text())
        assert report["telemetry_contract_replayed"] is True
        assert report["robot_camera_parameters"] == geometry()["robot_camera_parameters"]
        assert report["registered_geometry_audit"]["passed"] is True
        prefix = f"./static/recordings/{task}/{version}/"
        fields = ("video_url", "trace_url", "report_url") + (() if kind == "button" else ("poster_url",))
        for field in fields:
            assert report[field].startswith(prefix)
            assert (destination / report[field].removeprefix(prefix)).is_file()
        assert all(stream["src"].startswith(prefix) for stream in report["camera_streams"])
        assert (destination / "trace.json").read_bytes() == (source / "trace.json").read_bytes()
        with pytest.raises(FileExistsError):
            module["main"](args, root=tmp_path)
        assert report["observer_encoding"]["frames"] == len(rows)
        assert report["observer_encoding"]["full_decode_verified"] is True
        assert len(calls) == (1 if kind == "button" else 2)
    assert sentinel.read_bytes() == b"untouched"


@pytest.mark.parametrize("fault", [None, "no_opt_in", "wrong_version", "wrong_log", "two_missing", "last_only_success"])
def test_hatch_only_recovers_evidenced_continuous_prefix(tmp_path, monkeypatch, fault):
    source = tmp_path / "source"
    rows = source_fixture(source, "hatch")
    while len(rows) < 1700:
        row = copy.deepcopy(rows[-1])
        row["t"] = (len(rows) + 1) / 30
        rows.append(row)
    if fault == "last_only_success":
        # Keep a valid source whose first completion is on the omitted final frame.
        base = runpy.run_path(str(ROOT / "tests/test_hatch_recording.py"))["trace"]()
        rows = []
        for index in range(1700):
            row = copy.deepcopy(base[max(0, index - 1587)])
            row.update(t=(index + 1) / 30, phase=[4])
            rows.append(row)
    trace_path = source / "trace.json"
    trace_path.write_text(json.dumps(rows))
    original = trace_path.read_bytes()
    summary = json.loads((source / "summary.json").read_text())
    summary["moviepy_version"] = "1.0.3" if fault == "wrong_version" else "2.2.1"
    summary["camera_streams"].update(frames=1700, timestamps_s=[[r["t"], r["t"]] for r in rows])
    (source / "summary.json").write_text(json.dumps(summary))
    log = tmp_path / "recorder.log"
    log.write_text(f"[INFO]: [VideoRecorder] Wrote {1700 if fault == 'wrong_log' else 1701} frames "
                   f"to {source / 'hatch_expert_0000.mp4'}\n")

    def read_frames(path):
        count = 1699 if "/recordings/" in path else 1700
        if fault == "two_missing" and "expert_0000" in path:
            count -= 1
        yield {"fps": 30, "duration": count / 30}
        yield from (b"frame" for _ in range(count))

    ffmpeg = types.ModuleType("imageio_ffmpeg")
    ffmpeg.get_ffmpeg_exe = lambda: "ffmpeg"
    ffmpeg.read_frames = read_frames
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", ffmpeg)
    module = runpy.run_path(str(ROOT / "scripts/publish_hatch_demo.py"))

    def encode(command, *, check):
        Path(command[-1]).write_bytes(b"prefix")

    monkeypatch.setattr(module["subprocess"], "run", encode)
    args = [str(source), "--version", "registered-v1", "--recorder-log", str(log)]
    if fault != "no_opt_in":
        args.append("--allow-moviepy-tail-loss")
    dest = tmp_path / "website/public/static/recordings/open-hatch/registered-v1"
    if fault:
        with pytest.raises(ValueError):
            module["main"](args, root=tmp_path)
        assert not dest.exists()
    else:
        module["main"](args, root=tmp_path)
        report = json.loads((dest / "report.json").read_text())
        assert len(json.loads((dest / "trace.json").read_text())) == 1699
        assert report["duration_s"] == 1699 / 30
        assert report["telemetry_contract_replayed"]
        assert report["observer_encoding"]["frames"] == 1699
        assert all(stream["frames"] == 1699 for stream in report["camera_streams"])
        recovery = report["continuous_prefix_recovery"]
        assert recovery["source_telemetry_duration_s"] == 1700 / 30
        assert recovery["telemetry_dropped_tail_frames"] == recovery["base_dropped_tail_frames"] == 1
        assert recovery["source_trace_sha256"] != report["trace_sha256"]
        assert (dest / "encoder-tail-loss-evidence.txt").exists()
    assert trace_path.read_bytes() == original
