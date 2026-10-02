"""Freeze the three-condition study only after all full-cycle development gates."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--development", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


campaign = json.loads((a.development / "campaign.json").read_text())
assert campaign["status"] == "development gates passed"
assert campaign["seeds"] == [91000, 91001, 91002]
assert campaign["fixtures_spawned_at_reset_pose"]
for name, expected in campaign["source_sha256"].items():
    assert digest(ROOT / name) == expected, f"Candidate changed after development: {name}"
gates = {}
runtime_sources = None
for mode in ["free", "support", "two-hands"]:
    gate_path = a.development / f"{mode}-numerical-gate.json"
    gate = json.loads(gate_path.read_text())
    assert gate["passed"] and gate["scope"] == "full horizon"
    runs = []
    for hz in [240, 480]:
        run = a.development / f"{mode}-{hz}"
        report = json.loads((run / "report.json").read_text())
        replay = json.loads((run / "independent-replay.json").read_text())
        geometry = json.loads((run / "geometry-audit.json").read_text())
        assert report["seconds"] == 180 and abs(report["dt"] - 1 / hz) < 1e-12
        assert report["seeds"] == campaign["seeds"] and report["mode"] == mode
        assert report["fixtures_spawned_at_reset_pose"] and not report["hold_reset_commands"]
        assert not report.get("dry_diagnostic", False) and not report.get("isolated_fixture_diagnostic", False)
        assert report["feedback_frame"] == campaign["feedback_frame"]
        assert report["grasp_effort_command_Nm"] == campaign["grasp_effort_Nm"]
        assert report["robot_solver_iterations"] == [12, 2] and report["solver_type"] == 0
        assert not report["initial_load_balanced"]
        assert replay["complete"] and replay["finite"] and replay["recorded_success_agrees"]
        assert replay["last_time_s"] >= 180 - 1 / 30 - 1e-6
        assert not replay["episodes"][-1]["success"]
        if mode == "two-hands":
            assert all(row["success"] for row in replay["episodes"][:-1])
        for row in geometry["episodes"]:
            assert not row["cad_collisions"] and row["max_joint_limit_excess_rad"] <= 0.002
            assert max(row["tcp_fk_max_error_m"].values()) < 1e-4
            assert row["max_motor_force_N"] <= 1540.001
        if runtime_sources is None:
            runtime_sources = report["source_files"]
        assert runtime_sources == report["source_files"]
        for name, expected in report["source_files"].items():
            assert digest(ROOT / name) == expected, f"Runtime changed: {name}"
        imported = json.loads((run / "import_audit.json").read_text())
        fixture = json.loads((run / "fixture_audit.json").read_text())
        assert imported["passed"] and fixture["passed"]
        assert imported["urdf_sha256"] == digest(ROOT / "src/wasman/assets/data/robots/rexrov2_bimanual/rexrov2_bimanual.urdf")
        assert fixture["fixture_sha256"] == digest(ROOT / "src/wasman/assets/data/objects/industrial_valve/valve_bimanual_4x.usda")
        runs.append({"path": str(run), "report_sha256": digest(run / "report.json"),
                     "trace_sha256": digest(run / "trace.npz"), "geometry": geometry, "replay": replay})
    gates[mode] = {"sha256": digest(gate_path), "result": gate, "runs": runs}

files = list((ROOT / "src/wasman").rglob("*.py"))
for directory in ["src/wasman/physics/data", "src/wasman/assets/data/robots", "src/wasman/assets/data/objects",
                  "src/wasman/assets/data/environments/standard_pool", "src/wasman/assets/data/panels/ship_green"]:
    files.extend(f for f in (ROOT / directory).rglob("*")
                 if f.is_file() and "__pycache__" not in str(f) and not f.name.startswith("."))
files.extend((ROOT / "scripts").glob("*bimanual*.py"))
files.extend(ROOT / name for name in ["scripts/rex_rotor_presentation.py", "configs/studies/rex-centered-button-pose.json",
                                     "docs/studies/bimanual-valve-v1-protocol.md", "research/media-requirements.txt", "uv.lock"])
result = {
    "study": "bimanual-valve-v1", "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "final_seeds": list(range(92000, 92030)), "demo_seed": 91003, "development_seeds": campaign["seeds"],
    "modes": ["free", "support", "two-hands"], "seconds": 180, "dt": 1 / 240, "control_hz": 30,
    "feedback_frame": campaign["feedback_frame"], "grasp_effort_Nm": campaign["grasp_effort_Nm"],
    "fixtures_spawned_at_reset_pose": True, "solver_type": 0, "solver_iterations": [12, 2],
    "gates": gates, "source_sha256": {str(f.relative_to(ROOT)): digest(f) for f in sorted(set(files))},
    "isaaclab_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / ".deps/IsaacLab", text=True).strip(),
    "isaaclab_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT / ".deps/IsaacLab")).hexdigest(),
    "packages": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "pin", "imageio-ffmpeg", "warp-lang"]},
    "criterion": "sampled signed shaft angle >=170 degrees at 30 Hz",
    "scope": "Scripted controllers, custom twin-Oberon Rex and explicit 506 mm valve; no hardware validation.",
}
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(f"Frozen {len(result['source_sha256'])} source/asset files; final states may now be evaluated without tuning.")
