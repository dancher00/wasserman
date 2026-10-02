"""Sequential development gates; never launches final states or policy training."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--wait-for-pid", type=int)
p.add_argument("--after-short", type=Path, help="Wait for a matching first-turn diagnostic and require it to pass")
p.add_argument("--feedback-frame", choices=["world", "level"], default="world")
p.add_argument("--grasp-effort", type=float)
p.add_argument("--spawn-fixture-at-reset-pose", action="store_true")
p.add_argument("--modes", nargs="+", choices=["two-hands", "support", "free"], default=["two-hands", "support", "free"])
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
if (a.output / "campaign.json").exists():
    raise FileExistsError(a.output / "campaign.json")
sources = [
    ROOT / "scripts/probe_bimanual_valve.py",
    *sorted((ROOT / "src/wasman/controllers").glob("bimanual*.py")),
    ROOT / "src/wasman/assets/rexrov2_bimanual.py",
    ROOT / "src/wasman/physics/rexrov2_bimanual.py",
    ROOT / "src/wasman/physics/contact_wrench.py",
]
hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
state = dict(status="queued", scope="development only", seeds=[91000, 91001, 91002], source_sha256=hashes,
             feedback_frame=a.feedback_frame, grasp_effort_Nm=a.grasp_effort,
             fixtures_spawned_at_reset_pose=a.spawn_fixture_at_reset_pose, completed=[])
(a.output / "campaign_runner.py").write_bytes(Path(__file__).read_bytes())


def save():
    temporary = a.output / "campaign-next.json"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(a.output / "campaign.json")


def call(arguments, log):
    with log.open("w") as stream:
        subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            env={**os.environ, "OMP_NUM_THREADS": "4"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


save()
try:
    if a.wait_for_pid:
        while True:
            try:
                os.kill(a.wait_for_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(5)
    if a.after_short:
        while True:
            short = json.loads((a.after_short / "status.json").read_text())
            if short["status"] == "stopped":
                raise RuntimeError("Short diagnostic failed; full cycle not launched: " + short.get("error", ""))
            if short["status"] == "short diagnostic passed; full cycle not tested":
                break
            time.sleep(5)
        assert short["scope"] == "first-turn diagnostic only"
        assert short["feedback_frame"] == a.feedback_frame and short["grasp_effort_Nm"] == a.grasp_effort
        assert short["fixtures_spawned_at_reset_pose"] == a.spawn_fixture_at_reset_pose
        assert not short["balance_initial_load"] and short["solver_iterations"] == [12, 2]
        for name, expected in short["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, "Short-tested source changed"
        short_gate = a.after_short / "numerical-gate.json"
        assert json.loads(short_gate.read_text())["passed"]
        state["short_gate"] = {"path": str(short_gate), "sha256": hashlib.sha256(short_gate.read_bytes()).hexdigest()}
        save()
    for mode in a.modes:
        for hz in [240, 480]:
            for f, expected in hashes.items():
                if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Candidate changed: {f}")
            run = a.output / f"{mode}-{hz}"
            state.update(status="running", active=str(run))
            save()
            call(
                [
                    "scripts/probe_bimanual_valve.py",
                    "--fixture",
                    "large",
                    "--mode",
                    mode,
                    "--seeds",
                    "91000",
                    "91001",
                    "91002",
                    "--seconds",
                    "180",
                    "--dt",
                    str(1 / hz),
                    "--solver-type",
                    "0",
                    "--feedback-frame",
                    a.feedback_frame,
                    *(["--spawn-fixture-at-reset-pose"] if a.spawn_fixture_at_reset_pose else []),
                    *(["--grasp-effort", str(a.grasp_effort)] if a.grasp_effort is not None else []),
                    "--output-dir",
                    str(run),
                ],
                a.output / f"{mode}-{hz}.log",
            )
            call(["scripts/analyze_bimanual_trace.py", str(run)], a.output / f"{mode}-{hz}-replay.log")
            call(["scripts/audit_bimanual_trace.py", str(run)], a.output / f"{mode}-{hz}-geometry.log")
            replay = json.loads((run / "independent-replay.json").read_text())
            geometry = json.loads((run / "geometry-audit.json").read_text())
            assert replay["finite"] and replay["recorded_success_agrees"]
            assert not replay["episodes"][-1]["success"]
            for row in geometry["episodes"]:
                assert not row["cad_collisions"], f"CAD collision: {run}, {row['seed']}"
                assert max(row["tcp_fk_max_error_m"].values()) < 1e-4
                assert row["max_joint_limit_excess_rad"] <= 0.002, f"Native joint-limit violation: {row}"
            if mode == "two-hands":
                assert all(e["success"] for e in replay["episodes"][:-1]), "Two-hand feasibility failed"
            state["completed"].append(str(run))
            save()
        gate = a.output / f"{mode}-numerical-gate.json"
        call(
            [
                "scripts/audit_bimanual_numerics.py",
                str(a.output / f"{mode}-240"),
                str(a.output / f"{mode}-480"),
                "--output",
                str(gate),
            ],
            a.output / f"{mode}-numerical-gate.log",
        )
        assert json.loads(gate.read_text())["passed"], f"Numerical gate failed: {gate}"
    state.update(status="development gates passed", active=None)
    save()
except BaseException as exc:
    state.update(status="stopped", error=repr(exc))
    save()
    raise
