"""Continue an accepted development campaign into frozen tests and separate films.

No retries, tuning or publication. A missing development process, failed gate,
source mismatch or failed audit stops the queue while preserving every output.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-films", action="store_true")
    args = parser.parse_args()
    args.development = args.development.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "campaign_runner.py").write_bytes(Path(__file__).read_bytes())
    state = {
        "status": "waiting for full development acceptance",
        "development": str(args.development),
        "development_pid": args.development_pid,
        "pid": os.getpid(),
        "completed": [],
        "final_states_opened": False,
        "publication": "manual review required; no automatic upload",
    }

    def save():
        state["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = args.output / "status-next.json"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.output / "status.json")

    def call(command, label):
        state.update(status="running", active=label)
        save()
        with (args.output / (label + ".log")).open("x") as stream:
            subprocess.run(
                [sys.executable, *map(str, command)], cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "4"},
                stdout=stream, stderr=subprocess.STDOUT, check=True,
            )
        state["completed"].append(label)
        save()

    def audit(run, label):
        call(["scripts/analyze_bimanual_trace.py", run], label + "-replay")
        call(["scripts/audit_bimanual_trace.py", run], label + "-geometry")
        replay = json.loads((run / "independent-replay.json").read_text())
        geometry = json.loads((run / "geometry-audit.json").read_text())
        assert replay["complete"] and replay["finite"] and replay["recorded_success_agrees"]
        assert replay["last_time_s"] >= 180 - 1 / 30 - 1e-6
        assert not replay["episodes"][-1]["success"], "Motors-off control turned the valve"
        for row in geometry["episodes"]:
            assert not row["cad_collisions"], f"CAD collision: {label}, {row['seed']}"
            assert row["max_joint_limit_excess_rad"] <= 0.002
            assert row["max_motor_force_N"] <= 1540.001
            assert max(row["tcp_fk_max_error_m"].values()) < 1e-4

    save()
    try:
        while True:
            development = json.loads((args.development / "campaign.json").read_text())
            if development["status"] == "development gates passed":
                # Wait until the parent has exited: no second simulator process.
                try:
                    os.kill(args.development_pid, 0)
                except ProcessLookupError:
                    break
            elif development["status"] == "stopped":
                raise RuntimeError("Development did not pass: " + development.get("error", ""))
            else:
                command_file = Path(f"/proc/{args.development_pid}/cmdline")
                if not command_file.exists():
                    raise RuntimeError("Development process disappeared before acceptance; no final test launched")
                command = command_file.read_bytes().replace(b"\0", b" ")
                if b"scripts/run_bimanual_development.py" not in command:
                    raise RuntimeError("Development PID belongs to a different process")
            time.sleep(5)
        manifest = args.output / "frozen-sources.json"
        call(["scripts/freeze_bimanual_study.py", "--development", args.development,
              "--output", manifest], "freeze")
        runs = []
        for mode in ["two-hands", "support", "free"]:
            run = args.output / ("final-" + mode)
            state["final_states_opened"] = True
            call(["scripts/run_bimanual_study.py", "--manifest", manifest,
                  "--mode", mode, "--split", "final", "--output-dir", run], "final-" + mode)
            audit(run, "final-" + mode)
            runs.append(run)
        summary = args.output / "summary.json"
        call(["scripts/report_bimanual_study.py", "--runs", *runs, "--split", "final",
              "--output", summary], "summary")
        call(["scripts/plot_bimanual_study.py", summary,
              "--output-dir", args.output / "plots"], "plots")
        if not args.skip_films:
            for mode in ["two-hands", "support", "free"]:
                run = args.output / ("demo-" + mode)
                call(["scripts/run_bimanual_study.py", "--manifest", manifest,
                      "--mode", mode, "--split", "demo", "--physics-only", "--output-dir", run], "demo-" + mode)
                audit(run, "demo-" + mode)
                call(["scripts/render_bimanual_trace.py", "--run", run,
                      "--output-dir", run / "pool-render"], "render-" + mode)
        state.update(status="tests complete; inspect results and films before publication", active=None)
        save()
    except BaseException as exc:
        state.update(status="stopped", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
