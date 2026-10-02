"""Publish only evaluated learned models; expert demos remain separate media assets."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def write(path, data):
    (ROOT / path).write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-hatch", action="store_true")
    parser.add_argument("--include-shell", action="store_true")
    args = parser.parse_args()
    button = read("artifacts/press_button_t200_evaluation.json")
    rows = [
        dict(
            task="PressButton",
            model="PPO",
            input="State",
            successes=button["successes"],
            episodes=button["episodes"],
            median_success_time_s=None,
        )
    ]
    protocols = [
        dict(
            task="PressButton",
            description=(
                "Short-start task, legacy-v1 geometry, T200 actuation, 16 s horizon; "
                "1,024 episodes, evaluation seed 2031. State policy trained with seed 42. "
                "The registered-geometry approach film uses a different demonstration protocol."
            ),
        )
    ]
    task_roots = [("RotateValve", "valve_visual_20260923")]
    if args.include_hatch:
        task_roots.append(("OpenHatch", "hatch_visual_20260923"))
    if args.include_shell:
        task_roots.append(("CollectShell", "shell_visual_20260923"))
    for task, directory in task_roots:
        root = Path("artifacts") / directory
        verification = read(root / "final_verified.json")
        verified_runs = {
            r.get("run", Path(r.get("directory", "")).name): r
            for r in verification.get("runs", verification.get("reports", []))
        }
        report = read(root / "final_report/results.json")
        methods = [r for r in report["methods"] if r["model"] in ("ACT", "DP")]
        if len(methods) != 2 or report["final_methods_pending"]:
            raise ValueError(f"Incomplete model benchmark: {task}")
        for row in methods:
            for suffix in ("test31", "single42"):
                name = f"{row['model'].lower()}_{suffix}"
                proof = verified_runs[name]
                for file, digest in (
                    ("rollout.pt", proof.get("trace_sha256", proof.get("rollout_sha256"))),
                    ("summary.json", proof["summary_sha256"]),
                ):
                    with (ROOT / root / name / file).open("rb") as source:
                        actual = hashlib.file_digest(source, "sha256").hexdigest()
                    if actual != digest:
                        raise ValueError(f"Verified evidence changed: {task}/{name}/{file}")
            raw = read(root / f"{row['model'].lower()}_test31/summary.json")
            if (
                row["episodes"] != 30
                or row["successes"] != sum(raw["success_per_seed"][:30])
                or row.get("checkpoint_sha256", raw["checkpoint_sha256"]) != raw["checkpoint_sha256"]
            ):
                raise ValueError(f"Aggregate report differs from raw evaluation: {task}/{row['model']}")
            row["checkpoint_sha256"] = raw["checkpoint_sha256"]
            rows.append(
                dict(
                    task=task,
                    input="Wrist RGB + proprioception",
                    **{k: row[k] for k in ("model", "successes", "episodes", "median_success_time_s")},
                )
            )
        public = dict(report)
        public["methods"] = [
            {k: v for k, v in r.items() if k not in ("raw_report", "initial_pose_max_error_vs_expert")} for r in methods
        ]
        file = {"RotateValve": "rotate-valve", "OpenHatch": "open-hatch", "CollectShell": "collect-shell"}[task]
        write(f"website/public/static/{file}-model-benchmarks.json", public)
        if task == "RotateValve":
            description = (
                "Signed turn ≥170°, calm water, fixed valve position, registered-v1 geometry. "
                "75 s horizon; 74.67 s evaluated. Thirty unseen reset seeds (2500–2529), "
                "80 demonstrations, ACT 20,000 updates / DP 40 epochs. Separate single-environment "
                "seed 42 also succeeds for both models. RGB only; no object state or expert actions at evaluation."
            )
        elif task == "OpenHatch":
            description = (
                "Opening 80–105°, ≥75° grasped motion, ≤5° ungrasped motion, opposing grasp and "
                "a stable 1 s hold. Calm water, fixed hatch, registered-v1 geometry. 60 s horizon; "
                "59.67 s evaluated. Thirty unseen reset seeds (4400–4429), 80 demonstrations; "
                "ACT 20,000 updates / DP 40 epochs. Absolute base/joint/jaw targets preserve "
                "folded transit. No object state or expert actions at evaluation. Separate seed 42 is in the report. "
                "An instrumented ACT repeat scored16/30 versus the original10/30; exact rollout repeatability remains a limitation."
            )
        else:
            description = (
                "One target shell with six coral/algae habitat groups. Opposing grasp, whole-object lift ≥4 cm, "
                "carried distance ≥10 cm, full footprint inside hoop, released and settled for 1 s. "
                "Calm water; 240 s horizon; 239.67 s evaluated. Thirty unseen reset seeds (5500–5529), "
                "80 demonstrations; ACT 20,000 updates / DP 40 epochs. Wrist RGB and measured robot state only; "
                "absolute base/joint/jaw targets. Separate seed 42 is in the report."
            )
        protocols.append(dict(task=task, description=description))
    data = dict(rows=rows, protocols=protocols)
    write("website/src/modelBenchmarks.json", data)
    write("website/public/static/model-benchmarks.json", data)
    print(f"Published {len(rows)} model rows across {len(protocols)} tasks locally")


if __name__ == "__main__":
    main()
