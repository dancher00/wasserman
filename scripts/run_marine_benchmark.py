"""Serial, artifact-gated benchmark queue. Requires frozen expert/physics acceptance."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
BASE = ROOT / "artifacts/marine_mechanisms_20260924"
ENV = dict(
    os.environ,
    WARP_CACHE_PATH=str(ROOT / ".deps/valve-warp-cache"),
    PYTHONPATH=str(ROOT / ".deps/valve-policy-deps"),
    OMP_NUM_THREADS="4",
    MKL_NUM_THREADS="4",
)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["PullLever", "PushSlider"], required=True)
    args = parser.parse_args()
    task = args.task
    root = BASE / task
    root.mkdir(exist_ok=True)
    acceptance = BASE / "acceptance.json"
    if not acceptance.is_file():
        raise ValueError("Expert/physics acceptance has not been frozen")
    frozen = json.loads(acceptance.read_text())
    for path, digest in frozen["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError("Frozen source changed: " + path)
    (root / "acceptance.json").write_bytes(acceptance.read_bytes())
    state = root / "queue.json"

    def run(label, cmd, expected):
        for path, digest in frozen["source_sha256"].items():
            if sha(ROOT / path) != digest:
                raise ValueError("Frozen source changed: " + path)
        expected = Path(expected)
        if expected.exists():
            print("verified artifact already present", label, flush=True)
            return
        log = root / (label + ".log")
        write(
            state, dict(task=task, stage=label, status="running", started_unix=time.time(), command=list(map(str, cmd)))
        )
        with log.open("x") as stream:
            code = subprocess.call(
                [str(PYTHON), *map(str, cmd)], cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT
            )
        if code or not expected.is_file():
            write(
                state,
                dict(task=task, stage=label, status="failed", exit_code=code, expected=str(expected), log=str(log)),
            )
            raise RuntimeError(f"{label} did not produce its completion artifact")
        print("completed", label, flush=True)

    train_start, val_start, test_start = (8200, 8300, 8310) if task == "PushSlider" else (8400, 8500, 8510)
    for batch in range(13):
        successful = []
        for p in sorted(root.glob("train_batch*/seed_*/metadata.json")):
            if json.loads(p.read_text())["outcome"] == "success":
                successful.append(p)
        if len(successful) >= 80:
            break
        # At most100 predefined attempts; preserve every failure.
        seeds = list(range(train_start + batch * 8, min(train_start + batch * 8 + 8, train_start + 100)))
        if not seeds:
            break
        out = root / f"train_batch{batch:02d}"
        run(
            f"collect_{batch:02d}",
            [
                "scripts/rollout_marine_visual.py",
                "--task",
                task,
                "--mode",
                "collect",
                "--seeds",
                *map(str, seeds),
                "--output-dir",
                out,
            ],
            out / "report.json",
        )
        if batch == 0:
            pilot_episodes = [
                p.parent
                for p in sorted(out.glob("seed_*/metadata.json"))
                if json.loads(p.read_text())["outcome"] == "success"
            ][:2]
            if len(pilot_episodes) != 2:
                raise ValueError("Initial collection gate: fewer than two successes")
            for episode in pilot_episodes:
                seed = json.loads((episode / "metadata.json").read_text())["seed"]
                replay_out = root / f"packing_gate_{seed}"
                run(
                    replay_out.name,
                    [
                        "scripts/rollout_marine_visual.py",
                        "--task",
                        task,
                        "--mode",
                        "replay",
                        "--seeds",
                        str(seed),
                        "--replay-episode",
                        episode,
                        "--output-dir",
                        replay_out,
                    ],
                    replay_out / "report.json",
                )
                if json.loads((replay_out / "report.json").read_text())["success_per_seed"] != [True]:
                    raise ValueError("Early packing replay failed; collection stopped")
    run("dataset_audit", ["scripts/audit_marine_dataset.py", "--data", root], root / "dataset_audit.json")
    audit = json.loads((root / "dataset_audit.json").read_text())
    if not audit["passed"] or len(audit["selected_seeds"]) != 80:
        raise ValueError("Dataset gate")
    for seed in audit["selected_seeds"][:2]:
        episode = next(root.glob(f"train_batch*/seed_{seed}"))
        out = root / f"replay_{seed}"
        run(
            f"replay_{seed}",
            [
                "scripts/rollout_marine_visual.py",
                "--task",
                task,
                "--mode",
                "replay",
                "--seeds",
                str(seed),
                "--replay-episode",
                episode,
                "--output-dir",
                out,
            ],
            out / "report.json",
        )
        if json.loads((out / "report.json").read_text())["success_per_seed"] != [True]:
            raise ValueError("Lossless replay did not complete task; training blocked")
    selections = {}
    for model in ["act", "dp"]:
        logdir = ROOT / f"logs/{task.lower()}_{model}_v1"
        run(
            "train_" + model,
            [f"scripts/train_marine_{model}.py", "--task", task, "--data", root, "--output-dir", logdir],
            logdir / "completed.json",
        )
        budgets = [5000, 10000, 20000] if model == "act" else [10, 20, 40]
        candidates = []
        for budget in budgets:
            checkpoint = logdir / (f"model_{budget}.pt" if model == "act" else f"model_epoch_{budget}.pt")
            out = root / f"validation_{model}_{budget}"
            run(
                out.name,
                [
                    "scripts/rollout_marine_visual.py",
                    "--task",
                    task,
                    "--mode",
                    "policy",
                    "--purpose",
                    "validation",
                    "--checkpoint",
                    checkpoint,
                    "--seeds",
                    *map(str, range(val_start, val_start + 8)),
                    "--output-dir",
                    out,
                ],
                out / "report.json",
            )
            report = json.loads((out / "report.json").read_text())
            if not report["contract_replayed"] or report["expert_at_inference"]:
                raise ValueError("Validation provenance")
            candidates.append(
                dict(
                    budget=budget,
                    successes=sum(report["success_per_seed"]),
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=sha(checkpoint),
                )
            )
        selected = max(candidates, key=lambda r: (r["successes"], -r["budget"]))
        selections[model] = dict(selected=selected, validation=candidates, decision_unix=time.time(), test_opened=False)
        selection_path = root / f"selection_{model}.json"
        if selection_path.exists():
            previous = json.loads(selection_path.read_text())
            if previous["selected"] != selected or previous["validation"] != candidates:
                raise ValueError("Preserve original pre-test selection")
            selections[model] = previous
        else:
            write(selection_path, selections[model])
        for purpose, seeds in [("test", list(range(test_start, test_start + 30)) + [42]), ("single42", [42])]:
            out = root / f"{purpose}_{model}"
            run(
                out.name,
                [
                    "scripts/rollout_marine_visual.py",
                    "--task",
                    task,
                    "--mode",
                    "policy",
                    "--purpose",
                    purpose,
                    "--checkpoint",
                    selected["checkpoint"],
                    "--seeds",
                    *map(str, seeds),
                    "--output-dir",
                    out,
                ],
                out / "report.json",
            )
        out = root / f"video_{model}"
        run(
            out.name,
            [
                "scripts/rollout_marine_visual.py",
                "--task",
                task,
                "--mode",
                "policy",
                "--purpose",
                "single42",
                "--checkpoint",
                selected["checkpoint"],
                "--seeds",
                "42",
                "--observer-video",
                "--output-dir",
                out,
            ],
            out / "report.json",
        )
    out = root / "video_expert"
    run(
        "video_expert",
        [
            "scripts/rollout_marine_visual.py",
            "--task",
            task,
            "--mode",
            "expert",
            "--seeds",
            "42",
            "--observer-video",
            "--output-dir",
            out,
        ],
        out / "report.json",
    )
    write(
        root / "completed.json",
        dict(task=task, completed_unix=time.time(), selections=selections, acceptance_sha256=sha(acceptance)),
    )
    write(state, dict(task=task, status="complete", stage="experiments", website_published=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.exit(1)
