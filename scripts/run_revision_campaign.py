"""Resume a finite, frozen six-task campaign with bounded storage and three training runs."""

import argparse
import concurrent.futures
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

from wasman.learning.revision_protocol import PROTOCOL, TASKS, sha256, validate_runtime_snapshot

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv/bin/python")
SEEDS = (17, 43, 101)
MODELS = ("ACT", "DP", "BC")


def read(path):
    return json.loads(Path(path).read_text())


def run(root, label, command, output, *, training=False):
    receipts = root / "jobs"
    receipt = receipts / f"{label}.json"
    if receipt.exists() and read(receipt).get("returncode") == 0:
        return
    resume = output.exists() and training and (output / "last.pt").exists()
    if receipt.exists() or output.exists():
        if not resume:
            raise RuntimeError(f"Incomplete job requires investigation before retry: {label}")
        receipt = receipts / f"{label}-resume-{int(time.time())}.json"
    command = [PYTHON, *map(str, command)]
    if resume:
        command.append("--resume")
    subprocess.run(
        [
            PYTHON,
            "scripts/run_revision_job.py",
            "--record",
            str(receipt),
            "--output-dir",
            str(output),
            *(["--resume"] if resume else []),
            "--",
            *command,
        ],
        cwd=ROOT,
        check=True,
    )


def prune_optimizer(model):
    """Fixed-final-budget runs have no later training stage; retain final weights."""
    completed = read(model / "completed.json")
    if sha256(model / "final.pt") != completed["final_sha256"]:
        raise ValueError("Final checkpoint changed")
    last = model / "last.pt"
    if last.exists():
        receipt = dict(
            path="last.pt",
            bytes=last.stat().st_size,
            sha256=sha256(last),
            final_sha256=completed["final_sha256"],
            reason="Redundant completed fixed-budget optimizer state",
        )
        (model / "optimizer-pruned.json").write_text(json.dumps(receipt, indent=2) + "\n")
        last.unlink()


def fit(root, task, model, seed, data, *, interface=None):
    label = f"{task}-{model}-{seed}" + ("-ee" if interface == "ee" else "")
    output = root / "models" / label
    if (output / "completed.json").exists():
        if sha256(output / "final.pt") != read(output / "completed.json")["final_sha256"]:
            raise ValueError("Existing trained model changed")
    else:
        run(
            root,
            label,
            [
                "scripts/train_revision_policy.py",
                "--task",
                task,
                "--model",
                model,
                "--seed",
                seed,
                "--data",
                data,
                "--output-dir",
                output,
                *(["--interface", interface] if interface else []),
            ],
            output,
            training=True,
        )
    prune_optimizer(output)


def collect_and_train(root, task, workers):
    index = TASKS.index(task)
    data = root / "data" / task
    archived = root / "archives" / task / "rgb-manifest.json"
    if archived.exists() and (data / "completed.json").exists():
        return
    data.mkdir(parents=True, exist_ok=True)
    if not (data / "revision_audit.json").exists():
        success = 0
        for batch in range(50):
            start = 40000 + 1000 * index + batch * 8
            seeds = list(range(start, start + 8))
            output = data / f"train_batch{batch:02d}"
            run(
                root,
                f"{task}-collect-{batch:02d}",
                ["scripts/benchmark.py", "collect", "--task", task, "--seeds", *seeds, "--output-dir", output],
                output,
            )
            metas = [read(output / f"seed_{seed}" / "metadata.json") for seed in seeds]
            success += sum(meta["outcome"] == "success" for meta in metas)
            if success >= 80:
                break
        if success < 80:
            raise RuntimeError("Predeclared 400-candidate stream exhausted; do not cherry-pick new seeds")
        subprocess.run([PYTHON, "scripts/audit_revision_dataset.py", "--task", task, "--data", str(data)], check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fit, root, task, model, seed, data) for model in MODELS for seed in SEEDS]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    if task in ("PushSlider", "PullLever"):
        ee = root / "data" / f"{task}-ee"
        if not (ee / "view.json").exists():
            subprocess.run(
                [PYTHON, "scripts/prepare_revision_interface.py", "--data", str(data), "--output-dir", str(ee)],
                check=True,
            )
        if not (ee / "revision_audit.json").exists():
            subprocess.run([PYTHON, "scripts/audit_revision_dataset.py", "--task", task, "--data", str(ee)], check=True)
        seeds = [row["seed"] for row in read(data / "revision_audit.json")["episodes"][:8]]
        reports = {}
        for interface, dataset in (("actuator", data), ("ee", ee)):
            output = root / "interface-gates" / f"{task}-{interface}"
            run(
                root,
                f"{task}-{interface}-replay",
                [
                    "scripts/rollout_research_visual.py",
                    "--task",
                    task,
                    "--mode",
                    "replay",
                    "--interface",
                    interface,
                    "--seeds",
                    *seeds,
                    "--replay-dataset",
                    dataset,
                    "--output-dir",
                    output,
                ],
                output,
            )
            report = read(output / "report.json")
            if report["success_per_seed"] != [True] * 8:
                raise RuntimeError(f"Paired replay failed: {task}/{interface}; keep all outcomes and diagnose")
            reports[interface] = dict(
                path=os.path.relpath(output / "report.json", ee), sha256=sha256(output / "report.json")
            )
        gate = dict(
            passed=True,
            asset_profile="open-procedural-v1",
            seeds=seeds,
            reports=reports,
            dataset_audit_sha256=sha256(ee / "revision_audit.json"),
            native_audit_sha256=sha256(data / "revision_audit.json"),
        )
        (ee / "ee_replay_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fit, root, task, "DP", seed, ee, interface="ee") for seed in SEEDS]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    subprocess.run(
        [
            PYTHON,
            "scripts/archive_revision_rgb.py",
            "--data",
            str(data),
            "--archive",
            str(root / "archives" / task),
            "--prune-rgb",
            "--codec",
            "libx264rgb",
        ],
        check=True,
    )
    (data / "completed.json").write_text(
        json.dumps(
            dict(
                protocol=PROTOCOL,
                task=task,
                archived=True,
                training_seeds=SEEDS,
                models=MODELS,
                dataset_audit_sha256=sha256(data / "revision_audit.json"),
            ),
            indent=2,
        )
        + "\n"
    )


def evaluate(root, task, model, seed, *, purpose="test", multiplier=1.0, hydro="nominal", interface=None):
    index = TASKS.index(task)
    label = f"{task}-{model}-{seed}" + ("-ee" if interface == "ee" else "")
    checkpoint = root / "models" / label / "final.pt"
    if purpose == "test":
        reset_start = 60000 + 1000 * index
        group = "primary-evaluation"
    elif interface is not None:
        reset_start = 80000 + 1000 * index
        group = f"interface-evaluation/{interface}"
    else:
        reset_start = 70000 + 1000 * index
        group = f"controller-evaluation/ki{multiplier:g}-{hydro}"
    output = root / group / label
    command = [
        "scripts/benchmark.py",
        "evaluate",
        "--task",
        task,
        "--checkpoint",
        checkpoint,
        "--purpose",
        purpose,
        "--seeds",
        *range(reset_start, reset_start + 30),
        "--output-dir",
        output,
    ]
    if interface is not None:
        command += ["--interface", interface]
    elif purpose == "research":
        command += ["--integral-multiplier", multiplier, "--hydro-condition", hydro]
    run(root, f"eval-{group.replace('/', '-')}-{label}", command, output)


def evaluation_plan(stage):
    if stage == "evaluate":
        return [(task, model, seed, {}) for task, model, seed in itertools.product(TASKS, MODELS, SEEDS)]
    if stage != "diagnostics":
        raise ValueError("Unknown evaluation stage")
    jobs = []
    for task in ("OpenHatch", "RotateValve"):
        for seed in SEEDS:
            for multiplier in (0.0, 0.5, 1.0):
                jobs.append((task, "DP", seed, dict(purpose="research", multiplier=multiplier)))
            for hydro in ("published-damping", "published-added-mass", "published-both"):
                jobs.append((task, "DP", seed, dict(purpose="research", hydro=hydro)))
            jobs.append((task, "DP", seed, dict(purpose="research", multiplier=0.0, hydro="published-both")))
    for task in ("PushSlider", "PullLever"):
        for seed in SEEDS:
            for interface in ("actuator", "ee"):
                jobs.append((task, "DP", seed, dict(purpose="research", interface=interface)))
    return jobs


def evaluate_stage(root, stage, workers):
    jobs = evaluation_plan(stage)
    if workers == 1:
        for task, model, seed, options in jobs:
            evaluate(root, task, model, seed, **options)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate, root, task, model, seed, **options) for task, model, seed, options in jobs]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            # Retain and finish already running jobs; cancel work that has not started.
            for future in futures:
                future.cancel()
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=["collect-train", "evaluate", "diagnostics", "all"], default="all")
    p.add_argument("--training-workers", type=int, choices=[1, 2, 3], default=3)
    p.add_argument("--evaluation-workers", type=int, choices=[1, 2], default=1)
    a = p.parse_args()
    manifest = validate_runtime_snapshot()
    a.root.mkdir(parents=True, exist_ok=True)
    provenance = a.root / "campaign.json"
    record = dict(
        protocol=PROTOCOL,
        runtime_manifest_sha256=manifest,
        training_workers=a.training_workers,
        training_seeds=SEEDS,
        models=MODELS,
        tasks=TASKS,
        compute_note="Wall times are measured with concurrent training; sample budgets are matched, not FLOPs",
    )
    if provenance.exists() and read(provenance) != json.loads(json.dumps(record)):
        raise ValueError("Campaign configuration changed")
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    scheduling = a.root / f"execution-schedule-{time.time_ns()}.json"
    scheduling.write_text(
        json.dumps(
            dict(
                stage=a.stage,
                training_workers=a.training_workers,
                evaluation_workers=a.evaluation_workers,
                orchestrator_sha256=sha256(__file__),
                scope="Concurrent independent processes; each evaluation retains its complete ordered 30-reset cohort",
            ),
            indent=2,
        )
        + "\n"
    )
    if a.stage in ("collect-train", "all"):
        for task in TASKS:
            collect_and_train(a.root, task, a.training_workers)
    if a.stage in ("evaluate", "all"):
        evaluate_stage(a.root, "evaluate", a.evaluation_workers)
    if a.stage in ("diagnostics", "all"):
        evaluate_stage(a.root, "diagnostics", a.evaluation_workers)
    (a.root / f"completed-{a.stage}.json").write_text(
        json.dumps(dict(completed_unix=time.time(), protocol=PROTOCOL)) + "\n"
    )


if __name__ == "__main__":
    main()
