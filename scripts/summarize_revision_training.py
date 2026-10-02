"""Audit recorded training costs; timings are concurrent-run observations, not speed benchmarks."""

import argparse
import itertools
import json
import math
from pathlib import Path

from wasman.learning.revision_protocol import PROTOCOL, TASKS, sha256, validate_runtime_snapshot


def read(path):
    return json.loads(path.read_text())


def expected_runs():
    for task, model, seed in itertools.product(TASKS, ("ACT", "DP", "BC"), (17, 43, 101)):
        yield task, model, seed, ""
    for task, seed in itertools.product(("PushSlider", "PullLever"), (17, 43, 101)):
        yield task, "DP", seed, "-ee"


def checked_metrics(records, config, completed):
    """Keep every logged record, including any recomputation after an interruption."""
    expected_updates = 5000 if config["model"] == "DP" else 10000
    expected_batch = 64 if config["model"] == "DP" else 32
    if (
        not records
        or config["batch_size"] != expected_batch
        or config["sample_budget"] != 320000
        or completed["samples"] != 320000
        or completed["updates"] != expected_updates
        or completed["elapsed_s"] <= 0
        or not math.isfinite(completed["elapsed_s"])
        or records[-1]["step"] != expected_updates
    ):
        raise ValueError("Incomplete or incompatible final training budget")
    for row in records:
        if (
            row["samples"] != row["step"] * expected_batch
            or not 1 <= row["step"] <= expected_updates
            or row["elapsed_s"] <= 0
            or row["gpu_peak_bytes"] <= 0
            or not all(math.isfinite(row[key]) for key in ("loss", "elapsed_s", "gpu_peak_bytes"))
        ):
            raise ValueError("Invalid recorded training metric")
    if records[-1]["elapsed_s"] > completed["elapsed_s"]:
        raise ValueError("Final completion predates the final logged update")
    return dict(
        updates=expected_updates,
        samples=320000,
        batch_size=expected_batch,
        recorded_optimization_elapsed_s=completed["elapsed_s"],
        observed_max_allocated_bytes=max(row["gpu_peak_bytes"] for row in records),
        final_logged_training_loss=records[-1]["loss"],
        logged_records=len(records),
        repeated_logged_steps=len(records) - len({row["step"] for row in records}),
    )


def collect(root):
    rows, missing = [], []
    for task, model, seed, suffix in expected_runs():
        label = f"{task}-{model}-{seed}{suffix}"
        folder = root / "models" / label
        if not (folder / "completed.json").exists():
            missing.append(label)
            continue
        config, completed = read(folder / "config.json"), read(folder / "completed.json")
        validate_runtime_snapshot(config["runtime_manifest_sha256"])
        if (
            config["protocol"] != PROTOCOL
            or config["task"] != task
            or config["model"] != model
            or config["seed"] != seed
            or config["interface"] != ("ee" if suffix or task == "RotateValve" else "actuator")
            or config["pilot"]
            or completed["pilot"]
            or sha256(folder / "final.pt") != completed["final_sha256"]
        ):
            raise ValueError(f"Training identity mismatch: {label}")
        metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        evidence = {name: sha256(folder / name) for name in ("config.json", "completed.json", "metrics.jsonl")}
        rows.append(
            dict(
                label=label,
                task=task,
                model=model,
                training_seed=seed,
                interface=config["interface"],
                parameters=config["parameters"],
                training_precision=config["mixed_precision"],
                training_windows=config["training_windows"],
                microbatch=config["microbatch"],
                dataset_audit_sha256=config["dataset_audit_sha256"],
                final_sha256=completed["final_sha256"],
                evidence_sha256=evidence,
                **checked_metrics(metrics, config, completed),
            )
        )
    return dict(
        protocol=PROTOCOL,
        complete=not missing,
        rows=rows,
        missing=missing,
        expected_primary_runs=54,
        expected_additional_interface_runs=6,
        summarizer_sha256=sha256(__file__),
        interpretation=(
            "Equal sampled-window exposure does not match FLOPs, wall time, parameters, encoders or history. "
            "Elapsed time is the trainer's recorded optimization time under concurrent GPU/CPU work; "
            "it excludes setup and may exclude work lost before an interruption checkpoint. "
            "Memory is the maximum logged PyTorch allocation for that process, not total device usage. "
            "Final logged loss is a training-batch observation, not held-out performance; "
            "loss scales/objectives differ between families. Do not rank speed or convergence from this table."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = collect(args.root)
    if result["missing"] and not args.allow_incomplete:
        raise ValueError(f"Incomplete training campaign: {len(result['missing'])} missing runs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=result["complete"], runs=len(result["rows"]), missing=len(result["missing"]))))


if __name__ == "__main__":
    main()
