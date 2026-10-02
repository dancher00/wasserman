"""Score all predeclared native/EE DP runs with independently replayed success."""

import argparse
import json
from pathlib import Path

import numpy as np
from revision_scoring_evidence import checked_outcomes, initialization_fingerprint, require_matching_initialization
from score_revision_campaign import TRAINING_SEEDS, binomial_interval, crossed_interval, summarize

from wasman.learning.revision_protocol import (
    CONTROL_STEPS,
    PROTOCOL,
    TASKS,
    sha256,
    validate_evaluation,
    validate_runtime_snapshot,
)


def load(root, task, seed, interface):
    label = f"{task}-DP-{seed}" + ("-ee" if interface == "ee" else "")
    folder = root / "interface-evaluation" / interface / label
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text())
    model = root / "models" / label
    config = json.loads((model / "config.json").read_text())
    completed = json.loads((model / "completed.json").read_text())
    validate_runtime_snapshot(config["runtime_manifest_sha256"])
    start = 80000 + 1000 * TASKS.index(task)
    validate_evaluation(config, task, report["seeds"], report["purpose"], steps=CONTROL_STEPS[task])
    if (
        report["task"] != task
        or report["interface"] != interface
        or report["purpose"] != "research"
        or report["mode"] != "policy"
        or report["expert_at_inference"]
        or report["seeds"] != list(range(start, start + 30))
        or config["protocol"] != PROTOCOL
        or config["model"] != "DP"
        or config["seed"] != seed
        or config["interface"] != interface
        or config["pilot"]
        or completed["pilot"]
        or completed["samples"] != 320000
        or report["checkpoint_sha256"] != completed["final_sha256"]
        or sha256(model / "final.pt") != completed["final_sha256"]
        or config["rollout_seed"] != report["inference_random_seed"]
        or report["inference_precision"] != "fp32"
        or not report["contract_replayed"]
    ):
        raise ValueError(f"Interface evaluation identity mismatch: {label}")
    criteria = json.loads((folder / "contract.json").read_text())
    with np.load(folder / "trace.npz", allow_pickle=False) as trace:
        outcomes = checked_outcomes(report, trace, criteria, CONTROL_STEPS[task])
        initialization = initialization_fingerprint(trace["measured"][0], report)
    record = dict(
        **initialization,
        training_seed=seed,
        successes=int(outcomes.sum()),
        episodes=30,
        success_per_seed=outcomes.tolist(),
        reset_seeds=report["seeds"],
        conditional_binomial_95=binomial_interval(int(outcomes.sum()), 30),
        checkpoint_sha256=report["checkpoint_sha256"],
        report=str(report_path.relative_to(root)),
        report_sha256=sha256(report_path),
        trace_sha256=sha256(folder / "trace.npz"),
        contract_sha256=sha256(folder / "contract.json"),
        config_sha256=sha256(model / "config.json"),
    )
    return outcomes, record, config, criteria


def score_task(root, task):
    matrices, records, configs, criteria = {}, {}, {}, []
    for interface in ("actuator", "ee"):
        values, records[interface], configs[interface] = [], [], []
        for seed in TRAINING_SEEDS:
            outcomes, record, config, criterion = load(root, task, seed, interface)
            values.append(outcomes)
            records[interface].append(record)
            configs[interface].append(config)
            criteria.append(criterion)
        matrices[interface] = np.stack(values)
    if any(item != criteria[0] for item in criteria):
        raise ValueError("Success contract changed across interface runs")
    require_matching_initialization([record for runs in records.values() for record in runs])
    for native, ee in zip(configs["actuator"], configs["ee"], strict=True):
        for key in (
            "train_seeds",
            "validation_seeds",
            "sample_budget",
            "batch_size",
            "policy_hz",
            "n_action_steps",
            "chunk_size",
            "asset_profile",
            "runtime_manifest_sha256",
        ):
            if native[key] != ee[key]:
                raise ValueError(f"Interface comparison changed {key}")
    x, y = matrices["ee"], matrices["actuator"]
    return dict(
        task=task,
        contrast="ee-actuator",
        difference=float(x.mean() - y.mean()),
        descriptive_crossed_95=crossed_interval(x, y, paired_training=False),
        interfaces={key: dict(runs=records[key], **summarize(matrix)) for key, matrix in matrices.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows, missing = [], []
    for task in ("PushSlider", "PullLever"):
        try:
            rows.append(score_task(args.root, task))
        except FileNotFoundError:
            missing.append(task)
    if missing and not args.allow_incomplete:
        raise ValueError(f"Incomplete interface evaluations: {missing}")
    result = dict(
        protocol=PROTOCOL,
        complete=not missing,
        missing=missing,
        rows=rows,
        scorer_sha256=sha256(__file__),
        interpretation=(
            "Paired reset states; independently resampled training runs between interfaces. "
            "The intervention changes action representation and associated decoding together. "
            "It does not isolate IK, normalization or phase learning. Both tasks and all seeds are retained."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=result["complete"], tasks=len(rows), missing=missing)))


if __name__ == "__main__":
    main()
