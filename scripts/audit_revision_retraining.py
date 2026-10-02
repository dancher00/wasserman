"""Audit recorded same-seed full retraining without counting it as another independent seed."""

import argparse
import json
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from score_revision_campaign import load_run

from wasman.learning.revision_protocol import PROTOCOL, sha256, validate_runtime_snapshot


def tensor_differences(left, right):
    if left.keys() != right.keys():
        raise ValueError("Tensor keys differ")
    differences = {}
    for key in left:
        a, b = torch.as_tensor(left[key]), torch.as_tensor(right[key])
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Invalid tensor identity or values: {key}")
        if not torch.equal(a, b):
            differences[key] = float((a.to(torch.float64) - b.to(torch.float64)).abs().max())
    return differences


def load_model(folder, audit):
    config = json.loads((folder / "config.json").read_text())
    complete = json.loads((folder / "completed.json").read_text())
    if (
        config["protocol"] != PROTOCOL
        or config["pilot"]
        or complete["pilot"]
        or config["sample_budget"] != 320000
        or complete["samples"] != 320000
        or config["checkpoint_selection"] != "fixed-final-budget"
        or config["dataset_audit_sha256"] != sha256(audit)
        or complete["final_sha256"] != sha256(folder / "final.pt")
    ):
        raise ValueError("Retraining evidence identity/budget mismatch")
    validate_runtime_snapshot(config["runtime_manifest_sha256"])
    checkpoint = torch.load(folder / "final.pt", map_location="cpu", weights_only=True)
    if json.loads(json.dumps(checkpoint["config"])) != config:
        raise ValueError("Checkpoint and sidecar configurations differ")
    metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    coordinates = [(row["step"], row["samples"]) for row in metrics]
    if (
        not metrics
        or coordinates[0] != (1, config["batch_size"])
        or coordinates[-1] != (complete["updates"], 320000)
        or complete["updates"] * config["batch_size"] != 320000
        or any(step * config["batch_size"] != samples for step, samples in coordinates)
        or any(a[0] >= b[0] for a, b in pairwise(coordinates))
        or not np.isfinite([row["loss"] for row in metrics]).all()
    ):
        raise ValueError("Invalid or incomplete logged training coordinates")
    return config, checkpoint, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "repeated", "original-audit", "repeated-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repeated-rollout", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    left_config, left, left_metrics = load_model(args.original, args.original_audit)
    right_config, right, right_metrics = load_model(args.repeated, args.repeated_audit)
    if left_config != right_config:
        raise ValueError("Same-seed reproduction changed its configuration")
    if [(r["step"], r["samples"]) for r in left_metrics] != [(r["step"], r["samples"]) for r in right_metrics]:
        raise ValueError("Training metric sampling differs")
    weights = tensor_differences(left["state_dict"], right["state_dict"])
    stats = tensor_differences(left["statistics"], right["statistics"])
    loss_difference = np.array([r["loss"] for r in left_metrics]) - np.array([r["loss"] for r in right_metrics])
    result = dict(
        protocol=PROTOCOL,
        task=left_config["task"],
        model=left_config["model"],
        training_seed=left_config["seed"],
        configs_equal=True,
        dataset_audits_equal=sha256(args.original_audit) == sha256(args.repeated_audit),
        state_dict_bitwise_equal=not weights,
        differing_tensors=len(weights),
        parameter_max_abs_differences=weights,
        statistics_bitwise_equal=not stats,
        statistics_max_abs_differences=stats,
        max_logged_loss_absolute_difference=float(np.abs(loss_difference).max()),
        first_logged_losses=[left_metrics[0]["loss"], right_metrics[0]["loss"]],
        final_logged_losses=[left_metrics[-1]["loss"], right_metrics[-1]["loss"]],
        original_checkpoint_sha256=sha256(args.original / "final.pt"),
        repeated_checkpoint_sha256=sha256(args.repeated / "final.pt"),
        input_hashes={
            side: {
                name: sha256(folder / name) for name in ("final.pt", "config.json", "completed.json", "metrics.jsonl")
            }
            for side, folder in (("original", args.original), ("repeated", args.repeated))
        },
        audit_hashes=[sha256(args.original_audit), sha256(args.repeated_audit)],
        auditor_sha256=sha256(__file__),
        scope=(
            "Recorded full same-seed retraining; no new execution or extra independent training replicate. "
            "Logged losses exclude unlogged steps; wall times are not compared across concurrent workloads."
        ),
    )
    if args.repeated_rollout:
        name = "report.json" if left_config["task"] in ("PressButton", "PushSlider", "PullLever") else "summary.json"
        report = json.loads((args.repeated_rollout / name).read_text())
        if report["checkpoint_sha256"] != result["repeated_checkpoint_sha256"] or report["purpose"] != "test":
            raise ValueError("Repeated rollout belongs to a different checkpoint or purpose")
        # Reuse every primary evaluation identity/precision/source guard through
        # a read-only temporary namespace; this does not add a primary run.
        label = f"{right_config['task']}-{right_config['model']}-{right_config['seed']}"
        with tempfile.TemporaryDirectory(prefix="wm-retraining-audit-") as temporary:
            namespace = Path(temporary)
            for group, target in (("models", args.repeated), ("primary-evaluation", args.repeated_rollout)):
                (namespace / group).mkdir()
                (namespace / group / label).symlink_to(target.resolve(), target_is_directory=True)
            outcomes, evidence = load_run(namespace, right_config["task"], right_config["model"], right_config["seed"])
        evidence["report"] = f"rollout/{name}"
        result["repeated_evaluation"] = dict(
            successes=int(outcomes.sum()),
            episodes=len(outcomes),
            evidence=evidence,
            report_sha256=sha256(args.repeated_rollout / name),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("configs_equal", "state_dict_bitwise_equal", "differing_tensors", "statistics_bitwise_equal")
            }
        )
    )


if __name__ == "__main__":
    main()
