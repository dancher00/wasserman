"""Describe same-checkpoint cross-workstation agreement without adding training replicates."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from revision_scoring_evidence import audit_saved_trace
from score_revision_campaign import MODELS, TRAINING_SEEDS, load_run

from wasman.learning.revision_protocol import TASKS, sha256


def initial_state(folder):
    if (folder / "trace.npz").exists():
        with np.load(folder / "trace.npz", allow_pickle=False) as trace:
            return trace["measured"][0]
    return torch.load(folder / "rollout.pt", weights_only=True, map_location="cpu")["trace"][0]["measured"].numpy()


def compare_one(core, external, task, model, seed):
    label = f"{task}-{model}-{seed}"
    primary, record = load_run(core, task, model, seed)
    primary_folder = core / "primary-evaluation" / label
    other_folder = external / label / "rollout"
    filename = "report.json" if task in ("PressButton", "PushSlider", "PullLever") else "summary.json"
    left = json.loads((primary_folder / filename).read_text())
    right = json.loads((other_folder / filename).read_text())
    for key in ("seeds", "purpose", "checkpoint_sha256", "inference_precision", "expert_at_inference"):
        if left[key] != right[key]:
            raise ValueError(f"Repetition identity differs: {label}/{key}")
    for key in ("policy_random_seed", "inference_random_seed", "diagnostic_condition", "interface", "policy_hz"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Repetition configuration differs: {label}/{key}")
    repeated, provenance = audit_saved_trace(other_folder, task, right)
    a, b = initial_state(primary_folder), initial_state(other_folder)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid initial measured state")
    return dict(
        task=task,
        model=model,
        training_seed=seed,
        primary=record,
        repeated_report_sha256=sha256(other_folder / filename),
        repeated_evidence=provenance,
        primary_successes=int(primary.sum()),
        repeated_successes=int(repeated.sum()),
        paired_agreement=int(np.sum(primary == repeated)),
        paired_disagreements=[
            dict(reset_seed=state, primary=bool(x), repeated=bool(y))
            for state, x, y in zip(left["seeds"], primary, repeated, strict=True)
            if x != y
        ],
        initial_measured_states_bitwise_equal=bool(np.array_equal(a, b)),
        initial_measured_state_max_abs_difference=float(np.abs(a.astype(float) - b.astype(float)).max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows, missing = [], []
    for task, model, seed in itertools.product(TASKS, MODELS, TRAINING_SEEDS):
        try:
            rows.append(compare_one(args.core, args.external, task, model, seed))
        except FileNotFoundError:
            missing.append(f"{task}-{model}-{seed}")
    if missing and not args.allow_incomplete:
        raise ValueError(f"Incomplete reproduction: {len(missing)} pairs missing")
    result = dict(
        complete=not missing,
        missing=missing,
        pairs=rows,
        scope=(
            "Same final checkpoints and full fixed test cohorts across workstations; "
            "descriptive portability check, not independent training or additional statistical sample size"
        ),
        script_sha256=sha256(__file__),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=result["complete"], paired_runs=len(rows), missing=len(missing))))


if __name__ == "__main__":
    main()
