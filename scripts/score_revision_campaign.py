"""Score frozen runs without treating reset repetitions as independent trainings."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from revision_scoring_evidence import audit_saved_trace, require_matching_initialization
from scipy.stats import beta

from wasman.learning.revision_protocol import (
    CONTROL_STEPS,
    PROTOCOL,
    TASKS,
    sha256,
    validate_evaluation,
    validate_runtime_snapshot,
)

TRAINING_SEEDS = (17, 43, 101)
MODELS = ("ACT", "DP", "BC")
ANALYSIS_SEED = 20260929


def binomial_interval(k, n):
    if not 0 <= k <= n or n <= 0:
        raise ValueError("Invalid binomial counts")
    return [
        0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)),
    ]


def crossed_interval(left, right=None, *, paired_training=False, repeats=20000):
    """Resample training runs and shared reset IDs as separate crossed factors.

    Between algorithms training indices are independent. Interventions of the
    same frozen models share training indices. Never flatten the3x30 array.
    """
    left = np.asarray(left, dtype=float)
    if left.ndim != 2 or not np.isfinite(left).all():
        raise ValueError("Expected finite training-run by reset matrix")
    if right is not None:
        right = np.asarray(right, dtype=float)
        if right.shape != left.shape or not np.isfinite(right).all():
            raise ValueError("Contrast requires the same reset dimensions")
    rng = np.random.default_rng(ANALYSIS_SEED)
    runs, resets = left.shape
    reset_ids = rng.integers(resets, size=(repeats, resets))
    left_ids = rng.integers(runs, size=(repeats, runs))
    draws = left[left_ids[:, :, None], reset_ids[:, None, :]].mean(axis=(1, 2))
    if right is not None:
        right_ids = left_ids if paired_training else rng.integers(runs, size=(repeats, runs))
        draws -= right[right_ids[:, :, None], reset_ids[:, None, :]].mean(axis=(1, 2))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def summarize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    rates = matrix.mean(axis=1)
    return dict(
        training_runs=len(rates),
        resets_per_run=matrix.shape[1],
        mean=float(rates.mean()),
        training_run_sd=float(rates.std(ddof=1)),
        per_run_means=rates.tolist(),
        descriptive_crossed_95=crossed_interval(matrix),
    )


def load_run(root, task, model, seed):
    label = f"{task}-{model}-{seed}"
    folder = root / "primary-evaluation" / label
    report_path = folder / ("report.json" if task in ("PressButton", "PushSlider", "PullLever") else "summary.json")
    model_dir = root / "models" / label
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    report = json.loads(report_path.read_text())
    completed = json.loads((model_dir / "completed.json").read_text())
    validate_runtime_snapshot(config["runtime_manifest_sha256"])
    validate_evaluation(
        config,
        task,
        report["seeds"],
        report["purpose"],
        steps=report.get("evaluated_steps", CONTROL_STEPS[task]),
    )
    if (
        report["purpose"] != "test"
        or config["protocol"] != PROTOCOL
        or config["seed"] != seed
        or config["model"] != model
        or config["pilot"]
        or completed["pilot"]
        or completed["samples"] != 320000
        or report["expert_at_inference"]
        or report["inference_precision"] != "fp32"
        or (
            task != "PressButton"
            and report.get("policy_random_seed", report.get("inference_random_seed")) != config["rollout_seed"]
        )
        or report["checkpoint_sha256"] != completed["final_sha256"]
    ):
        raise ValueError(f"Scored identity mismatch: {label}")
    if sha256(model_dir / "final.pt") != completed["final_sha256"]:
        raise ValueError("Final checkpoint bytes changed")
    if "diagnostic_condition" in report:
        condition = report["diagnostic_condition"]
        if condition["integral_multiplier"] != 1 or condition["hydro_condition"] != "nominal":
            raise ValueError("Primary score contains a diagnostic intervention")
    if "interface" in report and report["interface"] != config["interface"]:
        raise ValueError("Primary action interface differs from its trained configuration")
    # The frozen Button report predates a separate inference-seed field. Its
    # pinned runtime reads the seed from the checkpoint; preserve that format.
    seed_provenance = "reported"
    if task == "PressButton":
        manifest_path = Path(__file__).resolve().parents[1] / "research/revision-v2-runtime.json"
        manifest = json.loads(manifest_path.read_text())
        expected = manifest["source_sha256"]["scripts/rollout_button_visual.py"]
        if (
            sha256(manifest_path) != config["runtime_manifest_sha256"]
            or expected not in report["source_sha256"].values()
            or config["rollout_seed"] != 42
        ):
            raise ValueError("Button inference seed is not bound to the frozen runtime")
        seed_provenance = "checkpoint config and pinned Button runtime; report omits a separate seed field"
    outcomes, physical_evidence = audit_saved_trace(folder, task, report)
    count = int(outcomes.sum())
    return outcomes, dict(
        **physical_evidence,
        inference_seed_provenance=seed_provenance,
        training_seed=seed,
        successes=count,
        episodes=30,
        conditional_binomial_95=binomial_interval(count, 30),
        reset_seeds=report["seeds"],
        success_per_seed=outcomes.tolist(),
        report=str(report_path.relative_to(root)),
        report_sha256=sha256(report_path),
        config_sha256=sha256(config_path),
        checkpoint_sha256=report["checkpoint_sha256"],
        samples=completed["samples"],
        training_elapsed_s=completed["elapsed_s"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true", help="Explicit progress report; never a complete score")
    a = p.parse_args()
    matrices, rows, missing, initializations = {}, [], [], {}
    for task, model in itertools.product(TASKS, MODELS):
        outcomes, runs = [], []
        for seed in TRAINING_SEEDS:
            try:
                values, record = load_run(a.root, task, model, seed)
            except FileNotFoundError:
                missing.append(f"{task}-{model}-{seed}")
                continue
            outcomes.append(values)
            runs.append(record)
            initializations.setdefault(task, []).append(record)
        if len(runs) == 3:
            matrix = np.stack(outcomes)
            matrices[task, model] = matrix
            rows.append(dict(task=task, model=model, runs=runs, **summarize(matrix)))
    for records in initializations.values():
        require_matching_initialization(records)
    if missing and not a.allow_incomplete:
        raise ValueError(f"Incomplete campaign: {len(missing)} missing runs")
    contrasts = []
    for task in TASKS:
        for left, right in itertools.combinations(MODELS, 2):
            if (task, left) in matrices and (task, right) in matrices:
                x, y = matrices[task, left], matrices[task, right]
                contrasts.append(
                    dict(
                        task=task,
                        contrast=f"{left}-{right}",
                        difference=float(x.mean() - y.mean()),
                        descriptive_crossed_95=crossed_interval(x, y),
                    )
                )
    macro = {}
    for model in MODELS:
        if all((task, model) in matrices for task in TASKS):
            macro[model] = dict(
                mean=float(np.mean([matrices[task, model].mean() for task in TASKS])),
                task_weights="equal; no pooled binomial interval",
            )
    result = dict(
        protocol=PROTOCOL,
        complete=not missing,
        missing=missing,
        rows=rows,
        contrasts=contrasts,
        macro=macro,
        analysis_seed=ANALYSIS_SEED,
        bootstrap_repeats=20000,
        scorer_sha256=sha256(__file__),
        uncertainty_note=(
            "Three training runs provide limited population uncertainty. "
            "Crossed percentile intervals are descriptive, marginal, and not multiplicity-adjusted. "
            "Reset intervals are conditional on each checkpoint."
        ),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=result["complete"], cells=len(rows), missing=len(missing))))


if __name__ == "__main__":
    main()
