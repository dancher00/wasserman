"""Physically audit the fixed eight-reset, post-test validation extension without selecting checkpoints."""

import argparse
import itertools
import json
from pathlib import Path

from revision_scoring_evidence import audit_saved_trace
from score_revision_campaign import MODELS, TRAINING_SEEDS, binomial_interval

from wasman.learning.revision_protocol import (
    CONTROL_STEPS,
    PROTOCOL,
    TASKS,
    sha256,
    validate_evaluation,
    validate_runtime_snapshot,
)


def load(core, external, task, model, seed):
    label = f"{task}-{model}-{seed}"
    folder = external / (label + "-validation") / "rollout"
    report_path = folder / ("report.json" if task in ("PressButton", "PushSlider", "PullLever") else "summary.json")
    report = json.loads(report_path.read_text())
    model_folder = core / "models" / label
    config = json.loads((model_folder / "config.json").read_text())
    completed = json.loads((model_folder / "completed.json").read_text())
    validate_runtime_snapshot(config["runtime_manifest_sha256"])
    validate_evaluation(config, task, report["seeds"], report["purpose"], steps=CONTROL_STEPS[task])
    if (
        report["purpose"] != "validation"
        or config["protocol"] != PROTOCOL
        or config["model"] != model
        or config["seed"] != seed
        or config["pilot"]
        or completed["pilot"]
        or completed["samples"] != 320000
        or report["checkpoint_sha256"] != completed["final_sha256"]
        or sha256(model_folder / "final.pt") != completed["final_sha256"]
        or report["expert_at_inference"]
        or report["inference_precision"] != "fp32"
        or (
            task != "PressButton"
            and report.get("policy_random_seed", report.get("inference_random_seed")) != config["rollout_seed"]
        )
    ):
        raise ValueError(f"Validation execution identity mismatch: {label}")
    if "diagnostic_condition" in report:
        condition = report["diagnostic_condition"]
        if condition["integral_multiplier"] != 1 or condition["hydro_condition"] != "nominal":
            raise ValueError("Validation contains a controller/model intervention")
    if "interface" in report and report["interface"] != config["interface"]:
        raise ValueError("Validation action interface changed")
    if task == "PressButton":
        manifest = json.loads((Path(__file__).resolve().parents[1] / "research/revision-v2-runtime.json").read_text())
        if (
            config["rollout_seed"] != 42
            or manifest["source_sha256"]["scripts/rollout_button_visual.py"] not in report["source_sha256"].values()
        ):
            raise ValueError("Button noise seed is not bound to the pinned runtime")
    outcomes, physical = audit_saved_trace(folder, task, report, expected_episodes=8)
    return dict(
        task=task,
        model=model,
        training_seed=seed,
        successes=int(outcomes.sum()),
        episodes=8,
        reset_seeds=report["seeds"],
        success_per_seed=outcomes.tolist(),
        conditional_binomial_95=binomial_interval(int(outcomes.sum()), 8),
        report=str(report_path.relative_to(external)),
        report_sha256=sha256(report_path),
        checkpoint_sha256=completed["final_sha256"],
        **physical,
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
            rows.append(load(args.core, args.external, task, model, seed))
        except FileNotFoundError:
            missing.append(f"{task}-{model}-{seed}")
    if missing and not args.allow_incomplete:
        raise ValueError(f"Incomplete validation extension: {len(missing)} missing runs")
    result = dict(
        protocol=PROTOCOL,
        complete=not missing,
        rows=rows,
        missing=missing,
        scorer_sha256=sha256(__file__),
        scope=(
            "Fixed eight-reset validation cohorts on the second workstation; "
            "execution scheduled after first test exposure. "
            "No tuning, checkpoint selection or training-seed replacement. These are supplemental checks, "
            "not primary test scores or additional independent trainings."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=not missing, runs=len(rows), missing=len(missing))))


if __name__ == "__main__":
    main()
