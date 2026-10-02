"""Paired frozen-policy diagnostics with a jointly censored 20-second prefix."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from revision_scoring_evidence import check_embedded_report, checked_tensor_outcomes
from score_revision_campaign import TRAINING_SEEDS, crossed_interval, summarize

from wasman.learning.revision_protocol import PROTOCOL, TASKS, sha256, validate_evaluation, validate_runtime_snapshot

CONDITIONS = (
    "ki0-nominal",
    "ki0.5-nominal",
    "ki1-published-damping",
    "ki1-published-added-mass",
    "ki1-published-both",
    "ki0-published-both",
)


def paired_metrics(left_trace, right_trace, *, prefix_steps=600, dt=1 / 30):
    steps = min(prefix_steps, len(left_trace), len(right_trace))
    if not steps:
        raise ValueError("Empty traces")

    def arrays(trace):
        return {
            key: np.stack([np.asarray(row[key]) for row in trace[:steps]])
            for key in (
                "active",
                "reset",
                "grasped",
                "angle",
                "station_position_error",
                "base_attitude",
                "motor_force",
                "motor_saturation_scale",
            )
        }

    left, right = arrays(left_trace), arrays(right_trace)
    valid = left["active"] & right["active"] & ~left["reset"] & ~right["reset"]
    if valid.ndim != 2:
        raise ValueError("Expected time by reset active mask")
    count = valid.sum(axis=0)
    # Zero exposure is explicit NaN internally, serialized as null by the caller.
    divisor = np.where(count > 0, count, np.nan)

    def mean(value):
        return np.where(valid, value, 0).sum(axis=0) / divisor

    def metrics(data):
        motor = data["motor_force"].reshape(steps, valid.shape[1], -1)
        saturation = data["motor_saturation_scale"].reshape(steps, valid.shape[1], -1).min(axis=-1) < (1 - 1e-6)
        delta = np.diff(data["angle"], axis=0, prepend=np.zeros_like(data["angle"][:1]))
        return dict(
            opposing_grasp_fraction=mean(data["grasped"]),
            station_position_rms_m=np.sqrt(mean(np.square(data["station_position_error"]).sum(axis=-1))),
            base_attitude_rms_deg=np.rad2deg(np.sqrt(mean(np.square(data["base_attitude"])))),
            motor_force_rms_N=np.sqrt(mean(np.square(motor).mean(axis=-1))),
            saturation_fraction=mean(saturation),
            signed_progress_deg=np.where(count > 0, np.rad2deg(np.where(valid, delta, 0).sum(axis=0)), np.nan),
            progress_during_opposing_grasp_deg=np.where(
                count > 0, np.rad2deg(np.where(valid & data["grasped"], delta, 0).sum(axis=0)), np.nan
            ),
        )

    return metrics(left), metrics(right), count * dt


def load(root, task, seed, condition):
    folder = root / "controller-evaluation" / condition / f"{task}-DP-{seed}"
    report_path = folder / "summary.json"
    report = json.loads(report_path.read_text())
    index = TASKS.index(task)
    if report["purpose"] != "research" or report["seeds"] != list(range(70000 + 1000 * index, 70030 + 1000 * index)):
        raise ValueError("Diagnostic cohort mismatch")
    checkpoint = root / "models" / f"{task}-DP-{seed}"
    config = json.loads((checkpoint / "config.json").read_text())
    completed = json.loads((checkpoint / "completed.json").read_text())
    validate_runtime_snapshot(config["runtime_manifest_sha256"])
    validate_evaluation(config, task, report["seeds"], report["purpose"], steps=report["evaluated_steps"])
    if (
        config["protocol"] != PROTOCOL
        or config["pilot"]
        or completed["pilot"]
        or config["task"] != task
        or config["model"] != "DP"
        or config["seed"] != seed
        or completed["samples"] != 320000
        or report["expert_at_inference"]
        or report["inference_precision"] != "fp32"
        or report["policy_random_seed"] != config["rollout_seed"]
        or report["checkpoint_sha256"] != completed["final_sha256"]
        or sha256(checkpoint / "final.pt") != completed["final_sha256"]
    ):
        raise ValueError("Diagnostic model mismatch")
    outcomes = np.asarray(report["success_per_seed"])
    if outcomes.shape != (30,) or outcomes.dtype != bool:
        raise ValueError("Require 30 explicit Boolean diagnostic outcomes")
    declared = report["diagnostic_condition"]
    actual = f"ki{declared['integral_multiplier']:g}-{declared['hydro_condition']}"
    if actual != condition:
        raise ValueError("Condition label disagrees with intervention")
    payload = torch.load(folder / "rollout.pt", map_location="cpu", weights_only=True)
    check_embedded_report(payload["summary"], report)
    checked_tensor_outcomes(task, report, payload["trace"])
    return (
        report,
        payload["trace"],
        dict(
            path=str(report_path.relative_to(root)),
            sha256=sha256(report_path),
            trace_sha256=sha256(folder / "rollout.pt"),
        ),
    )


def nullable(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def initialization_comparison(task, reference, variant, reference_trace, variant_trace):
    """Separate the matched reset snapshot from observations after condition-active warmup."""
    left, right = reference["initial_reset"], variant["initial_reset"]
    if left.keys() != right.keys():
        raise ValueError("Initial paired reset fields differ")
    reset_differences = {}
    reset_exact = {}
    for key in left:
        x, y = np.asarray(left[key], dtype=float), np.asarray(right[key], dtype=float)
        if x.shape != y.shape or not x.size or not np.allclose(x, y, atol=1e-7, rtol=0):
            raise ValueError(f"Initial paired reset mismatch: {key}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Nonfinite initial reset: {key}")
        reset_differences[key] = float(np.abs(x - y).max())
        reset_exact[key] = bool(np.array_equal(x, y))
    channels = (
        ["tool_x_m", "tool_y_m", "tool_z_m", "tool_qx", "tool_qy", "tool_qz", "tool_qw"]
        if task == "RotateValve"
        else ["base_x_normalized", "base_y_normalized", "base_z_normalized"]
        + [f"arm_{i}_normalized" for i in range(4)]
        + ["jaw_normalized"]
    )
    x = np.asarray(reference_trace[0]["measured"], dtype=float)
    y = np.asarray(variant_trace[0]["measured"], dtype=float)
    if (
        task not in ("RotateValve", "OpenHatch")
        or x.shape != (len(reference["seeds"]), len(channels))
        or y.shape != x.shape
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Invalid post-warmup measured state")
    return dict(
        reset_max_abs_difference_by_field=reset_differences,
        reset_bitwise_equal_by_field=reset_exact,
        post_warmup_measured_bitwise_equal=bool(np.array_equal(x, y)),
        post_warmup_max_abs_difference_by_channel=dict(zip(channels, np.abs(x - y).max(axis=0).tolist(), strict=True)),
        scope=(
            "Reset snapshots precede one 30-Hz camera-warmup step under the assigned condition. "
            "Measured states follow warmup and precede the first learned command; they need not match "
            "across interventions. Differences are reported in each native channel, not a mixed-unit norm. "
            "The scored continuous prefix starts after warmup."
        ),
    )


def score_pair(root, task, condition, *, reference_condition="ki1-nominal"):
    left_runs, right_runs, successes_left, successes_right, evidence, exposure = {}, {}, [], [], [], []
    initialization = []
    for seed in TRAINING_SEEDS:
        reference, reference_trace, reference_source = load(root, task, seed, reference_condition)
        variant, variant_trace, variant_source = load(root, task, seed, condition)
        if reference["checkpoint_sha256"] != variant["checkpoint_sha256"]:
            raise ValueError("Intervention changed policy weights")
        initialization.append(
            dict(
                training_seed=seed,
                **initialization_comparison(task, reference, variant, reference_trace, variant_trace),
            )
        )
        if (
            reference["diagnostic_condition"]["original_integral_gains"]
            != variant["diagnostic_condition"]["original_integral_gains"]
        ):
            raise ValueError("Controller reference gains changed")
        if (
            reference["diagnostic_condition"]["hydro_condition"] == variant["diagnostic_condition"]["hydro_condition"]
            and reference["diagnostic_condition"]["base_coefficients"]
            != variant["diagnostic_condition"]["base_coefficients"]
        ):
            raise ValueError("Integral-only comparison changed hydrodynamic coefficients")
        left, right, seconds = paired_metrics(variant_trace, reference_trace)
        for key in left:
            left_runs.setdefault(key, []).append(left[key])
            right_runs.setdefault(key, []).append(right[key])
        successes_left.append(variant["success_per_seed"])
        successes_right.append(reference["success_per_seed"])
        exposure.append(dict(training_seed=seed, joint_seconds=seconds.tolist()))
        evidence.append(dict(training_seed=seed, reference=reference_source, variant=variant_source))
    measurements = {}
    for key in left_runs:
        x, y = np.asarray(left_runs[key]), np.asarray(right_runs[key])
        row = dict(variant_per_reset=[nullable(r) for r in x], reference_per_reset=[nullable(r) for r in y])
        if np.isfinite(x).all() and np.isfinite(y).all():
            row.update(
                variant=summarize(x),
                reference=summarize(y),
                difference=float((x - y).mean()),
                descriptive_paired_crossed_95=crossed_interval(x, y, paired_training=True),
            )
        else:
            row["status"] = "Undefined continuous summary: at least one paired reset has zero observed exposure"
        measurements[key] = row
    x, y = np.asarray(successes_left), np.asarray(successes_right)
    return dict(
        task=task,
        contrast=condition + " minus " + reference_condition,
        evidence=evidence,
        initialization=initialization,
        exposure=exposure,
        success=dict(
            variant_per_reset=x.tolist(),
            reference_per_reset=y.tolist(),
            variant_per_run=x.sum(axis=1).tolist(),
            reference_per_run=y.sum(axis=1).tolist(),
            episodes_per_run=30,
            difference=float(x.mean() - y.mean()),
            descriptive_paired_crossed_95=crossed_interval(x, y, paired_training=True),
        ),
        measurements=measurements,
    )


def success_effect_sensitivity(nominal, published):
    if nominal["task"] != published["task"] or any(
        [item["training_seed"] for item in row["evidence"]] != list(TRAINING_SEEDS) for row in (nominal, published)
    ):
        raise ValueError("Model-sensitivity comparison changed task or training alignment")
    if (
        nominal["contrast"] != "ki0-nominal minus ki1-nominal"
        or published["contrast"] != "ki0-published-both minus ki1-published-both"
    ):
        raise ValueError("Require within-coefficient-model integral contrasts")

    def difference(row):
        success = row["success"]
        x = np.asarray(success["variant_per_reset"])
        y = np.asarray(success["reference_per_reset"])
        if x.shape != (3, 30) or y.shape != x.shape or x.dtype != bool or y.dtype != bool:
            raise ValueError("Require three paired full 30-reset Boolean controller cohorts")
        return x.astype(float) - y.astype(float)

    delta = difference(published) - difference(nominal)
    return dict(
        task=nominal["task"],
        contrast="(Ki0-Ki1 at published-both) minus (Ki0-Ki1 at nominal)",
        per_training_run=delta.mean(axis=1).tolist(),
        difference=float(delta.mean()),
        descriptive_paired_crossed_95=crossed_interval(delta),
        evidence=dict(nominal=nominal["evidence"], published=published["evidence"]),
        scope="Full-horizon success only; continuous-metric pairs can have different censored exposures",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    a = p.parse_args()
    rows, missing, matched, sensitivity = [], [], [], []
    for task in ("OpenHatch", "RotateValve"):
        for condition in CONDITIONS:
            try:
                rows.append(score_pair(a.root, task, condition))
            except FileNotFoundError:
                missing.append(f"{task}/{condition}")
        try:
            alternate = score_pair(a.root, task, "ki0-published-both", reference_condition="ki1-published-both")
            nominal = next(
                (row for row in rows if row["task"] == task and row["contrast"] == "ki0-nominal minus ki1-nominal"),
                None,
            )
            if nominal is None:
                raise FileNotFoundError("Nominal integral contrast is incomplete")
            matched.append(alternate)
            sensitivity.append(success_effect_sensitivity(nominal, alternate))
        except FileNotFoundError:
            missing.append(f"{task}/matched-published-integral-effect")
    if missing and not a.allow_incomplete:
        raise ValueError(f"Incomplete diagnostic contrasts: {missing}")
    result = dict(
        protocol=PROTOCOL,
        complete=not missing,
        missing=missing,
        contrasts=rows,
        matched_integral_contrasts=matched,
        success_effect_sensitivity=sensitivity,
        prefix_seconds=20,
        control_hz=30,
        scorer_sha256=sha256(__file__),
        interpretation=(
            "Frozen-policy interventions; contact association is not causal mediation. "
            "Published hydrodynamic coefficients are sensitivity conditions, not physical calibration."
        ),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=not missing, contrasts=len(rows), missing=len(missing))))


if __name__ == "__main__":
    main()
