"""Recompute declared task outcomes from saved physical evidence, without simulation."""

import hashlib
import json
import math

import numpy as np
import torch

from wasman.controllers.hatch_contract import HatchContract
from wasman.controllers.marine_evidence import verify as marine_verify
from wasman.controllers.shell_collection_contract import ShellCollectionContract
from wasman.learning.revision_protocol import CONTROL_STEPS, sha256


def check_embedded_report(embedded, report):
    # torch.save preserves coefficient tuples; JSON represents them as lists.
    # Compare their JSON meaning without tolerating numeric/source changes.
    if json.loads(json.dumps(embedded, allow_nan=False)) != report:
        raise ValueError("Trace/report mismatch")


def initialization_fingerprint(measured, report):
    values = np.asarray(measured, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(report["seeds"]) or not np.isfinite(values).all():
        raise ValueError("Invalid initial measured state")
    payload = dict(seeds=report["seeds"], measured=values.tolist(), initial_reset=report.get("initial_reset"))
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return dict(
        initialization_sha256=hashlib.sha256(encoded).hexdigest(),
        initialization_scope=(
            "Pre-command measured state, ordered seeds and reset randomization fields when recorded; "
            "not the complete simulator state"
        ),
    )


def require_matching_initialization(records):
    if not records or len({row["initialization_sha256"] for row in records}) != 1:
        raise ValueError("Paired reset initialization mismatch")


def checked_outcomes(report, trace, criteria, horizon, *, verifier=marine_verify, expected_episodes=30):
    outcomes = np.asarray(report["success_per_seed"])
    if outcomes.shape != (expected_episodes,) or outcomes.dtype != bool:
        raise ValueError(f"Require {expected_episodes} explicit Boolean outcomes")
    replay = verifier(trace, criteria)
    if replay["success_per_seed"] != outcomes.tolist() or replay["first_success_step"] != report["first_success_step"]:
        raise ValueError("Physical trace/report success mismatch")
    if not 0 < report["steps"] <= horizon or len(trace["q"]) != report["steps"]:
        raise ValueError("Trace horizon mismatch")
    if report["steps"] < horizon and np.any(trace["active"][-1] & ~trace["terminal"][-1] & ~outcomes):
        raise ValueError("Incomplete evaluation horizon")
    return outcomes


def checked_tensor_outcomes(task, report, trace, *, expected_episodes=30):
    n, horizon = len(report["seeds"]), CONTROL_STEPS[task]
    if n != expected_episodes:
        raise ValueError(f"Require {expected_episodes} declared episodes")
    if not 0 < len(trace) <= horizon or report["evaluated_steps"] != horizon or report["step_dt"] != 1 / 30:
        raise ValueError("Trace horizon mismatch")
    if task == "OpenHatch":
        contract = HatchContract(n, "cpu", report["step_dt"])
    elif task == "CollectShell":
        if report["target_shells"] != 1:
            raise ValueError("Core shell count changed")
        contract = ShellCollectionContract(n, 1, "cpu", report["step_dt"])
    elif task != "RotateValve" or report["success_contract"]["version"] != "ambench-angle-170-v1":
        raise ValueError("Unsupported success contract")
    active = torch.ones(n, dtype=torch.bool)
    first = torch.full((n,), -1, dtype=torch.long)
    resets = first.clone()
    for step, row in enumerate(trace):
        values = list(row.values())
        while values:
            value = values.pop()
            if isinstance(value, dict):
                values.extend(value.values())
            elif (
                isinstance(value, torch.Tensor)
                and not torch.isfinite(value).all()
                or isinstance(value, float)
                and not math.isfinite(value)
            ):
                raise ValueError("Nonfinite physical evidence")
        if not torch.equal(row["active"], active):
            raise ValueError("Trace active mask mismatch")
        if task == "RotateValve":
            if not torch.isfinite(row["angle"]).all():
                raise ValueError("Nonfinite valve evidence")
            physical = row["angle"] >= math.radians(170)
        elif task == "OpenHatch":
            physical = contract.update(
                row["angle"], row["speed"], row["forces"], row["near_handle"], row["tool_speed"], row["stable"]
            )
        else:
            physical = contract.update(**row["inputs"])["success"]
        success = physical & active & ~row["reset"]
        if not torch.equal(success, row["success"]):
            raise ValueError("Physical trace/report success mismatch")
        first[success] = step + 1
        resets[active & row["reset"]] = step + 1
        active &= ~(success | row["reset"])
    if active.any() and len(trace) != horizon:
        raise ValueError("Incomplete evaluation horizon")
    if first.tolist() != report["first_success_step"] or resets.tolist() != report["first_reset_step"]:
        raise ValueError("Episode censoring mismatch")
    outcomes = np.asarray(report["success_per_seed"])
    if outcomes.shape != (expected_episodes,) or outcomes.dtype != bool or (first > 0).tolist() != outcomes.tolist():
        raise ValueError("Physical outcome mismatch")
    if report["successes"] != int(outcomes.sum()):
        raise ValueError("Aggregate success mismatch")
    return outcomes


def audit_saved_trace(folder, task, report, *, expected_episodes=30):
    if task in ("PressButton", "PushSlider", "PullLever"):
        verifier = marine_verify
        if task == "PressButton":
            from button_visual_support import verify

            verifier = verify
        contract_path = folder / "contract.json"
        trace_path = folder / "trace.npz"
        with np.load(trace_path, allow_pickle=False) as trace:
            outcomes = checked_outcomes(
                report,
                trace,
                json.loads(contract_path.read_text()),
                CONTROL_STEPS[task],
                verifier=verifier,
                expected_episodes=expected_episodes,
            )
            initialization = initialization_fingerprint(trace["measured"][0], report)
        provenance = dict(contract_sha256=sha256(contract_path))
    else:
        trace_path = folder / "rollout.pt"
        payload = torch.load(trace_path, map_location="cpu", weights_only=True)
        check_embedded_report(payload["summary"], report)
        outcomes = checked_tensor_outcomes(task, report, payload["trace"], expected_episodes=expected_episodes)
        initialization = initialization_fingerprint(payload["trace"][0]["measured"].numpy(), report)
        provenance = {}
    return outcomes, dict(
        **provenance, **initialization, trace_sha256=sha256(trace_path), physical_contract_replayed=True
    )
