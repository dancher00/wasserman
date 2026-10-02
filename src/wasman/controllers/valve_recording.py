"""Replay measured contact telemetry before publishing a valve demonstration."""

import math

import torch

from .valve_contract import ValveContract
from .valve_success import STRICT_VALVE_SUCCESS, select_valve_success, valve_success_metadata


def verified_recording(record):
    """Return media metadata only for a complete, reset-free single trial.

    Chapter times come from measured contacts and task state, not an animation
    timeline or the expert's commanded phase. This is a demonstration gate,
    never an estimate of policy success rate.
    """
    if record["num_envs"] != 1 or record["failed_resets"] != 0:
        raise ValueError("A recording must contain one uninterrupted trial")
    frames = record["trace"]
    dt = record["protocol"]["policy_dt_s"]
    if len(frames) != record["steps"] or not frames:
        raise ValueError("Incomplete telemetry")
    contract = ValveContract(1, "cpu", dt)
    scoring = record.get("success_contract", {}).get("version", record["protocol"].get("version", STRICT_VALVE_SUCCESS))
    scoring_metadata = valve_success_metadata(scoring)
    observed_success = False
    chapters = [{"time": 0.0, "label": "Swim in" if frames[0]["navigation"] else "Approach"}]
    seen = set()

    def mark(name, frame):
        if name not in seen:
            chapters.append({"time": round(max(0.0, frame["t"] - dt), 3), "label": name})
            seen.add(name)

    for i, frame in enumerate(frames):
        if not math.isclose(frame["t"], (i + 1) * dt, abs_tol=1e-5):
            raise ValueError("Telemetry has a gap or repeated time")
        stable = frame["base_attitude_rad"][0] < 0.25
        stable &= frame["base_angular_speed_rad_s"][0] < 0.35
        stable &= frame["tool_axis_alignment"][0] > 0.70
        contract.update(
            torch.tensor(frame["angle_rad"]),
            torch.tensor(frame["wheel_speed_rad_s"]),
            torch.tensor(frame["finger_forces_n"]),
            torch.tensor(frame["wheel_clearance_m"]),
            torch.tensor(frame["tool_speed_m_s"]),
            torch.tensor([stable]),
            torch.tensor(frame["opposing_contacts"]),
        )
        measured_success = select_valve_success(torch.tensor(frame["angle_rad"]), contract.success, scoring)
        if bool(measured_success[0]) != frame["success"][0]:
            raise ValueError("Reported success differs from measured-contract replay")
        if "strict_valve_success" in frame and bool(contract.success[0]) != frame["strict_valve_success"][0]:
            raise ValueError("Reported physical diagnostic differs from measured-contract replay")
        observed_success |= bool(measured_success[0])
        if contract.grasp_steps[0] >= 15:
            mark("Grasp", frame)
        if contract.grasp_turn[0] >= math.radians(5):
            mark("Turn", frame)
        if contract.held[0]:
            mark("Hold", frame)
            if not contract.contact.any() and frame["wheel_clearance_m"][0] >= 0.08:
                mark("Release", frame)
        if measured_success[0]:
            mark("Complete", frame)
    if not observed_success or record["successes"] != 1:
        raise ValueError("The selected valve success criterion was not met")
    final = frames[-1]
    if scoring == STRICT_VALVE_SUCCESS and not (
        math.radians(170) <= final["angle_rad"][0] <= math.radians(175)
        and max(final["finger_forces_n"][0]) <= 0.05
        and final["wheel_clearance_m"][0] >= 0.08
        and abs(final["wheel_speed_rad_s"][0]) <= 0.05
        and stable
    ):
        raise ValueError("The recording must end with the valve settled and gripper clear")
    return {
        "task": record["task"],
        "controller": record["controller"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "seed": record["seed"],
        "episodes": 1,
        "successes": 1,
        "evaluation_role": "Single demonstration, not a benchmark success rate",
        "telemetry_contract_replayed": True,
        "success_contract": scoring_metadata,
        "automatic_resets": 0,
        "duration_s": round(len(frames) * dt, 3),
        "first_success_time_s": round(next(f["t"] for f in frames if f["success"][0]), 3),
        "final_angle_deg": math.degrees(frames[-1]["angle_rad"][0]),
        "max_base_attitude_deg": math.degrees(max(f["base_attitude_rad"][0] for f in frames)),
        "chapters": sorted(chapters, key=lambda c: c["time"]),
        "protocol": record["protocol"],
        "source_sha256_at_start": record["source_sha256_at_start"],
    }
