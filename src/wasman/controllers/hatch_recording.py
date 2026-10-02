"""Replay the hatch acceptance contract from raw measurements, not expert phases."""

import math

import torch

from wasman.controllers.hatch_contract import HatchContract


def replay_hatch_trace(trace, dt=1 / 30):
    if not trace:
        raise ValueError("Empty hatch trace")
    num_envs = len(trace[0]["angle_rad"])
    contract = HatchContract(num_envs, "cpu", dt)
    first = [-1] * num_envs
    for i, row in enumerate(trace):
        if not math.isclose(row["t"], (i + 1) * dt, abs_tol=1e-5):
            raise ValueError("Discontinuous hatch trace")
        fields = {}
        for name, shape in (
            ("angle_rad", (num_envs,)),
            ("speed_rad_s", (num_envs,)),
            ("force_vectors", (num_envs, 2, 3)),
            ("distance", (num_envs,)),
            ("tool_speed", (num_envs,)),
            ("attitude", (num_envs,)),
            ("base_angular_speed", (num_envs,)),
            ("action", (num_envs, 11)),
        ):
            value = torch.tensor(row[name], dtype=torch.float32)
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid hatch measurement: {name}")
            fields[name] = value
        navigating = ~torch.tensor(row["navigation_ready"], dtype=torch.bool)
        if navigating.any() and fields["action"][navigating, 6:10].abs().max().item() > 1e-6:
            raise ValueError("Arm deployed before navigation settled")
        success = contract.update(
            fields["angle_rad"],
            fields["speed_rad_s"],
            fields["force_vectors"],
            fields["distance"] < 0.06,
            fields["tool_speed"],
            (fields["attitude"] < 0.25) & (fields["base_angular_speed"] < 0.35),
        )
        if success.tolist() != row["success"]:
            raise ValueError("Reported hatch success disagrees with measured-contract replay")
        for env_id, value in enumerate(success.tolist()):
            if value and first[env_id] < 0:
                first[env_id] = i + 1
    return {"first_success_step": first, "successes": sum(x >= 0 for x in first)}
