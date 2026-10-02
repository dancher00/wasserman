"""Versioned task success; the legacy physical contract remains a diagnostic."""

import math

import torch

AMB_VALVE_SUCCESS = "ambench-angle-170-v1"
STRICT_VALVE_SUCCESS = "valve-grasp-turn-release-v1"
VALVE_SUCCESS_CONTRACTS = (AMB_VALVE_SUCCESS, STRICT_VALVE_SUCCESS)


def valve_angle_criteria(angle: torch.Tensor) -> dict[str, torch.Tensor]:
    """AM-Bench criteria: signed shaft angle, no upper bound or contact gate."""
    target = math.radians(170)
    return {"valve_engaged": angle >= 0.05 * target, "valve_rotated": angle >= target}


def valve_success_metadata(contract: str) -> dict:
    if contract == AMB_VALVE_SUCCESS:
        return {
            "version": contract,
            "goal_angle_deg": [170, None],
            "engagement_angle_deg": 8.5,
            "requires_bilateral_contact": False,
            "requires_hold": False,
            "requires_release": False,
            "success_ends_episode": False,
        }
    if contract == STRICT_VALVE_SUCCESS:
        return {
            "version": contract,
            "goal_angle_deg": [170, 175],
            "minimum_bilateral_turn_deg": 165,
            "hold_s": 1.0,
            "release_s": 0.5,
            "withdrawal_clearance_m": 0.08,
            "success_ends_episode": False,
        }
    raise ValueError(f"Unknown valve success contract: {contract}")


def select_valve_success(angle: torch.Tensor, strict_success: torch.Tensor, contract: str) -> torch.Tensor:
    if contract == AMB_VALVE_SUCCESS:
        return valve_angle_criteria(angle)["valve_rotated"]
    if contract == STRICT_VALVE_SUCCESS:
        return strict_success
    raise ValueError(f"Unknown valve success contract: {contract}")
