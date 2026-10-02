"""Independent replay must preserve all original smooth-button success gates."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "button_support", Path(__file__).parents[1] / "scripts/button_visual_support.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    n = 35
    data = {
        k: np.zeros((n, 1))
        for k in ["q", "distance", "alignment", "attitude", "angular_speed", "tool_speed", "mechanism_speed"]
    }
    data["q"][:] = 0.005
    data["alignment"][:] = 1.0
    data.update(active=np.ones((n, 1), bool), terminal=np.zeros((n, 1), bool), success=np.arange(n)[:, None] >= 29)
    c = dict(
        threshold=0.004,
        max_travel=0.007,
        distance=0.13,
        alignment=0.7,
        attitude=0.25,
        angular_speed=0.35,
        tool_speed=0.035,
        mechanism_speed=float("inf"),
        hold_steps=30,
    )
    return data, c


def test_original_sustained_completion():
    d, c = sample()
    assert module.verify(d, c)["first_success_step"] == [30]


@pytest.mark.parametrize(
    "field,value", [("q", 0.008), ("tool_speed", 0.04), ("alignment", 0.6), ("attitude", 0.26), ("distance", 0.14)]
)
def test_each_physical_gate_blocks_success(field, value):
    d, c = sample()
    d[field][:] = value
    d["success"][:] = False
    assert module.verify(d, c)["success_per_seed"] == [False]


def test_hold_is_consecutive_and_flags_are_not_trusted():
    d, c = sample()
    d["q"][15] = 0
    with pytest.raises(ValueError, match="Contract mismatch"):
        module.verify(d, c)
    d["success"][:] = False
    assert module.verify(d, c)["success_per_seed"] == [False]
