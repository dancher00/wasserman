"""Success replay must reject proximity, stability and censoring mistakes."""

import numpy as np
import pytest

from wasman.controllers.marine_evidence import verify


def fixture():
    data = dict(
        q=np.full((5, 1), 0.71),
        distance=np.full((5, 1), 0.01),
        alignment=np.ones((5, 1)),
        attitude=np.zeros((5, 1)),
        angular_speed=np.zeros((5, 1)),
        forces=np.ones((5, 1, 2, 3)),
        active=np.ones((5, 1), dtype=bool),
        terminal=np.zeros((5, 1), dtype=bool),
        success=np.array([[False], [False], [False], [True], [True]]),
    )
    criteria = dict(
        initial=0,
        direction=1,
        threshold=np.deg2rad(40),
        distance=0.06,
        alignment=0.7,
        attitude=0.25,
        angular_speed=0.35,
        hold_steps=4,
    )
    return data, criteria


def test_consecutive_hold_and_first_completion():
    d, c = fixture()
    assert verify(d, c)["first_success_step"] == [4]


@pytest.mark.parametrize(
    "key,value", [("q", 0.1), ("distance", 0.1), ("alignment", 0.5), ("attitude", 0.3), ("angular_speed", 0.5)]
)
def test_false_success_rejected(key, value):
    d, c = fixture()
    d[key][2] = value
    with pytest.raises(ValueError, match="contract mismatch"):
        verify(d, c)


def test_reset_is_not_completion():
    d, c = fixture()
    d["terminal"][3:] = True
    assert verify(d, c)["success_per_seed"] == [False]


def test_nonfinite_physics_rejected():
    d, c = fixture()
    d["q"][0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        verify(d, c)
