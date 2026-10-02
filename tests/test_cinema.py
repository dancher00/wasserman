import math

import pytest

from wasman.controllers.cinema import approach_camera, hatch_camera, valve_camera


@pytest.mark.unit
def test_approach_camera_clamps_and_settles():
    assert approach_camera(-1) == approach_camera(0)
    assert approach_camera(16) == approach_camera(28)
    assert approach_camera(0)[1] == (0.02, -0.04, 0.76)
    assert approach_camera(16)[1] == pytest.approx((0.48, -0.04, 0.76))
    for t in range(29):
        eye, target = approach_camera(t)
        assert all(math.isfinite(v) for v in (*eye, *target))
        assert eye[0] < target[0] and eye[1] < target[1] and eye[2] > target[2]


@pytest.mark.unit
def test_approach_camera_is_continuous_at_video_rate():
    previous = approach_camera(0)
    for step in range(1, 840):
        current = approach_camera(step / 30)
        for before, after in zip(previous, current, strict=True):
            assert math.dist(before, after) < 0.01
        previous = current


@pytest.mark.unit
def test_valve_camera_settles_before_the_turn_and_never_cuts():
    assert valve_camera(-1) == valve_camera(0)
    assert valve_camera(18) == valve_camera(75)
    previous = valve_camera(0)
    for step in range(1, 2250):
        current = valve_camera(step / 30)
        for before, after in zip(previous, current, strict=True):
            assert math.dist(before, after) < 0.01
        previous = current


@pytest.mark.unit
def test_hatch_camera_is_a_continuous_orbit_and_settles():
    assert hatch_camera(-1) == hatch_camera(0)
    assert hatch_camera(26) == hatch_camera(60)
    previous = hatch_camera(0)
    for step in range(1, 1800):
        current = hatch_camera(step / 30)
        for before, after in zip(previous, current, strict=True):
            assert math.dist(before, after) < 0.015
        previous = current
