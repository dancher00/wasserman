import math

import pytest
import torch

from wasman.controllers.valve_contract import ValveContract

pytestmark = pytest.mark.unit


def step(contract, degrees, forces=(0.5, 0.5), clearance=0.01, speed=0.0, stable=True):
    return contract.update(
        torch.tensor([math.radians(degrees)]),
        torch.tensor([speed]),
        torch.tensor([forces]),
        torch.tensor([clearance]),
        torch.zeros(1),
        torch.tensor([stable]),
    )


def test_valve_requires_turn_hold_and_release_not_just_angle():
    c = ValveContract(1, "cpu", 1 / 30)
    step(c, 0)
    for angle in range(1, 174):
        step(c, angle)
    for _ in range(30):
        assert not step(c, 173).item()
    assert c.held.item()
    for _ in range(15):
        step(c, 173, forces=(0.0, 0.0), clearance=0.1)
    assert c.success.item()
    c.reset([0])
    assert not c.success.item() and not c.held.item()
    assert c.grasp_turn.item() == 0


def test_one_sided_push_overshoot_and_fast_rotation_cannot_pass():
    for forces, angle, speed in [((0.5, 0.0), 173, 0), ((0.5, 0.5), 185, 0), ((0.5, 0.5), 173, 0.3)]:
        c = ValveContract(1, "cpu", 1 / 30)
        step(c, 0, forces=forces)
        for _ in range(60):
            step(c, angle, forces=forces, speed=speed)
        assert not c.held.item()


def test_motion_before_grasp_and_repeated_wiggles_do_not_count_as_a_turn():
    c = ValveContract(1, "cpu", 1 / 30)
    step(c, 160, forces=(0, 0))
    step(c, 160)
    for _ in range(60):
        step(c, 173)
        step(c, 160)
    for _ in range(60):
        step(c, 173)
    assert c.grasp_turn.item() < math.radians(14)
    assert not c.held.item()


def test_releasing_between_reverse_strokes_cannot_accumulate_fake_grasp_progress():
    c = ValveContract(1, "cpu", 1 / 30)
    step(c, 0)
    step(c, 90)
    step(c, 0, forces=(0, 0))
    step(c, 0)
    for angle in range(1, 174):
        step(c, angle)
    for _ in range(60):
        step(c, 173)
    assert c.grasp_turn.item() > math.pi
    assert not c.held.item()


def test_two_same_direction_contacts_are_not_a_pinch_grasp():
    c = ValveContract(1, "cpu", 1 / 30)
    for degrees in range(174):
        c.update(
            torch.tensor([math.radians(degrees)]),
            torch.zeros(1),
            torch.ones(1, 2),
            torch.zeros(1),
            torch.zeros(1),
            torch.ones(1, dtype=torch.bool),
            opposed=torch.zeros(1, dtype=torch.bool),
        )
    assert c.grasp_turn.item() == 0
    assert not c.held.item()
