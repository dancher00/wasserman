import math

import torch

from wasman.controllers.hatch_contract import HatchContract


def update(contract, deg, *, grasp=True, opposed=True, near=True, speed=0.0, stable=True):
    forces = torch.tensor([[[1.0, 0, 0], [-1.0 if opposed else 1.0, 0, 0]]]) * grasp
    return bool(
        contract.update(
            torch.tensor([math.radians(deg)]),
            torch.tensor([speed]),
            forces,
            torch.tensor([near]),
            torch.tensor([0.0]),
            torch.tensor([stable]),
        ).item()
    )


def test_true_grasp_open_and_continuous_hold():
    c = HatchContract(1, "cpu", 1 / 30)
    for deg in range(83):
        assert not update(c, deg, speed=0.2)
    for _ in range(29):
        assert not update(c, 82)
    assert update(c, 82)
    c.reset([0])
    assert not c.success.item() and c.grasp_motion.item() == 0


def test_external_opening_then_grasp_is_not_success():
    c = HatchContract(1, "cpu", 1 / 30)
    for deg in range(83):
        assert not update(c, deg, grasp=False)
    for _ in range(60):
        assert not update(c, 82)


def test_same_side_contact_and_distant_contact_are_not_grasps():
    for kwargs in ({"opposed": False}, {"near": False}):
        c = HatchContract(1, "cpu", 1 / 30)
        for deg in range(83):
            assert not update(c, deg, **kwargs)
        for _ in range(60):
            assert not update(c, 82, **kwargs)


def test_wiggling_does_not_accumulate_opening():
    c = HatchContract(1, "cpu", 1 / 30)
    for deg in [0, 20, 0, 20, 0, 20, 0, 20, 0]:
        assert not update(c, deg)
    assert abs(c.grasp_motion.item()) < 1e-5


def test_unstable_or_lost_grasp_resets_hold():
    c = HatchContract(1, "cpu", 1 / 30)
    for deg in range(83):
        update(c, deg, speed=0.2)
    for _ in range(29):
        assert not update(c, 82)
    assert not update(c, 82, stable=False)
    assert c.hold_steps.item() == 0
    for _ in range(29):
        assert not update(c, 82)
    assert not update(c, 82, grasp=False)
    assert c.hold_steps.item() == 0
