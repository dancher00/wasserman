import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from score_revision_interfaces import checked_outcomes  # noqa: E402

pytestmark = pytest.mark.unit


def fixture(success=False):
    shape = (4, 30)
    trace = {key: np.zeros(shape) for key in ("q", "distance", "attitude", "angular_speed")}
    trace.update(
        alignment=np.ones(shape),
        forces=np.zeros((*shape, 2, 3)),
        active=np.ones(shape, dtype=bool),
        terminal=np.zeros(shape, dtype=bool),
        success=np.zeros(shape, dtype=bool),
    )
    if success:
        trace["q"][:] = 1
        trace["success"][-1] = True
    criteria = dict(
        initial=0,
        direction=1,
        threshold=0.5,
        distance=0.1,
        alignment=0.7,
        attitude=0.25,
        angular_speed=0.35,
        hold_steps=4,
    )
    report = dict(success_per_seed=[success] * 30, first_success_step=[4 if success else -1] * 30, steps=4)
    return report, trace, criteria


def test_interface_success_requires_physical_evidence():
    report, trace, criteria = fixture(True)
    assert checked_outcomes(report, trace, criteria, 100).all()
    trace["q"][:] = 0
    with pytest.raises(ValueError, match="Success contract mismatch"):
        checked_outcomes(report, trace, criteria, 100)


def test_interface_failure_requires_completed_horizon():
    report, trace, criteria = fixture(False)
    with pytest.raises(ValueError, match="Incomplete evaluation horizon"):
        checked_outcomes(report, trace, criteria, 100)
    assert not checked_outcomes(report, trace, criteria, 4).any()
