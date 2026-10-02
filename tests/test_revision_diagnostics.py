from dataclasses import asdict
from types import SimpleNamespace

import pytest

from wasman.learning.revision_diagnostics import ADDED, LINEAR, apply_condition, validate_condition_cohort
from wasman.physics.hydrodynamics import LinkHydrodynamics

pytestmark = pytest.mark.unit


def test_interventions_leave_arm_and_other_controller_terms_identical():
    base = LinkHydrodynamics("base_link", 0.013, (0, 0, 0.035), (1,) * 6, (2,) * 6, (3,) * 6)
    arm = LinkHydrodynamics("arm", 0.001, (0, 0, 0), (4,) * 6, (5,) * 6, (6,) * 6)
    cfg = SimpleNamespace(
        station_position_ki=(40, 40, 45),
        station_rotation_ki=(2, 2, 2),
        station_position_kp=(300, 300, 350),
        station_rotation_kd=(7, 8, 10),
    )
    apply_condition(cfg, integral_multiplier=0, hydro_condition="published-both", links=(base, arm))
    assert cfg.station_position_ki == (0, 0, 0)
    assert cfg.station_rotation_ki == (0, 0, 0)
    assert cfg.station_position_kp == (300, 300, 350)
    assert cfg.station_rotation_kd == (7, 8, 10)
    assert cfg.link_hydrodynamics[1] is arm
    assert cfg.link_hydrodynamics[0].volume == base.volume
    assert cfg.link_hydrodynamics[0].center_of_buoyancy == base.center_of_buoyancy
    assert cfg.link_hydrodynamics[0].added_mass == ADDED
    assert cfg.link_hydrodynamics[0].linear_damping == LINEAR
    assert asdict(base)["added_mass"] == (1,) * 6


def test_non_nominal_conditions_cannot_enter_primary_test():
    config = dict(task="OpenHatch", model="DP", pilot=False)
    with pytest.raises(ValueError, match="nominal"):
        validate_condition_cohort(config, list(range(62000, 62030)), "test", 0, "nominal")
    validate_condition_cohort(config, list(range(72000, 72030)), "research", 0, "published-both")
    with pytest.raises(ValueError, match="complete paired"):
        validate_condition_cohort(config, [72000], "research", 0, "nominal")
