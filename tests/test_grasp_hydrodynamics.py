from wasman.assets.bluerov2_alpha import BLUEROV2_ALPHA_HYDRODYNAMICS
from wasman.assets.grasp_hydrodynamics import geometry_scaled_arm_hydrodynamics


def test_grasp_fluid_inertia_scales_with_link_geometry_without_changing_legacy():
    old = BLUEROV2_ALPHA_HYDRODYNAMICS
    new = geometry_scaled_arm_hydrodynamics(old)
    assert new[0] == old[0]
    assert [x.name for x in new] == [x.name for x in old]
    assert old[-3].added_mass[-1] == 0.02
    assert 0 < new[-3].added_mass[-1] < 2e-5
    assert new[-3].added_mass[:3] == old[-3].added_mass[:3]
    assert new[-1].added_mass == (0.0,) * 6
    assert new[-1].linear_damping == (0.0,) * 6
