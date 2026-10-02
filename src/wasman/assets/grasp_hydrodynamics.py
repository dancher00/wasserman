"""Geometry-scaled rotational coefficients for the articulated grasp variant.

The original button coefficients remain frozen. These are engineering estimates,
not identified hardware data. A small finger must not inherit a 0.02 kg m² fluid
inertia (hundreds of times its dry inertia) from a generic arm-link constant.
"""

from dataclasses import replace


def geometry_scaled_arm_hydrodynamics(links):
    lengths = (0.15, 0.04, 0.15, 0.04, 0.10, 0.03, 0.01, 0.09, 0.09, 0.01)
    result = [links[0]]
    for link, length in zip(links[1:], lengths, strict=True):
        second_moment = length**2 / 12
        # Conservative isotropic rotational bound using the largest transverse
        # coefficient. The mass/drag longitudinal lever arm has units m².
        added = max(link.added_mass[:3]) * second_moment
        linear = max(link.linear_damping[:3]) * second_moment
        quadratic = max(link.quadratic_damping[:3]) * length**3 / 32
        if link.name in ("alpha_tool_link", "alpha_push_rod_link"):
            # Bookkeeping frames/internal rod: not independently wetted bodies.
            result.append(replace(link, added_mass=(0.0,) * 6, linear_damping=(0.0,) * 6, quadratic_damping=(0.0,) * 6))
        else:
            result.append(
                replace(
                    link,
                    added_mass=(*link.added_mass[:3], added, added, added),
                    linear_damping=(*link.linear_damping[:3], linear, linear, linear),
                    quadratic_damping=(*link.quadratic_damping[:3], quadratic, quadratic, quadratic),
                )
            )
    return tuple(result)
