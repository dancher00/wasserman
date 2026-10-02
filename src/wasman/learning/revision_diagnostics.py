"""Predeclared controller and externally anchored base-model interventions."""

from dataclasses import asdict, replace

from wasman.learning.revision_protocol import TASKS

CONDITIONS = ("nominal", "published-damping", "published-added-mass", "published-both")
REFERENCE = "https://doi.org/10.3390/jmse10121898"
ADDED = (6.36, 7.12, 18.68, 0.189, 0.135, 0.222)
LINEAR = (13.7, 0.0, 33.0, 0.0, 0.8, 0.0)
QUADRATIC = (141.0, 217.0, 190.0, 1.19, 0.47, 1.5)


def apply_condition(cfg, *, integral_multiplier=1.0, hydro_condition="nominal", links=None):
    if integral_multiplier not in (0.0, 0.5, 1.0) or hydro_condition not in CONDITIONS:
        raise ValueError("Unregistered diagnostic condition")
    if links is None:
        from wasman.assets.bluerov2_alpha import BLUEROV2_ALPHA_HYDRODYNAMICS

        links = getattr(cfg, "link_hydrodynamics", BLUEROV2_ALPHA_HYDRODYNAMICS)
    if links[0].name != "base_link":
        raise ValueError("Expected the base as the first hydrodynamic link")
    changes = {}
    if hydro_condition in ("published-damping", "published-both"):
        changes.update(linear_damping=LINEAR, quadratic_damping=QUADRATIC)
    if hydro_condition in ("published-added-mass", "published-both"):
        changes["added_mass"] = ADDED
    cfg.link_hydrodynamics = (replace(links[0], **changes), *links[1:])
    original_gains = {}
    for key in ("station_position_ki", "station_rotation_ki"):
        original_gains[key] = list(getattr(cfg, key))
        setattr(cfg, key, tuple(integral_multiplier * x for x in getattr(cfg, key)))
    return dict(
        integral_multiplier=integral_multiplier,
        hydro_condition=hydro_condition,
        original_integral_gains=original_gains,
        base_coefficients=asdict(cfg.link_hydrodynamics[0]),
        coefficient_reference=REFERENCE if changes else None,
        interpretation="Model sensitivity; not calibration of this assembly",
    )


def validate_condition_cohort(config, seeds, purpose, integral_multiplier, hydro_condition):
    changed = integral_multiplier != 1.0 or hydro_condition != "nominal"
    if purpose in ("test", "validation") and changed:
        raise ValueError("Primary/validation cohorts use nominal physics and gains")
    if purpose == "research":
        index = TASKS.index(config["task"])
        expected = set(range(70000 + 1000 * index, 70030 + 1000 * index))
        if list(seeds) != sorted(expected) or config.get("pilot") or config["model"] != "DP":
            raise ValueError("Diagnostics require the complete paired cohort and a primary DP checkpoint")


def trace_state(raw):
    """Post-control samples, censored by each evaluator's active/reset mask."""
    return {
        "station_position_error": raw._station_position_error_w.detach().cpu().clone(),
        "station_orientation_error": raw._station_orientation_error_b.detach().cpu().clone(),
        "base_attitude": raw._base_attitude_error.detach().cpu().clone(),
        "motor_force": raw._thrusters.force.detach().cpu().clone(),
        "motor_saturation_scale": raw._thrusters.saturation_scale.detach().cpu().clone(),
        "contact_force_vectors": raw.finger_force_vectors().detach().cpu().clone(),
    }


def reset_state(raw):
    return dict(
        base_position=(raw.robot.data.body_link_pos_w.torch[:, raw._base_body_id] - raw.scene.env_origins).tolist(),
        base_quaternion=raw.robot.data.body_link_quat_w.torch[:, raw._base_body_id].tolist(),
        joint_position=raw.robot.data.joint_pos.torch.tolist(),
        volume_scale=raw._hydrodynamics.volume_scale.tolist(),
        damping_scale=raw._hydrodynamics.damping_scale.tolist(),
        added_mass_scale=raw._hydrodynamics.added_mass_scale.tolist(),
    )
