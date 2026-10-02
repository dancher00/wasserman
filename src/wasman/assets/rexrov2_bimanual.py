"""Experimental floating Rex with two native Oberon7 arms at symmetric mounts."""

from wasman.assets.rexrov2_centered import REXROV2_CENTERED_FLOATING_CFG
from wasman.physics.rexrov2_bimanual import URDF

BIMANUAL_REX_CFG = REXROV2_CENTERED_FLOATING_CFG.copy()
BIMANUAL_REX_CFG.spawn.asset_path = str(URDF)
BIMANUAL_REX_CFG.joint_ordering = tuple(
    side + "_" + n for side in ["left", "right"] for n in REXROV2_CENTERED_FLOATING_CFG.joint_ordering
)
BIMANUAL_REX_CFG.actuators = {}
for side in ["left", "right"]:
    for name, original in REXROV2_CENTERED_FLOATING_CFG.actuators.items():
        cfg = original.copy()
        cfg.joint_names_expr = [side + "_" + n for n in original.joint_names_expr]
        for key in ["actuator_effort_limit", "joint_effort_limit", "actuator_velocity_limit", "joint_velocity_limit"]:
            value = getattr(cfg, key)
            if isinstance(value, dict):
                setattr(cfg, key, {side + "_" + n: v for n, v in value.items()})
        BIMANUAL_REX_CFG.actuators[side + "_" + name] = cfg
BIMANUAL_REX_CFG.init_state.joint_pos = {
    side + "_" + n: v
    for side in ["left", "right"]
    for n, v in REXROV2_CENTERED_FLOATING_CFG.init_state.joint_pos.items()
}

# Native jaw closure reaches 0 rad. The generic 0.96 soft range leaves a gap
# larger than the original 8.4 mm valve rim; use the native limits explicitly.
BIMANUAL_REX_CFG.soft_joint_pos_limit_factor = 1.0
