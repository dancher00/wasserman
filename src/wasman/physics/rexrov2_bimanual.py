"""Explicit twin-arm parameter variant, retaining native per-link properties."""

from dataclasses import replace
from pathlib import Path
import numpy as np
import pinocchio as pin
from wasman.physics.rexrov2 import load_parameters

URDF = Path(__file__).parents[1] / "assets/data/robots/rexrov2_bimanual/rexrov2_bimanual.urdf"


def load_bimanual_parameters(joint_positions=None):
    p = load_parameters()
    names = ["base_link"] + [side + "_" + n for side in ["left", "right"] for n in p.names[1:]]

    def twice(x):
        return np.concatenate([x[:1], x[1:], x[1:]])

    model = pin.buildModelFromUrdf(str(URDF), pin.JointModelFreeFlyer())
    q = pin.neutral(model)
    if joint_positions:
        for n, v in joint_positions.items():
            q[model.joints[model.getJointId(n)].idx_q] = v
    return replace(
        p,
        names=names,
        mass=twice(p.mass),
        com=twice(p.com),
        volume=twice(p.volume),
        cob=twice(p.cob),
        quadratic_arm=twice(p.quadratic_arm),
        rigid_composite=pin.crba(model, model.createData(), q)[:6, :6].copy(),
    )
