"""Custom centerline variant; native off-center validated asset stays unchanged."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
from isaaclab_physx.sim.schemas import PhysxCollisionCfg
from pxr import PhysxSchema, Usd, UsdPhysics

from wasman.assets.rexrov2_oberon7 import REXROV2_OBERON7_PREVIEW_CFG

ASSET = Path(__file__).parent / "data/robots/rexrov2_oberon7_centered"


@clone
def spawn_centered_rex(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Author actual narrow collision skins after the converter's mesh instancing.

    Current URDF conversion instances collision meshes even with the legacy
    make_instanceable flag disabled; generic fragment overrides skip them.
    This explicit local deinstancing affects only this custom variant.
    """
    importer_cfg = cfg.copy()
    importer_cfg.collision_props = None
    prim = sim_utils.spawn_from_urdf(prim_path, importer_cfg, translation, orientation, **kwargs)
    while instances := [item for item in Usd.PrimRange(prim) if item.IsInstance()]:
        for item in instances:
            item.SetInstanceable(False)
    for item in Usd.PrimRange(prim):
        if item.HasAPI(UsdPhysics.CollisionAPI):
            api = PhysxSchema.PhysxCollisionAPI.Apply(item)
            api.CreateContactOffsetAttr(0.001)
            api.CreateRestOffsetAttr(0.0)
    return prim


REXROV2_CENTERED_REACH_CFG = REXROV2_OBERON7_PREVIEW_CFG.copy()
REXROV2_CENTERED_REACH_CFG.spawn.func = spawn_centered_rex
REXROV2_CENTERED_REACH_CFG.spawn.asset_path = str(ASSET / "rexrov2_oberon7_centered.urdf")
REXROV2_CENTERED_REACH_CFG.spawn.self_collision = True
REXROV2_CENTERED_REACH_CFG.spawn.make_instanceable = False
REXROV2_CENTERED_REACH_CFG.spawn.activate_contact_sensors = True
REXROV2_CENTERED_REACH_CFG.spawn.articulation_props.enabled_self_collisions = True
REXROV2_CENTERED_REACH_CFG.spawn.collision_props = [
    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
    # Two 1-mm offsets fit inside the measured 3.758-mm CAD clearance at the
    # pedestal/shoulder, unlike the generic 20-mm contact skin.
    PhysxCollisionCfg(contact_offset=0.001, rest_offset=0.0),
]
REXROV2_CENTERED_REACH_CFG.init_state.pos = (9.8, 0, 1.5)
REXROV2_CENTERED_REACH_CFG.init_state.joint_pos = {
    "oberon_azimuth": 0,
    "oberon_shoulder": 1.5,
    "oberon_elbow": -1.5,
    "oberon_roll": 1.5707963267948966,
    "oberon_pitch": 1.5,
    "oberon_wrist": 0,
    "oberon_finger_left_joint": 0.25,
    "oberon_finger_right_joint": 0.25,
}
# Reach smoke is fixed-base/zero-G on purpose: it validates articulated reach
# and enabled self-collisions, not free-floating manipulation/hydrodynamics.
REXROV2_CENTERED_FLOATING_CFG = REXROV2_CENTERED_REACH_CFG.copy()
REXROV2_CENTERED_FLOATING_CFG.spawn.fix_base = False
REXROV2_CENTERED_FLOATING_CFG.spawn.fix_root_link = False
for properties in REXROV2_CENTERED_FLOATING_CFG.spawn.rigid_props:
    if hasattr(properties, "disable_gravity"):
        properties.disable_gravity = False
        properties.linear_damping = properties.angular_damping = 0
        properties.enable_gyroscopic_forces = True
