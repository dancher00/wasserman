"""Compare open native fingers against ring/spokes at alternative acquisition rolls."""

import json, numpy as np, pinocchio as pin, coal
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation
from pathlib import Path
from wasman.controllers.bimanual_kinematics import BimanualKinematics

w = BimanualKinematics()
q = np.zeros(w.model.nq)
for s in w.indices:
    q[w.indices[s]] = [0, 0.794, 0.253, 0, 0.524, 0, 0.5, 0.5]
stage = Usd.Stage.Open("src/wasman/assets/data/objects/industrial_valve/valve_bimanual_4x.usda")
cache = UsdGeom.XformCache()
wheel_center = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/Valve/handle_link")))[3, :3]
shapes = []
for prim in stage.Traverse():
    if prim.GetPath().HasPrefix("/Valve/handle_link") and prim.HasAPI(UsdPhysics.CollisionAPI):
        if prim.GetTypeName() != "Capsule":
            continue
        shape = coal.Capsule(prim.GetAttribute("radius").Get(), prim.GetAttribute("height").Get())
        mat = np.array(cache.GetLocalToWorldTransform(prim))
        pos = mat[3, :3] - wheel_center + np.array([3, 0, 1.1])
        R = mat[:3, :3].T
        shapes.append((prim.GetName(), shape, coal.Transform3s(R, pos)))
rows = []
for phi in [0, -np.pi / 8]:
    R = Rotation.from_rotvec([phi, 0, 0]).as_matrix()
    goal = np.array([3, 0, 1.1]) + R @ np.array([0, -0.236, 0])
    q, error = w.solve_arm("right", goal - np.array([0.5, 0, 1.5]), R, q)
    pin.updateGeometryPlacements(w.model, w.data, w.geometry, w.geometry_data, q)
    contacts = []
    for i, g in enumerate(w.geometry.geometryObjects):
        if not w.links[i].startswith("right_oberon_finger"):
            continue
        T = w.geometry_data.oMg[i]
        x = coal.Transform3s(T.rotation, T.translation + np.array([0.5, 0, 1.5]))
        for name, shape, placement in shapes:
            result = coal.CollisionResult()
            coal.collide(g.geometry, x, shape, placement, coal.CollisionRequest(), result)
            if result.isCollision():
                contacts.append([w.links[i], name])
    rows.append(
        {
            "roll_deg": float(np.rad2deg(phi)),
            "ik_error": error,
            "open_finger_contacts": sorted(set(tuple(x) for x in contacts)),
        }
    )
out = Path("artifacts/bimanual_valve_v1_20260927/grasp-clearance.json")
out.write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
