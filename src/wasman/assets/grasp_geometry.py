"""Task-local CAD finger collisions; the released button asset stays unchanged."""

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def grasp_robot_urdf(source: Path) -> Path:
    """Cache a derived URDF referencing the user's locally installed CAD meshes.

    Meshes are not copied or redistributed. The importer convex-decomposes the
    curved fingers instead of filling their grasp aperture with one large box.
    """
    from wasman.assets.open_geometry import PROFILE, public_robot_urdf, selected_profile

    if selected_profile() == PROFILE:
        # The base config may already hold the native public derivative. Build
        # grasp geometry from the tracked source, never from a CAD mesh.
        if PROFILE in source.parts:
            source = Path(__file__).parent / "data/robots/bluerov2_alpha/bluerov2_alpha.urdf"
        return public_robot_urdf(source, grasp=True)
    tree = ET.parse(source)
    for mesh in tree.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.attrib["filename"]).resolve()))
    for name in ("alpha_left_finger_link", "alpha_right_finger_link"):
        link = tree.find(f"link[@name='{name}']")
        for collision in link.findall("collision"):
            link.remove(collision)
        collision = ET.SubElement(link, "collision")
        geometry = ET.SubElement(collision, "geometry")
        mesh = link.find("visual/geometry/mesh")
        ET.SubElement(geometry, "mesh", dict(mesh.attrib))
    contents = ET.tostring(tree.getroot(), encoding="unicode")
    digest = hashlib.sha256(contents.encode()).hexdigest()[:16]
    destination = source.parents[6] / ".asset-cache" / "grasp-robot" / digest / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_text(contents)
    return destination


def repair_mixed_unit_mimic(root):
    """Preserve URDF's 51 rad/m coupling in USD's degrees/distance units.

    Isaac Sim 6.1's imported NewtonMimicAPI currently retains the numeric URDF
    multiplier unchanged. A 4.9 mm stroke then opens only 0.00436 rad, not
    0.2499 rad. Set the documented USD coefficient, never a joint state.
    """
    from pxr import Usd

    followers = [p for p in Usd.PrimRange(root) if p.GetName() == "alpha_left_finger_joint"]
    if len(followers) != 1:
        raise RuntimeError("Expected one Alpha mixed-unit mimic follower")
    follower = followers[0]
    coefficient = follower.GetAttribute("newton:mimicCoef1")
    expected = math.degrees(51.0)
    if not coefficient or not any(math.isclose(coefficient.Get(), x, rel_tol=1e-5) for x in (51.0, expected)):
        raise RuntimeError("Imported mimic schema changed; review the mixed-unit coupling")
    coefficient.Set(expected)


def spawn_grasp_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from pxr import Sdf, Usd, UsdPhysics

    root = sim_utils.spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
    for robot in sim_utils.find_matching_prims(prim_path, stage=root.GetStage()):
        repair_mixed_unit_mimic(robot)
        colliders = [
            p
            for p in Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.MeshCollisionAPI)
            and any(f"alpha_{side}_finger_link/" in str(p.GetPath()) for side in ("left", "right"))
        ]
        paths = [p.GetPath() for p in colliders]
        for collider in colliders:
            parent = collider.GetParent()
            while parent != robot:
                if parent.IsInstance():
                    parent.SetInstanceable(False)
                parent = parent.GetParent()
        for path in paths:
            collider = robot.GetStage().GetPrimAtPath(path)
            UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Set("convexDecomposition")
            collider.ApplyAPI("PhysxConvexDecompositionCollisionAPI")
            collider.CreateAttribute("physxConvexDecompositionCollision:shrinkWrap", Sdf.ValueTypeNames.Bool).Set(True)
            collider.CreateAttribute("physxConvexDecompositionCollision:errorPercentage", Sdf.ValueTypeNames.Float).Set(
                0.5
            )
    return root
