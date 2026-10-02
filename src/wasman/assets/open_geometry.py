"""Distributable procedural arm geometry, explicitly separate from historical CAD.

No restricted mesh is read. Joint frames, inertias and non-finger collisions come
from the MIT-derived URDF. Grasp fingers are new two-segment primitives: neither
CAD approximations nor a claim of physical equivalence to the historical asset.
"""

import copy
import hashlib
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

PROFILE = "open-procedural-v1"


def selected_profile():
    profile = os.environ.get("WASMAN_ASSET_PROFILE", "historical-cad-v1")
    if profile not in (PROFILE, "historical-cad-v1"):
        raise ValueError(f"Unknown WASMAN_ASSET_PROFILE: {profile}")
    return profile


def public_robot_urdf(source: Path, *, grasp=False) -> Path:
    """Materialize an open URDF and bind the variant into its content/cache key."""
    source = Path(source).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("name", f"{PROFILE}-{'grasp' if grasp else 'native'}")
    for link in root.findall("link"):
        visuals = link.findall("visual")
        restricted = any("/alpha/" in m.get("filename", "") for v in visuals for m in v.findall(".//mesh"))
        if not restricted:
            continue
        material = next((copy.deepcopy(v.find("material")) for v in visuals if v.find("material") is not None), None)
        for visual in visuals:
            link.remove(visual)
        if grasp and link.get("name") in ("alpha_left_finger_link", "alpha_right_finger_link"):
            for collision in link.findall("collision"):
                link.remove(collision)
            sign = -1 if link.get("name") == "alpha_left_finger_link" else 1
            # Generic tapered jaw centreline, metres; 10 mm pad width and 12 mm thickness.
            points = ((0.0, 0.0), (sign * 0.006, 0.04), (sign * 0.014, 0.09))
            for index, ((y0, z0), (y1, z1)) in enumerate(zip(points[:-1], points[1:], strict=True)):
                collision = ET.SubElement(link, "collision", name=f"open_finger_segment_{index}")
                ET.SubElement(
                    collision,
                    "origin",
                    xyz=f"0 {(y0 + y1) / 2} {(z0 + z1) / 2}",
                    rpy=f"{-math.atan2(y1 - y0, z1 - z0)} 0 0",
                )
                ET.SubElement(
                    ET.SubElement(collision, "geometry"), "box", size=f"0.012 0.010 {math.hypot(y1 - y0, z1 - z0)}"
                )
        for index, collision in enumerate(link.findall("collision")):
            visual = ET.SubElement(link, "visual", name=f"open_visual_{index}")
            for child in collision:
                visual.append(copy.deepcopy(child))
            if material is not None:
                visual.append(copy.deepcopy(material))
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if "/alpha/" in filename:
            raise ValueError("Restricted mesh survived the public geometry conversion")
        mesh.set("filename", str((source.parent / filename).resolve()))
    contents = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    digest = hashlib.sha256(contents).hexdigest()
    project = Path(__file__).resolve().parents[3]
    destination = project / ".asset-cache" / PROFILE / digest / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(contents)
    return destination
