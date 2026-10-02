"""Deterministic custom twin-Oberon mount; original assets remain immutable."""

from pathlib import Path
import copy, json, hashlib, xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "src/wasman/assets/data/robots/rexrov2_oberon7_centered/rexrov2_oberon7_centered.urdf"
out = source.parent.parent / "rexrov2_bimanual"
out.mkdir(exist_ok=True)
src = ET.parse(source).getroot()
robot = ET.Element("robot", name="rexrov2_bimanual_custom")
for child in src:
    if child.tag == "link" and child.get("name") == "base_link":
        robot.append(copy.deepcopy(child))
for side, y in [("left", 0.50), ("right", -0.50)]:
    for item in src:
        if item.tag not in ["link", "joint"] or (item.tag == "link" and item.get("name") == "base_link"):
            continue
        node = copy.deepcopy(item)
        for elem in node.iter():
            for key, value in list(elem.attrib.items()):
                if key in ["name", "link", "joint"] and value.startswith("oberon_"):
                    elem.set(key, side + "_" + value)
        if item.tag == "joint" and item.get("name") == "oberon_mount":
            node.find("origin").set("xyz", f"1.3 {y} -0.665")
        robot.append(node)
# Resolve every mesh against the old asset; emit repository-relative paths.
import os

for mesh in robot.findall(".//mesh"):
    old = mesh.get("filename")
    mesh.set("filename", os.path.relpath((source.parent / old).resolve(), out))
ET.indent(robot)
target = out / "rexrov2_bimanual.urdf"
ET.ElementTree(robot).write(target, encoding="utf-8", xml_declaration=True)
(out / "provenance.json").write_text(
    json.dumps(
        {
            "custom_mount": True,
            "source": str(source.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "mounts_m": [[1.3, 0.5, -0.665], [1.3, -0.5, -0.665]],
            "changes": "Duplicate native Oberon links/actuators; one unchanged Rex base; no buoyancy compensation or motor increase. Not a manufactured platform claim.",
        },
        indent=2,
    )
    + "\n"
)
print(target)
