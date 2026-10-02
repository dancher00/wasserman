import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from pxr import Sdf, Usd

from wasman.assets.grasp_geometry import grasp_robot_urdf, repair_mixed_unit_mimic


def test_cad_grasp_variant_preserves_original_and_does_not_copy_restricted_meshes():
    source = Path(__file__).parents[1] / "src/wasman/assets/data/robots/bluerov2_alpha/bluerov2_alpha.urdf"
    original = source.read_bytes()
    derived = grasp_robot_urdf(source)
    assert source.read_bytes() == original
    assert derived != source
    tree = ET.parse(derived)
    assert not list(derived.parent.glob("*.stl"))
    assert len(tree.findall("joint")) == len(ET.fromstring(original).findall("joint"))
    for side in ("left", "right"):
        link = tree.find(f"link[@name='alpha_{side}_finger_link']")
        assert link.find("collision/geometry/box") is None
        assert link.find("collision/geometry/mesh").attrib == link.find("visual/geometry/mesh").attrib
    for mesh in tree.findall(".//mesh"):
        assert Path(mesh.attrib["filename"]).is_absolute()


def test_mixed_mimic_units_and_repeated_spawning():
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Robot")
    joint = stage.DefinePrim("/Robot/alpha_left_finger_joint")
    coef = joint.CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float)
    coef.Set(51.0)
    repair_mixed_unit_mimic(root)
    assert math.radians(coef.Get() * 0.0049) == pytest.approx(0.2499, rel=1e-6)
    repair_mixed_unit_mimic(root)
    assert math.radians(coef.Get() * 0.0049) == pytest.approx(0.2499, rel=1e-6)
    coef.Set(12.0)
    with pytest.raises(RuntimeError, match="schema changed"):
        repair_mixed_unit_mimic(root)
