"""The distributable profile must never require restricted files."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from wasman.assets.open_geometry import public_robot_urdf, selected_profile

SOURCE = Path(__file__).resolve().parents[1] / "src/wasman/assets/data/robots/bluerov2_alpha/bluerov2_alpha.urdf"


@pytest.mark.unit
@pytest.mark.parametrize("grasp", [False, True])
def test_open_geometry_resolves_without_restricted_cad(grasp):
    source = ET.parse(SOURCE)
    result = ET.parse(public_robot_urdf(SOURCE, grasp=grasp))
    assert [ET.tostring(j) for j in source.findall("joint")] == [ET.tostring(j) for j in result.findall("joint")]
    for mesh in result.findall(".//mesh"):
        assert "/alpha/" not in mesh.get("filename")
        assert Path(mesh.get("filename")).is_file()
    for link in source.findall("link"):
        target = result.find(f"link[@name='{link.get('name')}']")
        assert ET.tostring(link.find("inertial")) == ET.tostring(target.find("inertial"))
        if not grasp or "finger_link" not in link.get("name"):
            assert [ET.tostring(x) for x in link.findall("collision")] == [
                ET.tostring(x) for x in target.findall("collision")
            ]
        else:
            assert len(target.findall("collision")) == 2
            assert len(target.findall("collision/geometry/box")) == 2


@pytest.mark.unit
def test_unknown_profile_is_an_error(monkeypatch):
    monkeypatch.setenv("WASMAN_ASSET_PROFILE", "typo")
    with pytest.raises(ValueError):
        selected_profile()
