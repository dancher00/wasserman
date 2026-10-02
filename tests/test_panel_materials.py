import hashlib
import json

import pytest
import torch
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from wasman.assets.panels import PANEL_STYLES, centered_panel_pose, panel_path


@pytest.mark.parametrize("style", PANEL_STYLES)
def test_panel_has_sourced_maps_uvs_and_only_one_slab_collider(style):
    path = panel_path(style)
    source = json.loads(path.with_name("provenance.json").read_text())
    assert source["license"] == "CC0-1.0"
    for name, metadata in source["maps"].items():
        assert hashlib.sha256(path.with_name(f"{name}.jpg").read_bytes()).hexdigest() == metadata["sha256"]
    stage = Usd.Stage.Open(str(path))
    assert stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    assert [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)] == ["/Panel/Slab"]
    face = UsdGeom.Mesh(stage.GetPrimAtPath("/Panel/Surface"))
    assert list(face.GetFaceVertexCountsAttr().Get()) == [4] * 6
    assert len(UsdGeom.PrimvarsAPI(face).GetPrimvar("st").Get()) == 24
    points = torch.tensor(list(face.GetPointsAttr().Get())).reshape(6, 4, 3)
    normals = torch.linalg.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals /= normals.norm(dim=-1, keepdim=True)
    assert set(map(tuple, normals.tolist())) == {
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
    }
    assert UsdGeom.Imageable(stage.GetPrimAtPath("/Panel/Slab")).GetPurposeAttr().Get() == "guide"
    material, _ = UsdShade.MaterialBindingAPI(face).ComputeBoundMaterial()
    assert material


def test_panel_centering_preserves_plane_and_input_poses():
    panel = torch.tensor([[0.86, 2.0, 0.8, 0, 0, 0, 1], [0.86, 4.0, 0.8, 0, 0, 0, 1]])
    mechanism = panel.clone()
    mechanism[:, 1:3] += torch.tensor([-0.1, -0.04])
    centered = centered_panel_pose(panel, mechanism)
    assert torch.equal(centered[:, 1:3], mechanism[:, 1:3])
    assert torch.equal(centered[:, [0, 3, 4, 5, 6]], panel[:, [0, 3, 4, 5, 6]])
    assert not torch.equal(panel, centered)


def test_panel_style_validation(monkeypatch):
    monkeypatch.setenv("WASMAN_PANEL_STYLE", "harbor_concrete")
    assert panel_path() == panel_path("harbor_concrete")
    with pytest.raises(ValueError, match="Unknown panel"):
        panel_path("not-a-style")
