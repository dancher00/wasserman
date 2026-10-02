import torch

from wasman.assets.variation import PLACEMENTS, tilt_assembly
from wasman.tasks.underwater_panel.env_cfg import UnderwaterRotateValveT200EnvCfg
from wasman.tasks.underwater_press_button.config.bluerov2_alpha.env_cfg import UnderwaterPressButtonEnvCfg


def test_tilt_rotates_attachment_without_resizing_or_mutating_input():
    panel = torch.tensor([[0.86, -0.10, 0.78, 0, 0, 0, 1.0]])
    fixture = panel.clone()
    fixture[:, 0] -= 0.05
    tilted_panel, tilted_fixture = tilt_assembly(panel, fixture, 25)
    assert torch.equal(tilted_panel[:, :3], panel[:, :3])
    assert torch.allclose((tilted_fixture[:, :3] - tilted_panel[:, :3]).norm(dim=-1), torch.tensor([0.05]))
    assert torch.allclose(tilted_fixture[:, 3:], tilted_panel[:, 3:])
    assert torch.allclose(tilted_panel[:, 3:].norm(dim=-1), torch.ones(1))
    restored_panel, restored_fixture = tilt_assembly(tilted_panel, tilted_fixture, -25)
    assert torch.allclose(restored_panel, panel, atol=1e-6)
    assert torch.allclose(restored_fixture, fixture, atol=1e-6)
    assert panel[0, 6] == fixture[0, 6] == 1


def test_gallery_placements_are_separated_and_do_not_change_training_defaults():
    positions = torch.tensor(PLACEMENTS)
    assert torch.pdist(positions).min() >= 0.49
    for cfg, y_range, z_range in (
        (UnderwaterPressButtonEnvCfg(), (-0.17, -0.03), (0.70, 0.82)),
        (UnderwaterRotateValveT200EnvCfg(), (-0.08, 0.0), (0.72, 0.80)),
    ):
        assert cfg.scene.panel.init_state.rot == (0.0, 0.0, 0.0, 1.0)
        assert cfg.button_y_range == y_range
        assert cfg.button_z_range == z_range
    assert UnderwaterRotateValveT200EnvCfg().scene.button.spawn.scale == (1.0, 1.0, 1.0)
