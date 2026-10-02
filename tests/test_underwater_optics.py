import pytest
import torch

from wasman.assets.underwater_optics import WATER_PRESETS, WaterOptics, underwater_rgb


def test_zero_range_is_identity():
    rgb = torch.randint(0, 256, (2, 8, 8, 3), dtype=torch.uint8)
    assert torch.equal(rgb, underwater_rgb(rgb, torch.zeros(2, 8, 8, 1), WATER_PRESETS["harbor"]))


def test_depth_is_not_uniform_tint():
    rgb = torch.full((1, 1, 3, 3), 200, dtype=torch.uint8)
    depth = torch.tensor([[[0.0, 1.0, 4.0]]])
    output = underwater_rgb(rgb, depth, WATER_PRESETS["harbor"])
    assert (output[0, 0, 0] > output[0, 0, 1]).all()
    assert (output[0, 0, 1] > output[0, 0, 2]).all()
    assert output[0, 0, 2, 0] < output[0, 0, 2, 2]


def test_dark_water_does_not_emit_light():
    dark = torch.zeros((2, 2, 3), dtype=torch.uint8)
    assert underwater_rgb(dark, torch.ones(2, 2), WATER_PRESETS["harbor"], illumination=0).sum() == 0


def test_invalid_depth_is_finite_and_bounded():
    rgb = torch.full((1, 3, 3), 128, dtype=torch.uint8)
    output = underwater_rgb(rgb, torch.tensor([[float("inf"), float("nan"), -1.0]]), WATER_PRESETS["coastal"])
    assert output.dtype == torch.uint8
    assert torch.equal(output[0, 2], rgb[0, 2])


def test_coefficients_are_validated():
    with pytest.raises(ValueError):
        WaterOptics((-1, 0, 0), (0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError):
        underwater_rgb(torch.zeros(3, 3, 3), torch.ones(3, 3), WATER_PRESETS["clear"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_water_parameters_are_rejected(bad):
    with pytest.raises(ValueError):
        WaterOptics((bad, 0, 0), (0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError):
        WaterOptics((0, 0, 0), (0, bad, 0), (0, 0, 0))
    with pytest.raises(ValueError):
        WaterOptics((0, 0, 0), (0, 0, 0), (bad, 0, 0))
    with pytest.raises(ValueError):
        WaterOptics((0, 0, 0), (0, 0, 0), (0, 0, 0), max_range_m=bad)


def test_three_channels_required():
    with pytest.raises(ValueError, match="three RGB"):
        WaterOptics((0, 0), (0, 0, 0), (0, 0, 0))
