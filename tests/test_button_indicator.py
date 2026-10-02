import torch

from wasman.controllers.button_indicator import ButtonIndicatorState


def test_indicator_tracks_mechanical_press_with_hysteresis_and_reset():
    state = ButtonIndicatorState(2, "cpu")
    state.update(torch.tensor([0.004, 0.0039]))
    assert state.pressed.tolist() == [True, False]
    state.update(torch.tensor([0.0035, 0.0041]))
    assert state.pressed.tolist() == [True, True]
    state.update(torch.tensor([0.003, 0.005]))
    assert state.pressed.tolist() == [False, True]
    state.reset(torch.tensor([1]))
    assert not state.pressed.any()


def test_indicator_materials_are_independent_and_release_turns_light_off():
    from pxr import Usd, UsdGeom

    from wasman.controllers.button_indicator import ButtonIndicator

    stage = Usd.Stage.CreateInMemory()
    for i in range(2):
        UsdGeom.Cylinder.Define(stage, f"/World/envs/env_{i}/Button/plunger/plunger")
    indicator = ButtonIndicator(stage, 2, "cpu")
    indicator.update(torch.tensor([0.005, 0.0]), render=True)
    assert indicator.visuals[0][2].Get() > 0
    assert indicator.visuals[1][2].Get() == 0
    indicator.update(torch.tensor([0.002, 0.006]), render=True)
    assert indicator.visuals[0][2].Get() == 0
    assert indicator.visuals[1][2].Get() > 0
    indicator.reset(torch.tensor([1]))
    assert indicator.visuals[1][2].Get() == 0
