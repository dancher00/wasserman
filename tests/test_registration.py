"""Task registration contract."""

import gymnasium as gym
import pytest

import wasman.tasks  # noqa: F401


@pytest.mark.unit
def test_underwater_press_button_registration() -> None:
    spec = gym.spec("Wasman-Underwater-PressButton-Direct")
    assert spec.entry_point == (
        "wasman.tasks.underwater_press_button.config.bluerov2_alpha.env:UnderwaterPressButtonEnv"
    )
    assert spec.kwargs["env_cfg_entry_point"] == (
        "wasman.tasks.underwater_press_button.config.bluerov2_alpha.env_cfg:UnderwaterPressButtonEnvCfg"
    )
    assert spec.kwargs["default_agent"] == "rsl_rl"


@pytest.mark.unit
def test_smooth_button_registration() -> None:
    spec = gym.spec("Wasman-Underwater-PressButton-Smooth-Direct")
    assert spec.entry_point.endswith(".smooth:SmoothPressButtonEnv")


@pytest.mark.unit
def test_t200_task_variants_registered():
    for task in ("PressButton", "RotateValve"):
        spec = gym.spec(f"Wasman-Underwater-{task}-T200-Direct")
        assert spec.kwargs["default_agent"] == "rsl_rl"


@pytest.mark.unit
def test_valve_has_its_own_contract_and_optional_swim_in():
    assert gym.spec("Wasman-Underwater-RotateValve-T200-Direct").entry_point.endswith(".valve:UnderwaterRotateValveEnv")
    spec = gym.spec("Wasman-Underwater-RotateValve-Approach-T200-Direct")
    assert spec.entry_point.endswith(".approach:ApproachRotateValveEnv")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(".agents:ValvePPORunnerCfg")


@pytest.mark.unit
def test_approach_uses_its_own_controller_without_replacing_short_start():
    spec = gym.spec("Wasman-Underwater-PressButton-Approach-T200-Direct")
    assert spec.entry_point.endswith(".approach:ApproachPressButtonEnv")
    assert spec.kwargs["env_cfg_entry_point"].endswith(".approach_cfg:ApproachPressButtonEnvCfg")
    assert gym.spec("Wasman-Underwater-PressButton-T200-Direct").entry_point.endswith(".smooth:SmoothPressButtonEnv")


@pytest.mark.unit
@pytest.mark.parametrize("task", ["PushSlider", "PullLever", "RotateValve"])
def test_experimental_panel_registration(task: str) -> None:
    spec = gym.spec(f"Wasman-Underwater-{task}-Direct")
    assert spec.kwargs["env_cfg_entry_point"] == f"wasman.tasks.underwater_panel.env_cfg:Underwater{task}EnvCfg"
    assert spec.kwargs["default_agent"] == "rsl_rl"
