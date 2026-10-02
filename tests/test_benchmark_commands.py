"""The common command preserves task-specific benchmark backends and interfaces."""

import pytest

from wasman.benchmark import TASKS, command

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("model", ["ACT", "DP"])
def test_visual_training_keeps_original_backends(task, model):
    stem = {
        "RotateValve": "valve",
        "OpenHatch": "hatch",
        "CollectShell": "shell",
        "PressButton": "button",
        "PushSlider": "marine",
        "PullLever": "marine",
    }[task]
    result = command(task, "train", model)
    assert result[0] == f"scripts/train_{stem}_{model.lower()}.py"
    assert ("--task" in result) == (task in {"PressButton", "PushSlider", "PullLever"})
    assert not any(x in result for x in ["--steps", "--epochs", "--batch-size", "--seed"])


def test_ee_interface_is_explicit_and_does_not_turn_collection_into_policy_replay():
    assert command("PushSlider", "train", "DP", "ee")[0] == "scripts/train_research_dp.py"
    assert command("PullLever", "evaluate", interface="ee")[-2:] == ["--interface", "ee"]
    with pytest.raises(ValueError, match="Collect native"):
        command("PushSlider", "collect", interface="ee")
    with pytest.raises(ValueError, match="no benchmarked"):
        command("OpenHatch", "evaluate", interface="ee")


def test_evaluation_reads_architecture_from_checkpoint():
    assert command("RotateValve", "evaluate") == ["scripts/evaluate_valve_visual.py"]
    with pytest.raises(ValueError, match="only for training"):
        command("RotateValve", "evaluate", "ACT")
