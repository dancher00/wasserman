import copy

import pytest

from wasman.learning.hatch_reporting import publication_version, summarize_episode_splits


def inputs():
    train = [{"seed": 1}]
    validation = [{"seed": 2, "mode": "collect", "num_envs": 2, "successes": 2}]
    episode = {
        "seed": 3,
        "mode": "evaluate",
        "teacher_probability": 0,
        "checkpoint_sha256": "model",
        "num_envs": 2,
        "successes": 1,
        "success_per_env": [True, False],
        "terminal_per_env": [False, False],
        "censored_per_env": [False, False],
        "steps": 1700,
        "contract_replay": {"successes": 1},
    }
    other = copy.deepcopy(episode)
    other["seed"] = 4
    fit = {"train_seeds": [1], "validation_seeds": [2], "checkpoint_sha256": "model"}
    return train, validation, [episode, other], fit


def test_multiseed_summary_keeps_each_attempt_and_batch():
    result = summarize_episode_splits(*inputs())
    assert result["evaluation_episodes"] == 4 and result["evaluation_successes"] == 2
    assert result["evaluation_seeds"] == [3, 4]
    assert len(result["evaluation_batches"]) == 2
    assert result["evaluation_censored_per_env"] == [False] * 4


@pytest.mark.parametrize(
    "fault", ["teacher", "checkpoint", "seed", "replay", "replay_env", "conditions", "boundary", "validation"]
)
def test_invalid_publication_rejected(fault):
    train, validation, evaluation, fit = inputs()
    if fault == "teacher":
        evaluation[1]["teacher_probability"] = 0.5
    elif fault == "checkpoint":
        evaluation[1]["checkpoint_sha256"] = "other"
    elif fault == "seed":
        evaluation[1]["seed"] = 1
    elif fault == "replay":
        evaluation[1]["contract_replay"]["successes"] = 2
    elif fault == "replay_env":
        evaluation[1]["contract_replay"]["first_success_step"] = [-1, 1500]
    elif fault == "conditions":
        evaluation[1]["camera_appearance"] = {"water": "harbor"}
    elif fault == "boundary":
        evaluation[1]["boundary_effects_enabled"] = True
    else:
        fit["validation_seeds"].append(5)
    with pytest.raises(ValueError):
        summarize_episode_splits(train, validation, evaluation, fit)


def test_version_is_not_a_filesystem_path():
    assert publication_version("v12") == "v12"
    for value in ("../../oops", "v0", "v1/draft", "v2\n"):
        with pytest.raises(ValueError):
            publication_version(value)
