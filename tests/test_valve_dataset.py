import pytest
import torch

from wasman.learning.valve_dataset import incremental_huber, phase_pools, split_legacy_episodes, validate_dataset


def test_split_is_whole_episode_disjoint_across_rounds():
    train, val, episodes = split_legacy_episodes(2 * 4 * 11, num_envs=4, steps_per_round=11, validation_envs=1, seed=7)
    assert train.sum() == 66 and val.sum() == 22
    assert not torch.isin(episodes[train].unique(), episodes[val].unique()).any()
    for episode in episodes.unique():
        mask = episodes == episode
        assert train[mask].all() or val[mask].all()
    phases = torch.arange(88) % 8
    for pool in phase_pools(phases, train):
        assert train[pool].all() and not val[pool].any()


@pytest.mark.parametrize("length,envs,steps,held", [(89, 4, 11, 1), (88, 4, 11, 4), (88, 1, 88, 1)])
def test_bad_layout_or_empty_partition_rejected(length, envs, steps, held):
    with pytest.raises(ValueError):
        split_legacy_episodes(length, num_envs=envs, steps_per_round=steps, validation_envs=held, seed=7)


def test_data_contract_rejects_out_of_range_labels_and_nonfinite():
    data = {"observations": torch.zeros(8, 50), "targets": torch.zeros(8, 11), "phases": torch.arange(8)}
    validate_dataset(data)
    data["targets"][0, 0] = 2
    with pytest.raises(ValueError, match="action"):
        validate_dataset(data)
    data["targets"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        validate_dataset(data)


def test_loss_resolves_small_increment_error_without_absolute_target_scale():
    previous = torch.full((2, 11), 0.7)
    scale = torch.full((11,), 0.002)
    assert torch.allclose(incremental_huber(previous + scale, previous, scale), torch.tensor(0.5), atol=1e-5)


@pytest.mark.parametrize("invalid_phase", [0.5, float("nan"), float("inf"), -1.0, 8.0])
def test_phase_labels_must_be_finite_integers(invalid_phase):
    data = {"observations": torch.zeros(8, 50), "targets": torch.zeros(8, 11), "phases": torch.arange(8).float()}
    data["phases"][0] = invalid_phase
    with pytest.raises(ValueError, match="phase"):
        validate_dataset(data)
