import json

import pytest
import torch

from wasman.learning.revision_protocol import PROTOCOL, split_episodes, validate_evaluation, verify_asset_profile
from wasman.learning.revision_rgb import DPImageTransform


@pytest.mark.unit
def test_split_has_fixed_whole_episode_membership(tmp_path):
    episodes = []
    for seed in range(80):
        path = tmp_path / str(seed)
        path.mkdir()
        (path / "metadata.json").write_text(json.dumps({"seed": seed}))
        episodes.append(path)
    first = split_episodes(episodes)
    torch.manual_seed(43)
    second = split_episodes(episodes)
    assert first == second
    assert tuple(map(len, first)) == (76, 4)
    assert not set(first[0]) & set(first[1])
    assert set(first[0]) | set(first[1]) == set(episodes)
    with pytest.raises(ValueError, match="Duplicate"):
        split_episodes(episodes + [episodes[0]])


@pytest.mark.unit
def test_eval_preprocessing_does_not_consume_rng_and_normalizes_clip():
    transform = DPImageTransform().eval()
    image = torch.full((2, 3, 224, 224), 0.5)
    before = torch.get_rng_state().clone()
    actual = transform(image)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(actual, transform(image))
    expected = (torch.tensor([0.5] * 3) - torch.tensor([0.48145466, 0.4578275, 0.40821073])) / torch.tensor(
        [0.26862954, 0.26130258, 0.27577711]
    )
    torch.testing.assert_close(actual[0, :, 100, 100], expected)


@pytest.mark.unit
def test_wrong_asset_profile_rejected(monkeypatch):
    monkeypatch.setenv("WASMAN_ASSET_PROFILE", "open-procedural-v1")
    with pytest.raises(ValueError, match="asset profile"):
        verify_asset_profile({})


@pytest.mark.unit
def test_test_cohort_is_protected_from_tuning_and_pilot_weights():
    config = dict(protocol=PROTOCOL, task="PressButton", pilot=False, train_seeds=[40000], validation_seeds=[40001])
    validate_evaluation(config, "PressButton", list(range(60000, 60030)), "test", steps=470)
    with pytest.raises(ValueError, match="complete declared control horizon"):
        validate_evaluation(config, "PressButton", list(range(60000, 60030)), "test", steps=2)
    for seeds, purpose in [
        ([60000], "development"),
        ([60000], "test"),
        ([40001], "development"),
        ([50000], "validation"),
    ]:
        with pytest.raises(ValueError):
            validate_evaluation(config, "PressButton", seeds, purpose)
    with pytest.raises(ValueError, match="non-pilot"):
        validate_evaluation(dict(config, pilot=True), "PressButton", list(range(60000, 60030)), "test", steps=470)
