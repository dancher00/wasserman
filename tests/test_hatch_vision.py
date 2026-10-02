import json

import numpy as np
import pytest
import torch

from wasman.controllers.hatch_vision import PROPRIO_DIM, PROPRIO_FIELDS, HatchVisionActor
from wasman.learning.hatch_dataset import SCHEMA, CachedHatchBatches, HatchDataset, assert_disjoint, validate_batch


def batch(tmp_path, seed=1):
    path = tmp_path / str(seed)
    path.mkdir()
    meta = {
        "schema": SCHEMA,
        "seed": seed,
        "num_envs": 2,
        "steps": 12,
        "sample_every": 3,
        "dt_s": 1 / 30,
        "success_per_env": [True, False],
    }
    (path / "metadata.json").write_text(json.dumps(meta))
    for name, data in {
        "rgb": np.zeros((4, 2, 2, 3, 16, 16), dtype=np.uint8),
        "proprio": np.zeros((4, 2, PROPRIO_DIM), dtype=np.float32),
        "action": np.arange(12 * 2 * 11, dtype=np.float32).reshape(12, 2, 11) / 1000,
        "sample_step": np.arange(4) * 3,
        "camera_time": np.broadcast_to(np.arange(4)[:, None, None] / 10, (4, 2, 2)).copy(),
        "camera_frame": np.broadcast_to(np.arange(4)[:, None, None], (4, 2, 2)).copy(),
        "terminal": np.zeros((12, 2), dtype=bool),
    }.items():
        np.save(path / f"{name}.npy", data)
    state = np.load(path / "proprio.npy")
    state[1:, :, -11:] = np.load(path / "action.npy")[np.arange(1, 4) * 3 - 1]
    np.save(path / "proprio.npy", state)
    return path


def test_actor_two_views_and_robot_only_schema():
    torch.set_num_threads(2)
    actor = HatchVisionActor(6)
    rgb = torch.zeros((2, 2, 3, 32, 32), dtype=torch.uint8)
    state = torch.zeros(2, PROPRIO_DIM)
    result = actor(rgb, state)
    assert result.shape == (2, 6, 11) and torch.isfinite(result).all()
    rgb[:, 1] = 255
    assert not torch.allclose(result, actor(rgb, state))
    assert all(not any(token in name for token in ("target", "phase", "hatch", "time")) for name, _ in PROPRIO_FIELDS)


def test_zero_residual_keeps_previous_command():
    actor = HatchVisionActor(3)
    torch.nn.init.zeros_(actor.head[-1].weight)
    torch.nn.init.zeros_(actor.head[-1].bias)
    state = torch.randn(2, PROPRIO_DIM)
    output = actor(torch.zeros((2, 2, 3, 32, 32), dtype=torch.uint8), state)
    assert torch.allclose(output, state[:, None, -11:].expand(-1, 3, -1))


def test_absolute_actor_does_not_add_its_previous_prediction():
    actor = HatchVisionActor(3, "absolute")
    torch.nn.init.zeros_(actor.head[-1].weight)
    torch.nn.init.zeros_(actor.head[-1].bias)
    output = actor(torch.zeros((2, 2, 3, 32, 32), dtype=torch.uint8), torch.randn(2, PROPRIO_DIM))
    assert torch.equal(output, torch.zeros_like(output))


def test_bounded_state_and_level_attitude_interface():
    actor = HatchVisionActor(3, "absolute", state_clip=5.0, hold_level=True).eval()
    rgb = torch.zeros((2, 2, 3, 32, 32), dtype=torch.uint8)
    out = actor(rgb, torch.full((2, PROPRIO_DIM), 10.0))
    assert torch.equal(out[..., 3:6], torch.zeros_like(out[..., 3:6]))
    assert torch.equal(out, actor(rgb, torch.full((2, PROPRIO_DIM), 1000.0)))


def test_state_augmentation_is_disabled_at_evaluation():
    actor = HatchVisionActor(state_noise=0.1, previous_command_dropout=0.5).eval()
    rgb, state = torch.zeros((2, 2, 3, 32, 32), dtype=torch.uint8), torch.randn(2, PROPRIO_DIM)
    assert torch.equal(actor(rgb, state), actor(rgb, state))


def test_invalid_state_augmentation_rejected():
    with pytest.raises(ValueError):
        HatchVisionActor(state_clip=-1)
    with pytest.raises(ValueError):
        HatchVisionActor(previous_command_dropout=1.1)


def test_chunks_start_at_pre_action_step_and_do_not_cross_episode(tmp_path):
    path = batch(tmp_path)
    data = HatchDataset([path], chunk_size=6)
    assert len(data) == 3  # steps 0,3,6 of successful env0; step9 cannot provide six actions.
    _, _, actions = data[1]
    assert np.array_equal(actions.numpy(), np.load(path / "action.npy")[3:9, 0])


def test_episode_split_rejects_leakage(tmp_path):
    path = batch(tmp_path)
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(HatchDataset([path]), HatchDataset([path]))
    assert_disjoint(HatchDataset([path]), HatchDataset([batch(tmp_path, 2)]))


@pytest.mark.parametrize("field", ["camera_time", "camera_frame", "sample_step"])
def test_sync_validator_rejects_misalignment(tmp_path, field):
    path = batch(tmp_path)
    value = np.load(path / f"{field}.npy")
    value[1] = value[0]
    np.save(path / f"{field}.npy", value)
    with pytest.raises(ValueError):
        validate_batch(path)


def test_terminal_and_failure_records_not_silently_used_as_success(tmp_path):
    path = batch(tmp_path)
    terminal = np.load(path / "terminal.npy")
    terminal[5, 0] = True
    np.save(path / "terminal.npy", terminal)
    assert len(HatchDataset([path], chunk_size=6)) == 0
    assert len(HatchDataset([path], chunk_size=6, successful_only=False)) == 3


def test_future_command_leak_is_rejected(tmp_path):
    path = batch(tmp_path)
    state = np.load(path / "proprio.npy")
    state[1, :, -11:] = np.load(path / "action.npy")[3]
    np.save(path / "proprio.npy", state)
    with pytest.raises(ValueError, match="future actions"):
        validate_batch(path)


def test_dagger_uses_teacher_labels_not_executed_mixture(tmp_path):
    path = batch(tmp_path)
    action = np.load(path / "action.npy")
    labels = action + 0.25
    np.save(path / "teacher_action.npy", labels)
    _, _, target = HatchDataset([path], chunk_size=6)[0]
    assert np.array_equal(target.numpy(), labels[:6, 0])
    # Proprioception still contains the previously EXECUTED command.
    validate_batch(path)


def test_device_cache_matches_cpu_windows_and_teacher_labels(tmp_path):
    path = batch(tmp_path)
    np.save(path / "teacher_action.npy", np.load(path / "action.npy") + 0.25)
    data = HatchDataset([path], chunk_size=6, successful_only=False)
    cached = CachedHatchBatches(data, 4, device="cpu")
    observed = list(cached)
    assert [len(parts[0]) for parts in observed] == [4, 2]
    for column in range(3):
        assert torch.equal(
            torch.cat([parts[column] for parts in observed]), torch.stack([data[i][column] for i in range(len(data))])
        )
