import json

import numpy as np
import pytest
import torch

from wasman.controllers.hatch_actuator import pack_action, unpack_action
from wasman.learning.hatch_visual_dataset import HatchVisualDataset, interpolate_commands


def test_folded_transit_and_deployed_targets_roundtrip_without_a_phase_router():
    actions = torch.tensor(
        [
            [-0.4, 0.1, 0.0, 0, 0, 0, 0, 0, 0, 0, -1],
            [0.3, -0.1, 0.2, 0, 0, 0, 0.2, -0.3, 0.8, 0.9, 1],
        ]
    )
    torch.testing.assert_close(unpack_action(pack_action(actions)), actions, rtol=0, atol=0)
    actions[0, 4] = 0.1
    with pytest.raises(ValueError, match="level-attitude"):
        pack_action(actions)


def test_actuator_interpolation_does_not_treat_arm_joints_as_a_quaternion():
    targets = np.array([[0, 0, 0, 0, 0, 0, 0, -1], [1, 2, 3, 0.2, -0.4, 0.8, 0.6, 1]])
    result = interpolate_commands(targets, np.array([0.5, 100]))
    np.testing.assert_allclose(result[0], [0.5, 1, 1.5, 0.1, -0.2, 0.4, 0.3, 0])
    np.testing.assert_allclose(result[1], targets[1])


def recording(path, terminal=False):
    path.mkdir()
    n = 60
    state = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 8), dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    done[30] = terminal
    success = np.zeros(n, dtype=bool)
    success[-1] = True
    np.savez(
        path / "trajectory.npz",
        measured=state,
        command=state + 0.25,
        terminal=done,
        success_after=success,
        camera_frame=np.arange(n),
    )
    (path / "metadata.json").write_text(
        json.dumps(
            dict(
                schema="wasman-hatch-rgb-actuator-v1",
                outcome="success",
                seed=1,
                raw_hz=30,
                length=n,
                image_shape=[2, 2, 3],
            )
        )
    )
    (path / "wrist.rgb").write_bytes(
        np.broadcast_to(
            np.arange(n, dtype=np.uint8)[:, None, None, None],
            (n, 2, 2, 3),
        ).tobytes()
    )
    return path


def test_resampling_uses_past_image_and_absolute_targets_without_episode_leakage(tmp_path):
    data = HatchVisualDataset([recording(tmp_path / "episode")])
    item = data[1]  # logical .05 s selects frame1, not future frame2
    assert (item["observation.images.wrist"] == 1).all()
    np.testing.assert_array_equal(item["observation.state"], np.ones(8))
    np.testing.assert_allclose(item["action"][:3, 0], [1.25, 2.75, 4.25])
    assert data[-1]["action_is_pad"][1:].all()
    with pytest.raises(ValueError, match="first-episode"):
        HatchVisualDataset([recording(tmp_path / "reset", terminal=True)])
