import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wasman.learning.valve_visual_dataset import (
    ValveVisualDataset,
    interpolate_commands,
    local_state,
    relative_trajectory,
)


def test_relative_chunk_has_one_measured_anchor_and_is_rigid_transform_invariant():
    measured = np.array([[1, 2, 3, 0, 0, 0, 1, -0.2]], dtype=np.float32)
    command = np.array([[[2, 2, 3, 0, 0, 0, 1, -1], [3, 2, 3, 0, 0, 0, 1, 0.5]]], dtype=np.float32)
    expected = relative_trajectory(command, measured)
    np.testing.assert_allclose(expected[0, :, :3], [[1, 0, 0], [2, 0, 0]])
    rotation = Rotation.from_euler("xyz", [0.4, -0.8, 1.2])
    for value in (measured, command):
        value[..., :3] = rotation.apply(value[..., :3].reshape(-1, 3)).reshape(value[..., :3].shape) + [5, 6, 7]
        value[..., 3:7] = (
            (rotation * Rotation.from_quat(value[..., 3:7].reshape(-1, 4))).as_quat().reshape(value[..., 3:7].shape)
        )
    np.testing.assert_allclose(relative_trajectory(command, measured), expected, atol=1e-6)


def test_interpolation_treats_antipodal_quaternions_as_identical_and_pads_endpoint():
    commands = np.array([[0, 0, 0, 0, 0, 0, 1, -1], [2, 0, 0, 0, 0, 0, -1, 1]], dtype=np.float32)
    out = interpolate_commands(commands, np.array([0.5, 9]))
    np.testing.assert_allclose(out[0], [1, 0, 0, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(out[1], commands[-1])


def test_localized_state_keeps_only_measured_jaw_and_xyzw_identity():
    state = np.array([[1, 2, 3, 0, 0.6, 0, 0.8, 0.2]])
    np.testing.assert_allclose(local_state(state), [[0, 0, 0, 0, 0, 0, 1, 0.2]])
    assert state[0, 0] == 1


def recording(path, *, terminal=False):
    path.mkdir()
    length = 60
    measured = np.zeros((length, 8), dtype=np.float32)
    measured[:, 0] = np.arange(length) * 0.001
    measured[:, 6] = 1
    command = measured.copy()
    command[:, 0] += 0.1
    done = np.zeros(length, dtype=bool)
    done[30] = terminal
    success = np.zeros(length, dtype=bool)
    success[-1] = True
    np.savez(
        path / "trajectory.npz",
        measured=measured,
        command=command,
        terminal=done,
        success_after=success,
        camera_frame=np.arange(length),
    )
    (path / "metadata.json").write_text(
        json.dumps(
            dict(
                schema="wasman-valve-rgb-ee-v1",
                outcome="success",
                seed=123,
                raw_hz=30,
                length=length,
                image_shape=[2, 2, 3],
            )
        )
    )
    images = np.broadcast_to(np.arange(length, dtype=np.uint8)[:, None, None, None], (length, 2, 2, 3))
    (path / "wrist.rgb").write_bytes(images.tobytes())
    return path


def test_20hz_observation_is_past_frame_and_future_labels_share_its_actual_anchor(tmp_path):
    data = ValveVisualDataset([recording(tmp_path / "one")])
    # Logical t=.05 selects raw frame1 (t=.0333), not future frame2.
    sample = data[1]
    assert (sample["observation.images.wrist"] == 1).all()
    np.testing.assert_allclose(sample["action"][:3, 0], [0.1, 0.1015, 0.103], atol=1e-7)
    assert data[len(data) - 1]["action_is_pad"][1:].all()


def test_reset_spliced_recording_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="first-episode"):
        ValveVisualDataset([recording(tmp_path / "bad", terminal=True)])


def test_30hz_transport_preserves_each_recorded_command_and_frame(tmp_path):
    data = ValveVisualDataset([recording(tmp_path / "native")], policy_hz=30)
    sample = data[7]
    assert (sample["observation.images.wrist"] == 7).all()
    np.testing.assert_allclose(sample["action"][:3, 0], [0.1, 0.101, 0.102], atol=1e-7)
    assert len(data) == 60
