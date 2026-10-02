"""Versioned, pre-action two-camera episodes. Split by seed, never by image."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from wasman.controllers.hatch_vision import PROPRIO_DIM

SCHEMA = "wasman-hatch-rgb-proprio-v1"


def validate_batch(path):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    if meta["schema"] != SCHEMA:
        raise ValueError("Unknown demonstration schema")
    arrays = {
        k: np.load(path / f"{k}.npy", mmap_mode="r")
        for k in (
            "rgb",
            "proprio",
            "action",
            "sample_step",
            "camera_time",
            "camera_frame",
            "terminal",
        )
    }
    rgb, action = arrays["rgb"], arrays["action"]
    if (path / "teacher_action.npy").exists():
        arrays["teacher_action"] = np.load(path / "teacher_action.npy", mmap_mode="r")
        if arrays["teacher_action"].shape != action.shape or not np.isfinite(arrays["teacher_action"]).all():
            raise ValueError("Invalid teacher labels")
    elif meta.get("mode") == "dagger":
        raise ValueError("DAgger batch lacks teacher labels")
    samples, envs = rgb.shape[:2]
    if rgb.dtype != np.uint8 or rgb.shape[2:4] != (2, 3) or envs != meta["num_envs"]:
        raise ValueError("Invalid RGB layout")
    if action.shape != (meta["steps"], envs, 11) or not np.isfinite(action).all():
        raise ValueError("Invalid action labels")
    if arrays["proprio"].shape != (samples, envs, PROPRIO_DIM) or not np.isfinite(arrays["proprio"]).all():
        raise ValueError("Invalid proprioception")
    steps = arrays["sample_step"]
    if not np.array_equal(steps, np.arange(samples) * meta["sample_every"]):
        raise ValueError("Noncontiguous image samples")
    if arrays["terminal"].shape != action.shape[:2] or arrays["terminal"].dtype != np.bool_:
        raise ValueError("Invalid episode-boundary flags")
    if len(steps) > 1 and not np.array_equal(arrays["proprio"][1:, :, -11:], action[steps[1:] - 1]):
        raise ValueError("Previous-command field leaks future actions or is misaligned")
    times = arrays["camera_time"]
    if times.shape != (samples, envs, 2):
        raise ValueError("Missing camera timestamps")
    if not np.isfinite(times).all() or np.max(np.abs(times[..., 0] - times[..., 1])) > 1e-6:
        raise ValueError("Unsynchronized camera pair")
    if np.max(np.abs(times - steps[:, None, None] * meta["dt_s"])) > 0.02:
        raise ValueError("Image/state timestamp mismatch")
    if np.any(np.diff(arrays["camera_frame"], axis=0) <= 0):
        raise ValueError("Stale camera frame")
    return meta, arrays


class HatchDataset(Dataset):
    def __init__(self, paths, chunk_size=9, *, successful_only=True):
        self.chunk_size, self.batches, self.indices = chunk_size, [], []
        self.seeds = set()
        for path in paths:
            meta, arrays = validate_batch(path)
            if meta["seed"] in self.seeds:
                raise ValueError("Duplicate episode seed")
            self.seeds.add(meta["seed"])
            batch = len(self.batches)
            self.batches.append((meta, arrays))
            for sample, step in enumerate(arrays["sample_step"]):
                if step + chunk_size > meta["steps"]:
                    continue
                for env_id in range(meta["num_envs"]):
                    if successful_only and not meta["success_per_env"][env_id]:
                        continue
                    if arrays["terminal"][: step + chunk_size, env_id].any():
                        continue
                    self.indices.append((batch, sample, env_id, int(step)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        batch, sample, env_id, step = self.indices[index]
        arrays = self.batches[batch][1]
        return tuple(
            torch.from_numpy(np.array(value, copy=True))
            for value in (
                arrays["rgb"][sample, env_id],
                arrays["proprio"][sample, env_id],
                arrays.get("teacher_action", arrays["action"])[step : step + self.chunk_size, env_id],
            )
        )


def assert_disjoint(train, validation):
    if train.seeds & validation.seeds:
        raise ValueError("Training/validation episode leakage")


class CachedHatchBatches:
    """Keep selected windows resident on a device; avoid per-step CPU image copies.

    The indexing and successful-episode selection are inherited unchanged from
    HatchDataset. Cache construction uses bounded CPU staging chunks, not a
    second full-size RGB copy in host RAM. ``cpu`` is useful for equivalence tests.
    """

    def __init__(self, dataset, batch_size, *, device="cuda", shuffle=False):
        self.device, self.batch_size, self.shuffle = device, batch_size, shuffle
        self.count = len(dataset)
        if self.count == 0 or batch_size < 1:
            raise ValueError("Nonempty dataset and positive batch size required")
        shape = dataset.batches[0][1]["rgb"].shape[2:]
        self.rgb = torch.empty((self.count, *shape), dtype=torch.uint8, device=device)
        self.state = torch.empty((self.count, PROPRIO_DIM), device=device)
        self.target = torch.empty((self.count, dataset.chunk_size, 11), device=device)
        indices = np.asarray(dataset.indices)
        for batch, (_, arrays) in enumerate(dataset.batches):
            destinations = np.flatnonzero(indices[:, 0] == batch)
            labels = arrays.get("teacher_action", arrays["action"])
            for start in range(0, len(destinations), 512):
                dest = destinations[start : start + 512]
                _, sample, env_id, step = indices[dest].T
                dest_gpu = torch.as_tensor(dest, device=device)
                self.rgb[dest_gpu] = torch.as_tensor(arrays["rgb"][sample, env_id], device=device)
                self.state[dest_gpu] = torch.as_tensor(arrays["proprio"][sample, env_id], device=device)
                self.target[dest_gpu] = torch.as_tensor(
                    labels[step[:, None] + np.arange(dataset.chunk_size), env_id[:, None]], device=device
                )

    def __iter__(self):
        order = (
            torch.randperm(self.count, device=self.device)
            if self.shuffle
            else torch.arange(self.count, device=self.device)
        )
        for start in range(0, self.count, self.batch_size):
            index = order[start : start + self.batch_size]
            yield self.rgb[index], self.state[index], self.target[index]
