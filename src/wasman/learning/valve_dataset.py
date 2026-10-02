"""Explicit episode layout for legacy flattened valve demonstration tensors."""

import torch


def validate_dataset(data):
    obs, target, phase = (data[k] for k in ("observations", "targets", "phases"))
    if obs.ndim != 2 or obs.shape[1] != 50 or target.shape != (len(obs), 11) or phase.shape != (len(obs),):
        raise ValueError("Expected valve tensors (N,50), (N,11), (N,)")
    if not len(obs) or not torch.isfinite(obs).all() or not torch.isfinite(target).all():
        raise ValueError("Empty or nonfinite valve data")
    if (
        target.abs().max() > 1.000001
        or not torch.isfinite(phase).all()
        or not ((phase >= 0) & (phase < 8)).all()
        or not (phase == phase.long()).all()
    ):
        raise ValueError("Invalid normalized action or expert sampling phase")


def split_legacy_episodes(length, *, num_envs, steps_per_round, validation_envs, seed):
    """Split entire time-major environment episodes, never adjacent random rows.

    Caller explicitly supplies the documented layout: each round concatenates
    [step0/env0..N, step1/env0..N, ...]. Collection must have no mid-round resets;
    absent legacy episode IDs cannot independently establish that condition.
    """
    if num_envs < 2 or not 1 <= validation_envs < num_envs or steps_per_round < 1:
        raise ValueError("Require positive steps and nonempty train/validation environment groups")
    block = num_envs * steps_per_round
    if length < block or length % block:
        raise ValueError("Dataset length does not match the explicitly supplied collection layout")
    row = torch.arange(length)
    episodes = (row // block) * num_envs + row % num_envs
    generator = torch.Generator().manual_seed(seed)
    validation_episode_ids = torch.cat(
        [torch.randperm(num_envs, generator=generator)[:validation_envs] + r * num_envs for r in range(length // block)]
    )
    validation = torch.isin(episodes, validation_episode_ids)
    training = ~validation
    if torch.isin(episodes[training].unique(), episodes[validation].unique()).any():
        raise RuntimeError("Episode leakage between train and validation")
    return training, validation, episodes


def phase_pools(phases, mask):
    if mask.shape != phases.shape or mask.dtype != torch.bool:
        raise ValueError("Invalid sample-selection mask")
    return [(mask & (phases == phase)).nonzero().flatten() for phase in range(8)]


def incremental_huber(prediction, target, increment_scale):
    # Subtract first, avoiding cancellation between large separately normalized
    # absolute targets. This is the same mathematical increment error.
    return torch.nn.functional.smooth_l1_loss((prediction - target) / increment_scale, torch.zeros_like(target))
