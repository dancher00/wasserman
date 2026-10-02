"""Episode-level publication checks; assisted data never counts as actor success."""

import re


def publication_version(value):
    if not re.fullmatch(r"v[1-9][0-9]*", value):
        raise ValueError("Expected a version such as v4")
    return value


def summarize_episode_splits(trains, validations, evaluations, fit):
    groups = (trains, validations, evaluations)
    if not all(groups):
        raise ValueError("All episode splits are required")
    seeds = [{episode["seed"] for episode in group} for group in groups]
    if any(len(group) != len(ids) for group, ids in zip(groups, seeds, strict=True)):
        raise ValueError("Duplicate episode seed within split")
    if any(seeds[i] & seeds[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Episode split leakage")
    if seeds[0] != set(fit["train_seeds"]) or seeds[1] != set(fit["validation_seeds"]):
        raise ValueError("Report omitted fitted training or validation episodes")
    for episode in evaluations:
        if episode["mode"] != "evaluate" or episode.get("teacher_probability", 0) != 0:
            raise ValueError("Evaluation must be standalone, without teacher actions")
        if episode["checkpoint_sha256"] != fit["checkpoint_sha256"]:
            raise ValueError("Evaluated and fitted checkpoints differ")
        for field in ("success_per_env", "terminal_per_env", "censored_per_env"):
            if len(episode[field]) != episode["num_envs"]:
                raise ValueError("Invalid per-episode outcomes")
        if sum(episode["success_per_env"]) != episode["successes"]:
            raise ValueError("Success count disagrees with episode outcomes")
        replay = episode.get("contract_replay")
        if replay is not None and replay["successes"] != episode["successes"]:
            raise ValueError("Measured contact replay disagrees with reported successes")
        if (
            replay is not None
            and "first_success_step" in replay
            and [step >= 0 for step in replay["first_success_step"]] != episode["success_per_env"]
        ):
            raise ValueError("Contact replay disagrees with per-environment outcomes")
        first = evaluations[0]
        if episode.get("boundary_effects_enabled", False) != first.get("boundary_effects_enabled", False):
            raise ValueError("Do not aggregate different boundary-effect conditions")
        for key in ("conditions", "camera_appearance", "image_size", "sample_every", "dt_s"):
            if episode.get(key) != first.get(key):
                raise ValueError("Do not aggregate different evaluation conditions")
    return {
        "validation_episodes": sum(e["num_envs"] for e in validations),
        "validation_successes": sum(e["successes"] for e in validations),
        "validation_batches": [
            {key: e[key] for key in ("seed", "mode", "num_envs", "successes")} for e in validations
        ],
        "evaluation_episodes": sum(e["num_envs"] for e in evaluations),
        "evaluation_successes": sum(e["successes"] for e in evaluations),
        "evaluation_seed": evaluations[0]["seed"],  # Backward-compatible single-batch accessor.
        "evaluation_seeds": sorted(seeds[2]),
        "evaluation_steps": max(e["steps"] for e in evaluations),
        "evaluation_steps_range": [min(e["steps"] for e in evaluations), max(e["steps"] for e in evaluations)],
        "evaluation_batches": [
            {key: e[key] for key in ("seed", "num_envs", "successes", "steps")} for e in evaluations
        ],
        **{
            f"evaluation_{field}": [value for e in evaluations for value in e[field]]
            for field in ("terminal_per_env", "censored_per_env")
        },
    }
