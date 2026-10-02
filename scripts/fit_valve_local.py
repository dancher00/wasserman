"""Build a local-regression imitation baseline from train-only expert episodes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from wasman.learning.valve_dataset import split_legacy_episodes, validate_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("logs/valve_incremental_dagger/dataset_round_1.pt"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--base-envs", type=int, default=64, help="Explicit environment count of the first pure-expert block"
    )
    p.add_argument("--neighbors", type=int, default=64)
    p.add_argument("--ridge", type=float, default=0.1)
    p.add_argument(
        "--regression-float64",
        action="store_true",
        help="Form and solve the local normal equations in double precision",
    )
    p.add_argument("--match-held", action="store_true")
    p.add_argument("--exclude-previous-commands", action="store_true")
    p.add_argument("--exclude-commands-after-stop", action="store_true")
    p.add_argument(
        "--anchor-at-stop", action="store_true", help="Learn post-stop references relative to a latched public command"
    )
    p.add_argument(
        "--learn-mode-router",
        action="store_true",
        help="Learn late-stage routing from physical observations, not previous commands",
    )
    p.add_argument("--progress-weight", type=float, default=1.0)
    p.add_argument("--learn-sequential-router", action="store_true")
    p.add_argument("--latch-release-references", action="store_true")
    p.add_argument("--stage-age-features", action="store_true")
    p.add_argument("--incremental-hold", action="store_true")
    p.add_argument("--sequential-merge-complete", action="store_true")
    p.add_argument(
        "--router-withdrawal-frames",
        type=int,
        default=0,
        help="Limit router training to the first N withdrawal frames per episode; zero retains all",
    )
    p.add_argument(
        "--absolute-targets",
        action="store_true",
        help="Regularize toward demonstrated targets rather than zero increments",
    )
    p.add_argument("--extra-dataset", type=Path, help="Append completed train episodes from last 128-env DAgger block")
    p.add_argument("--extra-held-only", action="store_true", help="Add corrections only after the measured hold gate")
    p.add_argument(
        "--extra-stopped-only", action="store_true", help="Use extra labels from teacher stopping and release stages"
    )
    p.add_argument("--fixed-reference-normalization", action="store_true")
    p.add_argument(
        "--post-stop-blend", type=float, default=1.0, help="Blend learned outputs after training-fitted stop threshold"
    )
    p.add_argument(
        "--learn-stop-partition",
        action="store_true",
        help="Fit a one-split history classifier for demonstrated stopping modes",
    )
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output")
    if args.sequential_merge_complete and not args.learn_sequential_router:
        raise ValueError("Completion merge requires sequential routing")
    if args.latch_release_references and not args.learn_sequential_router:
        raise ValueError("Release-reference latching requires a sequential router")
    if (args.stage_age_features or args.incremental_hold) and not args.learn_sequential_router:
        raise ValueError("Stage-conditioned features/targets require sequential routing")
    if args.incremental_hold and not args.absolute_targets:
        raise ValueError("Mixed hold increments require absolute references in other stages")
    if args.learn_sequential_router and (
        args.learn_mode_router
        or args.learn_stop_partition
        or args.extra_dataset
        or args.anchor_at_stop
        or args.exclude_commands_after_stop
    ):
        raise ValueError("Sequential routing requires a pure base dataset and replaces other routers")
    if args.exclude_previous_commands and args.anchor_at_stop:
        raise ValueError("Command exclusion and reference anchoring are separate experiments")
    if args.exclude_commands_after_stop and (
        not args.learn_stop_partition or args.exclude_previous_commands or args.anchor_at_stop
    ):
        raise ValueError("Late command exclusion requires a binary partition and no other command transform")
    if args.router_withdrawal_frames < 0 or (args.router_withdrawal_frames and not args.learn_mode_router):
        raise ValueError("Withdrawal-window fitting requires a mode router and a nonnegative frame count")
    if args.anchor_at_stop and (not args.absolute_targets or not args.learn_stop_partition or args.extra_dataset):
        raise ValueError("Anchoring requires absolute targets, binary partition, and a pure base dataset")
    if args.learn_mode_router and (args.learn_stop_partition or args.extra_dataset):
        raise ValueError("Mode routing requires a pure base dataset and replaces the binary stop partition")
    if (
        args.extra_held_only or args.extra_stopped_only or args.fixed_reference_normalization
    ) and not args.extra_dataset:
        raise ValueError("Extra-data options require --extra-dataset")
    torch.set_num_threads(4)
    d = torch.load(args.dataset, map_location="cpu", weights_only=True)
    validate_dataset(d)
    base_envs = args.base_envs
    block = 2249 * base_envs
    layout = d.get("collection_layout", {})
    if layout and not layout["has_retained_prefix"]:
        if layout["num_envs"] != base_envs or layout["steps_per_round"] != 2249:
            raise ValueError("Explicit base layout differs from collection metadata")
        if layout["rounds"] == 1 and d["last_round_failed_resets"].any():
            raise ValueError("Dataset contains resets; whole-episode layout is not valid")
    if args.extra_dataset and (base_envs != 64 or d.get("geometry", "legacy-v1") != "legacy-v1"):
        raise ValueError("Extra-DAgger layout is only supported for legacy64-env base data")
    train, val, episodes = split_legacy_episodes(
        block, num_envs=base_envs, steps_per_round=2249, validation_envs=max(1, base_envs // 8), seed=2064
    )
    completed = d["phases"][:block].reshape(2249, base_envs)[-1] == 7
    selected = train & completed[torch.arange(block) % base_envs]
    metadata = {
        "algorithm": "state-only local-linear nonparametric imitation; NOT PPO or a neural actor",
        "actor_class": "ValveLocalPolicy",
        "expert_at_inference": False,
        "geometry": d.get("geometry", "legacy-v1"),
        "expert_training_options": d.get("expert_options", {}),
        "dataset_last_round_controller": d.get("last_round_collection_controller", {}),
        "base_block_environments": base_envs,
        "observation_dim": 50,
        "action_dim": 11,
        "previous_commands_in_regression": not args.exclude_previous_commands,
        "previous_commands_excluded_after_stop": args.exclude_commands_after_stop,
        "release_reference_constraint": args.latch_release_references,
        "hold_reference_increments": args.incremental_hold,
        "regression_stage_age_feature": args.stage_age_features,
        "regression_target": (
            "hold-stage increments; absolute references in other stages"
            if args.incremental_hold
            else "absolute actuator reference"
            if args.absolute_targets
            else "reference increment"
        ),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "training_episode_ids": episodes[selected].unique().tolist(),
        "excluded_validation_episode_ids": episodes[val].unique().tolist(),
        "prototypes": int(selected.sum()),
        "rollout_success": None,
    }
    observations, targets = d["observations"][:block][selected], d["targets"][:block][selected]
    stopped_modes = (d["phases"][:block] >= 4)[selected]
    history_all = d["observations"][:block, 38].reshape(2249, base_envs).cummax(0).values.flatten()
    history = history_all[selected]
    learned_threshold = None
    if not (args.learn_mode_router or args.learn_sequential_router) and (
        args.learn_stop_partition or args.post_stop_blend != 1
    ):
        order = history.argsort()
        values, modes = history[order], stopped_modes[order].long()
        positives = modes.cumsum(0)
        negatives = torch.arange(1, len(modes) + 1) - positives
        # Below split predicts moving, above split predicts stopping. Choose
        # minimum training classification error, excluding equal-value splits.
        errors = positives[:-1] + (1 - modes).sum() - negatives[:-1]
        errors[values[:-1] == values[1:]] = len(modes)
        best = int(errors.argmin())
        learned_threshold = float((values[best] + values[best + 1]) / 2)
        metadata["learned_mode_partition"] = {
            "feature": "running maximum of public wheel angle / pi",
            "teacher_mode_label": "phase >= 4 (training only)",
            "threshold": learned_threshold,
            "training_error_fraction": float(errors[best] / len(modes)),
            "expert_at_inference": False,
        }
    if args.extra_dataset:
        extra = torch.load(args.extra_dataset, map_location="cpu", weights_only=True)
        validate_dataset(extra)
        extra_block = 2249 * 128
        if len(extra["observations"]) != 3 * block + extra_block:
            raise ValueError("Expected retained-three-block plus 128-env collection layout")
        etrain, eval_mask, eid = split_legacy_episodes(
            extra_block,
            num_envs=128,
            steps_per_round=2249,
            validation_envs=16,
            seed=2076,
        )
        finished = extra["phases"][-extra_block:].reshape(2249, 128)[-1] == 7
        esel = etrain & finished[torch.arange(extra_block) % 128]
        if args.extra_held_only:
            esel &= extra["observations"][-extra_block:, 48] > 0.5
        if args.extra_stopped_only:
            esel &= extra["phases"][-extra_block:] >= 4
        observations = torch.cat((observations, extra["observations"][-extra_block:][esel]))
        targets = torch.cat((targets, extra["targets"][-extra_block:][esel]))
        stopped_modes = torch.cat((stopped_modes, (extra["phases"][-extra_block:] >= 4)[esel]))
        metadata["extra_collection"] = {
            "dataset_sha256": hashlib.sha256(args.extra_dataset.read_bytes()).hexdigest(),
            "prefix_rows_skipped": 3 * block,
            "training_episode_ids": eid[esel].unique().tolist(),
            "excluded_validation_episode_ids": eid[eval_mask].unique().tolist(),
            "kind": "teacher labels from actual block-mixed states, not pure demonstrations",
            "measured_held_only": args.extra_held_only,
            "teacher_stopping_and_release_only": args.extra_stopped_only,
        }
        metadata["prototypes"] = len(observations)
    feature_normalization = None
    if args.anchor_at_stop:
        # Preserve the unanchored acquisition metric so this experiment changes
        # late references, not the approach policy's feature weighting.
        mean = observations.mean(0)
        scale = observations.std(0).clamp_min(0.05)
        weight = torch.ones(50)
        weight[6:12] = weight[19:23] = weight[41:44] = 0.25
        scale = scale / weight
        scale[[38, 47]] /= args.progress_weight
        feature_normalization = {"mean": mean, "scale": scale}
        sequence = d["observations"][:block].reshape(2249, base_envs, 50)
        crossed = history_all.reshape(2249, base_envs) >= learned_threshold
        first = crossed.long().argmax(0)
        anchors = sequence[first, torch.arange(base_envs), 27:38]
        offsets = (anchors[None] * crossed[..., None]).reshape(block, 11)[selected]
        observations = observations.clone()
        observations[:, 27:38] -= offsets
        targets = targets - offsets
        metadata["reference_anchor"] = "previous public command at first crossing of training-fitted angle threshold"
    checkpoint = {
        "format": "wasman-valve-local-v1",
        "infos": metadata,
        "parameters": {
            "neighbors": args.neighbors,
            "ridge": args.ridge,
            "regression_float64": args.regression_float64,
            "match_held": args.match_held,
            "progress_weight": args.progress_weight,
            "post_stop_blend": args.post_stop_blend,
            "absolute_targets": args.absolute_targets,
            "anchor_at_stop": args.anchor_at_stop,
            "exclude_previous_commands": args.exclude_previous_commands,
            "exclude_commands_after_stop": args.exclude_commands_after_stop,
            "latch_release_references": args.latch_release_references,
            "incremental_hold": args.incremental_hold,
        },
        "observations": observations,
        "targets": targets,
    }
    if args.fixed_reference_normalization:
        checkpoint["normalization_count"] = int(selected.sum())
        metadata["normalization_count"] = int(selected.sum())
    if feature_normalization is not None:
        checkpoint["feature_normalization"] = feature_normalization
    if args.learn_stop_partition:
        checkpoint["stopped_modes"] = stopped_modes
    if learned_threshold is not None:
        checkpoint["learned_stop_threshold"] = learned_threshold
    if args.learn_mode_router:
        from wasman.controllers.valve_mode_router import ValveModeRouter
        from wasman.learning.valve_mode_tree import fit_mode_tree

        features = torch.cat((d["observations"][:block, ValveModeRouter.features], history_all[:, None]), -1)
        modes = (d["phases"][:block] - 3).clamp(0, 3).long()
        router_selected = selected.clone()
        if args.router_withdrawal_frames:
            withdrawal_count = (modes.reshape(2249, base_envs) == 3).long().cumsum(0).flatten()
            router_selected &= withdrawal_count <= args.router_withdrawal_frames
        tree = fit_mode_tree(features[router_selected].numpy(), modes[router_selected].numpy())
        router = ValveModeRouter(tree)
        predictions = router.classify(features[val].double())
        confusion = torch.bincount(modes[val] * 4 + predictions, minlength=16).reshape(4, 4)
        checkpoint["mode_tree"] = tree
        checkpoint["prototype_modes"] = modes[selected]
        metadata["learned_mode_router"] = {
            "algorithm": tree["algorithm"],
            "features": list(ValveModeRouter.features) + ["running max wheel angle"],
            "previous_commands_excluded": True,
            "expert_phase_is_input": False,
            "labels": ["acquire/turn", "hold", "release", "withdraw/complete"],
            "max_depth": tree["max_depth"],
            "nodes": len(tree["left"]),
            "withdrawal_training_window_frames": args.router_withdrawal_frames,
            "training_rows": int(router_selected.sum()),
            "held_out_episode_mode_confusion": confusion.tolist(),
            "validation_role": "offline teacher-mode classification; not a task success rate",
            "inference_memory": "running maximum angle and monotone predicted mode",
        }
    if args.learn_sequential_router:
        from wasman.controllers.valve_sequential_router import ValveSequentialRouter, previous_stages_and_ages
        from wasman.learning.valve_mode_tree import fit_mode_tree

        phases = d["phases"][:block].reshape(2249, base_envs).long()
        if args.sequential_merge_complete:
            phases = phases.clamp(max=6)
        previous, ages = previous_stages_and_ages(phases)
        if args.stage_age_features:
            action_ages = torch.where(phases != previous, 0, ages + 1)
            checkpoint["prototype_stage_age"] = action_ages.flatten()[selected].float() / 30
        if args.incremental_hold:
            hold_rows = phases.flatten()[selected] == 4
            checkpoint["targets"] = targets.clone()
            checkpoint["targets"][hold_rows] -= observations[hold_rows, 27:38]
        features = torch.cat(
            (d["observations"][:block, ValveSequentialRouter.features], ages.flatten()[:, None] / 30), -1
        )
        trees = []
        for stage in range(6 if args.sequential_merge_complete else 7):
            rows = selected & (previous.flatten() == stage)
            labels = (phases.flatten()[rows] > stage).long()
            trees.append(fit_mode_tree(features[rows].numpy(), labels.numpy(), num_classes=2, max_depth=6, min_leaf=8))
        checkpoint["transition_trees"] = trees
        checkpoint["prototype_modes"] = phases.flatten()[selected]
        router = ValveSequentialRouter(trees)
        heldout_ids = episodes[val].unique()
        heldout = d["observations"][:block].reshape(2249, base_envs, 50)[:, heldout_ids]
        with torch.inference_mode():
            predictions = torch.stack([router(observation).clone() for observation in heldout])
        truth = phases[:, heldout_ids]
        confusion = torch.bincount((truth * 8 + predictions).flatten(), minlength=64).reshape(8, 8)
        metadata["learned_sequential_router"] = {
            "labels": ["standoff", "approach", "grasp", "turn", "hold", "release", "withdraw", "complete"],
            "merge_withdraw_and_complete": args.sequential_merge_complete,
            "features": list(ValveSequentialRouter.features) + ["elapsed predicted-stage seconds"],
            "expert_phase_is_input_at_inference": False,
            "previous_commands_excluded_from_router": True,
            "architecture_prior": "start at stage0; advance by at most one learned transition per action",
            "heldout_closed_loop_router_confusion_on_teacher_observations": confusion.tolist(),
            "validation_role": (
                "router memory is autoregressive, physical observations are teacher trajectories; "
                "NOT physical rollout success"
            ),
            "nodes_per_transition": [len(tree["left"]) for tree in trees],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
