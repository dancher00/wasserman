# ruff: noqa: E402
"""Train pinned LeRobot ACT on successful wrist-RGB / measured-relative EE episodes."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".deps/valve-policy-deps"), str(ROOT / ".deps/lerobot-a16f34c/src")]

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from torch.utils.data import DataLoader

from wasman.learning.valve_visual_dataset import ValveVisualDataset, normalize_act_batch, successful_episodes


def make_actor(size=384, pretrained=True):
    return ACTPolicy(
        ACTConfig(
            input_features={
                "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, size, size)),
                "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
            chunk_size=16,
            n_action_steps=16,
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" if pretrained else None,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--state-mode", choices=("local", "absolute"), default="local")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error("Fresh output directory required")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    episodes = successful_episodes(args.data)
    if len(episodes) < args.episodes:
        raise ValueError(f"Need {args.episodes} successful episodes, have {len(episodes)}")
    episodes = episodes[: args.episodes]
    dataset = ValveVisualDataset(episodes, state_mode=args.state_mode)
    stats = dataset.statistics()
    size = dataset.trajectories[0]["meta"]["image_shape"][0]
    actor = make_actor(size).cuda()
    optimizer = torch.optim.AdamW(actor.get_optim_params(), lr=1e-5, weight_decay=1e-4)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    args.output_dir.mkdir(parents=True)
    seeds = [t["meta"]["seed"] for t in dataset.trajectories]
    provenance = dict(
        model="ACT",
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=1e-5,
        weight_decay=1e-4,
        kl_weight=10,
        chunk_size=16,
        image_size=size,
        policy_hz=20,
        state_mode=args.state_mode,
        train_seeds=seeds,
        validation_seeds=[],
        dataset_episodes=[str(x.resolve()) for x in episodes],
        seed=args.seed,
        lerobot_commit="a16f34c085c9597fcbdb9fde395a3334d78df716",
        source_sha256={
            str(x): hashlib.sha256(x.read_bytes()).hexdigest()
            for x in [Path(__file__), ROOT / "src/wasman/learning/valve_visual_dataset.py"]
        },
        mixed_precision="bfloat16",
        normalization="frozen training statistics + ImageNet RGB",
        inference="16 actions at 20 Hz, relative to one measured anchor; shared IK at 30 Hz",
        success_is_not_established_by_training=True,
    )
    (args.output_dir / "config.json").write_text(json.dumps(provenance, indent=2) + "\n")
    torch.save(
        dict(state_dict=actor.state_dict(), statistics=stats, config=provenance, step=0), args.output_dir / "model_0.pt"
    )
    iterator = iter(loader)
    start = time.perf_counter()
    with (args.output_dir / "metrics.jsonl").open("w", buffering=1) as log:
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = normalize_act_batch(batch, stats, "cuda")
            actor.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = actor(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 10)
            if not torch.isfinite(grad):
                raise RuntimeError(f"Nonfinite gradient at step {step}")
            optimizer.step()
            if step == 1 or step % 100 == 0:
                entry = dict(
                    step=step,
                    loss=float(loss.detach()),
                    grad_norm=float(grad),
                    **metrics,
                    elapsed_s=time.perf_counter() - start,
                    gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                )
                log.write(json.dumps(entry) + "\n")
                print(json.dumps(entry), flush=True)
            if step % 5000 == 0 or step == args.steps:
                checkpoint = dict(
                    state_dict=actor.state_dict(),
                    statistics=stats,
                    config=provenance,
                    step=step,
                    optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                )
                target = args.output_dir / f"model_{step}.pt"
                torch.save(checkpoint, target.with_suffix(".tmp"))
                target.with_suffix(".tmp").rename(target)
    (args.output_dir / "completed.json").write_text(
        json.dumps(dict(steps=args.steps, elapsed_s=time.perf_counter() - start))
    )


if __name__ == "__main__":
    main()
