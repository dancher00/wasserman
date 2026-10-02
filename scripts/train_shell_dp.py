# ruff: noqa: E402
"""Train the pinned AM-Bench CLIP/UNet DP on successful CollectShell episodes."""

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wasman.learning.shell_dp import ROOT, ShellDPDataset, make_dp
from wasman.learning.shell_visual_dataset import successful_episodes


def main():
    # shell_dp above establishes the isolated, pinned upstream import path.
    from diffusion_policy.model.diffusion.ema_model import EMAModel

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--max-updates", type=int)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output_dir.exists() or args.batch_size % args.microbatch:
        p.error("Fresh directory; batch size must be divisible by microbatch")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    episodes = successful_episodes(args.data)
    if len(episodes) < args.episodes:
        raise ValueError("Not enough successful demonstrations")
    episodes = episodes[: args.episodes]
    order = np.random.default_rng(args.seed).permutation(len(episodes))
    val_ids = set(order[: max(1, round(len(episodes) * 0.05))].tolist())
    train = ShellDPDataset([x for i, x in enumerate(episodes) if i not in val_ids])
    val = ShellDPDataset([x for i, x in enumerate(episodes) if i in val_ids], augment_start=False)
    model = make_dp()
    model.set_normalizer(train.normalizer())
    model.cuda()
    ema = EMAModel(copy.deepcopy(model), power=0.75)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.model.parameters(), "base_lr": 3e-4},
            {"params": model.obs_encoder.parameters(), "lr": 3e-5, "base_lr": 3e-5},
        ],
        lr=3e-4,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-6,
    )
    # The effective batch matches 64; accumulation keeps CLIP training within 16 GB.
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    val_loader = DataLoader(val, batch_size=args.microbatch, shuffle=False, num_workers=2, pin_memory=True)
    total_updates = args.epochs * len(loader)
    args.output_dir.mkdir(parents=True)
    config = dict(
        task="CollectShell",
        model="DP",
        epochs=args.epochs,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        seed=args.seed,
        learning_rate=3e-4,
        pretrained_encoder_learning_rate=3e-5,
        warmup_updates=2000,
        scheduler="cosine",
        policy_hz=30,
        image_size=384,
        network_image_size=224,
        chunk_size=16,
        n_action_steps=8,
        observation_steps=2,
        ddim_steps=16,
        train_diffusion_steps=50,
        mixed_precision="bfloat16",
        ema_power=0.75,
        train_seeds=[t["meta"]["seed"] for t in train.trajectories],
        validation_seeds=[t["meta"]["seed"] for t in val.trajectories],
        dataset_episodes=[str(x.resolve()) for x in episodes],
        total_updates=total_updates,
        max_updates=args.max_updates,
        ambench_commit="60bf5b73041df4eab571f7d9f0a297aeecbf2e0d",
        clip_weights_sha256=hashlib.sha256(
            (ROOT / ".deps/valve-weights/clip_vit_b16_timm.bin").read_bytes()
        ).hexdigest(),
        source_sha256={
            str(x): hashlib.sha256(x.read_bytes()).hexdigest()
            for x in [
                Path(__file__),
                ROOT / "src/wasman/learning/shell_dp.py",
                ROOT / "src/wasman/learning/shell_visual_dataset.py",
            ]
        },
        deviations=[
            "30 Hz raw data and policy trajectory; matched20Hz replay failed before training",
            "gradient accumulation for effective batch 64",
            "bfloat16",
            "absolute base/joint/jaw actions instead of relative EE",
            "240 second task",
        ],
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    start, update = time.perf_counter(), 0
    with (args.output_dir / "metrics.jsonl").open("w", buffering=1) as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in loader:
                update += 1
                factor = (
                    min(update / 2000, 1.0)
                    if update < 2000
                    else 0.5 * (1 + math.cos(math.pi * (update - 2000) / max(total_updates - 2000, 1)))
                )
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * factor
                optimizer.zero_grad(set_to_none=True)
                losses = []
                actual_batch_size = len(batch["action"])
                for lo in range(0, actual_batch_size, args.microbatch):
                    micro = {
                        "obs": {
                            k: v[lo : lo + args.microbatch].cuda(non_blocking=True) for k, v in batch["obs"].items()
                        },
                        "action": batch["action"][lo : lo + args.microbatch].cuda(non_blocking=True),
                    }
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model.compute_loss(micro)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss {update}")
                    fraction = len(micro["action"]) / actual_batch_size
                    (loss * fraction).backward()
                    losses.append(float(loss.detach()) * fraction)
                # Upstream DP has no gradient clipping; fail instead of changing its optimizer silently.
                torch.nn.utils.get_total_norm(
                    [x.grad for x in model.parameters() if x.grad is not None], error_if_nonfinite=True, foreach=True
                )
                optimizer.step()
                ema.step(model)
                if update == 1 or update % 50 == 0:
                    entry = dict(
                        epoch=epoch,
                        update=update,
                        loss=float(np.sum(losses)),
                        lr=3e-4 * factor,
                        elapsed_s=time.perf_counter() - start,
                        gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                    )
                    log.write(json.dumps(entry) + "\n")
                    print(json.dumps(entry), flush=True)
                if args.max_updates and update >= args.max_updates:
                    break
            ema.averaged_model.eval()
            validation = []
            with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
                torch.manual_seed(12345)
                for batch in val_loader:
                    micro = {
                        "obs": {k: v.cuda(non_blocking=True) for k, v in batch["obs"].items()},
                        "action": batch["action"].cuda(non_blocking=True),
                    }
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        validation.append(float(ema.averaged_model.compute_loss(micro)))
                    if args.max_updates:
                        break
            entry = dict(
                epoch=epoch,
                update=update,
                validation_loss=float(np.mean(validation)),
                elapsed_s=time.perf_counter() - start,
            )
            log.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
            payload = dict(
                state_dict=ema.averaged_model.state_dict(),
                training_state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
                config=config,
                epoch=epoch,
                step=update,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
            )
            last = args.output_dir / "last.pt"
            torch.save(payload, last.with_suffix(".tmp"))
            last.with_suffix(".tmp").rename(last)
            if epoch % 10 == 0 or epoch == args.epochs or (args.max_updates and update >= args.max_updates):
                os.link(last, args.output_dir / f"model_epoch_{epoch}.pt")
            if args.max_updates and update >= args.max_updates:
                break
    (args.output_dir / "completed.json").write_text(
        json.dumps(dict(epoch=epoch, updates=update, elapsed_s=time.perf_counter() - start))
    )


if __name__ == "__main__":
    main()
