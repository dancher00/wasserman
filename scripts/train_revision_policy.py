# ruff: noqa: E402
"""Fixed-split, fixed-sample-budget ACT/DP training for the separately versioned open core."""

import argparse
import copy
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".deps/valve-policy-deps"), str(ROOT / ".deps/lerobot-a16f34c/src")]

import numpy as np
import torch
from torch.utils.data import DataLoader

from wasman.assets.open_geometry import PROFILE, selected_profile
from wasman.learning.revision_protocol import (
    PROTOCOL,
    TASKS,
    install_dp_transform,
    sha256,
    split_episodes,
    validate_runtime_snapshot,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--model", choices=["ACT", "DP", "BC"], required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--samples", type=int, default=320000)
    p.add_argument("--microbatch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pilot", action="store_true", help="Unscored capacity smoke; never a primary result")
    p.add_argument("--interface", choices=["actuator", "ee"])
    p.add_argument("--resume", action="store_true", help="Continue this output directory's own last.pt")
    p.add_argument("--stop-after-updates", type=int, help="Pilot-only interruption/recovery check")
    args = p.parse_args()
    interface = args.interface or ("ee" if args.task == "RotateValve" else "actuator")
    if interface == "ee" and args.task not in ("RotateValve", "PushSlider", "PullLever"):
        p.error("No registered EE interface for this task")
    if interface == "actuator" and args.task == "RotateValve":
        p.error("RotateValve uses EE commands")
    batch_size = 32 if args.model in ("ACT", "BC") else 64
    if (
        args.output_dir.exists() != args.resume
        or args.samples <= 0
        or args.samples % batch_size
        or batch_size % args.microbatch
    ):
        p.error("Fresh output, positive full-batch sample budget, and divisible microbatch required")
    if args.stop_after_updates is not None and (not args.pilot or args.stop_after_updates <= 0):
        p.error("Positive --stop-after-updates is available only for engineering pilots")
    if (args.output_dir / "completed.json").exists():
        p.error("Completed runs are immutable")
    if selected_profile() != PROFILE:
        p.error("Set WASMAN_ASSET_PROFILE=open-procedural-v1 explicitly")
    if not args.pilot and args.samples != 320000:
        p.error("Primary budget is fixed; use --pilot for unscored engineering checks")
    runtime_hash = validate_runtime_snapshot() if not args.pilot else None
    torch.set_num_threads(8)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    stem = {
        "PressButton": "button",
        "RotateValve": "valve",
        "OpenHatch": "hatch",
        "CollectShell": "shell",
        "PushSlider": "marine",
        "PullLever": "marine",
    }[args.task]
    data_stem = "marine" if stem == "button" else stem
    data_module = importlib.import_module(f"wasman.learning.{data_stem}_visual_dataset")
    episodes = data_module.successful_episodes(args.data)[:80]
    if len(episodes) != 80 and not args.pilot:
        raise ValueError("Exactly 80 successful, audited episodes required")
    if any(json.loads((x / "metadata.json").read_text()).get("asset_profile") != PROFILE for x in episodes):
        raise ValueError("Dataset is not explicitly bound to open-procedural-v1")
    if args.task in ("PushSlider", "PullLever"):
        if any(
            json.loads((x / "metadata.json").read_text()).get("interface", "actuator") != interface for x in episodes
        ):
            raise ValueError("Dataset action interface mismatch")
        if interface == "ee":
            gate = json.loads((args.data / "ee_replay_gate.json").read_text())
            if not gate["passed"] or gate["asset_profile"] != PROFILE:
                raise ValueError("Paired native/EE replay gate required")
    audit = args.data / "revision_audit.json"
    audit_data = json.loads(audit.read_text())
    if not audit_data["passed"] or audit_data["task"] != args.task or (audit_data.get("pilot") and not args.pilot):
        raise ValueError("Successful independent dataset audit required")
    audited = {row["seed"]: row for row in audit_data["episodes"]}
    selected_seeds = [json.loads((x / "metadata.json").read_text())["seed"] for x in episodes]
    if set(selected_seeds) != set(audited):
        raise ValueError("Selected episodes differ from independent audit")
    for episode, seed in zip(episodes, selected_seeds, strict=True):
        for name, digest in audited[seed]["files"].items():
            if sha256(episode / name) != digest:
                raise ValueError(f"Dataset changed after audit: {episode / name}")
    if args.task in ("PushSlider", "PullLever") and interface == "ee":
        if gate["dataset_audit_sha256"] != sha256(audit):
            raise ValueError("Replay gate refers to a different dataset")
        if gate["seeds"] != selected_seeds[:8] or len(gate["seeds"]) != 8:
            raise ValueError("Replay gate must use the first eight selected demonstrations")
        view = json.loads((args.data / "view.json").read_text())
        if gate["native_audit_sha256"] != view["source_audit_sha256"]:
            raise ValueError("Replay gate native source mismatch")
        for replay_interface in ("actuator", "ee"):
            record = gate["reports"][replay_interface]
            report_path = args.data / record["path"]
            if sha256(report_path) != record["sha256"]:
                raise ValueError("Replay report changed")
            report = json.loads(report_path.read_text())
            if (
                report["task"] != args.task
                or report["interface"] != replay_interface
                or report["mode"] != "replay"
                or report["seeds"] != gate["seeds"]
                or report["success_per_seed"] != [True] * 8
                or not report["contract_replayed"]
            ):
                raise ValueError("Invalid paired replay report")
    train_paths, val_paths = split_episodes(episodes)
    cls = getattr(data_module, data_stem.title() + "VisualDataset")
    state_mode = "local" if stem == "valve" else "absolute"
    clock_kwargs = {"policy_hz": 30} if stem == "valve" else {}
    stats = None
    if args.model in ("ACT", "BC"):
        ds = cls(train_paths, state_mode=state_mode, **clock_kwargs)
        ds.items = [(e, i) for e, i in ds.items if not ds.trajectories[e]["pad"][i].any()]
        if args.model == "BC":
            from wasman.learning.chunk_bc import make_bc

            actor = make_bc(args.task, pretrained=True).cuda()
        else:
            actor = importlib.import_module(f"train_{stem}_act").make_actor().cuda()
        stats = ds.statistics()
        optimizer = torch.optim.AdamW(actor.get_optim_params(), lr=1e-5, weight_decay=1e-4)
        ema = None
    else:
        dp_module = importlib.import_module("button_dp" if stem == "button" else f"wasman.learning.{stem}_dp")
        dp_cls = getattr(dp_module, data_stem.title() + "DPDataset")
        ds = dp_cls(train_paths, augment_start=False, **clock_kwargs)
        actor = dp_module.make_dp()
        install_dp_transform(actor)
        actor.set_normalizer(ds.normalizer())
        actor.cuda()
        from diffusion_policy.model.diffusion.ema_model import EMAModel

        ema = EMAModel(copy.deepcopy(actor), power=0.75)
        optimizer = torch.optim.AdamW(
            [
                {"params": actor.model.parameters(), "lr": 3e-4, "base_lr": 3e-4},
                {"params": actor.obs_encoder.parameters(), "lr": 3e-5, "base_lr": 3e-5},
            ],
            betas=(0.95, 0.999),
            eps=1e-8,
            weight_decay=1e-6,
        )
    if len(ds) < batch_size:
        raise ValueError("Insufficient valid training windows")
    config = dict(
        protocol=PROTOCOL,
        runtime_manifest_sha256=runtime_hash,
        task=args.task,
        model=args.model,
        seed=args.seed,
        rollout_seed=42,
        asset_profile=PROFILE,
        pilot=args.pilot,
        sample_budget=args.samples,
        batch_size=batch_size,
        microbatch=args.microbatch,
        workers=args.workers,
        start_orientation_augmentation=False,
        split_seed=20260929,
        policy_hz=20 if stem == "hatch" else 30,
        image_size=384,
        state_mode=state_mode,
        n_action_steps=8,
        chunk_size=16,
        inference_precision="fp32",
        image_pipeline="clip-v2" if args.model == "DP" else "imagenet",
        checkpoint_selection="fixed-final-budget",
        interface=interface,
        train_seeds=[json.loads((x / "metadata.json").read_text())["seed"] for x in train_paths],
        validation_seeds=[json.loads((x / "metadata.json").read_text())["seed"] for x in val_paths],
        dataset_audit_sha256=sha256(audit),
        training_windows=len(ds),
        source_sha256={str(Path(__file__).relative_to(ROOT)): sha256(__file__)},
        protocol_sha256=sha256(ROOT / "docs/revision-v2-protocol.md"),
        parameters=sum(p.numel() for p in actor.parameters()),
        mixed_precision="bfloat16",
    )
    first_step, epoch, batch_offset, previous_elapsed = 1, 0, 0, 0.0
    if args.resume:
        # Only load the local run's own optimizer/RNG state, never a downloaded checkpoint.
        saved = torch.load(args.output_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("Resume configuration, dataset, protocol or trainer source changed")
        actor.load_state_dict(saved["training_state_dict"])
        # DP's dynamically reconstructed normalizer parameters follow the loaded
        # checkpoint device, so move the complete module after loading.
        actor.cuda()
        optimizer.load_state_dict(saved["optimizer"])
        if ema:
            ema.averaged_model.load_state_dict(saved["state_dict"])
            ema.averaged_model.cuda()
            ema.optimization_step = saved["ema_step"]
        first_step = saved["step"] + 1
        epoch, batch_offset = saved["epoch"], saved["batch_offset"]
        previous_elapsed = saved["elapsed_s"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        del saved
    else:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def epoch_iterator(number):
        # Dataset reads are deterministic; all random augmentation is in the model.
        # A separate generator makes sample order reproducible after interruption.
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(args.seed + number * 1000003),
        )
        return iter(loader)

    iterator = epoch_iterator(epoch)
    for _ in range(batch_offset):
        next(iterator)
    updates = args.samples // batch_size
    start = time.perf_counter()
    with (args.output_dir / "metrics.jsonl").open("a" if args.resume else "w", buffering=1) as log:
        for step in range(first_step, updates + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                batch_offset = 0
                iterator = epoch_iterator(epoch)
                batch = next(iterator)
            batch_offset += 1
            actor.train()
            optimizer.zero_grad(set_to_none=True)
            if args.model == "DP":
                warmup = max(1, round(updates * 0.2))
                factor = (
                    min(step / warmup, 1)
                    if step <= warmup
                    else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, updates - warmup)))
                )
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * factor
            loss_sum = 0.0
            for lo in range(0, batch_size, args.microbatch):
                if args.model in ("ACT", "BC"):
                    micro = data_module.normalize_act_batch(
                        {k: v[lo : lo + args.microbatch] for k, v in batch.items()}, stats, "cuda"
                    )
                else:
                    micro = {
                        "obs": {
                            k: v[lo : lo + args.microbatch].cuda(non_blocking=True) for k, v in batch["obs"].items()
                        },
                        "action": batch["action"][lo : lo + args.microbatch].cuda(non_blocking=True),
                    }
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = actor(micro)[0] if args.model in ("ACT", "BC") else actor.compute_loss(micro)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at {step}")
                (loss * args.microbatch / batch_size).backward()
                loss_sum += float(loss.detach()) * args.microbatch / batch_size
            if args.model in ("ACT", "BC"):
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 10, error_if_nonfinite=True)
            else:
                torch.nn.utils.get_total_norm(
                    [p.grad for p in actor.parameters() if p.grad is not None], error_if_nonfinite=True
                )
            optimizer.step()
            if ema:
                ema.step(actor)
            if step == 1 or step % 50 == 0 or step == updates:
                record = dict(
                    step=step,
                    samples=step * batch_size,
                    loss=loss_sum,
                    elapsed_s=previous_elapsed + time.perf_counter() - start,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                )
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            stopping = step == args.stop_after_updates
            if step % 1000 == 0 or step == updates or stopping:
                state = (ema.averaged_model if ema else actor).state_dict()
                payload = dict(
                    state_dict=state,
                    statistics=stats,
                    config=config,
                    step=step,
                    optimizer=optimizer.state_dict(),
                    training_state_dict=actor.state_dict(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(),
                    ema_step=ema.optimization_step if ema else None,
                    epoch=epoch,
                    batch_offset=batch_offset,
                    elapsed_s=previous_elapsed + time.perf_counter() - start,
                )
                temporary = args.output_dir / "last.tmp"
                torch.save(payload, temporary)
                temporary.replace(args.output_dir / "last.pt")
            if stopping and step < updates:
                return
        if first_step > updates:
            state = (ema.averaged_model if ema else actor).state_dict()
            step = updates
        torch.save(dict(state_dict=state, statistics=stats, config=config, step=step), args.output_dir / "final.pt")
    (args.output_dir / "completed.json").write_text(
        json.dumps(
            dict(
                updates=updates,
                samples=updates * batch_size,
                elapsed_s=previous_elapsed + time.perf_counter() - start,
                final_sha256=sha256(args.output_dir / "final.pt"),
                pilot=args.pilot,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
