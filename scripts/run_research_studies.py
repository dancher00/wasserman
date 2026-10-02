"""Finite two-task interface and data-budget studies; repeated training cancelled by user."""

import time
from pathlib import Path

from research_campaign_runtime import BASE, ROOT, read, run, select, sha, write


def evaluate(task, model, selected, folder, seeds, interface, label):
    config = read(Path(selected["checkpoint"]).parent / "config.json")
    if config["task"] != task or config["model"] != model.upper() or config.get("interface", "actuator") != interface:
        raise ValueError("Selected checkpoint configuration differs")
    receipt = folder.with_name(folder.name + "_selection.json")
    if receipt.exists():
        if read(receipt)["selected"] != selected:
            raise ValueError("Previously frozen evaluation selection differs")
    else:
        write(receipt, dict(selected=selected, frozen_unix=time.time(), config=config))
    run(
        label,
        [
            "scripts/rollout_research_visual.py",
            "--task",
            task,
            "--mode",
            "policy",
            "--purpose",
            "research",
            "--interface",
            interface,
            "--checkpoint",
            selected["checkpoint"],
            "--seeds",
            *seeds,
            "--output-dir",
            folder,
        ],
        folder / "report.json",
    )
    report = read(folder / "report.json")
    if report["checkpoint_sha256"] != selected["checkpoint_sha256"] or report["expert_at_inference"]:
        raise ValueError("Checkpoint/inference provenance mismatch")
    return str(folder)


def main():
    records = []
    for task in ["PushSlider", "PullLever"]:
        source = ROOT / "artifacts/marine_mechanisms_20260924" / task
        read(source / "completed.json")
        original = read(source / "dataset_audit.json")
        # This view may already have been produced for a declared development pilot.
        ee = BASE / "datasets" / f"{task}_ee80"
        if not (ee / "dataset_audit.json").exists():
            run(
                f"{task}_prepare_ee",
                [
                    "scripts/prepare_research_dataset.py",
                    "--source",
                    source,
                    "--output-dir",
                    ee,
                    "--episodes",
                    80,
                    "--interface",
                    "ee",
                ],
                ee / "dataset_audit.json",
            )
        else:
            view = read(ee / "dataset_audit.json")
            if (
                view["source_audit_sha256"] != sha(source / "dataset_audit.json")
                or view["selected_seeds"] != original["selected_seeds"]
            ):
                raise ValueError("Pre-existing EE view differs")
        seeds = original["selected_seeds"][:8]
        native_folder = BASE / "interface_replay" / task / "actuator_paired8"
        run(
            f"{task}_actuator_replay8",
            [
                "scripts/rollout_research_visual.py",
                "--task",
                task,
                "--mode",
                "replay",
                "--interface",
                "actuator",
                "--seeds",
                *seeds,
                "--replay-dataset",
                source,
                "--output-dir",
                native_folder,
            ],
            native_folder / "report.json",
        )
        if read(native_folder / "report.json")["success_per_seed"] != [True] * 8:
            raise ValueError("Actuator replay failed; interface comparison blocked")
        folder = BASE / "interface_replay" / task / "ee_paired8"
        run(
            f"{task}_ee_replay8",
            [
                "scripts/rollout_research_visual.py",
                "--task",
                task,
                "--mode",
                "replay",
                "--interface",
                "ee",
                "--seeds",
                *seeds,
                "--replay-dataset",
                ee,
                "--output-dir",
                folder,
            ],
            folder / "report.json",
        )
        report = read(folder / "report.json")
        if report["success_per_seed"] != [True] * 8:
            raise ValueError("EE replay failed; preserve diagnosis before training")
        write(
            ee / "ee_replay_gate.json",
            dict(
                passed=True,
                source_audit_sha256=sha(ee / "dataset_audit.json"),
                seeds=seeds,
                report=str(folder / "report.json"),
                report_sha256=sha(folder / "report.json"),
            ),
        )
        for model in ["act", "dp"]:
            selected = read(source / f"selection_{model}.json")["selected"]
            path = evaluate(
                task,
                model,
                selected,
                BASE / "interface" / task / f"actuator_{model}",
                range(10100, 10130),
                "actuator",
                f"{task}_interface_actuator_{model}",
            )
            records.append(
                dict(
                    study="interface",
                    task=task,
                    model=model.upper(),
                    interface="actuator",
                    episodes=80,
                    training_seed=42,
                    path=path,
                )
            )
            logs = ROOT / f"logs/research_{task.lower()}_ee_{model}_s42"
            work = BASE / "interface" / task / f"ee_{model}"
            run(
                f"{task}_train_ee_{model}",
                [f"scripts/train_research_{model}.py", "--task", task, "--data", ee, "--output-dir", logs]
                + (["--microbatch", 16] if model == "dp" else []),
                logs / "completed.json",
            )
            selected = select(task, model, work, logs, interface="ee", label=f"{task}_ee")
            path = evaluate(
                task, model, selected, work / "test", range(10100, 10130), "ee", f"{task}_interface_ee_{model}"
            )
            records.append(
                dict(
                    study="interface",
                    task=task,
                    model=model.upper(),
                    interface="ee",
                    episodes=80,
                    training_seed=42,
                    path=path,
                )
            )
        for count in [10, 20, 40]:
            view = BASE / "datasets" / f"{task}_actuator{count}"
            run(
                f"{task}_prepare_n{count}",
                ["scripts/prepare_research_dataset.py", "--source", source, "--output-dir", view, "--episodes", count],
                view / "dataset_audit.json",
            )
            logs = ROOT / f"logs/research_{task.lower()}_dp_n{count}_s42"
            work = BASE / "scaling" / task / f"n{count}"
            run(
                f"{task}_train_dp_n{count}",
                [
                    "scripts/train_research_dp.py",
                    "--task",
                    task,
                    "--data",
                    view,
                    "--episodes",
                    count,
                    "--microbatch",
                    16,
                    "--output-dir",
                    logs,
                ],
                logs / "completed.json",
            )
            selected = select(task, "dp", work, logs, label=f"{task}_n{count}")
            path = evaluate(
                task, "dp", selected, work / "test", range(10200, 10230), "actuator", f"{task}_scaling_n{count}"
            )
            records.append(
                dict(
                    study="scaling",
                    task=task,
                    model="DP",
                    interface="actuator",
                    episodes=count,
                    training_seed=42,
                    path=path,
                )
            )
        # Reuse the original seed42 DP as the80-demo endpoint on the SAME fresh scaling resets.
        selected = read(source / "selection_dp.json")["selected"]
        work = BASE / "scaling" / task / "n80"
        path = evaluate(task, "dp", selected, work / "test", range(10200, 10230), "actuator", f"{task}_scaling_n80")
        records.append(
            dict(study="scaling", task=task, model="DP", interface="actuator", episodes=80, training_seed=42, path=path)
        )
        write(BASE / "study_records.json", dict(complete=False, records=records))
    write(BASE / "study_records.json", dict(complete=True, records=records, completed_unix=time.time()))
    run("publish_research_studies", ["scripts/publish_research_studies.py"], BASE / "studies_published.json")


if __name__ == "__main__":
    main()
