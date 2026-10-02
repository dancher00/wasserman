"""Artifact-gated PressButton RGB baseline on the unchanged smooth T200 task."""

import time

from research_campaign_runtime import BASE, ROOT, read, run, select, sha, write


def main():
    root = BASE / "PressButton"
    gates = [
        root / "teacher_pilot/report.json",
        root / "rgb_pilot_v3/dataset_audit.json",
        root / "replay_pilot/report.json",
        root / "zero_pilot/report.json",
    ]
    if not all(p.exists() for p in gates):
        raise ValueError("Complete development gates first")
    if (
        not all(read(gates[0])["success_per_seed"])
        or not read(gates[1])["passed"]
        or not all(read(gates[2])["success_per_seed"])
        or any(read(gates[3])["success_per_seed"])
    ):
        raise ValueError("Development gate failure")
    write(
        root / "acceptance.json",
        dict(
            passed=True,
            gate_sha256={str(p): sha(p) for p in gates},
            protocol_sha256=sha(ROOT / "docs/research-core-protocol-20260924.md"),
        ),
    )
    for batch in range(13):
        successful = [p for p in root.glob("train_batch*/seed_*/metadata.json") if read(p)["outcome"] == "success"]
        if len(successful) >= 80:
            break
        out = root / f"train_batch{batch:02d}"
        seeds = list(range(9200 + batch * 8, min(9208 + batch * 8, 9300)))
        run(
            f"button_collect_{batch:02d}",
            ["scripts/rollout_button_visual.py", "--mode", "collect", "--seeds", *seeds, "--output-dir", out],
            out / "report.json",
        )
    run("button_dataset_audit", ["scripts/audit_button_dataset.py", "--data", root], root / "dataset_audit.json")
    audited = read(root / "dataset_audit.json")
    if not audited["passed"] or len(audited["selected_seeds"]) != 80:
        raise ValueError("Final dataset incomplete")
    for seed in audited["selected_seeds"][:2]:
        episode = next(root.glob(f"train_batch*/seed_{seed}"))
        out = root / f"replay_{seed}"
        run(
            f"button_replay_{seed}",
            [
                "scripts/rollout_button_visual.py",
                "--mode",
                "replay",
                "--seeds",
                seed,
                "--replay-episode",
                episode,
                "--output-dir",
                out,
            ],
            out / "report.json",
        )
        if not all(read(out / "report.json")["success_per_seed"]):
            raise ValueError("Lossless replay failed")
    for model in ["act", "dp"]:
        logs = ROOT / f"logs/button_rgb_{model}_v1"
        run(
            f"button_train_{model}",
            [f"scripts/train_button_{model}.py", "--task", "PressButton", "--data", root, "--output-dir", logs]
            + (["--microbatch", 16] if model == "dp" else []),
            logs / "completed.json",
        )
        selected = select("PressButton", model, root, logs, button=True, label="button")
        for purpose, seeds, video in [
            ("test", list(range(9110, 9140)) + [42], False),
            ("single42", [42], False),
            ("video", [42], True),
        ]:
            out = root / f"{purpose}_{model}"
            cmd = [
                "scripts/rollout_button_visual.py",
                "--mode",
                "policy",
                "--purpose",
                "test" if purpose == "test" else "single42",
                "--checkpoint",
                selected["checkpoint"],
                "--seeds",
                *seeds,
                "--output-dir",
                out,
            ]
            if video:
                cmd += ["--observer-video"]
                cmd = ["scripts/with_grounded_presentation.py", "--audit-path", out / "presentation.json", *cmd]
            run(f"button_{purpose}_{model}", cmd, out / "report.json")
    write(root / "completed.json", dict(complete=True, completed_unix=time.time()))
    run("button_publish", ["scripts/publish_button_visual.py"], root / "published.json")


if __name__ == "__main__":
    main()
