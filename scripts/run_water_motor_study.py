"""Six predeclared one-factor conditions; frozen DP; new paired reset seeds."""

from research_campaign_runtime import BASE, ROOT, run, sha, write


def main():
    rows = []
    for task, stem in [("RotateValve", "valve"), ("OpenHatch", "hatch")]:
        checkpoint = ROOT / f"logs/{stem}_dp_v1/model_epoch_40.pt"
        for condition in ["reference", "current_y_010", "current_y_020", "current_y_m020", "motor_080", "motor_060"]:
            out = BASE / "water_motor" / task / condition
            run(
                f"{task}_{condition}",
                [
                    "scripts/evaluate_water_motor_study.py",
                    "--task",
                    task,
                    "--condition",
                    condition,
                    "--checkpoint",
                    checkpoint,
                    "--seeds",
                    *range(10000, 10030),
                    "--output-dir",
                    out,
                ],
                out / "summary.json",
            )
            rows.append(dict(task=task, condition=condition, path=str(out), checkpoint_sha256=sha(checkpoint)))
            write(BASE / "water_motor_records.json", dict(complete=False, records=rows))
    write(BASE / "water_motor_records.json", dict(complete=True, records=rows))
    run("publish_water_motor", ["scripts/publish_water_motor_study.py"], BASE / "water_motor_published.json")


if __name__ == "__main__":
    main()
