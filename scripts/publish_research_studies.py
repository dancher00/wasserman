"""Recompute every study score from raw physical traces, then publish local plot data."""

import numpy as np
from research_campaign_runtime import BASE, ROOT, read, sha, write
from scipy.stats import binomtest

from wasman.controllers.marine_evidence import verify


def main():
    registry = read(BASE / "study_records.json")
    if not registry["complete"] or len(registry["records"]) != 16:
        raise ValueError("Study matrix incomplete")
    results = []
    initials = {}
    for record in registry["records"]:
        from pathlib import Path

        path = Path(record["path"])
        report = read(path / "report.json")
        selection = read(path.with_name(path.name + "_selection.json"))
        config = selection["config"]
        if (
            selection["selected"]["checkpoint_sha256"] != report["checkpoint_sha256"]
            or selection["frozen_unix"] > (path / "source_manifest.json").stat().st_mtime
            or config["seed"] != record["training_seed"]
            or config["model"] != record["model"]
            or config.get("interface", "actuator") != record["interface"]
            or len(config["dataset_episodes"]) != record["episodes"]
        ):
            raise ValueError("Selection/configuration provenance mismatch")
        criteria = read(path / "contract.json")
        d = dict(np.load(path / "trace.npz"))
        v = verify(d, criteria)
        if v["success_per_seed"] != report["success_per_seed"] or report["expert_at_inference"]:
            raise ValueError("Physical result mismatch")
        start = 10100 if record["study"] == "interface" else 10200
        if report["seeds"] != list(range(start, start + 30)):
            raise ValueError("Wrong final seeds")
        key = (record["task"], start)
        initial = d["measured"][0]
        if key in initials and not np.array_equal(initials[key], initial):
            raise ValueError("Unpaired initial measured states")
        initials[key] = initial
        k = sum(v["success_per_seed"])
        ci = binomtest(k, 30).proportion_ci(method="exact")
        times = np.asarray(v["first_success_step"])
        times = times[times >= 0] * report["dt"]
        results.append(
            dict(
                **record,
                num_demos=record["episodes"],
                training_demos=len(config["train_seeds"]),
                offline_holdout_demos=len(config.get("validation_seeds", [])),
                test_episodes=30,
                successes=k,
                success_percent=k / 30 * 100,
                ci95=[ci.low * 100, ci.high * 100],
                median_success_time_s=float(np.median(times)) if len(times) else None,
                trace_sha256=sha(path / "trace.npz"),
                checkpoint_sha256=report["checkpoint_sha256"],
            )
        )
    value = dict(
        complete=True,
        training_repetitions_per_configuration=1,
        records=results,
        scope=(
            "Two marine tasks; one training seed per configuration. "
            "Intervals describe reset variation, not training variability."
        ),
        protocol="docs/research-core-protocol-20260924.md",
    )
    write(ROOT / "website/public/static/research-core-studies.json", value)
    write(BASE / "studies_published.json", value)


if __name__ == "__main__":
    main()
