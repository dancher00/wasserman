"""Publish the separate RGB button baseline; preserve the historical state PPO."""

from pathlib import Path

import numpy as np
import publish_marine_benchmark as publication
from button_visual_support import verify
from scipy.stats import binomtest

publication.verify = verify
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/research_core_20260924/PressButton"
PUBLIC = ROOT / "website/public/static"


def main():
    publication.read(BASE / "completed.json")
    rows = []
    movies = []
    initials = []
    methods = []
    for key in ["act", "dp"]:
        report, contract, d = publication.verified_run(BASE / f"test_{key}")
        selection = publication.read(BASE / f"selection_{key}.json")
        if report["seeds"] != list(range(9110, 9140)) + [42] or report["expert_at_inference"]:
            raise ValueError("Wrong final split/controller")
        if selection["decision_unix"] > (BASE / f"test_{key}/source_manifest.json").stat().st_mtime:
            raise ValueError("Test influenced selection")
        if report["checkpoint_sha256"] != selection["selected"]["checkpoint_sha256"]:
            raise ValueError("Checkpoint mismatch")
        initials.append(d["measured"][0, :30])
        successes = np.asarray(report["success_per_seed"][:30])
        steps = np.asarray(report["first_success_step"][:30])
        k = int(successes.sum())
        ci = binomtest(k, 30).proportion_ci(method="exact")
        row = dict(
            task="PressButton",
            model=key.upper(),
            input="Wrist RGB + proprioception",
            successes=k,
            episodes=30,
            median_success_time_s=float(np.median(steps[successes] * report["dt"])) if k else None,
            exact_binomial_95_ci=[ci.low, ci.high],
        )
        rows.append(row)
        methods.append(
            dict(
                **row,
                checkpoint_sha256=report["checkpoint_sha256"],
                contract={k: (v if not isinstance(v, float) or np.isfinite(v) else None) for k, v in contract.items()},
                source_report=str(BASE / f"test_{key}/report.json"),
                trace_sha256=publication.sha(BASE / f"test_{key}/trace.npz"),
            )
        )
        presentation = publication.read(BASE / f"video_{key}/presentation.json")
        if (
            not presentation["physics_unchanged"]
            or not presentation["changes"]
            or presentation["physics_before_sha256"] != presentation["physics_after_sha256"]
        ):
            raise ValueError("Unverified grounded presentation")
        movie = publication.movie(BASE / f"video_{key}", "PressButton", key.upper())
        movie["presentation"] = "Grounded visual panel; unchanged physics; separate presentation take"
        movies.append(movie)
    if not np.array_equal(*initials):
        raise ValueError("ACT/DP initial measured states differ")
    result = dict(
        complete=True,
        task="PressButton",
        methods=methods,
        teacher="Frozen state PPO for demonstrations only; no privileged input at student inference",
        protocol="docs/research-core-protocol-20260924.md",
    )
    publication.write(PUBLIC / "press-button-model-benchmarks.json", result)
    for path in [PUBLIC / "model-benchmarks.json", ROOT / "website/src/modelBenchmarks.json"]:
        data = publication.read(path)
        data["rows"] = publication.merge(data["rows"], rows, ["task", "model"])
        data["protocols"].append(
            dict(
                task="PressButton",
                models=["ACT", "DP"],
                description=(
                    "Separate visual baseline: original16s smooth T200 task,21 robot-only measurements "
                    "and wrist RGB,10 full actuator targets.80 frozen-PPO demonstrations;30 unseen paired "
                    "resets; no teacher at inference."
                ),
            )
        )
        publication.write(path, data)
    path = PUBLIC / "policy-rollouts/manifest.json"
    publication.write(path, publication.merge(publication.read(path), movies, ["task", "model"]))
    publication.write(BASE / "published.json", dict(complete=True, local_only=True, methods=methods, movies=movies))


if __name__ == "__main__":
    main()
