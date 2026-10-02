# ruff: noqa: E402
"""Publish only physically verified marine evidence to the local website."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".deps/valve-policy-deps"))

import imageio_ffmpeg
import numpy as np
from scipy.stats import binomtest

from wasman.controllers.marine_evidence import verify

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/marine_mechanisms_20260924"
PUBLIC = ROOT / "website/public/static"


def read(p):
    return json.loads(p.read_text())


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(p)


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verified_run(folder):
    report, criteria = read(folder / "report.json"), read(folder / "contract.json")
    raw = dict(np.load(folder / "trace.npz"))
    result = verify(raw, criteria)
    if result["success_per_seed"] != report["success_per_seed"]:
        raise ValueError("Report differs from raw physics")
    return report, criteria, raw


def movie(folder, task, model=None):
    report, _, _ = verified_run(folder)
    if report["seeds"] != [42] or report["video_frames"] != report["steps"]:
        raise ValueError("Incomplete standalone film")
    if report["expert_at_inference"] != (model is None):
        raise ValueError("Wrong controller in film")
    if model is None and report["success_per_seed"] != [True]:
        raise ValueError("Expert film did not solve task")
    target = PUBLIC / ("policy-rollouts" if model else "marine-mechanisms")
    target.mkdir(exist_ok=True)
    stem = task.lower() + "-" + (model.lower() if model else "expert")
    video, poster = target / (stem + ".mp4"), target / (stem + ".png")
    shutil.copy2(folder / "observer.mp4", video)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-ss",
            "3",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(poster),
        ],
        check=True,
    )
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(video))
    if frames != report["steps"]:
        raise ValueError("Film frame count differs from trace")
    item = dict(
        task=task,
        success=report["success_per_seed"][0],
        seed=42,
        video="./static/" + str(video.relative_to(PUBLIC)),
        poster="./static/" + str(poster.relative_to(PUBLIC)),
        video_sha256=sha(video),
        checkpoint_sha256=report["checkpoint_sha256"],
        frames=report["steps"],
        duration_s=report["steps"] * report["dt"],
        view="external",
        note="Uncut physical rollout, standalone seed42; separate from the30-seed benchmark.",
    )
    if model:
        item["model"] = model
    return item


def merge(items, new, keys):
    return [x for x in items if not any(all(x.get(k) == n.get(k) for k in keys) for n in new)] + new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["PushSlider", "PullLever"], required=True)
    parser.add_argument("--expert-only", action="store_true")
    a = parser.parse_args()
    root = BASE / a.task
    expert = movie(root / "video_expert", a.task)
    manifest = PUBLIC / "marine-mechanisms/experts.json"
    write(manifest, merge(read(manifest) if manifest.exists() else [], [expert], ["task"]))
    if a.expert_only:
        print("Published verified expert", a.task)
        return
    complete = read(root / "completed.json")
    acceptance = read(BASE / "acceptance.json")
    if complete["acceptance_sha256"] != sha(BASE / "acceptance.json"):
        raise ValueError("Acceptance changed")
    methods = []
    movies = []
    initials = []
    rows = []
    curves = []
    test_start = 8310 if a.task == "PushSlider" else 8510
    for model in ["ACT", "DP"]:
        key = model.lower()
        selection = read(root / f"selection_{key}.json")
        selected = selection["selected"]
        if selection["decision_unix"] > (root / f"test_{key}/source_manifest.json").stat().st_mtime:
            raise ValueError("Checkpoint selection was not frozen before test")
        r, c, d = verified_run(root / f"test_{key}")
        if r["seeds"] != list(range(test_start, test_start + 30)) + [42] or r["expert_at_inference"]:
            raise ValueError("Final split/controller mismatch")
        if r["checkpoint_sha256"] != selected["checkpoint_sha256"]:
            raise ValueError("Wrong selected checkpoint")
        for path, digest in r["source_sha256"].items():
            relative = str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else path
            if acceptance["source_sha256"][relative] != digest:
                raise ValueError("Final run not from frozen source: " + relative)
        single, _, _ = verified_run(root / f"single42_{key}")
        if r["inference_precision"] != "fp32" or r["cudnn_benchmark"] or not r["cudnn_deterministic"] or r["tf32"]:
            raise ValueError("Evaluation precision differs from protocol")
        initials.append(d["measured"][0, :30])
        good = np.array(r["success_per_seed"][:30])
        steps = np.array(r["first_success_step"][:30])
        times = steps[good] * r["dt"]
        k = int(good.sum())
        ci = binomtest(k, 30).proportion_ci(method="exact")
        row = dict(
            task=a.task,
            model=model,
            input="Wrist RGB + proprioception",
            successes=k,
            episodes=30,
            median_success_time_s=float(np.median(times)) if k else None,
            exact_binomial_95_ci=[ci.low, ci.high],
        )
        rows.append(row)
        trial_progress = []
        contacts = []
        diagnostics = []
        for j in range(30):
            valid = np.flatnonzero(d["active"][:, j] & ~d["terminal"][:, j])
            if not len(valid):
                raise ValueError("No physical samples")
            progress = (d["q"][valid, j] - c["initial"]) * c["direction"] / c["threshold"]
            padded = np.pad(progress, (0, max(0, 1790 - len(progress))), mode="edge")[:1790]
            trial_progress.append(padded)
            force = np.linalg.norm(d["forces"][valid, j], axis=-1)
            contacts.append(float(np.mean((force > 0.12).all(-1))))
            bilateral = (force > 0.12).all(-1)
            sustained = (
                bool(np.convolve(bilateral.astype(int), np.ones(10, dtype=int), mode="valid").max() >= 10)
                if len(bilateral) >= 10
                else False
            )
            peak = float(progress.max())
            category = (
                "success"
                if good[j]
                else "goal_without_full_contract"
                if peak >= 1
                else "no_sustained_bilateral_grasp"
                if not sustained
                else "insufficient_travel_after_grasp"
            )
            diagnostics.append(
                dict(
                    category=category,
                    peak_goal_fraction=peak,
                    last_goal_fraction=float(progress[-1]),
                    sustained_bilateral_contact=sustained,
                    closest_tool_distance_m=float(d["distance"][valid, j].min()),
                    peak_attitude_error_rad=float(d["attitude"][valid, j].max()),
                )
            )
        indices = np.unique(np.r_[np.arange(0, 1790, 30), 1789])
        p = np.array(trial_progress)
        curve = dict(
            task=a.task,
            model=model,
            time_s=((indices + 1) * r["dt"]).tolist(),
            success_pct=[float(np.mean(good & (steps <= i + 1)) * 100) for i in indices],
            progress_median=np.median(p[:, indices], axis=0).tolist(),
            progress_q25=np.quantile(p[:, indices], 0.25, axis=0).tolist(),
            progress_q75=np.quantile(p[:, indices], 0.75, axis=0).tolist(),
        )
        curves.append(curve)
        methods.append(
            dict(
                **row,
                checkpoint=selected,
                seed_results=[
                    dict(
                        seed=s,
                        success=bool(ok),
                        first_success_s=float(step * r["dt"]) if ok else None,
                        bilateral_contact_fraction=contacts[i],
                        diagnostics=diagnostics[i],
                    )
                    for i, (s, ok, step) in enumerate(zip(r["seeds"][:30], good, steps, strict=True))
                ],
                batch42_success=r["success_per_seed"][30],
                standalone42_success=single["success_per_seed"][0],
                raw_trace_sha256=sha(root / f"test_{key}/trace.npz"),
                contract=c,
            )
        )
        film = movie(root / f"video_{key}", a.task, model)
        if film["checkpoint_sha256"] != selected["checkpoint_sha256"]:
            raise ValueError("Film checkpoint mismatch")
        movies.append(film)
    mismatch = float(np.max(np.abs(initials[0] - initials[1])))
    if mismatch > 1e-3:
        raise ValueError(f"Paired initial-state mismatch {mismatch}")
    report = dict(
        task=a.task,
        scene="Marine-v1",
        methods=methods,
        curves=curves,
        paired_initial_state_max_abs_difference=mismatch,
        acceptance_sha256=sha(BASE / "acceptance.json"),
        expert_at_policy_inference=False,
        limitations=[
            "One training seed; intervals describe reset variability only.",
            "Calm water; engineering mass/material assumptions, not calibrated hardware.",
            "60s horizon, 80 successful expert demos selected from at most100 attempts.",
            "Failed model trials retained; seed42 films are separate diagnostics.",
        ],
    )
    write(root / "benchmark_report.json", report)
    write(PUBLIC / "marine-mechanisms" / f"{a.task.lower()}-benchmark.json", report)
    curvepath = PUBLIC / "marine-mechanisms/curves.json"
    write(curvepath, merge(read(curvepath) if curvepath.exists() else [], curves, ["task", "model"]))
    for path in [ROOT / "website/src/modelBenchmarks.json", PUBLIC / "model-benchmarks.json"]:
        data = read(path)
        data["rows"] = merge(data["rows"], rows, ["task", "model"])
        desc = dict(
            task=a.task,
            description="Marine-v1: grounded wall, passive mechanism, T200; 60s. "
            "80 native30Hz wrist-RGB demonstrations; ACT20k / DP40epochs, selected on8 validation seeds. "
            "30 fresh paired test seeds and standalone42; one training seed. "
            "Raw physical criteria replayed independently.",
        )
        data["protocols"] = merge(data["protocols"], [desc], ["task"])
        write(path, data)
    lines = [
        f"# {a.task} Marine-v1 — verified benchmark",
        "",
        "Versioned grounded panel and detailed passive mechanism; original progress and stability criteria.",
        "80 successful native30Hz RGB/actuator demonstrations; one fixed training seed.",
        "Separate8-seed validation selects checkpoints before the30-seed test. Seed42 is excluded from the aggregate.",
        "",
        "| Model | Success | Exact95% interval | Median successful time | Standalone42 |",
        "|---|---|---|---|---|",
    ]
    for m in methods:
        lo, hi = m["exact_binomial_95_ci"]
        timing = f"{m['median_success_time_s']:.2f}s" if m["median_success_time_s"] is not None else "—"
        lines.append(
            f"| {m['model']} | {m['successes']}/30 | {lo:.1%}–{hi:.1%} | {timing} | {m['standalone42_success']} |"
        )
    lines += [
        "",
        "No expert acts at policy inference. All results are reconstructed from raw physical traces.",
        "Uncut expert movies belong to the task gallery; actual model movies and plots belong to Results.",
        "Failures are retained. A completed benchmark does not imply100% policy success.",
        "",
        f"Raw report: `artifacts/marine_mechanisms_20260924/{a.task}/benchmark_report.json`.",
        f"Paired initial-state maximum absolute difference: {mismatch:.6g}.",
        "",
        "Limits: one training seed, calm water, assumed physical masses/materials; "
        "no hardware calibration or transfer claim.",
        "Site publication is local/private only.",
    ]
    (ROOT / f"docs/{a.task.lower()}-marine-results.md").write_text("\n".join(lines) + "\n")
    if {"PushSlider", "PullLever"} <= {row["task"] for row in data["rows"]}:
        roadmap = ROOT / "docs/benchmark-roadmap.md"
        text = roadmap.read_text().replace(
            "Four tasks have learned-policy evaluations: PressButton, RotateValve, OpenHatch\nand CollectShell.",
            "Six tasks have learned-policy evaluations: PressButton, RotateValve, OpenHatch,\n"
            "CollectShell, PushSlider and PullLever.",
            1,
        )
        start = text.find("PullLever / PushSlider Marine-v1 is in execution:")
        end = text.find("Remaining paper/release work:", start)
        if start >= 0 and end > start:
            text = (
                text[:start]
                + (
                    "Marine-v1 PullLever and PushSlider have completed fixed ACT/DP runs, separate\n"
                    "validation, fresh30-seed tests and standalone42. Expert films, actual model\n"
                    "films, success intervals and progress curves are published locally. See\n"
                    "[PullLever results](pulllever-marine-results.md) and\n"
                    "[PushSlider results](pushslider-marine-results.md). Model failures remain reported.\n\n"
                )
                + text[end:]
            )
        roadmap.write_text(text)
    moviepath = PUBLIC / "policy-rollouts/manifest.json"
    write(moviepath, merge(read(moviepath), movies, ["task", "model"]))
    write(
        root / "published.json",
        dict(task=a.task, scope="local/private", report_sha256=sha(root / "benchmark_report.json")),
    )
    print("Published verified benchmark", a.task, [(r["model"], r["successes"]) for r in rows])


if __name__ == "__main__":
    main()
