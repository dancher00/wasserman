"""Package the two recorded full ACT retraining checks separately from primary trainings."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from package_revision_evaluation import ALLOWED
from package_revision_task import pack

from wasman.learning.revision_protocol import PROTOCOL, sha256, validate_runtime_snapshot

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ("final.pt", "config.json", "completed.json", "metrics.jsonl")


def run_audit(core, repeated, audit, rollout, output):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_revision_retraining.py"),
            "--original",
            str(core / "models/PressButton-ACT-17"),
            "--repeated",
            str(repeated),
            "--original-audit",
            str(core / "data/PressButton/revision_audit.json"),
            "--repeated-audit",
            str(audit),
            "--repeated-rollout",
            str(rollout),
            "--output",
            str(output),
        ],
        check=True,
    )


def regular(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Require regular evidence file: {path}")
    return path


def case_files(name, model, audit, rollout, jobs):
    files = {f"{name}/model/{item}": regular(model / item) for item in MODEL_FILES}
    files[f"{name}/dataset-audit.json"] = regular(audit)
    if any(not (rollout / item).is_file() for item in ("report.json", "trace.npz", "contract.json")):
        raise ValueError("Missing full Button physical evidence")
    for path in rollout.iterdir():
        if path.name in ALLOWED:
            files[f"{name}/rollout/{path.name}"] = regular(path)
    for label, path in jobs.items():
        regular(path)
        if json.loads(path.read_text()).get("returncode") != 0:
            raise ValueError("Unsuccessful recorded reproduction job")
        files[f"{name}/jobs/{label}.json"] = path
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "clean", "external", "external-environment", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing package; use fresh output")
    runtime = validate_runtime_snapshot()
    cases = {
        "clean": dict(
            model=args.clean / "models/PressButton-ACT-17",
            audit=args.clean / "data/PressButton/revision_audit.json",
            rollout=args.clean / "evaluation/PressButton-ACT-17",
            jobs={
                "training": args.clean / "jobs/PressButton-ACT-17-retrain.json",
                "evaluation": args.clean / "jobs/PressButton-ACT-17-evaluate.json",
            },
        ),
        "external": dict(
            model=args.external / "result/model",
            audit=args.external / "result/dataset-audit.json",
            rollout=args.external / "retrained-evaluation/rollout",
            jobs={
                "training": args.external / "result/job.json",
                "evaluation": args.external / "retrained-evaluation/job.json",
            },
        ),
    }
    files = {name: case_files(name, **case) for name, case in cases.items()}
    files["clean"]["clean/jobs/training.gpu.jsonl"] = regular(args.clean / "jobs/PressButton-ACT-17-retrain.gpu.jsonl")
    files["clean"]["clean/jobs/training.log"] = regular(args.clean / "jobs/PressButton-ACT-17-retrain.log")
    files["external"]["external/jobs/training.gpu.jsonl"] = regular(args.external / "result/gpu.jsonl")
    files["external"]["external/jobs/training.log"] = regular(args.external / "result/training.log")
    args.output.mkdir(parents=True)
    analysis = {}
    for label, path in (("primary", args.core / "environment-observed.json"), ("external", args.external_environment)):
        environment = json.loads(regular(path).read_text())
        if environment.get("runtime_manifest_sha256") != runtime or environment.get("protocol") != PROTOCOL:
            raise ValueError("Incompatible environment observation")
        analysis[f"retraining-provenance/{label}-environment-observed.json"] = path
    for name, case in cases.items():
        target = args.output / f"{name}-retraining.json"
        run_audit(args.core, case["model"], case["audit"], case["rollout"], target)
        analysis[f"retraining-analysis/{target.name}"] = target
    sources = [
        "scripts/audit_revision_retraining.py",
        "scripts/score_revision_campaign.py",
        "scripts/audit_revision_dataset.py",
        "scripts/revision_scoring_evidence.py",
        "scripts/package_revision_retraining.py",
        "scripts/restore_revision_retraining.py",
        "research/revision-v2-runtime.json",
    ]
    for name in sources:
        analysis[f"retraining-provenance/{name}"] = regular(ROOT / name)
    assets = [pack(args.output / "retraining-analysis.tar.zst", analysis)]
    for name, members in files.items():
        assets.append(pack(args.output / f"retraining-{name}.tar.zst", members))
    index = dict(
        schema="wm-open-retraining-package-v1",
        protocol=PROTOCOL,
        complete=True,
        cases=list(cases),
        runtime_manifest_sha256=runtime,
        assets=assets,
        source_hashes={name: sha256(ROOT / name) for name in sources},
        scope=(
            "Two recorded full same-seed ACT PressButton retrainings and their full30-reset evaluations; "
            "not extra independent primary seeds. Primary model and input audit supplied separately "
            "by the Button task package. Other clean-checkout DP repetitions and the first-image "
            "diagnostic are separate evidence. Environment observations are not retrospective per-job attestations."
        ),
    )
    path = args.output / "retraining-index.json"
    path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(dict(assets=len(assets), index_sha256=sha256(path))))


if __name__ == "__main__":
    main()
