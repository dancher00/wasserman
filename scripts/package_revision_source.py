"""Create a checksummed source snapshot for the open runtime without repository history."""

import argparse
import json
import subprocess
from pathlib import Path

from package_revision_task import pack

from wasman.learning.revision_protocol import PROTOCOL, sha256, validate_runtime_snapshot

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "README.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    ".gitattributes",
}


def selected(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Nonrelative source path")
    if "meshes/alpha/" in name or path.suffix in {".pt", ".rgb", ".mp4"}:
        return False
    return (
        name in ROOT_FILES
        or path.parts[0] in {"src", "configs"}
        or (path.parts[0] == "scripts" and path.suffix in {".py", ".sh"})
        or (
            path.parts[0] == "research"
            and path.name
            in {
                "revision-v2-runtime.json",
                "revision-v2-runtime-dependencies.json",
                "revision-v2-primary-results.json",
                "revision-v2-portability-results.json",
                "revision-v2-controller-results.json",
                "revision-v2-interface-results.json",
                "revision-v2-clean-valve-discrepancy.json",
                "revision-v2-clean-shell-discrepancy.json",
                "revision-v2-clean-repeat-results.json",
                "policy-sources.json",
                "training-backbones.json",
                "policy-requirements.txt",
                "media-requirements.txt",
            }
        )
        or (path.parts[0] == "docs" and path.suffix == ".md")
        or (path.parts[0] == "tests" and path.name.startswith("test_revision") and path.suffix == ".py")
    )


def add_runtime_dependencies(files, root):
    manifest = json.loads((root / "research/revision-v2-runtime-dependencies.json").read_text())
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Runtime dependency protocol mismatch")
    for row in manifest["files"]:
        name = row["path"]
        if Path(name).is_absolute() or ".." in Path(name).parts or name in files:
            raise ValueError("Unsafe or duplicate runtime dependency")
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or sha256(path) != row["sha256"]
        ):
            raise ValueError("Missing or changed runtime dependency")
        files[name] = path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh source package output required")
    runtime = validate_runtime_snapshot()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT).strip():
        raise ValueError("Commit tracked source changes before creating a revision-bound snapshot")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files = {name: ROOT / name for name in names if name and selected(name)}
    add_runtime_dependencies(files, ROOT)
    frozen = json.loads((ROOT / "research/revision-v2-runtime.json").read_text())
    if not set(frozen["source_sha256"]) <= files.keys():
        raise ValueError("Source selection omitted frozen runtime files")
    for name, path in files.items():
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"Nonregular source: {name}")
    args.output.mkdir(parents=True)
    guide = args.output / "SOURCE_SNAPSHOT.md"
    guide.write_text(
        "# Open-core source snapshot\n\n"
        "This archive contains the frozen runtime, distributable assets, current analysis and packaging scripts, "
        "dependency locks, notices and text documentation. The pinned PressButton reference checkpoint is included "
        "because the frozen rollout hashes it in every mode. It has no Git history, "
        "learned visual-policy weights, RGB datasets, "
        "website media or restricted Reach meshes. Historical guide links may refer to excluded archival material.\n\n"
        "Install from this directory with `bash scripts/bootstrap.sh --asset-profile open-procedural-v1`, "
        "then run `.venv/bin/python scripts/setup_policy_dependencies.py` and "
        "`.venv/bin/python scripts/setup_training_backbones.py`. Install the media dependencies with "
        "`uv pip install --python .venv/bin/python --no-deps -r research/media-requirements.txt`. Set "
        "`WASMAN_ASSET_PROFILE=open-procedural-v1` for every runtime command. "
        "Follow docs/revision-v2-protocol.md and docs/revision-v2-packages.md. "
        "Data, checkpoints and scored traces are separate checksummed packages. "
        "Third-party dependencies are installed separately under their own terms.\n\n"
        "Source integrity is checked by the supplied index; this snapshot does not prove public access "
        "or a fresh dependency installation. The historical Git-worktree reproduction commands do not "
        "apply to this versioned open-core snapshot. Release authoring commands that record a Git commit "
        "require a Git checkout; benchmark collection, training, evaluation and scoring do not.\n"
    )
    files[guide.name] = guide
    record = pack(args.output / "open-core-source.tar.zst", files)
    index = dict(
        schema="wm-open-source-package-v1",
        protocol=PROTOCOL,
        runtime_manifest_sha256=runtime,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        files=len(files),
        assets=[record],
        scope=(
            "Selected open runtime source snapshot; separate data/model/evaluation packages required. "
            "No Git history or restricted Reach meshes."
        ),
    )
    path = args.output / "source-index.json"
    path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(dict(files=len(files), bytes=record["bytes"], index_sha256=sha256(path))))


if __name__ == "__main__":
    main()
