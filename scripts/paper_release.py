"""Prepare or evaluate the immutable, isolated runtimes used by the paper.

Uses only the standard library. Install each runtime using its own guide before
execution. Options after -- are forwarded unchanged to that runtime's evaluator.
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research/paper-release.json"


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def check_runtime(directory, commit):
    if git(directory, "rev-parse", "HEAD") != commit:
        raise ValueError(f"Wrong runtime revision: expected {commit}")
    changed = git(directory, "status", "--porcelain", "--untracked-files=no")
    if changed:
        raise ValueError("Runtime has tracked changes; use a separate clean worktree")
    # Untracked Python modules can shadow the recorded implementation.
    extra = git(directory, "ls-files", "--others", "--exclude-standard", "src", "scripts")
    if extra:
        raise ValueError("Untracked runtime source files found; use a separate clean worktree")


def prepare(directory, record):
    if directory.exists():
        check_runtime(directory, record["commit"])
        return
    try:
        git(ROOT, "cat-file", "-e", record["commit"] + "^{commit}")
    except subprocess.CalledProcessError:
        subprocess.run(["git", "-C", str(ROOT), "fetch", "origin", record["commit"]], check=True)
    subprocess.run(
        ["git", "-C", str(ROOT), "worktree", "add", "--detach", str(directory), record["commit"]], check=True
    )
    check_runtime(directory, record["commit"])


def evaluation(directory, record, arguments):
    check_runtime(directory, record["commit"])
    python = directory / ".venv/bin/python"
    return [str(python), *record["entrypoint"], *arguments]


def main():
    manifest = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("operation", choices=["list", "prepare", "run", "recorded"])
    parser.add_argument("--runtime", choices=list(manifest["runtimes"]))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--result", help="ID in research/paper-results-index.json, for recorded evaluation")
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory, for recorded evaluation")
    parser.add_argument("--evidence-dir", type=Path, help="Extracted core-metadata, needed for old Valve evaluator")
    import sys

    argv = sys.argv[1:]
    boundary = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:boundary])
    remainder = argv[boundary + 1 :]
    if args.operation == "list":
        print(json.dumps(manifest, indent=2))
        return
    selected = None
    if args.operation == "recorded":
        rows = json.loads((ROOT / "research/paper-results-index.json").read_text())["records"]
        selected = next((row for row in rows if row["id"] == args.result), None)
        if selected is None or args.output_dir is None or remainder:
            parser.error("recorded requires --result, --output-dir and no forwarded options")
        if args.runtime and args.runtime != selected["runtime"]:
            parser.error("Result belongs to a different runtime")
        args.runtime = selected["runtime"]
    if not args.runtime or not args.directory:
        parser.error("--runtime and --directory are required")
    if args.operation != "run" and remainder:
        parser.error("Forwarded evaluator arguments are allowed only for run")
    directory = args.directory.expanduser().absolute()
    record = manifest["runtimes"][args.runtime]
    try:
        if args.operation == "prepare":
            if args.dry_run:
                parser.error("prepare has no dry-run; list shows the pinned revisions")
            prepare(directory, record)
            print(f"Prepared {args.runtime}: {directory}\nInstall using {directory / record['guide']}")
            return
        if selected:
            check_runtime(directory, record["commit"])
            backend_root = directory
            if selected["backend_location"] == "evidence":
                if args.evidence_dir is None:
                    parser.error("This original evaluator requires extracted --evidence-dir core-metadata")
                backend_root = args.evidence_dir.absolute()
            backend = backend_root / selected["backend"]
            if not backend.is_file() or hashlib.sha256(backend.read_bytes()).hexdigest() != selected["backend_sha256"]:
                parser.error("Recorded evaluator is missing or has a different SHA-256")
            output = args.output_dir if args.output_dir.is_absolute() else directory / args.output_dir
            if output.exists():
                parser.error("Recorded evaluation requires a fresh output directory")
            command = [
                str(directory / ".venv/bin/python"),
                str(backend),
                *selected["arguments"],
                "--output-dir",
                str(output),
            ]
            if not args.dry_run:
                checkpoint = directory / selected["checkpoint"]
                with checkpoint.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != selected["checkpoint_sha256"]:
                        parser.error("Selected checkpoint SHA-256 mismatch")
        else:
            command = evaluation(directory, record, remainder)
        print(f"cwd={directory}\n{shlex.join(command)}", flush=True)
        if args.dry_run:
            return
        if not Path(command[0]).is_file():
            parser.error("Runtime .venv is missing; follow its installation guide")
        environment = dict(os.environ)
        # Prevent imports or cache reuse from the caller's other development checkout.
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(directory / "scripts"), str(directory / ".deps/valve-policy-deps")]
        )
        environment["WARP_CACHE_PATH"] = str(directory / ".deps/paper-release-warp-cache")
        environment.setdefault("OMP_NUM_THREADS", "4" if args.runtime == "core" else "1")
        raise SystemExit(subprocess.call(command, cwd=directory, env=environment))
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
