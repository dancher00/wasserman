"""Guard against wrong revisions, changed source, corrupt downloads and overwrites."""

import hashlib
import importlib.util
import io
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load("paper_release")
artifacts = load("fetch_paper_artifacts")


@pytest.fixture
def runtime(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/model.py").write_text("original\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src/model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return tmp_path, release.git(tmp_path, "rev-parse", "HEAD")


def test_wrong_revision_does_not_change_checkout(runtime):
    path, commit = runtime
    with pytest.raises(ValueError, match="Wrong runtime"):
        release.prepare(path, {"commit": "0" * 40})
    assert release.git(path, "rev-parse", "HEAD") == commit


@pytest.mark.parametrize("staged", [False, True])
def test_modified_tracked_source_is_rejected(runtime, staged):
    path, commit = runtime
    (path / "src/model.py").write_text("changed\n")
    if staged:
        release.git(path, "add", "src/model.py")
    with pytest.raises(ValueError, match="tracked changes"):
        release.check_runtime(path, commit)
    assert (path / "src/model.py").read_text() == "changed\n"


def test_untracked_source_is_rejected_but_outputs_allowed(runtime):
    path, commit = runtime
    (path / "output.json").write_text("{}")
    release.check_runtime(path, commit)
    (path / "src/shadow.py").write_text("pass")
    with pytest.raises(ValueError, match="Untracked runtime"):
        release.check_runtime(path, commit)


def record(content):
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def test_reassembly_is_exact_and_existing_file_preserved(tmp_path):
    target = tmp_path / "model.pt"
    artifacts.write_checked(target, record(b"onetwo"), lambda: iter([io.BytesIO(b"one"), io.BytesIO(b"two")]))
    assert target.read_bytes() == b"onetwo"
    with pytest.raises(ValueError, match="Existing file differs"):
        artifacts.write_checked(target, record(b"wrong"), lambda: iter([]))
    assert target.read_bytes() == b"onetwo"


def test_corrupt_transfer_never_installs_destination(tmp_path):
    target = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="checksum mismatch"):
        artifacts.write_checked(target, record(b"correct"), lambda: iter([io.BytesIO(b"corrupt")]))
    assert not target.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("relative", ["../escape", "/absolute"])
def test_unsafe_manifest_path_rejected(tmp_path, relative):
    with pytest.raises(ValueError, match="Unsafe destination"):
        artifacts.safe_path(tmp_path, relative)


def test_destination_symlink_escape_rejected(tmp_path):
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe destination"):
        artifacts.safe_path(tmp_path, "escape/file")


def test_mixed_artifact_releases_keep_original_models(monkeypatch, tmp_path):
    """Updating presentation assets must not redirect v1 weights to a new tag."""
    import json
    import sys
    (tmp_path / "research").mkdir()
    records = [{"id":"model-a"}, {"id":"site", "release_tag":"presentation-v2"}, {"id":"model-b"}]
    (tmp_path / "research/paper-artifacts.json").write_text(json.dumps({"release_tag":"models-v1", "artifacts":records}))
    loaded, fetched = [], []
    monkeypatch.setattr(artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(artifacts, "headers", lambda: {})
    def assets_for(tag, auth):
        loaded.append(tag)
        return {"source_tag":tag}
    monkeypatch.setattr(artifacts, "assets_for_release", assets_for)
    monkeypatch.setattr(artifacts, "fetch", lambda row, destination, cache, assets, auth: fetched.append((row["id"],assets["source_tag"])))
    monkeypatch.setattr(sys, "argv", ["fetch_paper_artifacts.py", "fetch", "--artifact", "model-a", "--artifact", "site", "--artifact", "model-b", "--destination", str(tmp_path / "output"), "--cache", str(tmp_path / "cache")])
    artifacts.main()
    assert loaded == ["models-v1", "presentation-v2"]
    assert fetched == [("model-a","models-v1"),("site","presentation-v2"),("model-b","models-v1")]
