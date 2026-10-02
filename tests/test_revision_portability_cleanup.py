import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from prune_revision_portability_copies import main, plan_duplicates

from wasman.learning.revision_protocol import PROTOCOL, sha256

pytestmark = pytest.mark.unit


def fixture(tmp_path):
    external, restored, package, download = [tmp_path / n for n in ["original", "restored", "package", "download"]]
    name = "external/task/rollout/trace.npz"
    for root, relative in [(external, "task/rollout/trace.npz"), (restored, name)]:
        p = root / relative
        p.parent.mkdir(parents=True)
        p.write_bytes(b"physical evidence")
    for root in [package, download]:
        root.mkdir()
        (root / "a.zst").write_bytes(b"archive bytes")
    assets = [dict(path="a.zst", bytes=13, sha256=sha256(package / "a.zst"), members={name: sha256(restored / name)})]
    index = package / "portability-index.json"
    index.write_text(
        json.dumps(
            dict(
                schema="wm-open-portability-package-v1",
                protocol=PROTOCOL,
                complete=True,
                test_cohorts=54,
                validation_cohorts=54,
                assets=assets,
            )
        )
    )
    verification = restored / "portability-verified.json"
    verification.write_text(
        json.dumps(dict(index_sha256=sha256(index), both_analyses_byte_identical=True, all_members_verified=True))
    )
    completion = tmp_path / "completion.json"
    completion.write_text(json.dumps(dict(index_sha256=sha256(index), verification_sha256=sha256(verification))))
    return external, restored, package, download, completion, assets


@pytest.mark.parametrize("corrupt", [None, "original", "archive", "receipt"])
def test_cleanup_requires_all_retained_copies_and_proof(tmp_path, monkeypatch, corrupt):
    external, restored, package, download, completion, assets = fixture(tmp_path)
    if corrupt == "original":
        (external / "task/rollout/trace.npz").write_bytes(b"changed")
    if corrupt == "archive":
        (package / "a.zst").write_bytes(b"changed")
    if corrupt == "receipt":
        (restored / "portability-verified.json").write_text("{}")
    output = tmp_path / "cleanup.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prune",
            "--external",
            str(external),
            "--restored",
            str(restored),
            "--package",
            str(package),
            "--download",
            str(download),
            "--completion",
            str(completion),
            "--output",
            str(output),
            "--apply",
        ],
    )
    if corrupt:
        with pytest.raises((ValueError, KeyError)):
            main()
        assert (restored / "external/task/rollout/trace.npz").exists() and (download / "a.zst").exists()
    else:
        main()
        assert json.loads(output.read_text())["status"] == "completed"
        assert not (restored / "external/task/rollout/trace.npz").exists() and not (download / "a.zst").exists()
    assert (
        (external / "task/rollout/trace.npz").exists()
        and (package / "a.zst").exists()
        and (restored / "portability-verified.json").exists()
    )


def test_rejects_same_tree_and_symlink(tmp_path):
    external, restored, _, _, _, assets = fixture(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        plan_duplicates(external, external, assets)
    p = restored / "external/task/rollout/trace.npz"
    p.unlink()
    p.symlink_to(external / "task/rollout/trace.npz")
    with pytest.raises(ValueError, match="symlink"):
        plan_duplicates(external, restored, assets)
