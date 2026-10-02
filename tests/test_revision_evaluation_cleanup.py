import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from prune_revision_evaluation_copies import plan_duplicates
from wasman.learning.revision_protocol import sha256

pytestmark = pytest.mark.unit


def fixture(tmp_path):
    canonical, restored = tmp_path / "canonical", tmp_path / "restored"
    name = "primary-evaluation/task/trace.npz"
    for root in (canonical, restored):
        (root / name).parent.mkdir(parents=True)
        (root / name).write_bytes(b"physical evidence")
    assets = [{"members": {name: sha256(canonical / name), "models/final.pt": "not-a-cleanup-target"}}]
    return canonical, restored, name, assets


def test_only_verified_evaluation_copies_are_planned_without_mutation(tmp_path):
    canonical, restored, name, assets = fixture(tmp_path)
    plan = plan_duplicates(canonical, restored, assets)
    assert [p["path"] for p in plan] == [name]
    assert (canonical / name).read_bytes() == (restored / name).read_bytes()


@pytest.mark.parametrize("side", ["canonical", "restored"])
def test_either_corrupted_copy_prevents_cleanup(tmp_path, side):
    canonical, restored, name, assets = fixture(tmp_path)
    target = canonical if side == "canonical" else restored
    (target / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="differ"):
        plan_duplicates(canonical, restored, assets)
    assert (canonical / name).exists() and (restored / name).exists()


def test_symlink_and_same_tree_refused(tmp_path):
    canonical, restored, name, assets = fixture(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        plan_duplicates(canonical, canonical, assets)
    (restored / name).unlink()
    (restored / name).symlink_to(canonical / name)
    with pytest.raises(ValueError, match="symlink"):
        plan_duplicates(canonical, restored, assets)
    assert (canonical / name).exists()


@pytest.mark.parametrize("apply,corrupt_archive", [(False, False), (True, False), (True, True)])
def test_cli_keeps_canonical_evidence_and_requires_intact_archive(tmp_path, monkeypatch, apply, corrupt_archive):
    import json
    from prune_revision_evaluation_copies import main
    from wasman.learning.revision_protocol import PROTOCOL
    canonical, restored, name, assets = fixture(tmp_path)
    reproduction = tmp_path / "reproduction"
    reproduction.mkdir()
    restored.rename(reproduction / "restored")
    restored = reproduction / "restored"
    package = tmp_path / "package"
    package.mkdir()
    archive = package / "evidence.tar.zst"
    archive.write_bytes(b"synthetic archive for deletion-boundary test")
    assets[0].update(path=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive))
    index = package / "evaluation-index.json"
    index.write_text(json.dumps(dict(schema="wm-open-evaluation-package-v1", protocol=PROTOCOL,
        complete=True, primary_cohorts=54, controller_cohorts=42, interface_cohorts=12, assets=assets)))
    verification = reproduction / "analysis-check/verified.json"
    verification.parent.mkdir()
    verification.write_text(json.dumps(dict(index_sha256=sha256(index), all_four_analyses_byte_identical=True)))
    completion = tmp_path / "complete.json"
    completion.write_text(json.dumps(dict(index_sha256=sha256(index), verification_sha256=sha256(verification))))
    download = reproduction / "download"
    download.mkdir()
    (download / archive.name).write_bytes(archive.read_bytes())
    output = tmp_path / "cleanup.json"
    args = ["prune", "--campaign", str(canonical), "--package", str(package), "--reproduction", str(reproduction), "--completion", str(completion), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args + (["--apply"] if apply else []))
    if corrupt_archive:
        archive.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="archive changed"):
            main()
        assert not output.exists()
        assert (restored / name).exists()
    else:
        main()
        receipt = json.loads(output.read_text())
        assert receipt["status"] == ("completed" if apply else "verified-plan")
        assert (restored / name).exists() is not apply
        assert (download / archive.name).exists() is not apply
        assert archive.exists()
    assert (canonical / name).read_bytes() == b"physical evidence"
    assert verification.exists() and index.exists()
