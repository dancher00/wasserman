import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from package_revision_portability import cohort_files  # noqa: E402

pytestmark = pytest.mark.unit


def fixture_cohort(root):
    label = "PressButton-ACT-17"
    folder = root / label
    (folder / "rollout").mkdir(parents=True)
    names = ["job.json", "gpu.jsonl", "job.log", "rollout/report.json", "rollout/trace.npz", "rollout/contract.json"]
    manifest = {}
    for name in names:
        path = folder / name
        path.write_bytes(b'{"returncode": 0}')
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / "retrieval-manifest.json").write_text(json.dumps(manifest))
    (folder / "local-retrieval.json").write_text(json.dumps(dict(members_verified=True, files=len(names))))
    return label, folder, manifest


def test_portability_package_detects_corrupt_retrieved_evidence(tmp_path):
    label, folder, _ = fixture_cohort(tmp_path)
    (folder / "private.stl").write_text("unrelated asset")
    files = cohort_files(tmp_path, label)
    assert len(files) == 8 and all(not name.endswith(".stl") for name in files)
    (folder / "rollout/trace.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed"):
        cohort_files(tmp_path, label)


@pytest.mark.parametrize("member", ["../secret", "rollout/private.stl", "/absolute"])
def test_portability_package_rejects_unexpected_manifest_member(tmp_path, member):
    label, folder, manifest = fixture_cohort(tmp_path)
    manifest[member] = "0" * 64
    (folder / "retrieval-manifest.json").write_text(json.dumps(manifest))
    (folder / "local-retrieval.json").write_text(json.dumps(dict(members_verified=True, files=len(manifest))))
    with pytest.raises(ValueError, match="Unexpected"):
        cohort_files(tmp_path, label)


def test_portability_package_rejects_trace_symlink(tmp_path):
    label, folder, _ = fixture_cohort(tmp_path)
    trace = folder / "rollout/trace.npz"
    trace.unlink()
    trace.symlink_to(folder / "job.json")
    with pytest.raises(ValueError, match="nonregular"):
        cohort_files(tmp_path, label)
