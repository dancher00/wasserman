import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from package_revision_evaluation import group_files, groups  # noqa: E402

pytestmark = pytest.mark.unit


def test_evaluation_release_requires_each_declared_cohort_exactly_once():
    plan = dict(groups())
    paths = [path for folders in plan.values() for path in folders]
    assert len(paths) == len(set(paths)) == 108
    for prefix, count in (("primary-", 54), ("controller-", 42), ("interface-", 12)):
        assert sum(len(v) for k, v in plan.items() if k.startswith(prefix)) == count


def test_evaluation_package_excludes_unrelated_files_and_refuses_symlinks(tmp_path):
    name = "primary-evaluation/PressButton-ACT-17"
    folder = tmp_path / name
    folder.mkdir(parents=True)
    for filename in ("report.json", "trace.npz", "contract.json", "unrelated-private-asset.stl"):
        (folder / filename).write_bytes(b"synthetic package fixture")
    receipt = tmp_path / "jobs/eval-primary-evaluation-PressButton-ACT-17.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps(dict(returncode=0)))
    files = group_files(tmp_path, [name])
    assert len(files) == 4
    assert all(not key.endswith(".stl") for key in files)
    (folder / "wrist_0.png").symlink_to(folder / "unrelated-private-asset.stl")
    with pytest.raises(ValueError, match="regular file"):
        group_files(tmp_path, [name])
    (folder / "wrist_0.png").unlink()
    receipt.write_text(json.dumps(dict(returncode=1)))
    with pytest.raises(ValueError, match="Unfinished"):
        group_files(tmp_path, [name])
