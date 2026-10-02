import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from package_revision_source import add_runtime_dependencies, selected

pytestmark = pytest.mark.unit


def make_manifest(root):
    p = root / "checkpoints/reference.pt"
    p.parent.mkdir()
    p.write_bytes(b"fixed reference")
    (root / "research").mkdir()
    (root / "research/revision-v2-runtime-dependencies.json").write_text(
        json.dumps(
            dict(
                protocol="wm-open-v2-20260929",
                files=[
                    dict(
                        path="checkpoints/reference.pt",
                        bytes=p.stat().st_size,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    )
                ],
            )
        )
    )
    return p


def test_runtime_reference_is_explicitly_pinned_while_other_models_are_excluded(tmp_path):
    p = make_manifest(tmp_path)
    files = {}
    add_runtime_dependencies(files, tmp_path)
    assert files == {"checkpoints/reference.pt": p}
    assert not selected("checkpoints/arbitrary.pt")
    assert selected("research/revision-v2-runtime-dependencies.json")


@pytest.mark.parametrize("mutation", ["missing", "changed", "symlink"])
def test_missing_or_changed_reference_prevents_packaging(tmp_path, mutation):
    p = make_manifest(tmp_path)
    if mutation == "missing":
        p.unlink()
    elif mutation == "changed":
        p.write_bytes(b"wrong reference")
    else:
        q = tmp_path / "elsewhere"
        p.rename(q)
        p.symlink_to(q)
    with pytest.raises(ValueError):
        add_runtime_dependencies({}, tmp_path)
