import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from package_revision_task import pack
from revision_package_assets import delivery_records, split_archive, verified_asset_paths
from restore_revision_task import extract_verified
from stage_revision_supplement import package_records
from wasman.learning.revision_protocol import PROTOCOL

pytestmark = pytest.mark.unit


def build(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(os.urandom(140_000))
    package = tmp_path / "package"
    package.mkdir()
    original = pack(package / "evidence.tar.zst", {"cohort/trace.npz": source})
    record = split_archive(package, original, max_part_bytes=32_768)
    return source, package, record


def test_multipart_roundtrip_without_aggregate_archive(tmp_path):
    source, package, record = build(tmp_path)
    assert len(record["parts"]) >= 4
    assert all(row["bytes"] <= 32_768 for row in record["parts"])
    assert not (package / record["path"]).exists()
    assert verified_asset_paths(package, record)
    output = tmp_path / "restored"
    output.mkdir()
    extract_verified(package, record, output)
    assert (output / "cohort/trace.npz").read_bytes() == source.read_bytes()
    assert sorted(p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()) == ["cohort/trace.npz"]


@pytest.mark.parametrize("damage", ["reorder", "corrupt", "missing", "symlink"])
def test_bad_parts_refused_before_any_extraction(tmp_path, damage):
    source, package, record = build(tmp_path)
    first = package / record["parts"][0]["path"]
    if damage == "reorder":
        record["parts"].reverse()
    elif damage == "corrupt":
        first.write_bytes(b"x" * first.stat().st_size)
    elif damage == "missing":
        first.unlink()
    else:
        first.unlink()
        first.symlink_to(source)
    output = tmp_path / "restored"
    with pytest.raises(ValueError):
        extract_verified(package, record, output)
    assert not output.exists()
    assert source.exists()


def test_uploader_enumerates_transport_parts_and_index(tmp_path):
    _, package, record = build(tmp_path)
    index = package / "portability-index.json"
    index.write_text(json.dumps(dict(schema="wm-open-portability-package-v1", protocol=PROTOCOL,
        complete=True, test_cohorts=54, validation_cohorts=54, assets=[record])))
    rows = package_records(package, index.name)
    assert [r["path"] for r in rows] == [r["path"] for r in record["parts"]] + [index.name]


def test_duplicate_delivery_paths_and_bad_totals_refused():
    asset = dict(path="a", bytes=1, sha256="unused")
    with pytest.raises(ValueError, match="duplicate"):
        delivery_records([asset, asset])
    with pytest.raises(ValueError, match="byte total"):
        delivery_records([dict(asset, parts=[dict(asset, bytes=2)])])
