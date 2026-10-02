import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from fetch_revision_evidence import RECEIPT, download_package

pytestmark = pytest.mark.unit


def fixture():
    contents = {"shell.part-0000": b"first", "shell.part-0001": b"second"}
    parts = [dict(path=k, bytes=len(v), sha256=hashlib.sha256(v).hexdigest()) for k, v in contents.items()]
    index = dict(
        schema="wm-open-evaluation-package-v1",
        protocol="wm-open-v2-20260929",
        complete=True,
        assets=[dict(path="shell.tar.zst", bytes=11, sha256=hashlib.sha256(b"firstsecond").hexdigest(), parts=parts)],
    )
    return contents, index


def execute(tmp_path, contents, index):
    contents = dict(contents, **{"evaluation-index.json": json.dumps(index).encode()})
    return download_package(
        tmp_path,
        "evaluation-index.json",
        hashlib.sha256(contents["evaluation-index.json"]).hexdigest(),
        lambda name: io.BytesIO(contents[name]),
    )


def test_downloads_parts_and_verifies_combined_archive_without_reassembly(tmp_path):
    data, index = fixture()
    receipt = execute(tmp_path, data, index)
    assert receipt["delivery_files"] == 2 and receipt["logical_archives"] == 1
    assert (tmp_path / RECEIPT).is_file()
    assert not (tmp_path / "shell.tar.zst").exists()
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.part-*")} == data


@pytest.mark.parametrize("fault", ["corrupt_part", "reordered_parts", "wrong_whole_hash"])
def test_integrity_failures_never_write_success_receipt(tmp_path, fault):
    data, index = fixture()
    if fault == "corrupt_part":
        data["shell.part-0001"] = b"broken"
    elif fault == "reordered_parts":
        index["assets"][0]["parts"].reverse()
    else:
        index["assets"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        execute(tmp_path, data, index)
    assert not (tmp_path / RECEIPT).exists()


def test_refuses_symlink_and_unsafe_namespace_before_archive_download(tmp_path):
    data, index = fixture()
    target = tmp_path / "outside"
    target.write_bytes(b"untouched")
    (tmp_path / "shell.part-0001").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        execute(tmp_path, data, index)
    assert target.read_bytes() == b"untouched"
    assert not (tmp_path / "shell.part-0000").exists()


def test_refuses_incomplete_evidence(tmp_path):
    data, index = fixture()
    index["complete"] = False
    with pytest.raises(ValueError, match="Incomplete"):
        execute(tmp_path, data, index)
    assert not list(tmp_path.glob("*.part-*"))


def test_public_http_download(tmp_path):
    import functools
    import http.server
    import threading

    from fetch_revision_task import incoming

    data, index = fixture()
    index_bytes = json.dumps(index).encode()
    served = tmp_path / "served"
    served.mkdir()
    for name, content in {**data, "evaluation-index.json": index_bytes}.items():
        (served / name).write_bytes(content)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        receipt = download_package(
            tmp_path / "download",
            "evaluation-index.json",
            hashlib.sha256(index_bytes).hexdigest(),
            lambda name: incoming(f"http://127.0.0.1:{server.server_port}/{name}"),
        )
        assert receipt["all_downloaded_bytes_verified"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
