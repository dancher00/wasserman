"""Download selected private release artifacts and restore their exact original bytes.

Uses Git's configured GitHub credentials, never embeds credentials in a URL, and
never overwrites an existing different file. No dataset download is implicit.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com/repos/dancher00/WasserMan-Bench"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def matches(path, record):
    return path.is_file() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"]


def safe_path(root, relative):
    path = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(f"Unsafe destination: {relative}")
    return path


def headers():
    authentication = {"User-Agent": "WasserMan-artifacts"}
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
    )
    credential = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode == 0 and credential.get("password"):
        authentication["Authorization"] = "Bearer " + credential["password"]
    return authentication


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        return None


def asset_stream(url, authentication, *, repository_api=API):
    if not url.startswith(repository_api + "/releases/assets/"):
        raise ValueError("Unexpected release asset API URL")
    request = urllib.request.Request(url, headers={**authentication, "Accept": "application/octet-stream"})
    try:
        return urllib.request.build_opener(NoRedirect).open(request, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code not in {301, 302, 303, 307, 308}:
            raise
        location = error.headers["Location"]
        parsed = urllib.parse.urlparse(location)
        if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".githubusercontent.com"):
            raise ValueError("Unexpected asset redirect") from error
        # Signed CDN URL receives no GitHub credential.
        return urllib.request.urlopen(location, timeout=120)


def write_checked(destination, record, sources):
    if destination.exists():
        if matches(destination, record):
            return
        raise ValueError(f"Existing file differs; preserved: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".partial", dir=destination.parent)
    partial = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            for source in sources():
                with source:
                    shutil.copyfileobj(source, output, length=8 * 1024**2)
        if not matches(partial, record):
            raise ValueError(f"Download/reassembly checksum mismatch: {destination.name}")
        # Atomic create without replacing a concurrently created destination.
        os.link(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def fetch(record, destination, cache, assets, authentication):
    target = safe_path(destination, record["path"])
    if target.exists():
        if matches(target, record):
            print("Already verified:", target)
            return
        raise ValueError(f"Existing file differs; preserved: {target}")
    parts = []
    for part in record["parts"]:
        local = safe_path(cache, part["name"])
        remote = assets[part["name"]]
        if remote["size"] != part["bytes"] or remote.get("digest") != "sha256:" + part["sha256"]:
            raise ValueError(f"Remote digest mismatch: {part['name']}")
        write_checked(local, part, lambda url=remote["url"]: iter([asset_stream(url, authentication)]))
        parts.append(local)
        print("Verified part:", part["name"], flush=True)
    write_checked(target, record, lambda: (part.open("rb") for part in parts))
    print("Restored:", target, flush=True)


def assets_for_release(tag, authentication, *, repository_api=API):
    request = urllib.request.Request(
        repository_api + "/releases/tags/" + tag,
        headers={**authentication, "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            release = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        # Maintainer verification before publication: drafts have no tag endpoint.
        request = urllib.request.Request(repository_api + "/releases?per_page=100", headers=authentication)
        with urllib.request.urlopen(request, timeout=120) as response:
            candidates = [r for r in json.load(response) if r["tag_name"] == tag]
        if len(candidates) != 1:
            raise ValueError("Release unavailable with current GitHub credentials") from error
        release = candidates[0]
    assets = {row["name"]: row for row in release["assets"]}
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["list", "fetch"])
    parser.add_argument("--artifact", action="append", default=[], help="Artifact ID from list; repeat to select more")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "research/paper-artifacts.json").read_text())
    if args.operation == "list":
        for row in manifest["artifacts"]:
            print(f"{row['id']:45} {row['bytes'] / 2**30:6.3f} GiB  {row['path']}")
        return
    if not args.artifact or args.destination is None or args.cache is None:
        parser.error("fetch requires --artifact, --destination and --cache")
    records = {row["id"]: row for row in manifest["artifacts"]}
    if set(args.artifact) - records.keys():
        parser.error("Unknown artifact ID; use list")
    authentication = headers()
    release_assets = {}
    for identifier in args.artifact:
        row = records[identifier]
        tag = row.get("release_tag", manifest["release_tag"])
        if tag not in release_assets:
            release_assets[tag] = assets_for_release(tag, authentication)
        fetch(row, args.destination.absolute(), args.cache.absolute(), release_assets[tag], authentication)


if __name__ == "__main__":
    main()
