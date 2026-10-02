"""Restore a checksummed evaluation package alongside the separate six task packages."""

import argparse
import json
from pathlib import Path

from restore_revision_task import extract_verified

from wasman.learning.revision_protocol import PROTOCOL, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.package / "evaluation-index.json"
    if sha256(index_path) != args.index_sha256:
        raise ValueError("Evaluation index differs from the supplied digest")
    index = json.loads(index_path.read_text())
    if (
        index["schema"] != "wm-open-evaluation-package-v1"
        or index["protocol"] != PROTOCOL
        or not index["complete"]
        or [index[key] for key in ("primary_cohorts", "controller_cohorts", "interface_cohorts")] != [54, 42, 12]
    ):
        raise ValueError("Unsupported or incomplete evaluation package")
    args.output.mkdir(parents=True, exist_ok=True)
    for record in index["assets"]:
        extract_verified(args.package, record, args.output)
        print(record["path"], flush=True)
    receipt = dict(index_sha256=args.index_sha256, all_members_verified=True, protocol=PROTOCOL)
    (args.output / "evaluation-restored.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
