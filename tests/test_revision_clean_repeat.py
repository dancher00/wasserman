import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from score_revision_clean_repeat import normalized_sources

pytestmark = pytest.mark.unit


def test_normalizes_relocated_entry_script_only_when_frozen_hash_matches():
    hashes = json.loads((Path(__file__).parents[1] / "research/revision-v2-runtime.json").read_text())["source_sha256"]
    name = "scripts/rollout_button_visual.py"
    assert normalized_sources({"/new/checkout/" + name: hashes[name]}) == normalized_sources({name: hashes[name]})
    with pytest.raises(ValueError):
        normalized_sources({"/new/checkout/" + name: "0" * 64})
    with pytest.raises(ValueError):
        normalized_sources({"/new/other/rollout_button_visual.py": hashes[name]})
    with pytest.raises(ValueError):
        normalized_sources({"/new/checkout/" + name: hashes[name], name: hashes[name]})


def test_preserves_scalar_digest_schema_and_rejects_invalid_digest():
    assert normalized_sources("a" * 64) == "a" * 64
    with pytest.raises(ValueError):
        normalized_sources("not-a-sha256")
