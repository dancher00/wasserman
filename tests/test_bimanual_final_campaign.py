"""Final reset states must remain unopened when development evidence is invalid."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("campaign", [
    {"status": "stopped", "error": "Numerical gate failed"},
    {"status": "running"},
    {"status": "development gates passed", "seeds": [123]},
])
def test_incomplete_or_unverified_development_never_opens_final_resets(tmp_path, campaign):
    development = tmp_path / "development"
    development.mkdir()
    (development / "campaign.json").write_text(json.dumps(campaign))
    output = tmp_path / "evaluation"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_bimanual_final_campaign.py"),
         "--development", str(development), "--development-pid", "99999999",
         "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "stopped"
    assert not status["final_states_opened"]
    assert not list(output.glob("final-*"))
    assert not (output / "frozen-sources.json").exists()
