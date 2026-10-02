import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import score_revision_diagnostics as diagnostics  # noqa: E402
import score_revision_interfaces as interfaces  # noqa: E402

from wasman.learning.revision_protocol import PROTOCOL, sha256  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["controller", "interface"])
@pytest.mark.parametrize("damage", ["weights", "expert", "precision", "noise_seed"])
def test_interventions_reject_changed_execution_identity_before_scoring(tmp_path, monkeypatch, kind, damage):
    task = "RotateValve" if kind == "controller" else "PushSlider"
    reset_start = 71000 if kind == "controller" else 84000
    steps = 2240 if kind == "controller" else 1790
    label = f"{task}-DP-17"
    model = tmp_path / "models" / label
    model.mkdir(parents=True)
    checkpoint = model / "final.pt"
    checkpoint.write_bytes(b"synthetic test model identity")
    digest = sha256(checkpoint)
    config = dict(
        protocol=PROTOCOL,
        task=task,
        model="DP",
        seed=17,
        pilot=False,
        runtime_manifest_sha256="test-runtime",
        rollout_seed=42,
        interface="actuator",
        train_seeds=[40000],
        validation_seeds=[40001],
    )
    completed = dict(pilot=False, samples=320000, final_sha256=digest)
    report = dict(
        task=task,
        purpose="research",
        seeds=list(range(reset_start, reset_start + 30)),
        evaluated_steps=steps,
        checkpoint_sha256=digest,
        expert_at_inference=False,
        inference_precision="fp32",
        policy_random_seed=42,
        inference_random_seed=42,
        interface="actuator",
        mode="policy",
        contract_replayed=True,
    )
    if damage == "weights":
        checkpoint.write_bytes(b"different weights despite unchanged claimed digest")
    elif damage == "expert":
        report["expert_at_inference"] = True
    elif damage == "precision":
        report["inference_precision"] = "bfloat16"
    else:
        report.update(policy_random_seed=43, inference_random_seed=43)
    for filename, record in (("config.json", config), ("completed.json", completed)):
        (model / filename).write_text(json.dumps(record))
    module = diagnostics if kind == "controller" else interfaces
    monkeypatch.setattr(module, "validate_runtime_snapshot", lambda digest: None)
    group = "controller-evaluation/ki0-nominal" if kind == "controller" else "interface-evaluation/actuator"
    folder = tmp_path / group / label
    folder.mkdir(parents=True)
    (folder / ("summary.json" if kind == "controller" else "report.json")).write_text(json.dumps(report))
    # No trajectory exists: rejection must precede physical scoring, not rely on missing files.
    with pytest.raises(ValueError, match="mismatch"):
        module.load(tmp_path, task, 17, "ki0-nominal" if kind == "controller" else "actuator")
