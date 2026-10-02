import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import run_revision_campaign as campaign  # noqa: E402
from package_revision_evaluation import groups  # noqa: E402

pytestmark = pytest.mark.unit


def test_parallel_schedule_preserves_every_packaged_cohort_once():
    planned = []
    for stage in ("evaluate", "diagnostics"):
        for task, model, seed, options in campaign.evaluation_plan(stage):
            interface = options.get("interface")
            label = f"{task}-{model}-{seed}" + ("-ee" if interface == "ee" else "")
            if stage == "evaluate":
                group = "primary-evaluation"
            elif interface:
                group = f"interface-evaluation/{interface}"
            else:
                group = f"controller-evaluation/ki{options.get('multiplier', 1.0):g}-{options.get('hydro', 'nominal')}"
            planned.append(f"{group}/{label}")
    expected = [folder for _, folders in groups() for folder in folders]
    assert len(planned) == len(set(planned)) == 108
    assert set(planned) == set(expected)


def test_two_workers_preserve_arguments_and_run_independent_jobs(monkeypatch, tmp_path):
    plan = [("RotateValve", "DP", seed, dict(purpose="research", multiplier=0.5)) for seed in (17, 43)]
    monkeypatch.setattr(campaign, "evaluation_plan", lambda stage: plan)
    barrier, lock, seen = threading.Barrier(2, timeout=5), threading.Lock(), []

    def evaluate(root, task, model, seed, **options):
        barrier.wait()
        with lock:
            seen.append((root, task, model, seed, options))

    monkeypatch.setattr(campaign, "evaluate", evaluate)
    campaign.evaluate_stage(tmp_path, "diagnostics", 2)
    assert sorted(seen, key=lambda row: row[3]) == [(tmp_path, *row) for row in plan]
