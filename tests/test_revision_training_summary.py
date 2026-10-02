import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "training_summary", Path(__file__).parents[1] / "scripts/summarize_revision_training.py"
)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("damage", ["missing_final", "wrong_samples", "nonfinite_loss", "short_budget"])
def test_resource_table_refuses_incomplete_or_corrupt_training(damage):
    config = dict(model="DP", batch_size=64, sample_budget=320000)
    complete = dict(samples=320000, updates=5000, elapsed_s=100)
    rows = [dict(step=5000, samples=320000, elapsed_s=99, gpu_peak_bytes=1024, loss=0.2)]
    if damage == "missing_final":
        rows[0].update(step=4950, samples=316800)
    elif damage == "wrong_samples":
        rows[0]["samples"] = 160000
    elif damage == "nonfinite_loss":
        rows[0]["loss"] = float("nan")
    else:
        complete["samples"] = 160000
    with pytest.raises(ValueError):
        summary.checked_metrics(rows, config, complete)


def test_resumed_logged_work_is_retained_without_counting_as_extra_samples():
    config = dict(model="ACT", batch_size=32, sample_budget=320000)
    complete = dict(samples=320000, updates=10000, elapsed_s=100)
    rows = [
        dict(step=50, samples=1600, elapsed_s=1, gpu_peak_bytes=2048, loss=0.5),
        dict(step=50, samples=1600, elapsed_s=1.5, gpu_peak_bytes=1024, loss=0.5),
        dict(step=10000, samples=320000, elapsed_s=99, gpu_peak_bytes=1024, loss=0.2),
    ]
    result = summary.checked_metrics(rows, config, complete)
    assert result["samples"] == 320000
    assert result["repeated_logged_steps"] == 1
    assert result["observed_max_allocated_bytes"] == 2048
