import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
SPEC = importlib.util.spec_from_file_location("score", Path(__file__).parents[1] / "scripts/score_revision_campaign.py")
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)
pytestmark = pytest.mark.unit


def test_reset_replication_does_not_erase_training_variability():
    # Within each run every reset has exactly the same outcome.
    small = np.array([[0] * 30, [1] * 30, [1] * 30])
    large = np.repeat(small, 4, axis=1)
    assert score.crossed_interval(small) == [0.0, 1.0]
    assert score.crossed_interval(large) == [0.0, 1.0]
    assert score.summarize(small)["training_run_sd"] == pytest.approx(np.std([0, 1, 1], ddof=1))


def test_same_checkpoint_intervention_is_paired_but_algorithms_are_not():
    values = np.array([[0] * 30, [1] * 30, [1] * 30])
    assert score.crossed_interval(values, values, paired_training=True) == [0.0, 0.0]
    low, high = score.crossed_interval(values, values)
    assert low < 0 < high


def test_exact_reset_intervals_preserve_boundary_uncertainty():
    zero, full = score.binomial_interval(0, 30), score.binomial_interval(30, 30)
    assert zero[0] == 0 and 0 < zero[1] < 1
    assert full[1] == 1 and full[0] == pytest.approx(1 - zero[1])
