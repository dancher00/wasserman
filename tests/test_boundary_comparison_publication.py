"""Recomputed website comparison never invents motion or conflates force measures."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_published_comparison_exactly_matches_hashed_archived_traces():
    build = runpy.run_path(str(ROOT / "scripts/publish_boundary_comparison.py"))["build_comparison"]
    expected = build()
    actual = json.loads((ROOT / "website/public/static/boundary_comparison_v1.json").read_text())
    assert actual == expected
    assert actual["calibrated"] is False
    assert "not synchronized video" in actual["kind"]
    for mode in ("floor", "wall"):
        assert [case["seed"] for case in actual["modes"][mode]] == [2058, 2059, 2060]
        for case in actual["modes"][mode]:
            points = case["points"]
            assert len(points) == 240
            assert [point["t_s"] for point in points] == pytest.approx([(i + 1) / 30 for i in range(240)])
            assert max(point["removed_motor_thrust_n"] for point in points) == case["max_removed_motor_thrust_n"]
            # The report's net-force peak is sampled at 120Hz; plotted summed
            # motor losses at 30Hz. Do not compare their different-clock maxima.
            assert case["max_net_force_delta_n"] >= 0
