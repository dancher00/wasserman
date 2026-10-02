import runpy
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("fault", [None, "layout", "reset"])
def test_registered_collection_layout_is_preserved_and_checked(tmp_path, monkeypatch, fault):
    dataset = tmp_path / "data.pt"
    output = tmp_path / "model.pt"
    torch.save(
        {
            "observations": torch.zeros(4498, 50),
            "targets": torch.zeros(4498, 11),
            "phases": torch.full((4498,), 7),
            "geometry": "registered-v1",
            "collection_layout": {
                "num_envs": 3 if fault == "layout" else 2,
                "steps_per_round": 2249,
                "rounds": 1,
                "has_retained_prefix": False,
            },
            "last_round_failed_resets": torch.tensor([fault == "reset", False]),
        },
        dataset,
    )
    monkeypatch.setattr(
        sys, "argv", ["fit_valve_local.py", "--dataset", str(dataset), "--base-envs", "2", "--output", str(output)]
    )
    entry = Path(__file__).resolve().parents[1] / "scripts/fit_valve_local.py"
    if fault:
        with pytest.raises(ValueError):
            runpy.run_path(str(entry), run_name="__main__")
        assert not output.exists()
    else:
        runpy.run_path(str(entry), run_name="__main__")
        checkpoint = torch.load(output, weights_only=True)
        assert checkpoint["infos"]["geometry"] == "registered-v1"
        assert checkpoint["infos"]["base_block_environments"] == 2
        assert len(checkpoint["observations"]) == 2249
