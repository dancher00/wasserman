import importlib.util
from pathlib import Path


def test_release_diagnostic_measures_motion_not_controller_phase():
    path = Path(__file__).parents[1] / "scripts/diagnose_valve_release.py"
    spec = importlib.util.spec_from_file_location("release_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frames = []
    for t, x, force, angle in [(0, 1, 4, 3), (1, 0.995, 2, 3), (2, 0.98, 0.3, 3.02), (3, 0.85, 0, 3.1)]:
        frames.append(
            dict(
                t=t,
                angle_rad=[angle],
                held=[True],
                success=[False],
                tool_position_m=[[x, 0, 0]],
                finger_forces_n=[[force, 0]],
            )
        )
    result = module.summarize({"trace": frames})
    assert result["successes"] == 0
    assert result["withdrawing_under_contact"] == 1
    assert result["environments"][0]["withdrawal_time_s"] == 2
    assert result["environments"][0]["post_withdrawal_angle_change_deg"] > 4
