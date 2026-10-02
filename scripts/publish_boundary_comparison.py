"""Publish existing matched ON/OFF telemetry as independent quantitative evidence.

No simulation/video synthesis: every plotted point is recomputed from hashed
archived traces. Summed removed motor thrust is not the net vehicle wrench.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_comparison():
    modes = {mode: [] for mode in ("floor", "wall")}
    sources = {}
    for version in ("v1", "v2"):
        directory = ROOT / f"artifacts/boundary_clearance_{version}"
        path = directory / "report.json"
        report = json.loads(path.read_text())
        if not report["completed"]:
            raise ValueError("Only completed comparison experiments may be shown")
        sources[str(path.relative_to(ROOT))] = digest(path)
        cases = {case["id"]: case for case in report["cases"]}
        for pair in report["pairs"]:
            off, on = cases[pair["off"]], cases[pair["on"]]
            if off["protocol"] != "away_step" or off["base_to_surface_m"] != {"floor": 0.16, "wall": 0.45}[off["mode"]]:
                continue
            if (
                not pair["valid"]
                or not pair["initial_state_exact_match"]
                or off["initial_state"] != on["initial_state"]
            ):
                raise ValueError("Comparison requires valid, exactly matched initial states")
            traces = []
            for case in (off, on):
                path = directory / case["trace"]
                if digest(path) != case["trace_sha256"] or not case["passed"] or case["reset"]:
                    raise ValueError("Archived trace failed provenance/safety validation")
                sources[str(path.relative_to(ROOT))] = digest(path)
                traces.append(json.loads(path.read_text()))
            if len(traces[0]) != 240 or len(traces[1]) != 240:
                raise ValueError("Expected full eight-second comparison traces")
            points = []
            for a, b in zip(*traces, strict=True):
                if a["t_s"] != b["t_s"] or a["target_w_m"] != b["target_w_m"]:
                    raise ValueError("Compared trajectories must use the same clock and command")
                if not np.allclose(a["boundary_gain"], 1, atol=0, rtol=0):
                    raise ValueError("Boundary-OFF reference must have no thrust loss")
                delta = np.linalg.norm(np.asarray(b["position_w_m"]) - a["position_w_m"]) * 1000
                removed = np.sum(np.abs(b["motor_force_before_boundary_n"]) * (1 - np.asarray(b["boundary_gain"])))
                points.append(
                    {"t_s": a["t_s"], "position_delta_mm": float(delta), "removed_motor_thrust_n": float(removed)}
                )
            maximum = max(point["position_delta_mm"] for point in points)
            if not np.isclose(maximum, pair["max_on_off_position_delta_mm"], atol=1e-6):
                raise ValueError("Recomputed displacement differs from the archived report")
            modes[off["mode"]].append(
                {
                    "seed": off["seed"],
                    "base_to_surface_m": off["base_to_surface_m"],
                    "max_position_delta_mm": maximum,
                    "max_removed_motor_thrust_n": max(point["removed_motor_thrust_n"] for point in points),
                    "max_net_force_delta_n": on["max_boundary_force_delta_n"],
                    "points": points,
                }
            )
    if any(sorted(case["seed"] for case in cases) != [2058, 2059, 2060] for cases in modes.values()):
        raise ValueError("Expected three matched seeds for each surface")
    return {
        "schema_version": 1,
        "kind": "Independent matched boundary ON/OFF experiment; not synchronized video",
        "calibrated": False,
        "loss_coefficient": 0.2,
        "step_onset_s": 3.0,
        "step_distance_m": 0.12,
        "duration_s": 8,
        "samples_per_case": 240,
        "trace_hz": 30,
        "net_force_peak_hz": 120,
        "source_sha256": sources,
        "modes": modes,
    }


if __name__ == "__main__":
    destination = ROOT / "website/public/static/boundary_comparison_v1.json"
    destination.write_text(json.dumps(build_comparison(), indent=2, allow_nan=False) + "\n")
    print(destination)
