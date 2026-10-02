"""Export publication plots from a complete paired bimanual final report."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("summary", type=Path)
p.add_argument("--output-dir", type=Path, required=True)
a = p.parse_args()
r = json.loads(a.summary.read_text())
assert r["complete"] and r["evaluation_split"] == "final"
assert len(r["modes"]) == 3 and all(m["episodes"] == 30 for m in r["modes"])
a.output_dir.mkdir(parents=True, exist_ok=False)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": "#163e52",
        "axes.labelcolor": "#163e52",
    }
)
colors = ["#163e52", "#087fbd", "#26a4b8"]
metrics = [
    ("angle_mean_deg", "A  Valve rotation", "degrees", 1),
    ("base_attitude_mean_deg", "B  Base orientation error", "degrees", 1),
    ("base_displacement_mean_mm", "C  Base displacement", "mm", 1),
    ("motor_utilization_mean", "D  Motor force / native limit", "%", 100),
]
fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.6))
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.1, top=0.88, wspace=0.3, hspace=0.44)
for ax, (key, title, unit, scale) in zip(axes.flat, metrics, strict=True):
    for index, (mode, color) in enumerate(zip(r["modes"], colors, strict=True)):
        s = mode["series"]
        t = np.array([v["time_s"] for v in s])
        y = np.array([v[key] * scale for v in s])
        ax.plot(
            t,
            y,
            color=color,
            lw=1.5,
            ls="--" if index == 1 else "-",
            label=f"{mode['label']} ({mode['successes']}/{mode['episodes']})",
        )
        if key == "angle_mean_deg":
            ax.fill_between(
                t, [v["angle_low_deg"] for v in s], [v["angle_high_deg"] for v in s], color=color, alpha=0.1, lw=0
            )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(unit)
    ax.set_xlim(0, max(v["time_s"] for m in r["modes"] for v in m["series"]))
    ax.set_xlabel("Time (s)")
    ax.grid(axis="y", color="#dfe7eb", lw=0.5)
    if key != "angle_mean_deg":
        ax.set_ylim(bottom=0)
    else:
        ax.axhline(170, color="#697e89", ls=":", lw=0.9)
        ax.annotate(
            "170° threshold",
            xy=(0.98, 170),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color="#697e89",
        )
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.52, 0.995))
for ext in ["pdf", "svg", "png"]:
    fig.savefig(a.output_dir / f"bimanual-valve.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
manifest = dict(
    source_summary_sha256=hashlib.sha256(a.summary.read_bytes()).hexdigest(),
    plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    band="10th–90th percentiles over paired reset states, not training confidence intervals",
    files={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output_dir.iterdir()},
)
(a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
