"""Render website themes from the current manuscript teaser without changing it."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/build_focused_teaser.py"
DESTINATION = ROOT / "website/src/assets"
BACKUPS = ROOT.parent / "work/.maintenance/backups"
PALETTE = {"#dcecf7": "#183448", "#edf5fa": "#0e2536", "#f1f4f6": "#0c2231"}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    variants = {}
    native = json.loads((ROOT / "paper/revision-v8/figures/teaser-manifest.json").read_text())
    protected = {name: sha(ROOT / "paper" / name) for name in native["outputs"]}
    for theme in ("light", "dark"):
        spec = importlib.util.spec_from_file_location("current_manuscript_teaser", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.plt.rcParams["svg.hashsalt"] = "wasserman-website-teaser"
        if theme == "dark":
            module.INK = "#edf7ff"
            original_box = module.box
            module.box = lambda ax, x, y, w, h, color: original_box(ax, x, y, w, h, PALETTE.get(color, color))
        original_save = Figure.savefig

        def save(figure, filename, *args, **kwargs):
            suffix = Path(filename).suffix
            if suffix in {".svg", ".png"}:
                kwargs["transparent"] = True
                kwargs["dpi"] = 300 if suffix == ".svg" else 220
                if suffix == ".svg":
                    kwargs["metadata"] = {"Date": None}
            return original_save(figure, filename, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="website-teaser-", dir=BACKUPS) as temporary:
            module.OUT = Path(temporary)
            Figure.savefig = save
            try:
                module.main()
            finally:
                Figure.savefig = original_save
            output = DESTINATION / f"project-overview-{theme}.svg"
            shutil.copy2(module.OUT / "teaser.svg", output)
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
            shutil.copy2(module.OUT / "teaser.png", output.with_suffix(".png"))
        text = output.read_text()
        for label in ("2 × Oberon7", "Low light", "Water optics"):
            assert label in text, label
        assert "Two-arm Rex" not in text
        variants[theme] = {"asset": output.name, "sha256": sha(output),
                           "github_asset": output.with_suffix(".png").name,
                           "github_sha256": sha(output.with_suffix(".png"))}
    assert protected == {name: sha(ROOT / "paper" / name) for name in native["outputs"]}
    manifest = {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha(SOURCE),
                "renderer": "scripts/build_website_teaser.py", "renderer_sha256": sha(Path(__file__)),
                "scope": "Website themes of the current manuscript teaser; native layout, recorded frames and icon viewports preserved",
                "dark_panel_palette": PALETTE, "inputs": {"paper/" + name: digest for name, digest in native["inputs"].items()},
                "canonical_figure": "paper/revision-v8/figures/teaser.pdf",
                "canonical_figure_sha256": protected["revision-v8/figures/teaser.pdf"],
                "platforms": native["platforms"], "disturbances": native["disturbances"],
                "removed_from_layout": native["removed_from_layout"], "variants": variants}
    (DESTINATION / "project-overview-themes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(variants, indent=2))


if __name__ == "__main__":
    main()
