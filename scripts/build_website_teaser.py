"""Render theme variants from the editable figure, preserving source photographs.

Only the first figure and its existing helpers are evaluated. Publication
outputs go to the website; the manuscript and archived figures are untouched.
"""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/build_revision_overview.py"
DESTINATION = ROOT / "website/src/assets"
PALETTE = {"#dcecf7": "#183448", "#edf5fa": "#0e2536", "#f1f4f6": "#0c2231"}


class DarkPanels(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in PALETTE:
            return ast.copy_location(ast.Constant(PALETTE[node.value]), node)
        return node


def main():
    content = SOURCE.read_text()
    helpers, figures = content.split("# Figure 1.", 1)
    figure, _ = figures.split("# Figure 2.", 1)
    # The introductory comment belongs to the helper/figure boundary.
    figure = figure.split("\n", 1)[1]
    DESTINATION.mkdir(parents=True, exist_ok=True)
    variants = {}
    for theme in ("light", "dark"):
        namespace = {"__file__": str(SOURCE)}
        exec(compile(helpers, str(SOURCE), "exec"), namespace)
        namespace["plt"].rcParams["svg.hashsalt"] = "wasserman-website-teaser"
        if theme == "dark":
            namespace["INK"] = "#edf7ff"
        output = DESTINATION / f"project-overview-{theme}.svg"

        def save(fig, name, output=output, namespace=namespace):
            if name != "teaser":
                raise ValueError("Only the website teaser may be rendered")
            fig.savefig(output, dpi=300, transparent=True, metadata={"Date": None})
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
            fig.savefig(output.with_suffix(".png"), dpi=220, transparent=True)
            namespace["plt"].close(fig)

        namespace["save"] = save
        tree = ast.parse(figure, filename=str(SOURCE))
        if theme == "dark":
            tree = ast.fix_missing_locations(DarkPanels().visit(tree))
        exec(compile(tree, str(SOURCE), "exec"), namespace)
        variants[theme] = {"asset": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
        variants[theme]["github_asset"] = output.with_suffix(".png").name
        variants[theme]["github_sha256"] = hashlib.sha256(output.with_suffix(".png").read_bytes()).hexdigest()
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "renderer": "scripts/build_website_teaser.py",
        "scope": (
            "Website palette variants of the editable teaser; "
            "original photographs, viewports, geometry and labels retained"
        ),
        "dark_panel_palette": PALETTE,
        "inputs": namespace["inputs"],
        "variants": variants,
    }
    (DESTINATION / "project-overview-themes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(variants, indent=2))


if __name__ == "__main__":
    main()
