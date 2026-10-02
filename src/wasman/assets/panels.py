"""Local, sourced PBR panel variants; geometry is identical across styles."""

import os
from pathlib import Path

PANEL_STYLES = ("ship_green", "oxidized_steel", "harbor_concrete")
PANEL_ROOT = Path(__file__).parent / "data" / "panels"


def panel_path(style=None):
    style = style or os.environ.get("WASMAN_PANEL_STYLE", "ship_green")
    if style not in PANEL_STYLES:
        raise ValueError(f"Unknown panel style {style!r}; choose one of {PANEL_STYLES}")
    return PANEL_ROOT / style / "panel.usda"


def centered_panel_pose(default_pose, mechanism_pose):
    """Keep the wall plane's X coordinate; center its Y/Z on the fixture root."""
    pose = default_pose.clone()
    pose[:, 1:3] = mechanism_pose[:, 1:3]
    return pose
