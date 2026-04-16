"""Project-wide fixed constants for the target building setup."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


def _resolve_static_dir() -> Path:
    repo_static = PROJECT_ROOT / "static"
    package_static = PACKAGE_DIR / "static"
    cwd_static = Path.cwd() / "static"

    candidates = (repo_static, package_static, cwd_static)
    for candidate in candidates:
        if (candidate / "layout.json").exists() and (candidate / "color-mask.png").exists():
            return candidate

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return repo_static


STATIC_DIR = _resolve_static_dir()
OUTPUT_DIR = Path.cwd() / "output"

DEFAULT_LAYOUT_JSON_PATH = STATIC_DIR / "layout.json"
DEFAULT_MASK_IMAGE_PATH = STATIC_DIR / "color-mask.png"

DEFAULT_CANVAS_WIDTH = 4096
DEFAULT_CANVAS_HEIGHT = 606
DEFAULT_FPS = 25
