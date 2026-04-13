"""CLI: extract pane geometry from a building-facade mask image.

Reads a mask PNG where window panes appear as bright rectangles on a dark
background, detects their bounding boxes, organises them into the
row → block → half → pane-row hierarchy, and writes ``panes.json``.

Run this whenever the mask changes. The output file is referenced by all other
CLI tools and by ``Layout.from_json``.

Example::

    extract-panes --mask static/color-mask-bg.png --output static/panes.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from video_mapping.extract import extract_panes

DEFAULT_MASK = Path("static/color-mask-bg.png")
DEFAULT_OUTPUT = Path("static/panes.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pane geometry from a facade mask image into panes.json.",
    )
    _ = parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Extract pane geometry from a building-facade mask image."""
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = extract_panes(args.mask, args.output)
    print(f"Extracted {count} panes → {args.output}")


if __name__ == "__main__":
    main()
