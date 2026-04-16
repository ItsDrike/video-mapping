"""CLI: render a debug image showing each half-block colored by position.

Left halves are red and right halves are blue in the top row (reversed in the
bottom row) so you can visually confirm that block/half boundaries are correct.

Example::

    debug-draw-halves --output output/halves_debug.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from video_mapping.canvas import Canvas
from video_mapping.constants import (
    DEFAULT_CANVAS_HEIGHT,
    DEFAULT_CANVAS_WIDTH,
    DEFAULT_LAYOUT_JSON_PATH,
    DEFAULT_MASK_IMAGE_PATH,
)
from video_mapping.layout import Layout

DEFAULT_OUTPUT = Path("output/halves_debug.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a debug image with half-blocks color-coded by position.",
    )
    _ = parser.add_argument(
        "--mask",
        action="store_true",
        help="Render over the fixed building mask (default: transparent background).",
    )
    _ = parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT_JSON_PATH)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Render a debug image showing each half-block color-coded by position."""
    args = _parse_args()

    layout = Layout.from_json(args.layout)

    if args.mask:
        canvas = Canvas.from_image(DEFAULT_MASK_IMAGE_PATH)
    else:
        canvas = Canvas.transparent(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT)

    for row_idx, row in enumerate(layout.rows):
        for block in row.blocks:
            for half_idx, half in enumerate(block.halves):
                # Top row: left=red, right=blue. Bottom row: flipped.
                if row_idx == 0:
                    color = (255, 0, 0) if half_idx == 0 else (0, 0, 255)
                else:
                    color = (0, 0, 255) if half_idx == 0 else (255, 0, 0)

                canvas.color_half(half, color)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
