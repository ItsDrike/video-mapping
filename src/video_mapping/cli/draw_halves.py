"""CLI: render a debug image showing each half-block colored by position.

Left halves are red and right halves are blue in the top row (reversed in the
bottom row) so you can visually confirm that block/half boundaries are correct.

Example::

    draw-halves --output output/halves_debug.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from video_mapping.canvas import Canvas
from video_mapping.layout import Layout

DEFAULT_IMAGE = Path("static/color-mask.png")
DEFAULT_PANES_JSON = Path("static/panes.json")
DEFAULT_OUTPUT = Path("output/halves_debug.png")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a debug image with half-blocks color-coded by position.",
    )
    _ = parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    _ = parser.add_argument("--panes", type=Path, default=DEFAULT_PANES_JSON)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument(
        "--black",
        action="store_true",
        help="Use a black background instead of the mask image.",
    )
    return parser.parse_args()


def main() -> None:
    """Render a debug image showing each half-block color-coded by position."""
    args = _parse_args()

    layout = Layout.from_json(args.panes)

    if args.black:
        probe = Canvas.from_image(args.image)
        canvas = Canvas.black(probe.width, probe.height)
    else:
        canvas = Canvas.from_image(args.image)

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
