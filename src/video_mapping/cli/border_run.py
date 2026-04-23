"""CLI: animate a running band around the building perimeter.

A solid band of configurable length (default: one pillar's width) travels
clockwise around the outer boundary of the building facade:

  top wall (left to right) -> right pillar (top to bottom)
  -> bottom wall (right to left) -> left pillar (bottom to top)

At corners the band turns naturally: the band width becomes the band height as
it transitions from a horizontal wall to a vertical pillar, forming a smooth
L-shaped highlight.

Example::

    border-run --output output/border_run.webm
    border-run --speed 12 --loops 2 --output output/border_run_fast.webm
    border-run --band-len 60 --color 0 230 230 --output output/border_run_cyan.webm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from video_mapping.canvas import Canvas
from video_mapping.constants import DEFAULT_LAYOUT_JSON_PATH, DEFAULT_MASK_IMAGE_PATH
from video_mapping.layout import Layout
from video_mapping.perimeter import Perimeter
from video_mapping.render import VideoWriter

_FPS = 25
_DEFAULT_SPEED = 8.0  # pixels per frame
_DEFAULT_COLOR = (255, 220, 0)  # warm yellow
_DEFAULT_ALPHA = 0.90


def border_run(
    mask_path: Path,
    layout_path: Path,
    output_path: Path,
    *,
    speed: float = _DEFAULT_SPEED,
    loops: int = 1,
    color: tuple[int, int, int] = _DEFAULT_COLOR,
    alpha: float = _DEFAULT_ALPHA,
    band_len: int | None = None,
) -> None:
    """Render the running-band perimeter animation to *output_path*.

    Args:
        mask_path: Path to the background mask image.
        layout_path: Path to the layout.json file.
        output_path: Destination for the output video.
        speed: Band travel speed in pixels per frame.
        loops: Number of complete circuits of the perimeter.
        color: RGB color tuple for the band.
        alpha: Opacity of the band (0.0 transparent, 1.0 fully opaque).
        band_len: Length of the band in pixels.  Defaults to one pillar's width.
    """
    layout = Layout.from_json(layout_path)
    perimeter = Perimeter.from_layout(layout)

    if band_len is None:
        band_len = layout.pillars[0].width

    total_frames = round(perimeter.total_length * loops / speed)
    base = Canvas.from_image(mask_path)

    with VideoWriter(
        output_path,
        width=base.width,
        height=base.height,
        fps=_FPS,
        total_frames=total_frames,
        transparent=False,
        progress_desc="Border run animation",
    ) as writer:
        for frame_idx in range(total_frames):
            canvas = base.copy()
            head = (frame_idx * speed) % perimeter.total_length
            for rect in perimeter.band_rects(head, band_len):
                canvas.color_region(rect, color, alpha)
            _ = writer.write_canvas(canvas)

    duration = total_frames / _FPS
    print(f"Saved border-run animation -> {output_path}  ({duration:.1f} s, {total_frames} frames)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate a running band around the building perimeter.",
    )
    _ = parser.add_argument(
        "--mask",
        type=Path,
        default=DEFAULT_MASK_IMAGE_PATH,
        help="Background mask image (default: color-mask.png)",
    )
    _ = parser.add_argument(
        "--layout",
        type=Path,
        default=DEFAULT_LAYOUT_JSON_PATH,
        help="Path to layout.json (default: static/layout.json)",
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/border_run.webm"),
        help="Output video path (default: output/border_run.webm)",
    )
    _ = parser.add_argument(
        "--speed",
        type=float,
        default=_DEFAULT_SPEED,
        help=f"Band travel speed in pixels per frame (default: {_DEFAULT_SPEED})",
    )
    _ = parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Number of complete circuits of the perimeter (default: 1)",
    )
    _ = parser.add_argument(
        "--color",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=list(_DEFAULT_COLOR),
        help="Band color as R G B values 0-255 (default: 255 220 0)",
    )
    _ = parser.add_argument(
        "--alpha",
        type=float,
        default=_DEFAULT_ALPHA,
        help=f"Band opacity 0.0-1.0 (default: {_DEFAULT_ALPHA})",
    )
    _ = parser.add_argument(
        "--band-len",
        type=int,
        default=None,
        help="Band length in pixels (default: one pillar width)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the border-run animation."""
    args = _parse_args()
    r, g, b = args.color
    border_run(
        args.mask,
        args.layout,
        args.output,
        speed=args.speed,
        loops=args.loops,
        color=(r, g, b),
        alpha=args.alpha,
        band_len=args.band_len,
    )


if __name__ == "__main__":
    main()
