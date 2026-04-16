"""CLI: render a video that highlights each pane one-by-one in scan order.

Each pane lights up briefly (left-to-right, top-to-bottom) so you can visually
verify that the extracted pane geometry maps correctly onto the building facade.

Example::

    debug-pane-scan --mask --output output/pane_scan.webm
"""

from __future__ import annotations

import argparse
import signal
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from video_mapping.canvas import Canvas
from video_mapping.constants import (
    DEFAULT_CANVAS_HEIGHT,
    DEFAULT_CANVAS_WIDTH,
    DEFAULT_FPS,
    DEFAULT_LAYOUT_JSON_PATH,
    DEFAULT_MASK_IMAGE_PATH,
)
from video_mapping.layout import Layout
from video_mapping.render import VideoWriter

if TYPE_CHECKING:
    from video_mapping.types import RGBColor

DEFAULT_OUTPUT = Path("output/pane_scan.webm")
DEFAULT_HIGHLIGHT_COLOR: RGBColor = (255, 225, 70)

stop_event = Event()


def _handle_sigint(_: int, __: object) -> None:
    stop_event.set()
    print("\nStopping after current pane...")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a video highlighting panes one-by-one in scan order.",
    )
    _ = parser.add_argument(
        "--mask",
        action="store_true",
        help="Render over the fixed building mask (default: transparent background).",
    )
    _ = parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT_JSON_PATH)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    _ = parser.add_argument("--blink-seconds", type=float, default=0.025, metavar="SECS")
    _ = parser.add_argument("--alpha", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    """Render a video highlighting each pane one-by-one in scan order."""
    _ = signal.signal(signal.SIGINT, _handle_sigint)
    args = _parse_args()

    layout = Layout.from_json(args.layout)
    panes = list(layout.iter_scan_order())
    print(f"Loaded {len(panes)} panes from {args.layout}")

    if args.mask:
        base = Canvas.from_image(DEFAULT_MASK_IMAGE_PATH)
        transparent_output = False
    else:
        base = Canvas.transparent(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT)
        transparent_output = True

    frames_per_pane = max(1, round(args.fps * args.blink_seconds))
    total_frames = len(panes) * frames_per_pane
    print(f"Rendering {len(panes)} panes x {frames_per_pane} frames = {total_frames} frames total...")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with VideoWriter(
        args.output,
        width=base.width,
        height=base.height,
        fps=args.fps,
        total_frames=total_frames,
        transparent=transparent_output,
    ) as writer:
        try:
            for pane_idx, pane in enumerate(panes):
                if stop_event.is_set():
                    writer.log(f"Interrupted at pane {pane_idx}")
                    break

                canvas = base.copy()
                canvas.color_pane(pane, DEFAULT_HIGHLIGHT_COLOR, alpha=args.alpha)
                frame_bytes = canvas.to_array()

                for _ in range(frames_per_pane):
                    if not writer.write_array(frame_bytes):
                        writer.log("FFmpeg terminated early.")
                        break
        except BrokenPipeError:
            writer.log("FFmpeg terminated early.")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
