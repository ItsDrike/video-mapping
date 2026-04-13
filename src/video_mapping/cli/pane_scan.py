"""CLI: render a video that highlights each pane one-by-one in scan order.

Each pane lights up briefly (left-to-right, top-to-bottom) so you can visually
verify that the extracted pane geometry maps correctly onto the building facade.

Example::

    pane-scan --image static/color-mask.png --output output/pane_scan.mp4
"""

from __future__ import annotations

import argparse
import signal
from pathlib import Path
from threading import Event

from video_mapping.canvas import Canvas
from video_mapping.layout import Layout
from video_mapping.render import VideoWriter

DEFAULT_PANES_JSON = Path("static/panes.json")
DEFAULT_IMAGE = Path("static/color-mask.png")
DEFAULT_OUTPUT = Path("output/pane_scan.mp4")
DEFAULT_HIGHLIGHT_COLOR: tuple[int, int, int] = (255, 225, 70)

stop_event = Event()


def _handle_sigint(_: int, __: object) -> None:
    stop_event.set()
    print("\nStopping after current pane...")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a video highlighting panes one-by-one in scan order.",
    )
    _ = parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    _ = parser.add_argument("--panes", type=Path, default=DEFAULT_PANES_JSON)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument("--fps", type=int, default=40)
    _ = parser.add_argument("--blink-seconds", type=float, default=0.025, metavar="SECS")
    _ = parser.add_argument("--alpha", type=float, default=0.85)
    _ = parser.add_argument("--preset", default="ultrafast", help="libx264 preset.")
    _ = parser.add_argument(
        "--black",
        action="store_true",
        help="Use a black background instead of the mask image.",
    )
    return parser.parse_args()


def main() -> None:
    """Render a video highlighting each pane one-by-one in scan order."""
    _ = signal.signal(signal.SIGINT, _handle_sigint)
    args = _parse_args()

    layout = Layout.from_json(args.panes)
    panes = list(layout.iter_scan_order())
    print(f"Loaded {len(panes)} panes from {args.panes}")

    if args.black:
        # Peek at image dimensions if we're not loading it as background
        probe = Canvas.from_image(args.image)
        base = Canvas.black(probe.width, probe.height)
    else:
        base = Canvas.from_image(args.image)

    frames_per_pane = max(1, round(args.fps * args.blink_seconds))
    total_frames = len(panes) * frames_per_pane
    print(f"Rendering {len(panes)} panes x {frames_per_pane} frames = {total_frames} frames total...")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with VideoWriter(
        args.output,
        width=base.width,
        height=base.height,
        fps=args.fps,
        preset=args.preset,
    ) as writer:
        try:
            for pane_idx, pane in enumerate(panes):
                if stop_event.is_set():
                    print(f"Interrupted at pane {pane_idx}")
                    break

                canvas = base.copy()
                canvas.color_pane(pane, DEFAULT_HIGHLIGHT_COLOR, alpha=args.alpha)
                frame_bytes = canvas.to_array()

                for _ in range(frames_per_pane):
                    if not writer.write_array(frame_bytes):
                        print("FFmpeg terminated early.")
                        break

                if pane_idx % 100 == 0:
                    print(f"  pane {pane_idx + 1}/{len(panes)}")
        except BrokenPipeError:
            print("FFmpeg terminated early.")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
