"""CLI: render a 30-second generative blinking animation.

Layers several simultaneous effects:
- 2D interference wave field (geometric, hue-shifting)
- Horizontal scan beam (ping-pong sweep)
- Block strobe (geometric, phase-offset per block)
- Seeded random sparkles
- Pillar bars with organic multi-harmonic motion

Example::

    blink-animation --output output/blink_animation.mp4
"""

from __future__ import annotations

import argparse
import math
import signal
from pathlib import Path
from threading import Event

import numpy as np

from video_mapping.canvas import Canvas
from video_mapping.constants import DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, DEFAULT_FPS, DEFAULT_MASK_IMAGE_PATH
from video_mapping.layout import Layout, Pane
from video_mapping.render import VideoWriter
from video_mapping.types import RGBColor

DURATION = 30.0
DEFAULT_OUTPUT = Path("output/blink_animation.webm")
SEED = 42

_NUM_BUILDING_ROWS = 2
_NUM_PANE_ROWS = 6
_TOTAL_PANE_ROWS = _NUM_BUILDING_ROWS * _NUM_PANE_ROWS  # 12
_NUM_COLS = 66  # 11 blocks x 2 halves x 3 columns
_END_SCAN_FRACTION = 4 / DURATION

stop_event = Event()


def _handle_sigint(_: int, __: object) -> None:
    stop_event.set()
    print("\nStopping after current frame...")


def _hsv_to_rgb(h: float, s: float, v: float) -> RGBColor:
    """Convert HSV (each 0.0-1.0) to an RGB triple (0-255)."""
    if s == 0.0:
        vi = int(v * 255)
        return (vi, vi, vi)
    h6 = h * 6.0
    i = int(h6) % 6
    f = h6 - int(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (int(r * 255), int(g * 255), int(b * 255))


def _pane(layout: Layout, building_row: int, pane_row: int, col: int) -> Pane:
    return layout.rows[building_row].blocks[col // 6].halves[(col % 6) // 3].pane_at(pane_row, col % 3)


def _render_frame(
    canvas: Canvas,
    base: Canvas,
    layout: Layout,
    t: float,
    rng: np.random.Generator,
    draw_pillar_bars: bool,
    end_scan_panes: list[Pane],
    end_scan_progress: float | None,
) -> None:
    # ------------------------------------------------------------------ #
    # Layer 0: 2D interference wave field
    # Three overlapping sine waves at different angles create a moiré-like
    # pattern that slowly drifts and rotates hue over 20 seconds.
    # ------------------------------------------------------------------ #
    global_hue = t / 20.0

    for br in range(_NUM_BUILDING_ROWS):
        for pr in range(_NUM_PANE_ROWS):
            for col in range(_NUM_COLS):
                col_n = col / _NUM_COLS
                row_n = (br * _NUM_PANE_ROWS + pr) / _TOTAL_PANE_ROWS

                w1 = math.sin(2 * math.pi * (col_n * 2.0 - t * 0.40))
                w2 = math.sin(2 * math.pi * (row_n * 3.0 - t * 0.20))
                w3 = math.sin(2 * math.pi * (col_n * 0.5 + row_n * 1.5 - t * 0.28))

                brightness = max(0.0, (w1 + w2 + w3) / 3.0 * 0.5 + 0.28)
                hue = (global_hue + col_n * 0.35 + row_n * 0.12) % 1.0
                canvas.color_pane(_pane(layout, br, pr, col), _hsv_to_rgb(hue, 0.88, brightness))

    # ------------------------------------------------------------------ #
    # Layer 1: horizontal scan beam (ping-pong, period 2.5 s)
    # A soft-edged white streak sweeps top-to-bottom and back.
    # ------------------------------------------------------------------ #
    scan_phase = (t % 2.5) / 2.5
    scan_row_f = (1.0 - abs(2.0 * scan_phase - 1.0)) * (_TOTAL_PANE_ROWS - 1)

    for col in range(_NUM_COLS):
        for row_idx in range(_TOTAL_PANE_ROWS):
            dist = abs(row_idx - scan_row_f)
            if dist >= 2.0:
                continue
            alpha = (1.0 - dist / 2.0) * 0.88
            br = row_idx // _NUM_PANE_ROWS
            pr = row_idx % _NUM_PANE_ROWS
            canvas.color_pane(_pane(layout, br, pr, col), (215, 235, 255), alpha=alpha)

    # ------------------------------------------------------------------ #
    # Layer 2: block strobe
    # Blocks strobe with a travelling phase wave (1.8 s cycle). At the peak
    # of each block's sine the block flashes a complementary hue.
    # ------------------------------------------------------------------ #
    n_blocks = len(layout.top_row.blocks)
    for block_idx, (block_top, block_bottom) in enumerate(
        zip(layout.top_row.blocks, layout.bottom_row.blocks, strict=True)
    ):
        phase = 2.0 * math.pi * block_idx / n_blocks
        s = math.sin(2.0 * math.pi * t / 1.8 + phase)
        if s <= 0.78:
            continue
        intensity = (s - 0.78) / 0.22
        hue = (0.62 + block_idx / n_blocks * 0.30 + t / 24.0) % 1.0
        color = _hsv_to_rgb(hue, 1.0, 1.0)
        canvas.color_block(block_top, color, alpha=intensity * 0.60)
        canvas.color_block(block_bottom, color, alpha=intensity * 0.60)

    # ------------------------------------------------------------------ #
    # Layer 3: random sparkles
    # The count pulses with a slow sine so density ebbs and flows.
    # ------------------------------------------------------------------ #
    n_sparkles = int(rng.integers(6, 8 + int(14 * abs(math.sin(t * 0.9)))))
    for _ in range(n_sparkles):
        br = int(rng.integers(0, _NUM_BUILDING_ROWS))
        pr = int(rng.integers(0, _NUM_PANE_ROWS))
        col = int(rng.integers(0, _NUM_COLS))
        bright = float(rng.random())
        # Warm white with a slight golden tint at lower brightness
        canvas.color_pane(
            _pane(layout, br, pr, col),
            (255, int(210 + 45 * bright), int(160 + 95 * bright)),
            alpha=bright * 0.92,
        )

    # ------------------------------------------------------------------ #
    # Layer 4: pillar bars — organic multi-harmonic motion
    # Three superimposed sine waves per pillar give a fluid, non-repeating
    # appearance within the 30-second window.
    # ------------------------------------------------------------------ #
    if draw_pillar_bars:
        for i, pillar in enumerate(layout.pillars):
            phase = i / len(layout.pillars) * 2.0 * math.pi
            h = (
                0.44
                + 0.26 * math.sin(2.0 * math.pi * t / 2.3 + phase)
                + 0.16 * math.sin(2.0 * math.pi * t / 1.05 + phase * 1.7)
                + 0.08 * math.sin(2.0 * math.pi * t / 0.52 + phase * 0.9)
            )
            bar_h = int(canvas.height * max(0.04, min(1.0, h)))
            pillar_hue = (0.07 + 0.04 * math.sin(t * 0.55 + i * 0.38)) % 1.0
            canvas.fill_pillar_bar(pillar, bar_h, _hsv_to_rgb(pillar_hue, 1.0, 1.0))

    if end_scan_progress is None:
        return

    progress = max(0.0, min(1.0, end_scan_progress))
    total = len(end_scan_panes)
    retired = min(total, int(progress * total))

    canvas_arr = canvas.to_array()
    base_arr = base.to_array()

    for pane in end_scan_panes[:retired]:
        canvas_arr[pane.y1 : pane.y2 + 1, pane.x1 : pane.x2 + 1] = base_arr[
            pane.y1 : pane.y2 + 1, pane.x1 : pane.x2 + 1
        ]

    if retired < total:
        canvas.color_pane(end_scan_panes[retired], (220, 240, 255), alpha=0.98)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 30-second generative blinking animation.",
    )
    _ = parser.add_argument(
        "--mask",
        action="store_true",
        help="Render over the fixed building mask (default: transparent background).",
    )
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    _ = parser.add_argument("--duration", type=float, default=DURATION, help="Duration in seconds.")
    _ = parser.add_argument("--seed", type=int, default=SEED, help="RNG seed for sparkles.")
    _ = parser.add_argument(
        "--no-pillar-bars",
        action="store_true",
        help="Disable rendering the pillar bar layer.",
    )
    return parser.parse_args()


def main() -> None:
    """Render a generative blinking animation video."""
    _ = signal.signal(signal.SIGINT, _handle_sigint)
    args = _parse_args()

    layout = Layout.default()
    rng = np.random.default_rng(args.seed)

    n_frames = int(args.duration * args.fps)
    if args.mask:
        base = Canvas.from_image(DEFAULT_MASK_IMAGE_PATH)
        transparent_output = False
    else:
        base = Canvas.transparent(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT)
        transparent_output = True

    print(f"Rendering {n_frames} frames ({args.duration:.0f}s @ {args.fps} fps)...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    end_scan_panes = list(layout.iter_scan_order())
    end_scan_seconds = args.duration * _END_SCAN_FRACTION
    end_scan_frames = max(1, round(end_scan_seconds * args.fps))
    end_scan_start_frame = max(0, n_frames - end_scan_frames)

    with VideoWriter(
        args.output,
        width=base.width,
        height=base.height,
        fps=args.fps,
        total_frames=n_frames,
        transparent=transparent_output,
    ) as writer:
        for frame_idx in range(n_frames):
            if stop_event.is_set():
                writer.log(f"Interrupted at frame {frame_idx}")
                break

            t = frame_idx / args.fps
            canvas = base.copy()
            if frame_idx >= end_scan_start_frame and n_frames > 1:
                end_scan_progress = (frame_idx - end_scan_start_frame) / max(1, n_frames - 1 - end_scan_start_frame)
            else:
                end_scan_progress = None

            _render_frame(
                canvas,
                base,
                layout,
                t,
                rng,
                draw_pillar_bars=not args.no_pillar_bars,
                end_scan_panes=end_scan_panes,
                end_scan_progress=end_scan_progress,
            )

            if not writer.write_canvas(canvas):
                writer.log("FFmpeg terminated early.")
                break

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
