"""CLI: render a debug pass that scans an outward blue glow across half-blocks.

Every window half is filled with a fixed cool blue variant. A single glow pulse
then walks the half-blocks in scan order, blooming outward from each half-block
bounding box before fading away and moving on to the next half.

Example::

    debug-draw-half-glow --output output/half_glow_debug.webm
    debug-draw-half-glow --duration 10 --output output/half_glow_long.webm
"""

from __future__ import annotations

import argparse
import math
import signal
from dataclasses import dataclass
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
from video_mapping.layout import Half, Layout
from video_mapping.render import VideoWriter

if TYPE_CHECKING:
    from video_mapping.types import RGBColor

DEFAULT_OUTPUT = Path("output/half_glow_debug.webm")
DEFAULT_DURATION = 8.0
_BASE_HALF_COLOR = (82, 132, 214)
_EDGE_HALF_COLOR = (132, 192, 250)
_OUTER_GLOW_COLOR = (28, 70, 132)
_OUTER_GLOW_EDGE_COLOR = (52, 114, 196)
_INNER_GLOW_COLOR = (122, 184, 246)
_INNER_GLOW_EDGE_COLOR = (186, 226, 255)
_HALF_FILL_ALPHA = 0.42

stop_event = Event()


@dataclass(frozen=True, slots=True)
class _HalfGlowState:
    half: Half
    fill_color: RGBColor
    outer_glow_color: RGBColor
    inner_glow_color: RGBColor


def _handle_sigint(_: int, __: object) -> None:
    stop_event.set()
    print("\nStopping after current frame...")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render blue-lit half-block windows with animated outward glow.",
    )
    _ = parser.add_argument(
        "--background",
        choices=("transparent", "mask", "black"),
        default="transparent",
        help="Background mode (default: transparent).",
    )
    _ = parser.add_argument(
        "--mask-path",
        type=Path,
        default=DEFAULT_MASK_IMAGE_PATH,
        help="Background mask image for --background mask (default: color-mask.png)",
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
        default=DEFAULT_OUTPUT,
        help="Output video path (default: output/half_glow_debug.webm)",
    )
    _ = parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    _ = parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Duration in seconds (default: {DEFAULT_DURATION:g})",
    )
    return parser.parse_args()


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = max(0.0, min(1.0, (value - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _mix_colors(color_a: RGBColor, color_b: RGBColor, amount: float) -> RGBColor:
    mix = max(0.0, min(1.0, amount))
    return (
        round(color_a[0] + (color_b[0] - color_a[0]) * mix),
        round(color_a[1] + (color_b[1] - color_a[1]) * mix),
        round(color_a[2] + (color_b[2] - color_a[2]) * mix),
    )


def _prepare_half_states(layout: Layout) -> tuple[_HalfGlowState, ...]:
    half_states: list[_HalfGlowState] = []
    for row_idx, row in enumerate(layout.rows):
        for block_idx, block in enumerate(row.blocks):
            for half_idx, half in enumerate(block.halves):
                edge_distance = min(block_idx, len(row.blocks) - 1 - block_idx)
                edge_mix = 1.0 - edge_distance / max(1, len(row.blocks) // 2)
                tone_phase = 0.5 + 0.5 * math.sin(row_idx * 0.9 + block_idx * 0.58 + half_idx * 1.45)
                fill_mix = min(1.0, 0.14 + edge_mix * 0.22 + tone_phase * 0.34)
                outer_mix = min(1.0, 0.08 + edge_mix * 0.18 + tone_phase * 0.28)
                inner_mix = min(1.0, 0.20 + edge_mix * 0.16 + tone_phase * 0.42)
                half_states.append(
                    _HalfGlowState(
                        half=half,
                        fill_color=_mix_colors(_BASE_HALF_COLOR, _EDGE_HALF_COLOR, fill_mix),
                        outer_glow_color=_mix_colors(_OUTER_GLOW_COLOR, _OUTER_GLOW_EDGE_COLOR, outer_mix),
                        inner_glow_color=_mix_colors(_INNER_GLOW_COLOR, _INNER_GLOW_EDGE_COLOR, inner_mix),
                    )
                )
    return tuple(half_states)


def _scan_pulse(progress: float, half_count: int) -> tuple[int | None, float]:
    if half_count <= 0:
        return (None, 0.0)

    scan_pos = progress * half_count
    active_idx = min(half_count - 1, int(scan_pos))
    local_progress = scan_pos - active_idx
    if progress >= 1.0:
        local_progress = 1.0

    pulse = math.sin(math.pi * local_progress)
    return (active_idx, max(0.0, pulse))


def _create_base_canvas(background: str, mask_path: Path) -> tuple[Canvas, bool]:
    if background == "mask":
        return Canvas.from_image(mask_path), False
    if background == "black":
        return Canvas.black(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT), False
    return Canvas.transparent(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT), True


def _draw_half_glow_frame(canvas: Canvas, halves: tuple[_HalfGlowState, ...], progress: float) -> None:
    for half_state in halves:
        canvas.color_half(half_state.half, half_state.fill_color, alpha=_HALF_FILL_ALPHA)

    active_idx, pulse = _scan_pulse(progress, len(halves))
    if active_idx is None or pulse <= 0.0:
        return

    active_half = halves[active_idx]
    pulse_strength = _smoothstep(0.0, 1.0, pulse)
    outer_alpha = 0.22 * pulse_strength
    inner_alpha = 0.30 * pulse_strength
    outer_radius = max(12, round(22 + 54 * pulse_strength))
    inner_radius = max(8, round(10 + 26 * pulse_strength))
    bbox = active_half.half.bbox()
    canvas.blend_outer_glow_rect(
        bbox,
        active_half.outer_glow_color,
        radius=outer_radius,
        alpha=outer_alpha,
        falloff_power=2.3,
    )
    canvas.blend_outer_glow_rect(
        bbox,
        active_half.inner_glow_color,
        radius=inner_radius,
        alpha=inner_alpha,
        falloff_power=1.4,
    )


def draw_half_glow_video(
    layout_path: Path,
    output_path: Path,
    *,
    background: str = "transparent",
    mask_path: Path = DEFAULT_MASK_IMAGE_PATH,
    fps: int = DEFAULT_FPS,
    duration: float = DEFAULT_DURATION,
) -> None:
    """Render the half-block glow debug video to ``output_path``."""
    if duration <= 0.0:
        msg = "Duration must be positive"
        raise ValueError(msg)

    layout = Layout.from_json(layout_path)
    half_states = _prepare_half_states(layout)
    base, transparent_output = _create_base_canvas(background, mask_path)
    total_frames = max(1, round(duration * fps))

    with VideoWriter(
        output_path,
        width=base.width,
        height=base.height,
        fps=fps,
        total_frames=total_frames,
        transparent=transparent_output,
        progress_desc="Half glow debug",
    ) as writer:
        for frame_idx in range(total_frames):
            if stop_event.is_set():
                writer.log(f"Interrupted at frame {frame_idx}")
                break

            progress = frame_idx / max(1, total_frames - 1)
            canvas = base.copy()
            _draw_half_glow_frame(canvas, half_states, progress)
            if not writer.write_canvas(canvas):
                writer.log("FFmpeg terminated early.")
                break

    print(f"Saved half-glow debug video -> {output_path}  ({total_frames / fps:.1f} s, {total_frames} frames)")


def main() -> None:
    """Generate the half-block glow debug video."""
    _ = signal.signal(signal.SIGINT, _handle_sigint)
    args = _parse_args()
    draw_half_glow_video(
        args.layout,
        args.output,
        background=args.background,
        mask_path=args.mask_path,
        fps=args.fps,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
