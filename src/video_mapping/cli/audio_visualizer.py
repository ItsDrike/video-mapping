"""CLI: render an audio-reactive bar visualiser onto the building facade.

Each structural pillar shows a frequency band as a rising bar whose height
tracks the audio energy in that band. By default, window panes also flash
with a warm glow on beats.

Example::

    audio-visualizer --audio audio.wav --output output/audio_vis.mp4
"""

from __future__ import annotations

import argparse
import signal
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

import numpy as np

from video_mapping.audio import process_audio
from video_mapping.canvas import Canvas
from video_mapping.constants import DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, DEFAULT_FPS, DEFAULT_MASK_IMAGE_PATH
from video_mapping.layout import Layout, Pillar
from video_mapping.render import VideoWriter

if TYPE_CHECKING:
    from video_mapping.layout import Pane
    from video_mapping.types import RGBColor

# Visualisation defaults
DEFAULT_BAR_BOTTOM_COLOR: RGBColor = (0, 255, 0)
DEFAULT_BAR_TOP_COLOR: RGBColor = (255, 0, 0)
DEFAULT_GLOW_COLOR: RGBColor = (255, 200, 50)  # warm yellow-orange
DEFAULT_OUTPUT = Path("output/audio_visualizer.webm")

stop_event = Event()


def _handle_sigint(_: int, __: object) -> None:
    stop_event.set()
    print("\nStopping after current frame...")


def _build_glow_pulses(beats: np.ndarray, *, fft_fps: float) -> np.ndarray:
    """Convert a dense beat envelope into sparser accent pulses for window glow."""
    if len(beats) == 0:
        return beats

    baseline_window = max(1, round(fft_fps * 0.45))
    baseline_kernel = np.ones(baseline_window, dtype=np.float32) / baseline_window
    baseline = np.convolve(beats, baseline_kernel, mode="same")

    novelty = np.maximum(beats - baseline * 1.1, 0.0)
    active_novelty = novelty[novelty > 0.0]
    scale = float(np.quantile(active_novelty, 0.95)) if active_novelty.size else 1.0
    novelty = np.clip(novelty / (scale + 1e-6), 0.0, 1.0)

    threshold = max(0.12, float(np.quantile(novelty, 0.8)) * 0.8)
    min_gap = max(1, round(fft_fps * 0.14))
    triggers = np.zeros_like(novelty)
    last_trigger_idx = -min_gap

    for idx in range(1, len(novelty) - 1):
        strength = float(novelty[idx])
        if strength < threshold:
            continue
        if strength < novelty[idx - 1] or strength < novelty[idx + 1]:
            continue

        if idx - last_trigger_idx < min_gap:
            if strength <= triggers[last_trigger_idx]:
                continue
            triggers[last_trigger_idx] = 0.0

        triggers[idx] = strength
        last_trigger_idx = idx

    decay = 0.82
    pulses = np.zeros_like(triggers)
    pulses[0] = triggers[0]
    for idx in range(1, len(pulses)):
        pulses[idx] = max(triggers[idx], pulses[idx - 1] * decay)

    return pulses


def _build_glow_bed(beats: np.ndarray) -> np.ndarray:
    """Normalize the dense beat envelope into a gentler continuous glow bed."""
    if len(beats) == 0:
        return beats

    floor = float(np.quantile(beats, 0.2))
    ceiling = float(np.quantile(beats, 0.9))
    normalized = np.clip((beats - floor) / max(ceiling - floor, 1e-6), 0.0, 1.0)
    return np.power(normalized, 1.6).astype(np.float32)


def _select_glow_panes(panes: list[Pane], *, coverage: float, frame_idx: int) -> list[Pane]:
    """Pick an evenly distributed moving subset of panes for partial-facade glow."""
    if coverage >= 0.999:
        return panes

    pane_count = len(panes)
    target_count = max(1, min(pane_count, round(pane_count * coverage)))
    start = (frame_idx * 7) % pane_count
    offsets = ((np.arange(target_count, dtype=np.int32) * pane_count) // target_count + start) % pane_count
    return [panes[int(idx)] for idx in offsets]


def _apply_glow(
    canvas: Canvas,
    panes: list[Pane],
    *,
    bed_strength: float,
    pulse_strength: float,
    frame_idx: int,
    color: RGBColor,
) -> None:
    combined_strength = max(bed_strength * 0.55, pulse_strength)
    if combined_strength < 0.04:
        return

    alpha = min(0.9, 0.08 + bed_strength * 0.20 + pulse_strength * 0.65)
    coverage = min(1.0, 0.12 + bed_strength * 0.30 + pulse_strength * 0.68)
    canvas.color_panes(_select_glow_panes(panes, coverage=coverage, frame_idx=frame_idx), color, alpha=alpha)


def _lerp_channel(start: int, end: int, t: float) -> int:
    return int(start + (end - start) * t)


def _bar_gradient_color_at_y(
    y: int,
    *,
    canvas_height: int,
    bottom_color: RGBColor,
    top_color: RGBColor,
) -> RGBColor:
    # y=canvas_height-1 (bottom) -> t=0.0 (green), y=0 (top) -> t=1.0 (red)
    if canvas_height <= 1:
        return bottom_color
    t = (canvas_height - 1 - y) / (canvas_height - 1)
    return (
        _lerp_channel(bottom_color[0], top_color[0], t),
        _lerp_channel(bottom_color[1], top_color[1], t),
        _lerp_channel(bottom_color[2], top_color[2], t),
    )


def _draw_gradient_pillar_bar(
    canvas: Canvas,
    pillar: Pillar,
    *,
    bar_height: int,
    bottom_color: RGBColor,
    top_color: RGBColor,
) -> None:
    clamped_height = max(0, min(int(bar_height), canvas.height))
    if clamped_height == 0:
        return

    y_start = canvas.height - clamped_height
    for y in range(y_start, canvas.height):
        row_color = _bar_gradient_color_at_y(
            y,
            canvas_height=canvas.height,
            bottom_color=bottom_color,
            top_color=top_color,
        )
        canvas.fill_rect(x1=pillar.x_start, y1=y, x2=pillar.x_end, y2=y, color=row_color)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an audio-reactive pillar visualiser video.",
    )
    _ = parser.add_argument("--audio", type=Path, required=True, help="Input WAV file.")
    _ = parser.add_argument(
        "--mask",
        action="store_true",
        help="Render over the fixed building mask (default: transparent background).",
    )
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    _ = parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional max output duration in seconds (default: full audio length).",
    )
    _ = parser.add_argument("--hop-size", type=int, default=1024)
    _ = parser.add_argument(
        "--pillars-only",
        action="store_true",
        help="Disable window glow effects and render only pillar bars.",
    )
    return parser.parse_args()


def main() -> None:
    """Render an audio-reactive bar visualiser onto the building facade."""
    _ = signal.signal(signal.SIGINT, _handle_sigint)
    args = _parse_args()

    layout = Layout.default()
    num_bands = len(layout.pillars)
    panes = layout.all_panes_flat()

    if args.mask:
        base = Canvas.from_image(DEFAULT_MASK_IMAGE_PATH)
        transparent_output = False
    else:
        base = Canvas.transparent(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT)
        transparent_output = True

    print("Processing audio...")
    sample_rate, band_heights, beats = process_audio(
        args.audio,
        num_bands=num_bands,
        bar_height=base.height - 1,
        hop_size=args.hop_size,
    )

    fft_fps = sample_rate / args.hop_size
    glow_bed = _build_glow_bed(beats)
    glow_pulses = _build_glow_pulses(beats, fft_fps=fft_fps)
    n_fft_frames = len(band_heights)
    n_video_frames = int(n_fft_frames / fft_fps * args.fps)
    if args.duration is not None:
        n_video_frames = min(n_video_frames, int(args.duration * args.fps))

    print(f"Rendering {n_video_frames} frames (Ctrl+C to stop)...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with VideoWriter(
        args.output,
        width=base.width,
        height=base.height,
        fps=args.fps,
        total_frames=n_video_frames,
        transparent=transparent_output,
        audio_path=args.audio,
    ) as writer:
        for frame_idx in range(n_video_frames):
            if stop_event.is_set():
                writer.log(f"Interrupted at frame {frame_idx}")
                break

            fft_idx = min(int(frame_idx / args.fps * fft_fps), n_fft_frames - 1)
            frame_heights = band_heights[fft_idx]

            canvas = base.copy()

            for pillar, bar_height in zip(layout.pillars, frame_heights, strict=True):
                _draw_gradient_pillar_bar(
                    canvas,
                    pillar,
                    bar_height=int(bar_height),
                    bottom_color=DEFAULT_BAR_BOTTOM_COLOR,
                    top_color=DEFAULT_BAR_TOP_COLOR,
                )

            if not args.pillars_only:
                _apply_glow(
                    canvas,
                    panes,
                    bed_strength=float(glow_bed[fft_idx]),
                    pulse_strength=float(glow_pulses[fft_idx]),
                    frame_idx=frame_idx,
                    color=DEFAULT_GLOW_COLOR,
                )

            if not writer.write_canvas(canvas):
                writer.log("FFmpeg finished (audio ended).")
                break

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
