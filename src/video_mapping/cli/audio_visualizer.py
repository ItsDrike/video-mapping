"""CLI: render an audio-reactive bar visualiser onto the building facade.

Each structural pillar shows a frequency band as a rising bar whose height
tracks the audio energy in that band. On beats, all window panes flash with
a warm glow.

Example::

    audio-visualizer --audio audio.wav --output output/audio_vis.mp4
"""

from __future__ import annotations

import argparse
import signal
from pathlib import Path
from threading import Event

from video_mapping.audio import process_audio
from video_mapping.canvas import Canvas
from video_mapping.constants import DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, DEFAULT_FPS, DEFAULT_MASK_IMAGE_PATH
from video_mapping.layout import Layout, Pillar
from video_mapping.render import VideoWriter

# Visualisation defaults
DEFAULT_BAR_BOTTOM_COLOR: tuple[int, int, int] = (0, 255, 0)
DEFAULT_BAR_TOP_COLOR: tuple[int, int, int] = (255, 0, 0)
DEFAULT_GLOW_COLOR: tuple[int, int, int] = (255, 200, 50)  # warm yellow-orange
DEFAULT_OUTPUT = Path("output/audio_visualizer.webm")

stop_event = Event()


def _handle_sigint(_: int, __: object) -> None:
    stop_event.set()
    print("\nStopping after current frame...")


def _apply_glow(
    canvas: Canvas,
    layout: Layout,
    strength: float,
    color: tuple[int, int, int],
) -> None:
    if strength < 0.05:
        return
    alpha = min(0.9, strength * 1.5)
    canvas.color_panes(layout.all_panes(), color, alpha=alpha)


def _lerp_channel(start: int, end: int, t: float) -> int:
    return int(start + (end - start) * t)


def _bar_gradient_color_at_y(
    y: int,
    canvas_height: int,
    bottom_color: tuple[int, int, int],
    top_color: tuple[int, int, int],
) -> tuple[int, int, int]:
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
    bar_height: int,
    bottom_color: tuple[int, int, int],
    top_color: tuple[int, int, int],
) -> None:
    clamped_height = max(0, min(int(bar_height), canvas.height))
    if clamped_height == 0:
        return

    y_start = canvas.height - clamped_height
    for y in range(y_start, canvas.height):
        row_color = _bar_gradient_color_at_y(y, canvas.height, bottom_color, top_color)
        canvas.fill_rect(pillar.x_start, y, pillar.x_end, y, row_color)


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
    return parser.parse_args()


def main() -> None:
    """Render an audio-reactive bar visualiser onto the building facade."""
    _ = signal.signal(signal.SIGINT, _handle_sigint)
    args = _parse_args()

    layout = Layout.default()
    num_bands = len(layout.pillars)

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
                    int(bar_height),
                    DEFAULT_BAR_BOTTOM_COLOR,
                    DEFAULT_BAR_TOP_COLOR,
                )

            glow_strength = float(beats[fft_idx])
            _apply_glow(canvas, layout, glow_strength, DEFAULT_GLOW_COLOR)

            if not writer.write_canvas(canvas):
                writer.log("FFmpeg finished (audio ended).")
                break

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
