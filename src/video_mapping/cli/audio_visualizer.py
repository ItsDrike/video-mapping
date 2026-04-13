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
from video_mapping.layout import Layout
from video_mapping.render import VideoWriter

# Visualisation defaults
DEFAULT_FPS = 30
DEFAULT_BAR_COLOR: tuple[int, int, int] = (0, 255, 0)
DEFAULT_GLOW_COLOR: tuple[int, int, int] = (255, 200, 50)  # warm yellow-orange
DEFAULT_PANES_JSON = Path("static/panes.json")
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an audio-reactive pillar visualiser video.",
    )
    _ = parser.add_argument("--audio", type=Path, required=True, help="Input WAV file.")
    _ = parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Background image (default: transparent canvas). Pass the color mask for debug rendering.",
    )
    _ = parser.add_argument("--panes", type=Path, default=DEFAULT_PANES_JSON)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    _ = parser.add_argument("--width", type=int, default=4096)
    _ = parser.add_argument("--height", type=int, default=606)
    _ = parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional max output duration in seconds (default: full audio length).",
    )
    _ = parser.add_argument("--hop-size", type=int, default=1024)
    _ = parser.add_argument(
        "--vf",
        default="pad=width=4096:height=606:x=0:y=0",
        help="ffmpeg -vf filter string.",
    )
    return parser.parse_args()


def main() -> None:
    """Render an audio-reactive bar visualiser onto the building facade."""
    _ = signal.signal(signal.SIGINT, _handle_sigint)
    args = _parse_args()

    layout = Layout.from_json(args.panes)
    num_bands = len(layout.pillars)

    print("Processing audio...")
    sample_rate, band_heights, beats = process_audio(
        args.audio,
        num_bands=num_bands,
        bar_height=args.height - 1,
        hop_size=args.hop_size,
    )

    fft_fps = sample_rate / args.hop_size
    n_fft_frames = len(band_heights)
    n_video_frames = int(n_fft_frames / fft_fps * args.fps)
    if args.duration is not None:
        n_video_frames = min(n_video_frames, int(args.duration * args.fps))

    if args.image is not None:
        base = Canvas.from_image(args.image)
        transparent_output = False
    else:
        base = Canvas.transparent(args.width, args.height)
        transparent_output = True

    print(f"Rendering {n_video_frames} frames (Ctrl+C to stop)...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with VideoWriter(
        args.output,
        width=base.width,
        height=base.height,
        fps=args.fps,
        audio_path=args.audio,
        vf_filter=None if transparent_output else args.vf,
        preset=None,
        input_pix_fmt="rgba" if transparent_output else "rgb24",
        output_codec="libvpx-vp9",
        output_pix_fmt="yuva420p" if transparent_output else "yuv420p",
        audio_codec="libopus",
    ) as writer:
        for frame_idx in range(n_video_frames):
            if stop_event.is_set():
                print(f"Interrupted at frame {frame_idx}")
                break

            fft_idx = min(int(frame_idx / args.fps * fft_fps), n_fft_frames - 1)
            frame_heights = band_heights[fft_idx]

            canvas = base.copy()

            for pillar, bar_height in zip(layout.pillars, frame_heights, strict=True):
                canvas.fill_pillar_bar(pillar, int(bar_height), DEFAULT_BAR_COLOR)

            glow_strength = float(beats[fft_idx])
            _apply_glow(canvas, layout, glow_strength, DEFAULT_GLOW_COLOR)

            if not writer.write_canvas(canvas):
                print("FFmpeg finished (audio ended).")
                break

            if frame_idx % 50 == 0:
                print(f"  frame {frame_idx}/{n_video_frames}  beat={glow_strength:.3f}")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
