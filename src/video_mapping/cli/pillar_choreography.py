"""CLI: render mirrored pillar color choreography synced to audio events.

Pillars start transparent. They appear in mirrored pairs (outside-in) on
significant beat events, with pair colors moving from light green to orange.
After all pairs are visible, colors continue shifting in sync with the music,
then disappear in mirrored center-out pairs near the end.

Example::

    pillar-choreography --audio audio.wav --output output/pillar_choreography.webm
"""

from __future__ import annotations

import argparse
import math
import signal
from pathlib import Path
from threading import Event

import numpy as np

from video_mapping.audio import process_audio
from video_mapping.canvas import Canvas
from video_mapping.constants import DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, DEFAULT_FPS, DEFAULT_MASK_IMAGE_PATH
from video_mapping.layout import Layout
from video_mapping.render import VideoWriter
from video_mapping.types import RGBColor

DEFAULT_DURATION = 15.0
DEFAULT_OUTPUT = Path("output/pillar_choreography.webm")

_START_HUE = 0.34  # brighter light green
_END_HUE = 0.04  # deeper orange

stop_event = Event()


def _handle_sigint(_: int, __: object) -> None:
    stop_event.set()
    print("\nStopping after current frame...")


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> RGBColor:
    """Convert HSV (0..1 floats) to RGB (0..255 ints)."""
    if saturation <= 0.0:
        gray = int(value * 255)
        return (gray, gray, gray)

    h6 = (hue % 1.0) * 6.0
    i = int(h6)
    f = h6 - i
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - saturation * (1.0 - f))

    if i == 0:
        r, g, b = value, t, p
    elif i == 1:
        r, g, b = q, value, p
    elif i == 2:
        r, g, b = p, value, t
    elif i == 3:
        r, g, b = p, q, value
    elif i == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q

    return (int(r * 255), int(g * 255), int(b * 255))


def _build_mirrored_pairs(num_pillars: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    left = 0
    right = num_pillars - 1
    while left < right:
        pairs.append((left, right))
        left += 1
        right -= 1
    if left == right:
        pairs.append((left, right))
    return pairs


def _select_event_frames(
    beat_strengths: np.ndarray,
    *,
    n_events: int,
    fps: int,
    window_start_ratio: float,
    window_end_ratio: float,
    min_start_frame: int = 1,
) -> list[int]:
    n_frames = len(beat_strengths)
    if n_frames <= 0 or n_events <= 0:
        return []

    window_start = max(1, int(n_frames * window_start_ratio), min_start_frame)
    window_end = max(window_start + 1, int(n_frames * window_end_ratio), min_start_frame + 1)
    window_end = min(window_end, n_frames)

    threshold = max(0.10, float(np.percentile(beat_strengths, 75)) * 0.75)
    min_gap = max(1, int(fps * 0.33))

    peaks: list[int] = []
    for idx in range(window_start, window_end - 1):
        strength = float(beat_strengths[idx])
        if strength < threshold:
            continue
        if strength < beat_strengths[idx - 1] or strength < beat_strengths[idx + 1]:
            continue
        if peaks and (idx - peaks[-1] < min_gap):
            if beat_strengths[idx] > beat_strengths[peaks[-1]]:
                peaks[-1] = idx
            continue
        peaks.append(idx)

    fallback = np.linspace(
        window_start,
        max(window_start, window_end - 1),
        n_events,
        dtype=np.int32,
    ).tolist()
    candidates = sorted(set(peaks + fallback))

    reveal_frames: list[int] = []
    for idx in candidates:
        if reveal_frames and idx - reveal_frames[-1] < min_gap:
            continue
        reveal_frames.append(int(idx))
        if len(reveal_frames) == n_events:
            return reveal_frames

    while len(reveal_frames) < n_events:
        if not reveal_frames:
            next_idx = window_start
        else:
            next_idx = min(n_frames - 1, reveal_frames[-1] + min_gap)
        reveal_frames.append(next_idx)

    return reveal_frames


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render mirrored pillar color choreography synced to audio.",
    )
    _ = parser.add_argument("--audio", type=Path, required=True, help="Input WAV file.")
    _ = parser.add_argument(
        "--mask",
        action="store_true",
        help="Render over the fixed building mask (default: transparent background).",
    )
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    _ = parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    _ = parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Max duration in seconds.")
    _ = parser.add_argument(
        "--audio-offset",
        type=float,
        default=0.0,
        help="Start audio (and visuals) at this offset in seconds.",
    )
    _ = parser.add_argument("--hop-size", type=int, default=1024)
    _ = parser.add_argument(
        "--start-end-animation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mirrored reveal/hide animation (use --no-start-end-animation to show all gradients immediately).",
    )
    return parser.parse_args()


def main() -> None:
    """Render mirrored, beat-driven pillar colors for a WAV track."""
    _ = signal.signal(signal.SIGINT, _handle_sigint)
    args = _parse_args()

    layout = Layout.default()
    num_pillars = len(layout.pillars)

    if args.mask:
        base = Canvas.from_image(DEFAULT_MASK_IMAGE_PATH)
        transparent_output = False
    else:
        base = Canvas.transparent(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT)
        transparent_output = True

    print("Processing audio...")
    sample_rate, band_heights, beats = process_audio(
        args.audio,
        num_bands=num_pillars,
        bar_height=base.height - 1,
        hop_size=args.hop_size,
    )

    fft_fps = sample_rate / args.hop_size
    n_fft_frames = len(band_heights)
    start_fft_idx = max(0, min(n_fft_frames - 1, int(args.audio_offset * fft_fps)))
    remaining_fft_frames = max(0, n_fft_frames - start_fft_idx)
    n_video_frames = int(remaining_fft_frames / fft_fps * args.fps)
    n_video_frames = min(n_video_frames, int(args.duration * args.fps))
    if n_video_frames <= 0:
        print("Audio offset is beyond available audio frames; nothing to render.")
        return

    video_indices = np.minimum(
        start_fft_idx + (np.arange(n_video_frames, dtype=np.float32) / args.fps * fft_fps).astype(np.int32),
        n_fft_frames - 1,
    )
    beat_strengths = beats[video_indices].astype(np.float32)
    frame_bands = band_heights[video_indices].astype(np.float32)

    pairs = _build_mirrored_pairs(num_pillars)
    n_pairs = len(pairs)
    if args.start_end_animation:
        reveal_frames = _select_event_frames(
            beat_strengths,
            n_events=n_pairs,
            fps=args.fps,
            window_start_ratio=0.0,
            window_end_ratio=0.7,
        )
        hide_frames = _select_event_frames(
            beat_strengths,
            n_events=n_pairs,
            fps=args.fps,
            window_start_ratio=0.72,
            window_end_ratio=1.0,
            min_start_frame=(reveal_frames[-1] + max(1, int(args.fps * 0.5))) if reveal_frames else 1,
        )
    else:
        reveal_frames = []
        hide_frames = []
    hide_order = list(reversed(range(n_pairs)))
    pair_hues = np.linspace(_START_HUE, _END_HUE, num=n_pairs, dtype=np.float32)

    print(f"Rendering {n_video_frames} frames (Ctrl+C to stop)...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with VideoWriter(
        args.output,
        width=base.width,
        height=base.height,
        fps=args.fps,
        audio_path=args.audio,
        vf_filter=None,
        preset=None,
        input_pix_fmt="rgba" if transparent_output else "rgb24",
        output_codec="libvpx-vp9",
        output_pix_fmt="yuva420p" if transparent_output else "yuv420p",
        audio_codec="libopus",
        audio_start_seconds=args.audio_offset,
        audio_duration_seconds=n_video_frames / args.fps,
    ) as writer:
        for frame_idx in range(n_video_frames):
            if stop_event.is_set():
                print(f"Interrupted at frame {frame_idx}")
                break

            canvas = base.copy()
            beat = float(beat_strengths[frame_idx])
            if args.start_end_animation:
                revealed_pairs = sum(1 for reveal in reveal_frames if frame_idx >= reveal)
                hidden_pairs_count = sum(1 for hide in hide_frames if frame_idx >= hide)
            else:
                revealed_pairs = n_pairs
                hidden_pairs_count = 0
            hidden_pair_indices = set(hide_order[:hidden_pairs_count])
            all_revealed = revealed_pairs >= n_pairs

            for pair_idx, (left_idx, right_idx) in enumerate(pairs):
                if pair_idx >= revealed_pairs or pair_idx in hidden_pair_indices:
                    continue

                base_hue = float(pair_hues[pair_idx])
                pair_progress = pair_idx / max(1, n_pairs - 1)

                for pillar_idx in (left_idx, right_idx):
                    energy = float(frame_bands[frame_idx, pillar_idx]) / max(1.0, base.height - 1)

                    if all_revealed:
                        phase = 2.0 * math.pi * frame_idx / args.fps
                        hue = (
                            base_hue
                            + 0.10 * (energy - 0.5)
                            + 0.06 * beat
                            + 0.03
                            * math.sin(
                                phase * 0.7 + pillar_idx * 0.45,
                            )
                        )
                        hue = max(base_hue - 0.05, min(base_hue + 0.05, hue))
                        saturation = max(0.65, min(1.0, 0.62 + 0.26 * pair_progress + 0.28 * energy + 0.10 * beat))
                        value = max(0.35, min(1.0, 0.42 + 0.45 * energy + 0.30 * beat))
                    else:
                        hue = base_hue
                        saturation = 0.60 + 0.30 * pair_progress
                        value = 0.75 - 0.12 * pair_progress + 0.22 * beat

                    color = _hsv_to_rgb(hue, saturation, value)
                    canvas.fill_pillar(layout.pillars[pillar_idx], color)

            if not writer.write_canvas(canvas):
                print("FFmpeg finished (audio ended).")
                break

            if frame_idx % 50 == 0:
                print(f"  frame {frame_idx}/{n_video_frames}  beat={beat:.3f}")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
