"""CLI: render beat-reactive random half-block lighting.

Only a small set of half-blocks are lit at once. Beat intensity controls how
many lit half-blocks relocate on each beat, while overall activity controls how
many are currently lit.

Example::

    half-block-beats --audio audio.wav --output output/half_block_beats.webm --seed 42
"""

from __future__ import annotations

import argparse
import colorsys
import random
import signal
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

import numpy as np

from video_mapping.audio import process_audio
from video_mapping.canvas import Canvas
from video_mapping.constants import DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, DEFAULT_FPS, DEFAULT_MASK_IMAGE_PATH
from video_mapping.layout import Half, Layout
from video_mapping.render import VideoWriter

if TYPE_CHECKING:
    from video_mapping.types import RGBColor

DEFAULT_OUTPUT = Path("output/half_block_beats.webm")
DEFAULT_BEAT_THRESHOLD = 0.18
DEFAULT_SWAP_COOLDOWN = 0.22

stop_event = Event()


@dataclass(frozen=True, slots=True)
class _PreparedFrames:
    """Store beat and band data already mapped to output video frames."""

    beat_strengths: np.ndarray
    frame_bands: np.ndarray
    n_video_frames: int


def _handle_sigint(_: int, __: object) -> None:
    stop_event.set()
    print("\nStopping after current frame...")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render random beat-reactive half-block lighting.",
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
    _ = parser.add_argument(
        "--audio-offset",
        type=float,
        default=0.0,
        help="Start audio (and visuals) at this offset in seconds.",
    )
    _ = parser.add_argument("--hop-size", type=int, default=1024)
    _ = parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic output.")
    _ = parser.add_argument("--min-lit", type=int, default=1, help="Minimum number of lit half-blocks.")
    _ = parser.add_argument("--max-lit", type=int, default=8, help="Maximum number of lit half-blocks.")
    _ = parser.add_argument(
        "--beat-threshold",
        type=float,
        default=DEFAULT_BEAT_THRESHOLD,
        help="Minimum beat strength required before any relocation can happen.",
    )
    _ = parser.add_argument(
        "--swap-cooldown",
        type=float,
        default=DEFAULT_SWAP_COOLDOWN,
        help="Minimum seconds between relocation events.",
    )
    return parser.parse_args()


def _all_halves(layout: Layout) -> list[Half]:
    halves: list[Half] = []
    for row in layout.rows:
        for block in row.blocks:
            halves.extend(block.halves)
    return halves


def _random_color(rng: random.Random) -> RGBColor:
    hue = rng.random()
    saturation = rng.uniform(0.65, 1.0)
    value = rng.uniform(0.70, 1.0)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def _target_lit_count(
    beat: float,
    mean_energy: float,
    min_lit: int,
    max_lit: int,
) -> int:
    if max_lit <= min_lit:
        return min_lit
    activity = max(0.0, min(1.0, 0.70 * beat + 0.30 * mean_energy))
    return min_lit + round((max_lit - min_lit) * activity)


def _swap_count_for_beat(active_count: int, beat: float) -> int:
    if active_count <= 0 or beat < 0.08:
        return 0
    if beat >= 0.90:
        return active_count
    ratio = (beat - 0.08) / 0.82
    return max(1, round(active_count * ratio))


def _smooth_value(previous: float, current: float, mix: float) -> float:
    return previous * (1.0 - mix) + current * mix


def _create_base_canvas(*, use_mask: bool) -> tuple[Canvas, bool]:
    """Create the base canvas and return whether output stays transparent."""
    if use_mask:
        return Canvas.from_image(DEFAULT_MASK_IMAGE_PATH), False
    return Canvas.transparent(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT), True


def _prepare_frames(
    args: argparse.Namespace,
    *,
    base_height: int,
    num_halves: int,
) -> _PreparedFrames | None:
    """Process audio and resample analysis arrays onto video-frame indices."""
    print("Processing audio...")
    sample_rate, band_heights, beats = process_audio(
        args.audio,
        num_bands=num_halves,
        bar_height=base_height - 1,
        hop_size=args.hop_size,
    )

    fft_fps = sample_rate / args.hop_size
    n_fft_frames = len(band_heights)
    if n_fft_frames <= 0:
        return None

    start_fft_idx = max(0, min(n_fft_frames - 1, int(args.audio_offset * fft_fps)))
    remaining_fft_frames = max(0, n_fft_frames - start_fft_idx)
    n_video_frames = int(remaining_fft_frames / fft_fps * args.fps)
    if args.duration is not None:
        n_video_frames = min(n_video_frames, int(args.duration * args.fps))
    if n_video_frames <= 0:
        return None

    video_indices = np.minimum(
        start_fft_idx + (np.arange(n_video_frames, dtype=np.float32) / args.fps * fft_fps).astype(np.int32),
        n_fft_frames - 1,
    )
    return _PreparedFrames(
        beat_strengths=beats[video_indices].astype(np.float32),
        frame_bands=band_heights[video_indices].astype(np.float32),
        n_video_frames=n_video_frames,
    )


def _inactive_half_indices(num_halves: int, active_colors: dict[int, RGBColor]) -> list[int]:
    """Return half indices that are not currently active."""
    return [idx for idx in range(num_halves) if idx not in active_colors]


def _activate_random_halves(
    active_indices: list[int],
    active_colors: dict[int, RGBColor],
    *,
    num_halves: int,
    count: int,
    rng: random.Random,
) -> None:
    """Activate random inactive halves and assign fresh colors."""
    inactive = _inactive_half_indices(num_halves, active_colors)
    for idx in rng.sample(inactive, k=min(count, len(inactive))):
        active_indices.append(idx)
        active_colors[idx] = _random_color(rng)


def _deactivate_random_halves(
    active_indices: list[int],
    active_colors: dict[int, RGBColor],
    *,
    count: int,
    rng: random.Random,
) -> None:
    """Deactivate random active halves and remove their colors."""
    for idx in rng.sample(active_indices, k=min(count, len(active_indices))):
        active_indices.remove(idx)
        del active_colors[idx]


def _retarget_active_halves(
    active_indices: list[int],
    active_colors: dict[int, RGBColor],
    *,
    target_count: int,
    num_halves: int,
    rng: random.Random,
) -> None:
    """Move active-half count toward the requested target."""
    current_count = len(active_indices)
    if target_count > current_count:
        _activate_random_halves(
            active_indices,
            active_colors,
            num_halves=num_halves,
            count=target_count - current_count,
            rng=rng,
        )
    elif target_count < current_count:
        _deactivate_random_halves(
            active_indices,
            active_colors,
            count=current_count - target_count,
            rng=rng,
        )


def _swap_active_halves(
    active_indices: list[int],
    active_colors: dict[int, RGBColor],
    *,
    num_halves: int,
    swap_count: int,
    rng: random.Random,
) -> None:
    """Swap out a subset of active halves for inactive ones."""
    _deactivate_random_halves(
        active_indices,
        active_colors,
        count=swap_count,
        rng=rng,
    )
    _activate_random_halves(
        active_indices,
        active_colors,
        num_halves=num_halves,
        count=swap_count,
        rng=rng,
    )


def _render_video(
    args: argparse.Namespace,
    *,
    base: Canvas,
    transparent_output: bool,
    halves: list[Half],
    frame_data: _PreparedFrames,
    min_lit: int,
    max_lit: int,
    rng: random.Random,
) -> None:
    """Run the frame loop and render beat-reactive half-block output."""
    active_indices: list[int] = []
    active_colors: dict[int, RGBColor] = {}
    beat_smoothed = 0.0
    energy_smoothed = 0.0
    last_swap_frame = -10_000
    swap_cooldown_frames = max(1, round(args.swap_cooldown * args.fps))
    energy_scale = max(1.0, base.height - 1)

    print(f"Rendering {frame_data.n_video_frames} frames (Ctrl+C to stop)...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with VideoWriter(
        args.output,
        width=base.width,
        height=base.height,
        fps=args.fps,
        total_frames=frame_data.n_video_frames,
        transparent=transparent_output,
        audio_path=args.audio,
        audio_start_seconds=args.audio_offset,
        audio_duration_seconds=frame_data.n_video_frames / args.fps,
    ) as writer:
        for frame_idx, (beat_value, frame_band_values) in enumerate(
            zip(frame_data.beat_strengths, frame_data.frame_bands, strict=True)
        ):
            if stop_event.is_set():
                writer.log(f"Interrupted at frame {frame_idx}")
                break

            beat = float(beat_value)
            mean_energy = float(frame_band_values.mean()) / energy_scale
            beat_smoothed = _smooth_value(beat_smoothed, beat, 0.20)
            energy_smoothed = _smooth_value(energy_smoothed, mean_energy, 0.12)
            target_count = _target_lit_count(beat_smoothed, energy_smoothed, min_lit, max_lit)

            _retarget_active_halves(
                active_indices,
                active_colors,
                target_count=target_count,
                num_halves=len(halves),
                rng=rng,
            )

            can_swap = frame_idx - last_swap_frame >= swap_cooldown_frames
            if can_swap and beat_smoothed >= args.beat_threshold:
                swap_count = min(
                    _swap_count_for_beat(len(active_indices), beat_smoothed),
                    len(halves) - len(active_indices),
                )
                if swap_count > 0:
                    last_swap_frame = frame_idx
                    _swap_active_halves(
                        active_indices,
                        active_colors,
                        num_halves=len(halves),
                        swap_count=swap_count,
                        rng=rng,
                    )

            canvas = base.copy()
            for half_idx in active_indices:
                canvas.color_half(halves[half_idx], active_colors[half_idx])

            if not writer.write_canvas(canvas):
                writer.log("FFmpeg finished (audio ended).")
                break


def main() -> None:
    """Render random half-block lighting that reacts to beats."""
    _ = signal.signal(signal.SIGINT, _handle_sigint)
    args = _parse_args()
    rng = random.Random(args.seed)  # noqa: S311 - deterministic visual randomness, not security-sensitive

    layout = Layout.default()
    halves = _all_halves(layout)
    base, transparent_output = _create_base_canvas(use_mask=args.mask)
    frame_data = _prepare_frames(args, base_height=base.height, num_halves=len(halves))
    if frame_data is None:
        print("Audio offset is beyond available audio frames; nothing to render.")
        return

    min_lit = max(0, min(args.min_lit, len(halves)))
    max_lit = max(min_lit, min(args.max_lit, len(halves)))

    _render_video(
        args,
        base=base,
        transparent_output=transparent_output,
        halves=halves,
        frame_data=frame_data,
        min_lit=min_lit,
        max_lit=max_lit,
        rng=rng,
    )

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
