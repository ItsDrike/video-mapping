"""CLI: render a mirrored loading animation across the building windows.

The outer four blocks on each side animate fully. The middle three blocks only
light in the building's outer vertical bands: the topmost three pane rows in
the top row and the bottommost three pane rows in the bottom row. The load
grows from the outer corners toward the center, while soft half-block scanner
sweeps and block breathing keep the active areas alive until a short fade-out
at the end.

Example::

    loading-animation --output output/loading_animation.webm
    loading-animation --duration 14 --mask
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
from video_mapping.constants import DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, DEFAULT_FPS, DEFAULT_MASK_IMAGE_PATH
from video_mapping.layout import Block, Half, Layout, Pane
from video_mapping.render import VideoWriter

if TYPE_CHECKING:
    from video_mapping.types import RGBColor

DEFAULT_OUTPUT = Path("output/loading_animation.webm")
DEFAULT_DURATION = 10.0
_FULL_BLOCKS_PER_SIDE = 4
_CENTER_ACTIVE_ROWS = 3
_LOAD_END_FRACTION = 0.84
_FADE_OUT_FRACTION = 0.12

_TOP_BASE = (70, 175, 255)
_BOTTOM_BASE = (82, 245, 210)
_SCAN_COLOR = (165, 232, 255)
_FRONT_COLOR = (248, 252, 255)

stop_event = Event()


@dataclass(frozen=True, slots=True)
class _BlockPair:
    pane_groups: tuple[tuple[Pane, ...], ...]
    load_rank: float
    pulse_phase: float
    color: RGBColor


@dataclass(frozen=True, slots=True)
class _HalfPair:
    pane_groups: tuple[tuple[Pane, ...], ...]
    load_rank: float
    pulse_phase: float
    color: RGBColor


@dataclass(frozen=True, slots=True)
class _PanePair:
    panes: tuple[Pane, Pane]
    load_rank: float
    pulse_phase: float
    color: RGBColor


@dataclass(frozen=True, slots=True)
class _AnimationGeometry:
    block_pairs: tuple[_BlockPair, ...]
    half_pairs: tuple[_HalfPair, ...]
    pane_pairs: tuple[_PanePair, ...]


def _handle_sigint(_: int, __: object) -> None:
    stop_event.set()
    print("\nStopping after current frame...")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a mirrored loading animation across the window facade.",
    )
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
        default=DEFAULT_DURATION,
        help=f"Duration in seconds (default: {DEFAULT_DURATION:g}).",
    )
    return parser.parse_args()


def _create_base_canvas(*, use_mask: bool) -> tuple[Canvas, bool]:
    if use_mask:
        return Canvas.from_image(DEFAULT_MASK_IMAGE_PATH), False
    return Canvas.transparent(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT), True


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return 1.0 if value >= edge1 else 0.0
    t = _clamp01((value - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _mix_colors(color_a: RGBColor, color_b: RGBColor, amount: float) -> RGBColor:
    mix = _clamp01(amount)
    return (
        round(color_a[0] + (color_b[0] - color_a[0]) * mix),
        round(color_a[1] + (color_b[1] - color_a[1]) * mix),
        round(color_a[2] + (color_b[2] - color_a[2]) * mix),
    )


def _row_color(row_idx: int) -> RGBColor:
    return _TOP_BASE if row_idx == 0 else _BOTTOM_BASE


def _active_pane_rows(
    *,
    row_idx: int,
    block_idx: int,
    num_blocks: int,
    num_pane_rows: int,
) -> tuple[int, ...]:
    active_row_count = min(_CENTER_ACTIVE_ROWS, num_pane_rows)
    middle_start = _FULL_BLOCKS_PER_SIDE
    middle_end = num_blocks - _FULL_BLOCKS_PER_SIDE
    if middle_start <= block_idx < middle_end:
        if row_idx == 0:
            return tuple(range(active_row_count))
        return tuple(range(num_pane_rows - active_row_count, num_pane_rows))
    return tuple(range(num_pane_rows))


def _half_selected_panes(half: Half, active_rows: tuple[int, ...]) -> tuple[Pane, ...]:
    return tuple(pane for row_idx in active_rows for pane in half.row(row_idx))


def _block_selected_panes(block: Block, active_rows: tuple[int, ...]) -> tuple[Pane, ...]:
    return _half_selected_panes(block.left, active_rows) + _half_selected_panes(block.right, active_rows)


def _prepare_row_block_pairs(
    blocks: tuple[Block, ...],
    active_rows_by_block: tuple[tuple[int, ...], ...],
    *,
    row_idx: int,
    base_color: RGBColor,
) -> list[_BlockPair]:
    block_pairs: list[_BlockPair] = []
    num_blocks = len(blocks)
    num_block_pairs = num_blocks // 2 + num_blocks % 2
    block_den = max(1, num_block_pairs - 1)

    for block_pair_idx in range(num_block_pairs):
        left_block_idx = block_pair_idx
        right_block_idx = num_blocks - 1 - block_pair_idx
        pane_groups = [_block_selected_panes(blocks[left_block_idx], active_rows_by_block[left_block_idx])]
        if right_block_idx != left_block_idx:
            pane_groups.append(_block_selected_panes(blocks[right_block_idx], active_rows_by_block[right_block_idx]))
        block_pairs.append(
            _BlockPair(
                pane_groups=tuple(pane_groups),
                load_rank=block_pair_idx / block_den,
                pulse_phase=row_idx * 1.30 + block_pair_idx * 0.72,
                color=base_color,
            )
        )

    return block_pairs


def _prepare_row_half_pairs(
    blocks: tuple[Block, ...],
    active_rows_by_block: tuple[tuple[int, ...], ...],
    *,
    row_idx: int,
    base_color: RGBColor,
) -> list[_HalfPair]:
    half_pairs: list[_HalfPair] = []
    num_blocks = len(blocks)
    num_half_pairs = num_blocks
    half_den = max(1, num_half_pairs - 1)
    halves_per_block = len(blocks[0].halves)

    for half_pair_idx in range(num_half_pairs):
        left_half_global_idx = half_pair_idx
        right_half_global_idx = num_blocks * halves_per_block - 1 - half_pair_idx
        left_block_idx, left_half_idx = divmod(left_half_global_idx, halves_per_block)
        right_block_idx, right_half_idx = divmod(right_half_global_idx, halves_per_block)
        left_half = blocks[left_block_idx].halves[left_half_idx]
        right_half = blocks[right_block_idx].halves[right_half_idx]
        half_pairs.append(
            _HalfPair(
                pane_groups=(
                    _half_selected_panes(left_half, active_rows_by_block[left_block_idx]),
                    _half_selected_panes(right_half, active_rows_by_block[right_block_idx]),
                ),
                load_rank=half_pair_idx / half_den,
                pulse_phase=row_idx * 1.15 + half_pair_idx * 0.55,
                color=_mix_colors(base_color, _SCAN_COLOR, 0.26 + 0.08 * left_half_idx),
            )
        )

    return half_pairs


def _pane_from_global_column(
    blocks: tuple[Block, ...],
    pane_row: int,
    global_col: int,
    *,
    panes_per_block: int,
    panes_per_half: int,
) -> tuple[Pane, int, int, int]:
    block_idx = global_col // panes_per_block
    block_col = global_col % panes_per_block
    half_idx, pane_col = divmod(block_col, panes_per_half)
    return (blocks[block_idx].halves[half_idx].pane_at(pane_row, pane_col), block_idx, half_idx, pane_col)


def _prepare_row_pane_pairs(
    blocks: tuple[Block, ...],
    active_rows_by_block: tuple[tuple[int, ...], ...],
    *,
    row_idx: int,
    base_color: RGBColor,
    num_pane_rows: int,
) -> list[_PanePair]:
    pane_pairs: list[_PanePair] = []
    pane_rows_den = max(1, num_pane_rows - 1)
    panes_per_half = blocks[0].left.num_pane_cols
    halves_per_block = len(blocks[0].halves)
    panes_per_block = halves_per_block * panes_per_half
    num_pane_cols = len(blocks) * panes_per_block
    num_pane_pairs = num_pane_cols // 2
    pane_cols_den = max(1, num_pane_pairs - 1)

    for pane_row in range(num_pane_rows):
        vertical_rank = pane_row / pane_rows_den if row_idx == 0 else (num_pane_rows - 1 - pane_row) / pane_rows_den
        for pane_pair_idx in range(num_pane_pairs):
            left_col = pane_pair_idx
            right_col = num_pane_cols - 1 - pane_pair_idx
            left_pane, left_block_idx, left_half_idx, left_pane_col = _pane_from_global_column(
                blocks,
                pane_row,
                left_col,
                panes_per_block=panes_per_block,
                panes_per_half=panes_per_half,
            )
            right_pane, right_block_idx, _, _ = _pane_from_global_column(
                blocks,
                pane_row,
                right_col,
                panes_per_block=panes_per_block,
                panes_per_half=panes_per_half,
            )
            if (
                pane_row not in active_rows_by_block[left_block_idx]
                or pane_row not in active_rows_by_block[right_block_idx]
            ):
                continue

            horizontal_rank = pane_pair_idx / pane_cols_den
            lattice_offset = 0.045 * ((pane_row + left_half_idx + left_pane_col) % 3) / 2.0
            pane_pairs.append(
                _PanePair(
                    panes=(left_pane, right_pane),
                    load_rank=_clamp01(horizontal_rank * 0.74 + vertical_rank * 0.26 + lattice_offset),
                    pulse_phase=(
                        row_idx * 1.05
                        + pane_pair_idx * 0.23
                        + pane_row * 0.33
                        + left_half_idx * 0.19
                        + left_pane_col * 0.11
                    ),
                    color=_mix_colors(base_color, _SCAN_COLOR, 0.10 + vertical_rank * 0.18),
                )
            )

    return pane_pairs


def _prepare_geometry(layout: Layout) -> _AnimationGeometry:
    block_pairs: list[_BlockPair] = []
    half_pairs: list[_HalfPair] = []
    pane_pairs: list[_PanePair] = []

    num_blocks = len(layout.top_row.blocks)
    num_pane_rows = layout.top_row.blocks[0].left.num_pane_rows

    for row_idx, row in enumerate(layout.rows):
        base_color = _row_color(row_idx)
        active_rows_by_block = tuple(
            _active_pane_rows(
                row_idx=row_idx,
                block_idx=block_idx,
                num_blocks=num_blocks,
                num_pane_rows=num_pane_rows,
            )
            for block_idx in range(num_blocks)
        )
        block_pairs.extend(
            _prepare_row_block_pairs(
                row.blocks,
                active_rows_by_block,
                row_idx=row_idx,
                base_color=base_color,
            )
        )
        half_pairs.extend(
            _prepare_row_half_pairs(
                row.blocks,
                active_rows_by_block,
                row_idx=row_idx,
                base_color=base_color,
            )
        )
        pane_pairs.extend(
            _prepare_row_pane_pairs(
                row.blocks,
                active_rows_by_block,
                row_idx=row_idx,
                base_color=base_color,
                num_pane_rows=num_pane_rows,
            )
        )

    return _AnimationGeometry(
        block_pairs=tuple(block_pairs),
        half_pairs=tuple(half_pairs),
        pane_pairs=tuple(pane_pairs),
    )


def _draw_block_breathing(
    canvas: Canvas,
    geometry: _AnimationGeometry,
    *,
    animation_progress: float,
    load_progress: float,
    fade: float,
) -> None:
    for pair in geometry.block_pairs:
        loaded = _smoothstep(pair.load_rank - 0.18, pair.load_rank + 0.08, load_progress)
        if loaded <= 0.0:
            continue
        pulse = 0.5 + 0.5 * math.sin(math.tau * (animation_progress * 3.5) + pair.pulse_phase)
        alpha = fade * loaded * (0.03 + 0.10 * pulse)
        if alpha <= 0.0:
            continue
        for pane_group in pair.pane_groups:
            canvas.color_panes(pane_group, pair.color, alpha)


def _draw_half_scanners(
    canvas: Canvas,
    geometry: _AnimationGeometry,
    *,
    animation_progress: float,
    load_progress: float,
    fade: float,
) -> None:
    scan_progress = 1.0 - abs((animation_progress * 4.0 % 2.0) - 1.0)
    for pair in geometry.half_pairs:
        readiness = _smoothstep(pair.load_rank - 0.16, pair.load_rank + 0.12, load_progress)
        sweep = _clamp01(1.0 - abs(pair.load_rank - scan_progress) / 0.28)
        if sweep <= 0.0 and readiness <= 0.0:
            continue
        shimmer = 0.5 + 0.5 * math.sin(math.tau * (animation_progress * 5.0) + pair.pulse_phase)
        alpha = fade * sweep * (0.03 + (0.04 + 0.09 * readiness) * shimmer)
        if alpha <= 0.0:
            continue
        for pane_group in pair.pane_groups:
            canvas.color_panes(pane_group, pair.color, alpha)


def _draw_loading_panes(
    canvas: Canvas,
    geometry: _AnimationGeometry,
    *,
    animation_progress: float,
    load_progress: float,
    fade: float,
) -> None:
    for pair in geometry.pane_pairs:
        loaded = _smoothstep(pair.load_rank - 0.20, pair.load_rank + 0.02, load_progress)
        if loaded <= 0.0:
            continue
        front = _clamp01(1.0 - abs(pair.load_rank - load_progress) / 0.12)
        breathe = 0.5 + 0.5 * math.sin(math.tau * (animation_progress * 6.5) + pair.pulse_phase)
        alpha = fade * min(0.95, 0.08 + 0.42 * loaded + 0.16 * loaded * breathe + 0.28 * front)
        if alpha <= 0.0:
            continue
        color = _mix_colors(pair.color, _FRONT_COLOR, 0.18 + 0.72 * front)
        for pane in pair.panes:
            canvas.color_pane(pane, color, alpha)


def _render_frame(canvas: Canvas, geometry: _AnimationGeometry, animation_progress: float) -> None:
    load_progress = _clamp01(animation_progress / _LOAD_END_FRACTION)
    fade = 1.0 - _smoothstep(1.0 - _FADE_OUT_FRACTION, 1.0, animation_progress)
    if fade <= 0.0:
        return

    _draw_block_breathing(
        canvas,
        geometry,
        animation_progress=animation_progress,
        load_progress=load_progress,
        fade=fade,
    )
    _draw_half_scanners(
        canvas,
        geometry,
        animation_progress=animation_progress,
        load_progress=load_progress,
        fade=fade,
    )
    _draw_loading_panes(
        canvas,
        geometry,
        animation_progress=animation_progress,
        load_progress=load_progress,
        fade=fade,
    )


def main() -> None:
    """Render the mirrored loading animation video."""
    _ = signal.signal(signal.SIGINT, _handle_sigint)
    args = _parse_args()

    if args.duration <= 0:
        msg = "Duration must be positive."
        raise ValueError(msg)

    layout = Layout.default()
    geometry = _prepare_geometry(layout)
    base, transparent_output = _create_base_canvas(use_mask=args.mask)

    n_frames = max(1, round(args.duration * args.fps))
    print(f"Rendering {n_frames} frames ({args.duration:.2f}s @ {args.fps} fps)...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with VideoWriter(
        args.output,
        width=base.width,
        height=base.height,
        fps=args.fps,
        total_frames=n_frames,
        transparent=transparent_output,
        progress_desc="Loading animation",
    ) as writer:
        for frame_idx in range(n_frames):
            if stop_event.is_set():
                writer.log(f"Interrupted at frame {frame_idx}")
                break

            animation_progress = frame_idx / max(1, n_frames - 1)
            canvas = base.copy()
            _render_frame(canvas, geometry, animation_progress)

            if not writer.write_canvas(canvas):
                writer.log("FFmpeg terminated early.")
                break

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
