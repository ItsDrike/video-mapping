"""CLI: generate a ~20-second debug video walking through every detected layout region.

The video renders each element type in sequence on top of the building mask so you
can visually verify that every detected region maps to the correct area of the facade.

Sequence
--------
Phase 1 - Pane scan (2.5 s)
    Panes are highlighted progressively in visual scan order (pane-row 0 -> pane-row 5
    sweeping left-to-right across all blocks for the top building row, then the bottom
    row).  Each pane turns white once visited and stays lit for the rest of this phase.
    Highlight color: white (255, 255, 255), alpha 0.70.

Phase 2 - Half-block pane colours (1.5 s)
    All 18 panes in every half-block are filled with the same colour.  Six colours
    rotate across halves so adjacent halves are always visually distinct.
    Palette (alpha 0.60):
      red (220, 60, 60)  /  green (50, 190, 80)  /  blue (60, 120, 240)
      yellow (230, 200, 0)  /  cyan (0, 200, 220)  /  purple (200, 60, 220)

Phase 3 - Inter-pane grid strips (1 s)
    Only the horizontal h_strips (between pane rows) and vertical v_strips (between
    pane columns) are drawn.  Panes are left clear so the mask shows through.
    Color: white (255, 255, 255), alpha 0.85.

Phase 4 - Panes / grid strips / block walls together (1.5 s)
    All three element classes shown simultaneously with distinct colours:
    - Panes:        cyan (0, 200, 220), alpha 0.45
    - Grid strips:  yellow (240, 210, 0), alpha 0.80
    - Block walls:  orange (255, 130, 0), alpha 0.80

Phase 5 - Gray vs red block walls (1 s)
    Per-block wall regions split by semantic role:
    - top_gray walls:              gray (140, 140, 155), alpha 0.75
    - vertical_red + bottom_red:   red (220, 40, 40), alpha 0.75

Phase 6 - Three distinct per-block wall colours (1.5 s)
    Each of the three per-block wall sections shown in its own colour:
    - top_gray:     gray (140, 140, 155), alpha 0.80
    - vertical_red: red (210, 40, 40), alpha 0.80
    - bottom_red:   magenta (220, 60, 200), alpha 0.80

Phase 7 - Pane-row sweep, 6 passes x 0.5 s (3 s total)
    For pane-row index 0 through 5, the current row is highlighted brightly in both
    the top and bottom building rows simultaneously.  Rows already swept stay dimly
    visible.
    - Current row:  warm yellow (255, 200, 0), alpha 0.80
    - Past rows:    warm yellow (255, 200, 0), alpha 0.25

Phase 8 - Half-block bounding boxes (1 s)
    The bounding box of every half-block is filled.  A 4-colour palette cycles so
    adjacent halves contrast visually (alpha 0.55):
      orange (220, 100, 60)  /  steel-blue (60, 180, 220)
      lime (180, 220, 60)  /  hot-pink (220, 60, 180)

Phase 9 - Block bounding boxes (1 s)
    The bounding box of every block is filled with a single colour.
    Color: violet (180, 60, 255), alpha 0.55.

Phase 10 - Global wall sections, revealed one at a time (4 s -- 4 x 1 s)
    Global wall strips are added in top-to-bottom order; each previous wall stays
    visible when the next is revealed:
    a) above:         blue (30, 120, 255), alpha 0.65
    b) middle_top:    red (220, 40, 40), alpha 0.65
    c) middle_bottom: gray (140, 140, 155), alpha 0.65
    d) below:         green (0, 200, 80), alpha 0.65

Phase 11 - Pillars (1.5 s)
    All 12 structural pillars highlighted at their full canvas-height spans.
    Color: cyan (0, 230, 230), alpha 0.70.

Example::

    debug-draw-layout --output output/layout_debug.webm
    debug-draw-layout --mask static/color-mask-bg.png --output output/layout_debug_bg.webm
"""

from __future__ import annotations

import argparse
from pathlib import Path

from video_mapping.canvas import Canvas
from video_mapping.constants import DEFAULT_LAYOUT_JSON_PATH, DEFAULT_MASK_IMAGE_PATH
from video_mapping.layout import Layout
from video_mapping.render import VideoWriter
from video_mapping.types import RGBColor

_FPS = 25

# Colour palette for half-block pane highlights (Phase 2)
_HALF_PALETTE: tuple[RGBColor, ...] = (
    (220, 60, 60),  # red
    (50, 190, 80),  # green
    (60, 120, 240),  # blue
    (230, 200, 0),  # yellow
    (0, 200, 220),  # cyan
    (200, 60, 220),  # purple
)

# Colour palette for half/block bounding-box highlights (Phases 8-9)
_BBOX_PALETTE: tuple[RGBColor, ...] = (
    (220, 100, 60),  # burnt-orange
    (60, 180, 220),  # steel-blue
    (180, 220, 60),  # lime
    (220, 60, 180),  # hot-pink
)


def _frames(seconds: float) -> int:
    """Convert seconds to a whole number of frames at _FPS."""
    return round(seconds * _FPS)


def _base(mask_path: Path) -> Canvas:
    """Return a fresh copy of the mask image as the canvas background."""
    return Canvas.from_image(mask_path)


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def _phase_1_scan(mask_path: Path, layout: Layout, writer: VideoWriter) -> None:
    """Phase 1: progressive pane scan (panes light up and stay lit)."""
    panes = list(layout.iter_scan_order())
    total = _frames(2.5)
    n = len(panes)
    canvas = _base(mask_path)
    prev = 0
    for i in range(total):
        end = round((i + 1) * n / total)
        for pane in panes[prev:end]:
            canvas.color_pane(pane, (255, 255, 255), 0.70)
        prev = end
        _ = writer.write_canvas(canvas)


def _phase_2_half_colours(mask_path: Path, layout: Layout, writer: VideoWriter) -> None:
    """Phase 2: all half-blocks filled with cycling colours."""
    canvas = _base(mask_path)
    halves = [half for row in layout.rows for block in row.blocks for half in block.halves]
    for idx, half in enumerate(halves):
        canvas.color_half(half, _HALF_PALETTE[idx % len(_HALF_PALETTE)], 0.60)
    for _ in range(_frames(1.5)):
        _ = writer.write_canvas(canvas)


def _phase_3_grid_strips(mask_path: Path, layout: Layout, writer: VideoWriter) -> None:
    """Phase 3: inter-pane grid strips only (panes left clear)."""
    canvas = _base(mask_path)
    for row in layout.rows:
        for block in row.blocks:
            for half in block.halves:
                canvas.fill_half_grid(half, (255, 255, 255), 0.85)
    for _ in range(_frames(1.0)):
        _ = writer.write_canvas(canvas)


def _phase_4_three_layers(mask_path: Path, layout: Layout, writer: VideoWriter) -> None:
    """Phase 4: panes / grid strips / block walls in three distinct colours."""
    canvas = _base(mask_path)
    for row in layout.rows:
        for block in row.blocks:
            for half in block.halves:
                canvas.color_half(half, (0, 200, 220), 0.45)
                canvas.fill_half_grid(half, (240, 210, 0), 0.80)
    for wall in layout.all_walls():
        canvas.color_region(wall, (255, 130, 0), 0.80)
    for _ in range(_frames(1.5)):
        _ = writer.write_canvas(canvas)


def _phase_5_gray_vs_red(mask_path: Path, layout: Layout, writer: VideoWriter) -> None:
    """Phase 5: top_gray (gray) vs vertical_red + bottom_red (red)."""
    canvas = _base(mask_path)
    for wall in layout.gray_walls():
        canvas.color_region(wall, (140, 140, 155), 0.75)
    for wall in layout.red_walls():
        canvas.color_region(wall, (220, 40, 40), 0.75)
    for _ in range(_frames(1.0)):
        _ = writer.write_canvas(canvas)


def _phase_6_three_wall_colours(mask_path: Path, layout: Layout, writer: VideoWriter) -> None:
    """Phase 6: three distinct per-block wall colours."""
    canvas = _base(mask_path)
    for row in layout.rows:
        for block in row.blocks:
            if block.walls is None:
                continue
            canvas.color_region(block.walls.top_gray, (140, 140, 155), 0.80)
            canvas.color_region(block.walls.vertical_red, (210, 40, 40), 0.80)
            canvas.color_region(block.walls.bottom_red, (220, 60, 200), 0.80)
    for _ in range(_frames(1.5)):
        _ = writer.write_canvas(canvas)


def _phase_7_pane_row_sweep(mask_path: Path, layout: Layout, writer: VideoWriter) -> None:
    """Phase 7: pane-row sweep, 6 passes (both building rows highlighted simultaneously)."""
    num_pane_rows = layout.top_row.blocks[0].left.num_pane_rows
    frames_per_pass = _frames(0.5)
    for pass_idx in range(num_pane_rows):
        canvas = _base(mask_path)
        # Dim previously swept rows
        for prev_idx in range(pass_idx):
            for row in layout.rows:
                for block in row.blocks:
                    for half in block.halves:
                        canvas.color_half_pane_row(half, prev_idx, (255, 200, 0), 0.25)
        # Highlight the current row brightly in both building rows
        for row in layout.rows:
            for block in row.blocks:
                for half in block.halves:
                    canvas.color_half_pane_row(half, pass_idx, (255, 200, 0), 0.80)
        for _ in range(frames_per_pass):
            _ = writer.write_canvas(canvas)


def _phase_8_half_bboxes(mask_path: Path, layout: Layout, writer: VideoWriter) -> None:
    """Phase 8: half-block bounding boxes, cycling through 4 colours."""
    canvas = _base(mask_path)
    halves = [half for row in layout.rows for block in row.blocks for half in block.halves]
    for idx, half in enumerate(halves):
        canvas.blend_bbox(half.bbox(), _BBOX_PALETTE[idx % len(_BBOX_PALETTE)], 0.55)
    for _ in range(_frames(1.0)):
        _ = writer.write_canvas(canvas)


def _phase_9_block_bboxes(mask_path: Path, layout: Layout, writer: VideoWriter) -> None:
    """Phase 9: block bounding boxes, single colour."""
    canvas = _base(mask_path)
    for row in layout.rows:
        for block in row.blocks:
            canvas.blend_bbox(block.bbox(), (180, 60, 255), 0.55)
    for _ in range(_frames(1.0)):
        _ = writer.write_canvas(canvas)


def _phase_10_wall_sections(mask_path: Path, layout: Layout, writer: VideoWriter) -> None:
    """Phase 10: global wall sections revealed one at a time (accumulated)."""
    canvas = _base(mask_path)
    wall_steps = [
        (layout.walls.above, (30, 120, 255)),
        (layout.walls.middle_top, (220, 40, 40)),
        (layout.walls.middle_bottom, (140, 140, 155)),
        (layout.walls.below, (0, 200, 80)),
    ]
    for wall, color in wall_steps:
        canvas.fill_wall(wall, color, 0.65)
        for _ in range(_frames(1.0)):
            _ = writer.write_canvas(canvas)


def _phase_11_pillars(mask_path: Path, layout: Layout, writer: VideoWriter) -> None:
    """Phase 11: all pillars at full canvas-height span."""
    canvas = _base(mask_path)
    for pillar in layout.pillars:
        canvas.color_region(pillar, (0, 230, 230), 0.70)
    for _ in range(_frames(1.5)):
        _ = writer.write_canvas(canvas)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def draw_layout_video(mask_path: Path, layout_path: Path, output_path: Path) -> None:
    """Render the layout-debug video to *output_path*."""
    layout = Layout.from_json(layout_path)

    total_frames = (
        _frames(2.5)  # phase 1  - pane scan
        + _frames(1.5)  # phase 2  - half colours
        + _frames(1.0)  # phase 3  - grid strips
        + _frames(1.5)  # phase 4  - panes + grid + block walls
        + _frames(1.0)  # phase 5  - gray vs red walls
        + _frames(1.5)  # phase 6  - three wall colours
        + 6 * _frames(0.5)  # phase 7  - pane-row sweep (6 passes)
        + _frames(1.0)  # phase 8  - half bboxes
        + _frames(1.0)  # phase 9  - block bboxes
        + 4 * _frames(1.0)  # phase 10 - wall sections (4 steps)
        + _frames(1.5)  # phase 11 - pillars
    )

    sample = Canvas.from_image(mask_path)

    with VideoWriter(
        output_path,
        width=sample.width,
        height=sample.height,
        fps=_FPS,
        total_frames=total_frames,
        transparent=False,
        progress_desc="Layout debug video",
    ) as writer:
        _phase_1_scan(mask_path, layout, writer)
        _phase_2_half_colours(mask_path, layout, writer)
        _phase_3_grid_strips(mask_path, layout, writer)
        _phase_4_three_layers(mask_path, layout, writer)
        _phase_5_gray_vs_red(mask_path, layout, writer)
        _phase_6_three_wall_colours(mask_path, layout, writer)
        _phase_7_pane_row_sweep(mask_path, layout, writer)
        _phase_8_half_bboxes(mask_path, layout, writer)
        _phase_9_block_bboxes(mask_path, layout, writer)
        _phase_10_wall_sections(mask_path, layout, writer)
        _phase_11_pillars(mask_path, layout, writer)

    duration = total_frames / _FPS
    print(f"Saved layout debug video -> {output_path}  ({duration:.1f} s, {total_frames} frames)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a layout-debug video walking through every detected region.",
    )
    _ = parser.add_argument(
        "--mask",
        type=Path,
        default=DEFAULT_MASK_IMAGE_PATH,
        help="Background mask image (default: color-mask.png)",
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
        default=Path("output/layout_debug.webm"),
        help="Output video path (default: output/layout_debug.webm)",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the layout-debug video."""
    args = _parse_args()
    draw_layout_video(args.mask, args.layout, args.output)


if __name__ == "__main__":
    main()
