"""CLI: extract building layout geometry from a facade mask image.

Reads a mask PNG where window panes appear as bright rectangles on a dark
background, detects their bounding boxes, organises them into the
row → block → half → pane-row hierarchy, computes inter-pane grid strips,
wall sections, and detects pillar positions from the colour mask image.

Example::

    debug-extract-layout --mask static/color-mask-bg.png \\
        --color-mask static/color-mask.png --output static/layout.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from video_mapping.constants import (
    DEFAULT_CANVAS_HEIGHT,
    DEFAULT_CANVAS_WIDTH,
    DEFAULT_LAYOUT_JSON_PATH,
    DEFAULT_MASK_IMAGE_PATH,
)

EXPECTED_PANES = 792
ROWS = 2
BLOCKS_PER_ROW = 11
PANES_PER_HALF = 18

DEFAULT_MONOCHROME_MASK = DEFAULT_LAYOUT_JSON_PATH.parent / "color-mask-bg.png"
DEFAULT_COLOR_MASK = DEFAULT_MASK_IMAGE_PATH
DEFAULT_OUTPUT = DEFAULT_LAYOUT_JSON_PATH


@dataclass(frozen=True)
class _Pane:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def _extract_raw_panes(mask_path: Path) -> list[_Pane]:
    img = Image.open(mask_path).convert("RGB")
    arr = np.array(img)

    gray = arr[:, :, 0]
    mask = gray >= 250

    labels, _ = ndimage.label(mask)
    slices = ndimage.find_objects(labels)

    panes: list[_Pane] = []
    for sl in slices:
        ys, xs = sl
        x1, x2 = xs.start, xs.stop - 1
        y1, y2 = ys.start, ys.stop - 1
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        if 30 <= width <= 50 and 20 <= height <= 35:
            panes.append(_Pane(x1, y1, x2, y2))

    panes.sort(key=lambda p: (p.cy, p.cx))

    if len(panes) != EXPECTED_PANES:
        msg = f"Expected {EXPECTED_PANES} panes, got {len(panes)}"
        raise RuntimeError(msg)

    return panes


def _split_rows(panes: list[_Pane]) -> list[list[_Pane]]:
    panes_sorted = sorted(panes, key=lambda p: p.cy)
    mid = len(panes_sorted) // 2
    top = sorted(panes_sorted[:mid], key=lambda p: p.cx)
    bottom = sorted(panes_sorted[mid:], key=lambda p: p.cx)

    if len(top) != EXPECTED_PANES // 2 or len(bottom) != EXPECTED_PANES // 2:
        raise RuntimeError("Unexpected row split sizes")

    return [top, bottom]


def _split_row_into_halves(row_panes: list[_Pane]) -> list[list[_Pane]]:
    panes_sorted = sorted(row_panes, key=lambda p: p.cx)

    columns: list[list[_Pane]] = []
    current: list[_Pane] = [panes_sorted[0]]
    column_threshold = 25

    for i in range(1, len(panes_sorted)):
        if abs(panes_sorted[i].cx - panes_sorted[i - 1].cx) < column_threshold:
            current.append(panes_sorted[i])
        else:
            columns.append(current)
            current = [panes_sorted[i]]
    columns.append(current)

    if len(columns) != 66:
        msg = f"Expected 66 columns, got {len(columns)}"
        raise RuntimeError(msg)

    halves: list[list[_Pane]] = []
    for i in range(0, len(columns), 3):
        half: list[_Pane] = []
        for col in columns[i : i + 3]:
            half.extend(col)
        halves.append(sorted(half, key=lambda p: (p.cy, p.cx)))

    if len(halves) != 22:
        msg = f"Expected 22 halves, got {len(halves)}"
        raise RuntimeError(msg)

    return halves


def _validate_half(half: list[_Pane], row_idx: int, half_idx: int) -> None:
    if len(half) != PANES_PER_HALF:
        msg = f"Row {row_idx}, half {half_idx}: expected {PANES_PER_HALF} panes, got {len(half)}"
        raise RuntimeError(msg)

    xs = sorted({round(p.cx / 25) for p in half})
    ys = sorted({round(p.cy / 15) for p in half})

    if len(xs) != 3:
        msg = f"Row {row_idx}, half {half_idx}: expected 3 pane columns, got {len(xs)}"
        raise RuntimeError(msg)
    if len(ys) != 6:
        msg = f"Row {row_idx}, half {half_idx}: expected 6 pane rows, got {len(ys)}"
        raise RuntimeError(msg)


def _half_to_grid(half: list[_Pane]) -> list[list[dict[str, int]]]:
    ordered = sorted(half, key=lambda p: (p.cy, p.cx))
    grid = []
    for row_start in range(0, len(ordered), 3):
        row_panes = ordered[row_start : row_start + 3]
        grid.append([{"x1": p.x1, "y1": p.y1, "x2": p.x2, "y2": p.y2} for p in row_panes])
    return grid


def _compute_grid_strips(
    half_panes: list[_Pane],
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    """Compute horizontal and vertical grid strips for one half-block.

    Returns (h_strips, v_strips) — the structural frame regions between panes.
    Horizontal strips sit between consecutive pane rows; vertical strips between
    consecutive pane columns.
    """
    ordered = sorted(half_panes, key=lambda p: (p.cy, p.cx))

    # Group into 6 rows x 3 columns
    pane_rows: list[list[_Pane]] = []
    for row_start in range(0, len(ordered), 3):
        row_panes = sorted(ordered[row_start : row_start + 3], key=lambda p: p.cx)
        pane_rows.append(row_panes)

    # X/Y extents of this half (covers the full frame area)
    x1 = min(p.x1 for p in half_panes)
    x2 = max(p.x2 for p in half_panes)
    y1 = min(p.y1 for p in half_panes)
    y2 = max(p.y2 for p in half_panes)

    # Horizontal strips between consecutive pane rows
    h_strips: list[dict[str, int]] = []
    for i in range(len(pane_rows) - 1):
        upper = pane_rows[i]
        lower = pane_rows[i + 1]
        sy1 = max(p.y2 for p in upper) + 1
        sy2 = min(p.y1 for p in lower) - 1
        if sy1 <= sy2:
            h_strips.append({"x1": x1, "y1": sy1, "x2": x2, "y2": sy2})

    # Vertical strips between consecutive pane columns
    cols = [[row[c] for row in pane_rows] for c in range(3)]
    v_strips: list[dict[str, int]] = []
    for i in range(len(cols) - 1):
        left_col = cols[i]
        right_col = cols[i + 1]
        sx1 = max(p.x2 for p in left_col) + 1
        sx2 = min(p.x1 for p in right_col) - 1
        if sx1 <= sx2:
            v_strips.append({"x1": sx1, "y1": y1, "x2": sx2, "y2": y2})

    return h_strips, v_strips


def _compute_wall_sections(
    rows: list[list[_Pane]],
    color_mask: np.ndarray,
    canvas_width: int,
    canvas_height: int,
) -> dict[str, dict[str, int]]:
    """Compute the three horizontal wall sections from pane positions.

    Returns a dict with keys 'above', 'middle', 'below', each an {x1,y1,x2,y2} rect
    spanning the full canvas width.
    """
    top_panes = rows[0]
    bottom_panes = rows[1]

    top_y1 = min(p.y1 for p in top_panes)
    top_y2 = max(p.y2 for p in top_panes)
    bot_y1 = min(p.y1 for p in bottom_panes)
    bot_y2 = max(p.y2 for p in bottom_panes)

    above = {"x1": 0, "y1": 0, "x2": canvas_width - 1, "y2": top_y1 - 1}
    middle = {"x1": 0, "y1": top_y2 + 1, "x2": canvas_width - 1, "y2": bot_y1 - 1}
    below = {"x1": 0, "y1": bot_y2 + 1, "x2": canvas_width - 1, "y2": canvas_height - 1}

    split_y = _detect_middle_split_y(color_mask, middle["y1"], middle["y2"])
    middle_top = {"x1": 0, "y1": middle["y1"], "x2": canvas_width - 1, "y2": split_y}
    middle_bottom = {"x1": 0, "y1": split_y + 1, "x2": canvas_width - 1, "y2": middle["y2"]}

    return {
        "above": above,
        "middle": middle,
        "middle_top": middle_top,
        "middle_bottom": middle_bottom,
        "below": below,
    }


def _detect_middle_split_y(color_mask: np.ndarray, y1: int, y2: int) -> int:
    """Detect the red->gray transition inside the middle wall strip."""
    if y2 <= y1:
        return y1

    middle_region = color_mask[y1 : y2 + 1, :, :3]
    red = middle_region[:, :, 0].astype(np.int32)
    green = middle_region[:, :, 1].astype(np.int32)
    blue = middle_region[:, :, 2].astype(np.int32)

    is_red = (red >= 165) & (green <= 130) & (blue <= 130)
    red_ratio = is_red.mean(axis=1).astype(np.float32)

    if len(red_ratio) < 3:
        return y1

    smooth = np.convolve(red_ratio, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")
    if float(smooth.max() - smooth.min()) < 0.03:
        return y1 + (y2 - y1) // 2

    diff = np.diff(smooth)
    drop_idx = int(np.argmin(diff))
    return max(y1, min(y2 - 1, y1 + drop_idx))


def _detect_pillars(
    top_row_panes: list[_Pane],
    color_mask: np.ndarray,
    canvas_width: int,
) -> list[tuple[int, int]]:
    """Detect pillar x-ranges from inter-block gaps using the colour mask image.

    The algorithm:
    1. Derive block x-extents from the already-detected pane positions.
    2. Identify the 12 gaps (left edge, 10 inter-block, right edge).
    3. Within each gap, find the span of red pixels (R>180, G<100, B<100) at a
       sample row inside the top building row — these are the pillar columns.
    4. Fall back to gap centre if no red is found in a gap.
    """
    # Build block x-ranges from half-block groupings
    half_blocks = _split_row_into_halves(top_row_panes)
    block_x_ranges: list[tuple[int, int]] = []
    for i in range(0, len(half_blocks), 2):
        all_panes = half_blocks[i] + half_blocks[i + 1]
        bx1 = min(p.x1 for p in all_panes)
        bx2 = max(p.x2 for p in all_panes)
        block_x_ranges.append((bx1, bx2))

    # Build the 12 gaps: left-edge + 10 inter-block + right-edge
    gaps: list[tuple[int, int]] = [(0, block_x_ranges[0][0] - 1)]
    gaps.extend((block_x_ranges[i][1] + 1, block_x_ranges[i + 1][0] - 1) for i in range(len(block_x_ranges) - 1))
    gaps.append((block_x_ranges[-1][1] + 1, canvas_width - 1))

    # Sample at median y of the top building row (within a pane row)
    sample_y = int(np.median([p.cy for p in top_row_panes]))
    row_pixels = color_mask[sample_y]
    r = row_pixels[:, 0].astype(np.int32)
    g = row_pixels[:, 1].astype(np.int32)
    b = row_pixels[:, 2].astype(np.int32)
    is_red = (r > 180) & (g < 100) & (b < 100)

    pillars: list[tuple[int, int]] = []
    for gap_x1, gap_x2 in gaps:
        if gap_x2 < gap_x1:
            continue
        segment = is_red[gap_x1 : gap_x2 + 1]
        if segment.any():
            indices = np.where(segment)[0]
            x_start = gap_x1 + int(indices.min())
            x_end = gap_x1 + int(indices.max())
        else:
            # Fallback: centre of the gap with a standard half-width of 13px
            centre = (gap_x1 + gap_x2) // 2
            hw = 13
            x_start = max(0, centre - hw)
            x_end = min(canvas_width - 1, centre + hw)
        pillars.append((x_start, x_end))

    return pillars


def _build_structure(
    panes: list[_Pane],
    color_mask_path: Path,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    canvas_height: int = DEFAULT_CANVAS_HEIGHT,
) -> dict[str, object]:
    row_groups = _split_rows(panes)

    color_mask = np.array(Image.open(color_mask_path).convert("RGB"))

    # Detect pillars from the colour mask image
    pillar_data = _detect_pillars(row_groups[0], color_mask, canvas_width)
    pillar_json = [{"x1": x1, "y1": 0, "x2": x2, "y2": canvas_height - 1} for x1, x2 in pillar_data]

    # Compute the three horizontal wall sections
    walls = _compute_wall_sections(row_groups, color_mask, canvas_width, canvas_height)

    rows_out: list[dict[str, object]] = []
    for row_idx, row_panes in enumerate(row_groups):
        half_blocks = _split_row_into_halves(row_panes)
        for half_idx, half in enumerate(half_blocks):
            _validate_half(half, row_idx, half_idx)

        blocks_out: list[dict[str, object]] = []
        for block_idx in range(BLOCKS_PER_ROW):
            left_half = half_blocks[block_idx * 2]
            right_half = half_blocks[block_idx * 2 + 1]

            left_h, left_v = _compute_grid_strips(left_half)
            right_h, right_v = _compute_grid_strips(right_half)

            left_x1 = min(p.x1 for p in left_half)
            left_x2 = max(p.x2 for p in left_half)
            right_x1 = min(p.x1 for p in right_half)
            right_x2 = max(p.x2 for p in right_half)
            row_y1 = min(p.y1 for p in left_half + right_half)
            row_y2 = max(p.y2 for p in left_half + right_half)

            block_x1 = min(left_x1, right_x1)
            block_x2 = max(left_x2, right_x2)
            between_x1 = left_x2 + 1
            between_x2 = right_x1 - 1
            if between_x1 > between_x2:
                mid_x = (left_x2 + right_x1) // 2
                between_x1 = mid_x
                between_x2 = mid_x

            if row_idx == 0:
                top_gray = {"x1": block_x1, "y1": walls["above"]["y1"], "x2": block_x2, "y2": walls["above"]["y2"]}
                bottom_red = {
                    "x1": block_x1,
                    "y1": walls["middle_top"]["y1"],
                    "x2": block_x2,
                    "y2": walls["middle_top"]["y2"],
                }
            else:
                top_gray = {
                    "x1": block_x1,
                    "y1": walls["middle_bottom"]["y1"],
                    "x2": block_x2,
                    "y2": walls["middle_bottom"]["y2"],
                }
                bottom_red = {"x1": block_x1, "y1": walls["below"]["y1"], "x2": block_x2, "y2": walls["below"]["y2"]}

            block_walls = {
                "top_gray": top_gray,
                "between_halves_red": {"x1": between_x1, "y1": row_y1, "x2": between_x2, "y2": row_y2},
                "bottom_red": bottom_red,
            }

            blocks_out.append(
                {
                    "halves": [
                        {
                            "rows": _half_to_grid(left_half),
                            "h_strips": left_h,
                            "v_strips": left_v,
                        },
                        {
                            "rows": _half_to_grid(right_half),
                            "h_strips": right_h,
                            "v_strips": right_v,
                        },
                    ],
                    "walls": block_walls,
                }
            )
        rows_out.append({"blocks": blocks_out})

    return {"rows": rows_out, "pillars": pillar_json, "walls": walls}


def extract_layout(mask_path: Path, color_mask_path: Path, output_path: Path) -> int:
    """Extract layout from *mask_path* and write structured JSON to *output_path*."""
    panes = _extract_raw_panes(mask_path)
    data = _build_structure(panes, color_mask_path)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return len(panes)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract layout geometry from a facade mask image into layout.json.",
    )
    _ = parser.add_argument("--mask", type=Path, default=DEFAULT_MONOCHROME_MASK)
    _ = parser.add_argument(
        "--color-mask",
        type=Path,
        default=DEFAULT_COLOR_MASK,
        help="Colour mask image used for pillar detection (default: color-mask.png)",
    )
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Extract building layout geometry from a facade mask image."""
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = extract_layout(args.mask, args.color_mask, args.output)
    print(f"Extracted {count} panes → {args.output}")


if __name__ == "__main__":
    main()
