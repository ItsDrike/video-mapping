"""CLI: extract building layout geometry from a facade mask image.

Reads a mask PNG where window panes appear as bright rectangles on a dark
background, detects their bounding boxes, organises them into the
row → block → half → pane-row hierarchy, and writes ``layout.json``.

Example::

    debug-extract-layout --mask static/color-mask-bg.png --output static/layout.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from video_mapping.constants import DEFAULT_LAYOUT_JSON_PATH

DEFAULT_PILLARS: list[tuple[int, int]] = [
    (4, 30),
    (374, 398),
    (744, 768),
    (1114, 1138),
    (1482, 1506),
    (1852, 1876),
    (2222, 2246),
    (2590, 2614),
    (2960, 2984),
    (3330, 3354),
    (3700, 3724),
    (4068, 4092),
]

EXPECTED_PANES = 792
ROWS = 2
BLOCKS_PER_ROW = 11
PANES_PER_HALF = 18

DEFAULT_MASK = DEFAULT_LAYOUT_JSON_PATH.parent / "color-mask-bg.png"
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


def _build_structure(panes: list[_Pane]) -> dict[str, object]:
    rows_out: list[dict[str, object]] = []

    for row_idx, row_panes in enumerate(_split_rows(panes)):
        half_blocks = _split_row_into_halves(row_panes)
        for half_idx, half in enumerate(half_blocks):
            _validate_half(half, row_idx, half_idx)

        blocks_out: list[dict[str, object]] = []
        for block_idx in range(BLOCKS_PER_ROW):
            left_half = half_blocks[block_idx * 2]
            right_half = half_blocks[block_idx * 2 + 1]
            blocks_out.append(
                {
                    "halves": [
                        {"rows": _half_to_grid(left_half)},
                        {"rows": _half_to_grid(right_half)},
                    ]
                }
            )
        rows_out.append({"blocks": blocks_out})

    pillar_data = [{"x_start": x_start, "x_end": x_end} for x_start, x_end in DEFAULT_PILLARS]
    return {"rows": rows_out, "pillars": pillar_data}


def extract_layout(mask_path: Path, output_path: Path) -> int:
    """Extract layout from *mask_path* and write structured JSON to *output_path*."""
    panes = _extract_raw_panes(mask_path)
    data = _build_structure(panes)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return len(panes)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract layout geometry from a facade mask image into layout.json.",
    )
    _ = parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Extract building layout geometry from a facade mask image."""
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = extract_layout(args.mask, args.output)
    print(f"Extracted {count} panes → {args.output}")


if __name__ == "__main__":
    main()
