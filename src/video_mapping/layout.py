"""Building layout data model: panes, halves, blocks, rows, pillars."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

from video_mapping.constants import DEFAULT_LAYOUT_JSON_PATH


@dataclass(frozen=True)
class Pane:
    """A single window pane. All coordinates are inclusive pixel positions."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        """Horizontal centre pixel."""
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        """Vertical centre pixel."""
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> int:
        """Pane width in pixels."""
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        """Pane height in pixels."""
        return self.y2 - self.y1 + 1

    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) — the bounding box of this pane."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Pillar:
    """A structural pillar between window blocks.

    Both x_start and x_end are inclusive pixel columns (same convention as Pane).
    """

    x_start: int
    x_end: int

    @property
    def width(self) -> int:
        """Pillar width in pixels."""
        return self.x_end - self.x_start + 1


@dataclass(frozen=True)
class Half:
    """A half-block: 6 pane-rows x 3 pane-columns = 18 panes.

    pane_rows[row_idx][col_idx] gives the Pane at that position.
    """

    pane_rows: tuple[tuple[Pane, ...], ...]

    def pane_at(self, row: int, col: int) -> Pane:
        """Return the pane at (row, col) within this half."""
        return self.pane_rows[row][col]

    def row(self, idx: int) -> tuple[Pane, ...]:
        """Return all panes in horizontal pane-row *idx*."""
        return self.pane_rows[idx]

    def all_panes(self) -> Iterator[Pane]:
        """Yield every pane, top-to-bottom then left-to-right."""
        for pane_row in self.pane_rows:
            yield from pane_row

    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) — the bounding box spanning all panes in this half.

        All coordinates are inclusive and cover the entire half-block region,
        even gaps between individual panes.
        """
        all_panes_list = list(self.all_panes())
        if not all_panes_list:
            return (0, 0, 0, 0)
        x1 = min(p.x1 for p in all_panes_list)
        y1 = min(p.y1 for p in all_panes_list)
        x2 = max(p.x2 for p in all_panes_list)
        y2 = max(p.y2 for p in all_panes_list)
        return (x1, y1, x2, y2)

    @property
    def num_pane_rows(self) -> int:
        """Number of pane-rows in this half (typically 6)."""
        return len(self.pane_rows)

    @property
    def num_pane_cols(self) -> int:
        """Number of pane columns in this half (typically 3)."""
        return len(self.pane_rows[0]) if self.pane_rows else 0


@dataclass(frozen=True)
class Block:
    """A window block consisting of a left and right half."""

    halves: tuple[Half, Half]

    @property
    def left(self) -> Half:
        """Left half-block."""
        return self.halves[0]

    @property
    def right(self) -> Half:
        """Right half-block."""
        return self.halves[1]

    def all_panes(self) -> Iterator[Pane]:
        """Yield every pane in this block."""
        for half in self.halves:
            yield from half.all_panes()

    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) — the bounding box spanning all panes in this block.

        All coordinates are inclusive and cover the entire block region,
        even gaps between individual panes.
        """
        all_panes_list = list(self.all_panes())
        if not all_panes_list:
            return (0, 0, 0, 0)
        x1 = min(p.x1 for p in all_panes_list)
        y1 = min(p.y1 for p in all_panes_list)
        x2 = max(p.x2 for p in all_panes_list)
        y2 = max(p.y2 for p in all_panes_list)
        return (x1, y1, x2, y2)


@dataclass(frozen=True)
class Row:
    """A horizontal building row containing 11 blocks."""

    blocks: tuple[Block, ...]

    def block_at(self, idx: int) -> Block:
        """Return the block at index *idx*."""
        return self.blocks[idx]

    def all_panes(self) -> Iterator[Pane]:
        """Yield every pane in this row."""
        for block in self.blocks:
            yield from block.all_panes()

    def iter_scan_order(self) -> Iterator[Pane]:
        """Yield panes in scan order: for each pane-row, sweep left-to-right across all blocks."""
        num_pane_rows = self.blocks[0].left.num_pane_rows
        for pane_row_idx in range(num_pane_rows):
            for block in self.blocks:
                for half in block.halves:
                    yield from half.pane_rows[pane_row_idx]

    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) — the bounding box spanning all panes in this row."""
        all_panes_list = list(self.all_panes())
        if not all_panes_list:
            return (0, 0, 0, 0)
        x1 = min(p.x1 for p in all_panes_list)
        y1 = min(p.y1 for p in all_panes_list)
        x2 = max(p.x2 for p in all_panes_list)
        y2 = max(p.y2 for p in all_panes_list)
        return (x1, y1, x2, y2)


@dataclass(frozen=True)
class Layout:
    """The complete building layout: two horizontal rows of blocks, plus the structural pillars."""

    _default_layout_cache: ClassVar[Layout | None] = None

    rows: tuple[Row, Row]
    pillars: tuple[Pillar, ...]

    @property
    def top_row(self) -> Row:
        """The building's top row of blocks."""
        return self.rows[0]

    @property
    def bottom_row(self) -> Row:
        """The building's bottom row of blocks."""
        return self.rows[1]

    def all_panes(self) -> Iterator[Pane]:
        """Yield every pane: top row then bottom row, blocks left-to-right, halves, then pane-rows."""
        for row in self.rows:
            yield from row.all_panes()

    def all_panes_flat(self) -> list[Pane]:
        """Return all panes as a flat list."""
        return list(self.all_panes())

    def iter_scan_order(self) -> Iterator[Pane]:
        """Yield all panes in visual scan order (top row first, then bottom row)."""
        for row in self.rows:
            yield from row.iter_scan_order()

    def pane_at(self, row: int, col: int) -> Pane:
        """Get a pane by global scan-order coordinates.

        Args:
            row: Pane row within a half-block (0-5).
            col: Global column index across all blocks in scan order (0-65).
                 Divmod by 6 to get (block_idx, half_idx, col_idx).

        Example::

            # Get pane at pane-row 2, column 35 (block 5, right half, col 2)
            pane = layout.pane_at(row=2, col=35)
        """
        block_idx = col // 6
        half_idx = (col % 6) // 3
        col_idx = col % 3
        return self.rows[0].blocks[block_idx].halves[half_idx].pane_at(row, col_idx)

    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) — the bounding box spanning the entire building layout."""
        all_panes_list = list(self.all_panes())
        if not all_panes_list:
            return (0, 0, 0, 0)
        x1 = min(p.x1 for p in all_panes_list)
        y1 = min(p.y1 for p in all_panes_list)
        x2 = max(p.x2 for p in all_panes_list)
        y2 = max(p.y2 for p in all_panes_list)
        return (x1, y1, x2, y2)

    @classmethod
    def from_json(cls, path: Path) -> Layout:
        """Load layout from a layout.json file.

        Args:
            path: Path to the layout.json produced by the extract pipeline.
        """
        with path.open() as f:
            data = json.load(f)

        return cls.from_data(data)

    @classmethod
    def from_data(cls, data: dict[str, object]) -> Layout:
        """Load layout from already-parsed pane data."""
        typed_data = cast("dict[str, Any]", data)

        pillar_src = _extract_pillars(typed_data)
        pillar_objects = tuple(Pillar(x_start, x_end) for x_start, x_end in pillar_src)

        rows = []
        for row_data in typed_data["rows"]:
            blocks = []
            for block_data in row_data["blocks"]:
                halves: list[Half] = []
                for half_data in block_data["halves"]:
                    pane_rows = tuple(
                        tuple(Pane(p["x1"], p["y1"], p["x2"], p["y2"]) for p in row_panes)
                        for row_panes in half_data["rows"]
                    )
                    halves.append(Half(pane_rows=pane_rows))
                blocks.append(Block(halves=(halves[0], halves[1])))
            rows.append(Row(blocks=tuple(blocks)))

        return cls(rows=(rows[0], rows[1]), pillars=pillar_objects)

    @classmethod
    def default(cls) -> Layout:
        """Return the hard-locked default building layout."""
        if cls._default_layout_cache is None:
            cls._default_layout_cache = cls.from_json(DEFAULT_LAYOUT_JSON_PATH)
        return cls._default_layout_cache


def _extract_pillars(data: dict[str, Any]) -> list[tuple[int, int]]:
    pillars_raw = data.get("pillars")
    if not isinstance(pillars_raw, list):
        msg = "Layout JSON must include a top-level 'pillars' list"
        raise TypeError(msg)

    pillars: list[tuple[int, int]] = []
    for entry in pillars_raw:
        if isinstance(entry, dict):
            x_start = int(entry["x_start"])
            x_end = int(entry["x_end"])
        else:
            x_start = int(entry[0])
            x_end = int(entry[1])
        pillars.append((x_start, x_end))
    return pillars
