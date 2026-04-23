"""Building layout data model: panes, halves, blocks, rows, pillars, walls, grids."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, cast

from video_mapping.constants import DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, DEFAULT_LAYOUT_JSON_PATH

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangular region. All coordinates are inclusive pixel positions.

    This is the base for all positioned building elements: Pane, Pillar, WallStrips members,
    and inter-pane grid strips are all Rect instances or subclasses.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Region width in pixels."""
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        """Region height in pixels."""
        return self.y2 - self.y1 + 1

    def bbox(self) -> Rect:
        """Return self — this object is already a Rect."""
        return self


@dataclass(frozen=True)
class WallStrips:
    """The three horizontal wall sections of the building facade.

    These are the structural wall regions that surround the two window rows:
    above the top row, between the two rows, and below the bottom row.
    Each strip spans the full canvas width.
    """

    above: Rect  # wall above the top window row
    middle: Rect  # wall between the two window rows
    middle_top: Rect  # top (red) band of the middle wall
    middle_bottom: Rect  # bottom (gray) band of the middle wall
    below: Rect  # wall below the bottom window row

    def strips(self) -> tuple[Rect, Rect, Rect, Rect]:
        """Return the four non-overlapping wall strips in top-to-bottom order.

        Returns ``(above, middle_top, middle_bottom, below)``.  The ``middle``
        field is the union of ``middle_top`` + ``middle_bottom`` and is
        intentionally omitted here to avoid double-counting.
        """
        return (self.above, self.middle_top, self.middle_bottom, self.below)


@dataclass(frozen=True)
class BlockWalls:
    """Per-block wall regions and their semantic grouping.

    For each block in each row:
    - ``top_gray``: horizontal gray wall above this block's windows
    - ``vertical_red``: vertical red wall between left/right halves
    - ``bottom_red``: horizontal red wall below this block's windows
    - ``outer_left_v``: thin vertical gap strip between the left pillar and the left half
    - ``outer_right_v``: thin vertical gap strip between the right half and the right pillar

    The outer strips span only the pane y-range of the block (not the full canvas height).
    They are ``None`` when no adjacent pillar is found (should not happen in a valid layout).
    """

    top_gray: Rect
    vertical_red: Rect
    bottom_red: Rect
    outer_left_v: Rect | None = None
    outer_right_v: Rect | None = None

    @property
    def middle_wall(self) -> Rect:
        """Vertical wall between left and right halves."""
        return self.vertical_red

    def rects(self) -> tuple[Rect, Rect, Rect]:
        """Return the three primary wall regions for this block (top_gray, vertical_red, bottom_red)."""
        return (self.top_gray, self.vertical_red, self.bottom_red)

    def outer_v_strips(self) -> tuple[Rect, ...]:
        """Return the outer vertical gap strips on either side of this block's pane area."""
        result: list[Rect] = []
        if self.outer_left_v is not None:
            result.append(self.outer_left_v)
        if self.outer_right_v is not None:
            result.append(self.outer_right_v)
        return tuple(result)

    def red(self) -> tuple[Rect, Rect]:
        """Return red wall regions for this block (vertical + bottom horizontal)."""
        return (self.vertical_red, self.bottom_red)

    def gray(self) -> tuple[Rect]:
        """Return gray wall regions for this block (top horizontal)."""
        return (self.top_gray,)


@dataclass(frozen=True)
class Pane(Rect):
    """A single window pane.

    Inherits x1, y1, x2, y2, width, height, bbox() from Rect.
    Adds cx/cy for the pane centre — useful for positioning effects.
    """

    @property
    def cx(self) -> float:
        """Horizontal centre pixel."""
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        """Vertical centre pixel."""
        return (self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class Pillar(Rect):
    """A structural pillar between window blocks.

    Inherits x1, y1, x2, y2, width from Rect. y1/y2 span the full canvas height
    (0 and canvas_height-1). x_start/x_end are aliases for x1/x2 for readability.
    """

    @property
    def x_start(self) -> int:
        """Left edge pixel column (alias for x1)."""
        return self.x1

    @property
    def x_end(self) -> int:
        """Right edge pixel column (alias for x2)."""
        return self.x2


@dataclass(frozen=True)
class Half:
    """A half-block: 6 pane-rows x 3 pane-columns = 18 panes.

    pane_rows[row_idx][col_idx] gives the Pane at that position.
    h_strips holds the 5 horizontal grid strips between consecutive pane rows.
    v_strips holds the 2 vertical grid strips between consecutive pane columns.
    Together, h_strips and v_strips form the structural grid (window frame) between panes.
    """

    pane_rows: tuple[tuple[Pane, ...], ...]
    h_strips: tuple[Rect, ...] = field(default=())
    v_strips: tuple[Rect, ...] = field(default=())

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

    def all_grid_strips(self) -> Iterator[Rect]:
        """Yield every grid strip (h_strips then v_strips) in this half."""
        yield from self.h_strips
        yield from self.v_strips

    def bbox(self) -> Rect:
        """Return the bounding box spanning all panes in this half.

        All coordinates are inclusive and cover the entire half-block region,
        even gaps between individual panes.
        """
        all_panes_list = list(self.all_panes())
        if not all_panes_list:
            return Rect(0, 0, 0, 0)
        return Rect(
            min(p.x1 for p in all_panes_list),
            min(p.y1 for p in all_panes_list),
            max(p.x2 for p in all_panes_list),
            max(p.y2 for p in all_panes_list),
        )

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
    walls: BlockWalls | None = None

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

    def _walls_or_raise(self) -> BlockWalls:
        if self.walls is None:
            msg = "Block wall metadata is not available"
            raise RuntimeError(msg)
        return self.walls

    @property
    def middle_wall(self) -> Rect:
        """Vertical wall between the left and right halves."""
        return self._walls_or_raise().middle_wall

    def all_walls(self) -> tuple[Rect, Rect, Rect]:
        """Return all wall regions associated with this block."""
        return self._walls_or_raise().rects()

    def red_walls(self) -> tuple[Rect, Rect]:
        """Return red wall regions for this block."""
        return self._walls_or_raise().red()

    def gray_walls(self) -> tuple[Rect]:
        """Return gray wall regions for this block."""
        return self._walls_or_raise().gray()

    def outer_v_strips(self) -> tuple[Rect, ...]:
        """Return outer vertical gap strips between pillars and this block's pane area."""
        if self.walls is None:
            return ()
        return self.walls.outer_v_strips()

    def bbox(self) -> Rect:
        """Return the bounding box spanning all panes in this block."""
        all_panes_list = list(self.all_panes())
        if not all_panes_list:
            return Rect(0, 0, 0, 0)
        return Rect(
            min(p.x1 for p in all_panes_list),
            min(p.y1 for p in all_panes_list),
            max(p.x2 for p in all_panes_list),
            max(p.y2 for p in all_panes_list),
        )

    def full_bbox(self) -> Rect:
        """Return the bounding box spanning the entire block including its top and bottom walls.

        Unlike bbox() which covers only the pane area, full_bbox() extends from the
        top of the top_gray wall to the bottom of the bottom_red wall, giving the full
        height of the block as a structural unit of the building facade.
        """
        walls = self._walls_or_raise()
        return Rect(walls.top_gray.x1, walls.top_gray.y1, walls.bottom_red.x2, walls.bottom_red.y2)


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

    def all_walls(self) -> Iterator[Rect]:
        """Yield all wall regions associated with this row's blocks."""
        for block in self.blocks:
            yield from block.all_walls()

    def red_walls(self) -> Iterator[Rect]:
        """Yield red wall regions associated with this row's blocks."""
        for block in self.blocks:
            yield from block.red_walls()

    def gray_walls(self) -> Iterator[Rect]:
        """Yield gray wall regions associated with this row's blocks."""
        for block in self.blocks:
            yield from block.gray_walls()

    def bbox(self) -> Rect:
        """Return the bounding box spanning all panes in this row."""
        all_panes_list = list(self.all_panes())
        if not all_panes_list:
            return Rect(0, 0, 0, 0)
        return Rect(
            min(p.x1 for p in all_panes_list),
            min(p.y1 for p in all_panes_list),
            max(p.x2 for p in all_panes_list),
            max(p.y2 for p in all_panes_list),
        )


@dataclass(frozen=True)
class Layout:
    """The complete building layout: two horizontal rows of blocks, pillars, and wall sections."""

    _default_layout_cache: ClassVar[Layout | None] = None

    rows: tuple[Row, Row]
    pillars: tuple[Pillar, ...]
    walls: WallStrips

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

    def all_walls(self) -> Iterator[Rect]:
        """Yield all block-associated wall regions in the full layout."""
        for row in self.rows:
            yield from row.all_walls()

    def red_walls(self) -> Iterator[Rect]:
        """Yield all red wall regions in the full layout."""
        for row in self.rows:
            yield from row.red_walls()

    def gray_walls(self) -> Iterator[Rect]:
        """Yield all gray wall regions in the full layout."""
        for row in self.rows:
            yield from row.gray_walls()

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

    def bbox(self) -> Rect:
        """Return the bounding box spanning the entire building layout."""
        all_panes_list = list(self.all_panes())
        if not all_panes_list:
            return Rect(0, 0, 0, 0)
        return Rect(
            min(p.x1 for p in all_panes_list),
            min(p.y1 for p in all_panes_list),
            max(p.x2 for p in all_panes_list),
            max(p.y2 for p in all_panes_list),
        )

    @classmethod
    def from_json(cls, path: Path) -> Layout:
        """Load layout from a layout.json file.

        Args:
            path: Path to the layout.json produced by the extract pipeline.
        """
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_data(data)

    @classmethod
    def from_data(cls, data: dict[str, object]) -> Layout:
        """Load layout from already-parsed pane data."""
        typed_data = cast("dict[str, Any]", data)

        pillar_objects = _extract_pillars(typed_data)

        rows: list[Row] = []
        for row_data in typed_data["rows"]:
            blocks: list[Block] = []
            for block_data in row_data["blocks"]:
                halves: list[Half] = []
                for half_data in block_data["halves"]:
                    pane_rows = tuple(
                        tuple(Pane(p["x1"], p["y1"], p["x2"], p["y2"]) for p in row_panes)
                        for row_panes in half_data["rows"]
                    )
                    h_strips_raw = half_data.get("h_strips", [])
                    v_strips_raw = half_data.get("v_strips", [])
                    if h_strips_raw or v_strips_raw:
                        h_strips = tuple(Rect(s["x1"], s["y1"], s["x2"], s["y2"]) for s in h_strips_raw)
                        v_strips = tuple(Rect(s["x1"], s["y1"], s["x2"], s["y2"]) for s in v_strips_raw)
                    else:
                        h_strips, v_strips = _compute_half_strips(pane_rows)
                    halves.append(Half(pane_rows=pane_rows, h_strips=h_strips, v_strips=v_strips))
                blocks.append(
                    Block(
                        halves=(halves[0], halves[1]),
                        walls=_extract_block_walls(block_data),
                    )
                )
            rows.append(Row(blocks=tuple(blocks)))

        walls = _extract_walls(typed_data, rows)
        rows_with_walls = _attach_computed_block_walls(rows, walls, pillar_objects)
        return cls(rows=(rows_with_walls[0], rows_with_walls[1]), pillars=pillar_objects, walls=walls)

    @classmethod
    def default(cls) -> Layout:
        """Return the hard-locked default building layout."""
        if cls._default_layout_cache is None:
            cls._default_layout_cache = cls.from_json(DEFAULT_LAYOUT_JSON_PATH)
        return cls._default_layout_cache


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_pillars(data: dict[str, Any]) -> tuple[Pillar, ...]:
    """Parse pillars from JSON, handling both new {x1,y1,x2,y2} and old {x_start,x_end} formats."""
    pillars_raw = data.get("pillars")
    if not isinstance(pillars_raw, list):
        msg = "Layout JSON must include a top-level 'pillars' list"
        raise TypeError(msg)

    y2_default = DEFAULT_CANVAS_HEIGHT - 1
    pillars: list[Pillar] = []
    for entry in pillars_raw:
        if isinstance(entry, dict):
            if "x1" in entry:
                pillars.append(Pillar(int(entry["x1"]), int(entry["y1"]), int(entry["x2"]), int(entry["y2"])))
            else:
                # Backward compat: old {x_start, x_end} format
                pillars.append(Pillar(int(entry["x_start"]), 0, int(entry["x_end"]), y2_default))
        else:
            pillars.append(Pillar(int(entry[0]), 0, int(entry[1]), y2_default))
    return tuple(pillars)


def _extract_walls(data: dict[str, Any], rows: list[Row]) -> WallStrips:
    """Load wall sections from JSON, or compute them from pane positions as fallback."""
    walls_raw = data.get("walls")
    if isinstance(walls_raw, dict):
        middle = Rect(**walls_raw["middle"])
        middle_top_raw = walls_raw.get("middle_top")
        middle_bottom_raw = walls_raw.get("middle_bottom")
        if isinstance(middle_top_raw, dict) and isinstance(middle_bottom_raw, dict):
            middle_top = Rect(**middle_top_raw)
            middle_bottom = Rect(**middle_bottom_raw)
        else:
            middle_top, middle_bottom = _split_middle_wall(middle)

        return WallStrips(
            above=Rect(**walls_raw["above"]),
            middle=middle,
            middle_top=middle_top,
            middle_bottom=middle_bottom,
            below=Rect(**walls_raw["below"]),
        )
    return _compute_walls_from_rows(rows)


def _compute_walls_from_rows(rows: list[Row]) -> WallStrips:
    """Compute wall section bounding boxes from pane positions."""
    top_panes = list(rows[0].all_panes())
    bot_panes = list(rows[1].all_panes())

    top_y1 = min(p.y1 for p in top_panes)
    top_y2 = max(p.y2 for p in top_panes)
    bot_y1 = min(p.y1 for p in bot_panes)
    bot_y2 = max(p.y2 for p in bot_panes)

    w = DEFAULT_CANVAS_WIDTH - 1
    h = DEFAULT_CANVAS_HEIGHT - 1

    middle = Rect(0, top_y2 + 1, w, bot_y1 - 1)
    middle_top, middle_bottom = _split_middle_wall(middle)

    return WallStrips(
        above=Rect(0, 0, w, top_y1 - 1),
        middle=middle,
        middle_top=middle_top,
        middle_bottom=middle_bottom,
        below=Rect(0, bot_y2 + 1, w, h),
    )


def _split_middle_wall(middle: Rect) -> tuple[Rect, Rect]:
    """Split the middle wall strip into top and bottom halves."""
    if middle.height <= 1:
        return (middle, middle)

    split_y = middle.y1 + (middle.height // 2) - 1
    split_y = max(middle.y1, min(middle.y2 - 1, split_y))

    return (
        Rect(middle.x1, middle.y1, middle.x2, split_y),
        Rect(middle.x1, split_y + 1, middle.x2, middle.y2),
    )


def _extract_block_walls(block_data: dict[str, Any]) -> BlockWalls | None:
    """Parse per-block wall metadata when present in layout JSON."""
    walls_raw = block_data.get("walls")
    if not isinstance(walls_raw, dict):
        return None

    outer_left_raw = walls_raw.get("outer_left_v")
    outer_right_raw = walls_raw.get("outer_right_v")
    return BlockWalls(
        top_gray=Rect(**walls_raw["top_gray"]),
        vertical_red=Rect(**walls_raw["between_halves_red"]),
        bottom_red=Rect(**walls_raw["bottom_red"]),
        outer_left_v=Rect(**outer_left_raw) if outer_left_raw else None,
        outer_right_v=Rect(**outer_right_raw) if outer_right_raw else None,
    )


def _attach_computed_block_walls(
    rows: list[Row],
    walls: WallStrips,
    pillars: tuple[Pillar, ...],
) -> list[Row]:
    """Fill missing per-block wall metadata from pane geometry, global wall strips, and pillars."""
    rows_out: list[Row] = []
    for row_idx, row in enumerate(rows):
        blocks_out: list[Block] = []
        for block in row.blocks:
            existing = block.walls

            # Skip blocks that already have all data including outer strips
            if existing is not None and existing.outer_left_v is not None:
                blocks_out.append(block)
                continue

            if existing is None:
                existing = _compute_block_walls(row_idx, block, walls)

            outer_left, outer_right = _compute_outer_v_strips(block, pillars)
            blocks_out.append(
                Block(
                    halves=block.halves,
                    walls=BlockWalls(
                        top_gray=existing.top_gray,
                        vertical_red=existing.vertical_red,
                        bottom_red=existing.bottom_red,
                        outer_left_v=outer_left,
                        outer_right_v=outer_right,
                    ),
                )
            )

        rows_out.append(Row(blocks=tuple(blocks_out)))

    return rows_out


def _compute_block_walls(row_idx: int, block: Block, walls: WallStrips) -> BlockWalls:
    """Compute per-block wall regions for top/bottom rows (without outer strips)."""
    left_bbox = block.left.bbox()
    right_bbox = block.right.bbox()
    block_bbox = block.bbox()

    between_x1 = left_bbox.x2 + 1
    between_x2 = right_bbox.x1 - 1
    if between_x1 > between_x2:
        mid_x = (left_bbox.x2 + right_bbox.x1) // 2
        between_x1 = mid_x
        between_x2 = mid_x

    if row_idx == 0:
        top_gray = Rect(block_bbox.x1, walls.above.y1, block_bbox.x2, walls.above.y2)
        bottom_red = Rect(block_bbox.x1, walls.middle_top.y1, block_bbox.x2, walls.middle_top.y2)
    else:
        top_gray = Rect(block_bbox.x1, walls.middle_bottom.y1, block_bbox.x2, walls.middle_bottom.y2)
        bottom_red = Rect(block_bbox.x1, walls.below.y1, block_bbox.x2, walls.below.y2)

    between_halves_red = Rect(between_x1, block_bbox.y1, between_x2, block_bbox.y2)
    return BlockWalls(
        top_gray=top_gray,
        vertical_red=between_halves_red,
        bottom_red=bottom_red,
    )


def _compute_outer_v_strips(block: Block, pillars: tuple[Pillar, ...]) -> tuple[Rect | None, Rect | None]:
    """Compute outer vertical gap strips between adjacent pillars and a block's pane area.

    Returns ``(outer_left, outer_right)``.  Either may be ``None`` if no adjacent
    pillar is found (e.g. when calling on synthetic/test data with no pillars).
    """
    left_bbox = block.left.bbox()
    right_bbox = block.right.bbox()
    pane_y1 = min(left_bbox.y1, right_bbox.y1)
    pane_y2 = max(left_bbox.y2, right_bbox.y2)

    # Left outer strip: rightmost pillar whose x2 < left half's x1
    left_candidates = [p.x2 for p in pillars if p.x2 < left_bbox.x1]
    outer_left: Rect | None = None
    if left_candidates:
        lx2 = max(left_candidates)
        x1, x2 = lx2 + 1, left_bbox.x1 - 1
        if x1 <= x2:
            outer_left = Rect(x1, pane_y1, x2, pane_y2)

    # Right outer strip: leftmost pillar whose x1 > right half's x2
    right_candidates = [p.x1 for p in pillars if p.x1 > right_bbox.x2]
    outer_right: Rect | None = None
    if right_candidates:
        rx1 = min(right_candidates)
        x1, x2 = right_bbox.x2 + 1, rx1 - 1
        if x1 <= x2:
            outer_right = Rect(x1, pane_y1, x2, pane_y2)

    return outer_left, outer_right


def _compute_half_strips(
    pane_rows: tuple[tuple[Pane, ...], ...],
) -> tuple[tuple[Rect, ...], tuple[Rect, ...]]:
    """Compute horizontal and vertical grid strips from pane rows (fallback for old JSON)."""
    if not pane_rows or not pane_rows[0]:
        return ((), ())

    x1 = min(p.x1 for row in pane_rows for p in row)
    x2 = max(p.x2 for row in pane_rows for p in row)
    y1 = min(p.y1 for row in pane_rows for p in row)
    y2 = max(p.y2 for row in pane_rows for p in row)

    h_strips: list[Rect] = []
    for i in range(len(pane_rows) - 1):
        upper = pane_rows[i]
        lower = pane_rows[i + 1]
        sy1 = max(p.y2 for p in upper) + 1
        sy2 = min(p.y1 for p in lower) - 1
        if sy1 <= sy2:
            h_strips.append(Rect(x1, sy1, x2, sy2))

    num_cols = len(pane_rows[0])
    v_strips: list[Rect] = []
    for c in range(num_cols - 1):
        left_col = [row[c] for row in pane_rows]
        right_col = [row[c + 1] for row in pane_rows]
        sx1 = max(p.x2 for p in left_col) + 1
        sx2 = min(p.x1 for p in right_col) - 1
        if sx1 <= sx2:
            v_strips.append(Rect(sx1, y1, sx2, y2))

    return (tuple(h_strips), tuple(v_strips))
