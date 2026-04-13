"""Canvas: a mutable numpy-backed drawing surface with building-layout-aware helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from video_mapping.layout import Block, Half, Pane, Pillar, Row
from video_mapping.types import RGBColor, U8Array


class Canvas:
    """A mutable RGB drawing surface backed by a numpy uint8 array.

    Coordinates follow image convention: (0, 0) is the top-left corner.
    All rectangle coordinates (including Pane and Pillar bounds) are *inclusive*
    on both ends.

    Usage::

        # Production: pure black background (what gets projected onto the building)
        canvas = Canvas.black(4096, 606)

        # Debug: draw on top of the building mask for spatial reference
        canvas = Canvas.from_image(Path("static/color-mask.png"))

        canvas.color_pane(layout.top_row.blocks[0].left.pane_at(0, 0), (255, 0, 0))
        canvas.fill_pillar_bar(layout.pillars[3], bar_height=200, color=(0, 255, 0))
    """

    def __init__(self, array: U8Array) -> None:
        self._arr = array

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def black(cls, width: int, height: int) -> Canvas:
        """Create a fully-black canvas (the projection surface)."""
        return cls(np.zeros((height, width, 3), dtype=np.uint8))

    @classmethod
    def from_image(cls, path: Path) -> Canvas:
        """Load a canvas from an image file.

        Useful for loading the building mask as a background when you want
        spatial context while developing a visualisation.
        """
        img = Image.open(path).convert("RGB")
        return cls(np.array(img, dtype=np.uint8))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Canvas width in pixels."""
        return int(self._arr.shape[1])

    @property
    def height(self) -> int:
        """Canvas height in pixels."""
        return int(self._arr.shape[0])

    def copy(self) -> Canvas:
        """Return an independent copy. Use this to cheaply clone a base frame."""
        return Canvas(self._arr.copy())

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_array(self) -> U8Array:
        """Expose the underlying array (shape: height x width x 3, dtype uint8)."""
        return self._arr

    def to_image(self) -> Image.Image:
        """Convert to a PIL Image."""
        return Image.fromarray(self._arr, mode="RGB")

    def save(self, path: Path) -> None:
        """Write the canvas to a PNG file."""
        self.to_image().save(path)

    # ------------------------------------------------------------------
    # Low-level drawing
    # ------------------------------------------------------------------

    def fill_rect(self, x1: int, y1: int, x2: int, y2: int, color: RGBColor) -> None:
        """Solid-fill a rectangle. All coords are inclusive."""
        self._arr[y1 : y2 + 1, x1 : x2 + 1] = color

    def blend_rect(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: RGBColor,
        alpha: float,
    ) -> None:
        """Alpha-blend *color* onto a rectangle (alpha=1.0 → fully opaque). Coords are inclusive."""
        region = self._arr[y1 : y2 + 1, x1 : x2 + 1].astype(np.float32)
        color_arr = np.array(color, dtype=np.float32)
        blended = region * (1.0 - alpha) + color_arr * alpha
        self._arr[y1 : y2 + 1, x1 : x2 + 1] = blended.clip(0, 255).astype(np.uint8)

    def _draw_rect(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: RGBColor,
        alpha: float,
    ) -> None:
        if alpha >= 1.0:
            self.fill_rect(x1, y1, x2, y2, color)
        else:
            self.blend_rect(x1, y1, x2, y2, color, alpha)

    # ------------------------------------------------------------------
    # Pane / window drawing
    # ------------------------------------------------------------------

    def color_pane(self, pane: Pane, color: RGBColor, alpha: float = 1.0) -> None:
        """Color a single window pane."""
        self._draw_rect(pane.x1, pane.y1, pane.x2, pane.y2, color, alpha)

    def color_panes(self, panes: Iterable[Pane], color: RGBColor, alpha: float = 1.0) -> None:
        """Color an arbitrary collection of panes with the same color."""
        for pane in panes:
            self.color_pane(pane, color, alpha)

    def color_half_pane_row(
        self,
        half: Half,
        row_idx: int,
        color: RGBColor,
        alpha: float = 1.0,
    ) -> None:
        """Color all panes in a single horizontal pane-row within a half-block."""
        for pane in half.pane_rows[row_idx]:
            self.color_pane(pane, color, alpha)

    def color_half(self, half: Half, color: RGBColor, alpha: float = 1.0) -> None:
        """Color all 18 panes in a half-block."""
        self.color_panes(half.all_panes(), color, alpha)

    def color_block(self, block: Block, color: RGBColor, alpha: float = 1.0) -> None:
        """Color all panes in a block (left and right halves)."""
        self.color_panes(block.all_panes(), color, alpha)

    def color_row(self, row: Row, color: RGBColor, alpha: float = 1.0) -> None:
        """Color every pane in an entire building row."""
        self.color_panes(row.all_panes(), color, alpha)

    # ------------------------------------------------------------------
    # Pillar drawing
    # ------------------------------------------------------------------

    def fill_pillar(self, pillar: Pillar, color: RGBColor) -> None:
        """Solid-fill a pillar from top to bottom of the canvas."""
        self._arr[:, pillar.x_start : pillar.x_end + 1] = color

    def fill_pillar_bar(
        self,
        pillar: Pillar,
        bar_height: int,
        color: RGBColor,
        alpha: float = 1.0,
    ) -> None:
        """Fill a pillar bar that rises *bar_height* pixels from the bottom of the canvas."""
        y_start = max(0, self.height - bar_height)
        self._draw_rect(pillar.x_start, y_start, pillar.x_end, self.height - 1, color, alpha)
