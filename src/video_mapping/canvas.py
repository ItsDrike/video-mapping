"""Canvas: a mutable numpy-backed drawing surface with building-layout-aware helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from video_mapping.layout import Block, Half, Pane, Pillar, Rect, Row, WallStrips
    from video_mapping.types import RGBColor, U8Array


class Canvas:  # noqa: PLR0904 - Canvas is intentionally a rich drawing facade
    """A mutable RGB/RGBA drawing surface backed by a numpy uint8 array.

    Coordinates follow image convention: (0, 0) is the top-left corner.
    All rectangle coordinates (including Pane and Pillar bounds) are *inclusive*
    on both ends.

    Usage::

        # Production: transparent background for alpha-enabled exports
        canvas = Canvas.transparent(4096, 606)

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
    def transparent(cls, width: int, height: int) -> Canvas:
        """Create a fully-transparent RGBA canvas."""
        return cls(np.zeros((height, width, 4), dtype=np.uint8))

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
        """Expose the underlying array (shape: height x width x 3|4, dtype uint8)."""
        return self._arr

    def to_image(self) -> Image.Image:
        """Convert to a PIL Image."""
        mode = "RGBA" if self._arr.shape[2] == 4 else "RGB"
        return Image.fromarray(self._arr, mode=mode)

    def save(self, path: Path) -> None:
        """Write the canvas to a PNG file."""
        self.to_image().save(path)

    # ------------------------------------------------------------------
    # Low-level drawing
    # ------------------------------------------------------------------

    def fill_rect(self, *, x1: int, y1: int, x2: int, y2: int, color: RGBColor) -> None:
        """Solid-fill a rectangle. All coords are inclusive."""
        region = self._arr[y1 : y2 + 1, x1 : x2 + 1]
        region[..., :3] = color
        if self._arr.shape[2] == 4:
            region[..., 3] = 255

    def blend_rect(
        self,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: RGBColor,
        alpha: float,
    ) -> None:
        """Alpha-blend *color* onto a rectangle (alpha=1.0 → fully opaque). Coords are inclusive."""
        region = self._arr[y1 : y2 + 1, x1 : x2 + 1]
        color_arr = np.array(color, dtype=np.float32)
        if self._arr.shape[2] == 3:
            region_rgb = region.astype(np.float32)
            blended = region_rgb * (1.0 - alpha) + color_arr * alpha
            region[:] = blended.clip(0, 255).astype(np.uint8)
            return

        region_f = region.astype(np.float32)
        dst_rgb = region_f[..., :3] / 255.0
        dst_alpha = region_f[..., 3:4] / 255.0

        src_alpha = np.float32(alpha)
        out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)

        src_rgb = color_arr / 255.0
        numer_rgb = src_rgb * src_alpha + dst_rgb * dst_alpha * (1.0 - src_alpha)
        out_rgb = np.zeros_like(dst_rgb)
        _ = np.divide(numer_rgb, out_alpha, out=out_rgb, where=out_alpha > 0.0)

        region[..., :3] = (out_rgb * 255.0).clip(0, 255).astype(np.uint8)
        region[..., 3] = (out_alpha[..., 0] * 255.0).clip(0, 255).astype(np.uint8)

    def _draw_rect(
        self,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: RGBColor,
        alpha: float,
    ) -> None:
        if alpha >= 1.0:
            self.fill_rect(x1=x1, y1=y1, x2=x2, y2=y2, color=color)
        else:
            self.blend_rect(x1=x1, y1=y1, x2=x2, y2=y2, color=color, alpha=alpha)

    # ------------------------------------------------------------------
    # Pane / window drawing
    # ------------------------------------------------------------------

    def color_pane(self, pane: Pane, color: RGBColor, alpha: float = 1.0) -> None:
        """Color a single window pane."""
        self._draw_rect(x1=pane.x1, y1=pane.y1, x2=pane.x2, y2=pane.y2, color=color, alpha=alpha)

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
    # Bounding box drawing (useful for overlays that cross pane boundaries)
    # ------------------------------------------------------------------

    def fill_bbox(self, bbox: Rect, color: RGBColor) -> None:
        """Fill a bounding box with a solid color.

        Useful for drawing solid regions even when individual panes don't perfectly
        cover the area (e.g. a half-block, block, or row overlay).
        """
        self.fill_rect(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2, color=color)

    def blend_bbox(self, bbox: Rect, color: RGBColor, alpha: float) -> None:
        """Alpha-blend a color onto a bounding box.

        Useful for translucent overlays across structural regions.
        """
        self.blend_rect(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2, color=color, alpha=alpha)

    # ------------------------------------------------------------------
    # Generic Rect / grid-strip / wall drawing
    # ------------------------------------------------------------------

    def color_region(self, region: Rect, color: RGBColor, alpha: float = 1.0) -> None:
        """Color any Rect — pane, pillar, wall section, grid strip, or bounding box."""
        self._draw_rect(x1=region.x1, y1=region.y1, x2=region.x2, y2=region.y2, color=color, alpha=alpha)

    def fill_half_grid(self, half: Half, color: RGBColor, alpha: float = 1.0) -> None:
        """Fill all inter-pane grid strips (window frame) in a half-block.

        Draws both the horizontal strips (between pane rows) and the vertical
        strips (between pane columns) with the same color.
        """
        for strip in half.all_grid_strips():
            self._draw_rect(x1=strip.x1, y1=strip.y1, x2=strip.x2, y2=strip.y2, color=color, alpha=alpha)

    def fill_wall(self, wall: Rect, color: RGBColor, alpha: float = 1.0) -> None:
        """Fill a single horizontal wall section (above/middle/below the window rows)."""
        self._draw_rect(x1=wall.x1, y1=wall.y1, x2=wall.x2, y2=wall.y2, color=color, alpha=alpha)

    def fill_walls(self, walls: WallStrips, color: RGBColor, alpha: float = 1.0) -> None:
        """Fill all wall sections with the same color (above, middle_top, middle_bottom, below)."""
        for w in walls.strips():
            self._draw_rect(x1=w.x1, y1=w.y1, x2=w.x2, y2=w.y2, color=color, alpha=alpha)

    # ------------------------------------------------------------------
    # Pillar drawing
    # ------------------------------------------------------------------

    def fill_pillar(self, pillar: Pillar, color: RGBColor) -> None:
        """Solid-fill the full pillar rect (spans y1 to y2 as stored in JSON)."""
        self.fill_rect(x1=pillar.x1, y1=pillar.y1, x2=pillar.x2, y2=pillar.y2, color=color)

    def fill_pillar_bar(
        self,
        pillar: Pillar,
        bar_height: int,
        color: RGBColor,
        alpha: float = 1.0,
    ) -> None:
        """Fill a pillar bar that rises *bar_height* pixels from the bottom of the pillar."""
        y_start = max(pillar.y1, pillar.y2 - bar_height + 1)
        self._draw_rect(x1=pillar.x1, y1=y_start, x2=pillar.x2, y2=pillar.y2, color=color, alpha=alpha)
