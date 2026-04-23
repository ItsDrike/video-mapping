"""Perimeter: a clockwise rectangular loop parameterized in pixels for running-band animations.

The loop follows the outer boundary of the building facade in order:
  top wall (left to right) -> right pillar (top to bottom)
  -> bottom wall (right to left) -> left pillar (bottom to top)

Each position on the perimeter is measured in pixels of travel distance.  Use
``band_rects()`` to get the screen rectangles occupied by a moving band of a given
pixel length at any head position.

Corner handling
---------------
Adjacent sides share corner pixels (e.g. the top-right corner belongs to both the
top strip and the right pillar strip).  When the band straddles a corner,
``band_rects()`` returns two overlapping rectangles that together form a natural
L-shape.  Drawing them in the same color on the same frame is visually seamless.

Example::

    from video_mapping import Layout
    from video_mapping.perimeter import Perimeter

    layout = Layout.default()
    perimeter = Perimeter.from_layout(layout)
    band_len = layout.pillars[0].width

    # Get rects for a band whose head is 200 px into the loop
    for rect in perimeter.band_rects(200, band_len):
        canvas.color_region(rect, (255, 220, 0), 0.90)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from video_mapping.layout import Rect

if TYPE_CHECKING:
    from video_mapping.layout import Layout


@dataclass(frozen=True)
class _Seg:
    """One side of the rectangular perimeter loop.

    The stripe region (x1/y1/x2/y2) is the full pixel extent of this side.
    Travel proceeds along the x-axis when ``h`` is True, y-axis otherwise.
    ``fwd`` indicates the travel direction (increasing x or y when True).
    ``start`` is the cumulative pixel offset at the beginning of this segment;
    ``length`` is how many pixels this side spans in the travel direction.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    start: int
    length: int
    h: bool
    fwd: bool


def _seg_rect(seg: _Seg, local_start: int, local_end: int) -> Rect:
    """Convert a local pixel range to a Rect within the segment stripe.

    Args:
        seg: The segment to convert within.
        local_start: Start offset from the segment's travel origin (inclusive, 0-based).
        local_end: End offset from the segment's travel origin (exclusive).

    Returns:
        The Rect covering ``[local_start, local_end)`` pixels of travel within ``seg``.
    """
    if seg.h:
        if seg.fwd:
            return Rect(seg.x1 + local_start, seg.y1, seg.x1 + local_end - 1, seg.y2)
        return Rect(seg.x2 - local_end + 1, seg.y1, seg.x2 - local_start, seg.y2)
    if seg.fwd:
        return Rect(seg.x1, seg.y1 + local_start, seg.x2, seg.y1 + local_end - 1)
    return Rect(seg.x1, seg.y2 - local_end + 1, seg.x2, seg.y2 - local_start)


class Perimeter:
    """A clockwise rectangular perimeter loop around the building facade.

    The loop runs: top wall (left to right) -> right pillar (top to bottom) ->
    bottom wall (right to left) -> left pillar (bottom to top).

    Parameterized as a 1D distance in pixels.  Use ``band_rects()`` to get the
    screen rectangles for a moving band of a given length at any head position.

    Construct via ``Perimeter.from_layout(layout)``.
    """

    def __init__(self, segs: tuple[_Seg, ...], total_length: int) -> None:
        self._segs = segs
        self._total_length = total_length

    @property
    def total_length(self) -> int:
        """Total perimeter length in pixels (sum of all four side lengths)."""
        return self._total_length

    def band_rects(self, head: float, band_len: int) -> tuple[Rect, ...]:
        """Return the rectangles covered by a band of length ``band_len`` at position ``head``.

        The band occupies positions ``[head - band_len + 1, head]`` (inclusive) along
        the perimeter, wrapping around if necessary.

        Args:
            head: Leading-edge position in pixels.  Values outside ``[0, total_length)``
                are automatically wrapped.
            band_len: Length of the band in pixels.  Should be less than ``total_length``.

        Returns:
            One Rect per segment the band overlaps.  When the band straddles a corner
            the two adjacent rects share corner pixels, forming a natural L-shape.
        """
        total = self._total_length
        head_i = int(head) % total
        tail_i = (head_i - band_len + 1) % total

        rects: list[Rect] = []

        ranges: list[tuple[int, int]]
        if tail_i <= head_i:
            ranges = [(tail_i, head_i)]
        else:
            # Band wraps around the end of the perimeter
            ranges = [(tail_i, total - 1), (0, head_i)]

        for range_start, range_end in ranges:
            for seg in self._segs:
                seg_end = seg.start + seg.length - 1
                lo = max(range_start, seg.start)
                hi = min(range_end, seg_end)
                if lo <= hi:
                    rects.append(_seg_rect(seg, lo - seg.start, hi - seg.start + 1))

        return tuple(rects)

    @classmethod
    def from_layout(cls, layout: Layout) -> Perimeter:
        """Build a Perimeter from a Layout using the outermost pillars and wall strips.

        The perimeter is bounded by ``layout.pillars[0]`` (left) and
        ``layout.pillars[-1]`` (right), and vertically by ``layout.walls.above``
        and ``layout.walls.below``.

        Args:
            layout: A fully loaded Layout instance.

        Returns:
            A ``Perimeter`` ready for use with ``band_rects()``.
        """
        lp = layout.pillars[0]
        rp = layout.pillars[-1]
        above = layout.walls.above
        below = layout.walls.below

        h_len = rp.x2 - lp.x1 + 1
        v_len = below.y2 - above.y1 + 1

        top_seg = _Seg(
            x1=lp.x1,
            y1=above.y1,
            x2=rp.x2,
            y2=above.y2,
            start=0,
            length=h_len,
            h=True,
            fwd=True,
        )
        right_seg = _Seg(
            x1=rp.x1,
            y1=above.y1,
            x2=rp.x2,
            y2=below.y2,
            start=h_len,
            length=v_len,
            h=False,
            fwd=True,
        )
        bottom_seg = _Seg(
            x1=lp.x1,
            y1=below.y1,
            x2=rp.x2,
            y2=below.y2,
            start=h_len + v_len,
            length=h_len,
            h=True,
            fwd=False,
        )
        left_seg = _Seg(
            x1=lp.x1,
            y1=above.y1,
            x2=lp.x2,
            y2=below.y2,
            start=2 * h_len + v_len,
            length=v_len,
            h=False,
            fwd=False,
        )

        return cls(
            segs=(top_seg, right_seg, bottom_seg, left_seg),
            total_length=2 * h_len + 2 * v_len,
        )
