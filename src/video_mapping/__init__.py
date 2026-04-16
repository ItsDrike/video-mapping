"""video_mapping — programmatic drawing API for the building-facade video mapping project.

Quick-start::

from pathlib import Path
    from video_mapping import Canvas, Layout, VideoWriter
    from video_mapping.constants import DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH

    layout = Layout.default()
    base = Canvas.transparent(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT)

    # Draw something on a single pane
    base.color_pane(layout.top_row.blocks[2].left.pane_at(0, 1), (255, 0, 0))

    # Render a video
    with VideoWriter(Path("output/out.mp4"), width=base.width, height=base.height, fps=25) as writer:
        for _ in range(75):  # 3 seconds at 25 fps
            frame = base.copy()
            writer.write_canvas(frame)
"""

from video_mapping.canvas import Canvas
from video_mapping.layout import Block, BlockWalls, Half, Layout, Pane, Pillar, Rect, Row, WallStrips
from video_mapping.render import VideoWriter

__all__ = [
    "Block",
    "BlockWalls",
    "Canvas",
    "Half",
    "Layout",
    "Pane",
    "Pillar",
    "Rect",
    "Row",
    "VideoWriter",
    "WallStrips",
]
