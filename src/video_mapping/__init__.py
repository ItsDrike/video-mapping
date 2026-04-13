"""video_mapping — programmatic drawing API for the building-facade video mapping project.

Quick-start::

    from pathlib import Path
    from video_mapping import Canvas, Layout, VideoWriter

    layout = Layout.from_json(Path("static/panes.json"))
    base = Canvas.black(4096, 606)

    # Draw something on a single pane
    base.color_pane(layout.top_row.blocks[2].left.pane_at(0, 1), (255, 0, 0))

    # Render a video
    with VideoWriter(Path("output/out.mp4"), width=base.width, height=base.height, fps=25) as writer:
        for _ in range(90):  # 3 seconds at 30 fps
            frame = base.copy()
            writer.write_canvas(frame)
"""

from video_mapping.canvas import Canvas
from video_mapping.layout import Block, Half, Layout, Pane, Pillar, Row
from video_mapping.render import VideoWriter

__all__ = [
    "Block",
    "Canvas",
    "Half",
    "Layout",
    "Pane",
    "Pillar",
    "Row",
    "VideoWriter",
]
