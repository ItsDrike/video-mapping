# video-mapping

`video-mapping` is a Python toolkit for generating animations that fit a specific building facade used for projection mapping.

The target geometry is defined by `static/color-mask.png` (with structure
metadata in `static/layout.json`). This project exists to make it easy to
create animated visuals that align with that facade by providing:

- layout-aware drawing primitives (panes, block halves, blocks, rows, pillars, walls)
- a numpy/Pillow-backed `Canvas` abstraction
- a `VideoWriter` wrapper for ffmpeg output
- audio analysis helpers for beat/energy-driven animation
- ready-to-run CLI scripts for several animation styles

## Installation

Python `3.14+` is required.

### 1) Install uv

Install `uv` from the official docs:

- <https://docs.astral.sh/uv/getting-started/installation/>

### 2) Install dependencies

From the repository root:

```bash
uv sync
```

This command is the same on Linux, macOS, and Windows.

## Running the built-in scripts

Scripts are exposed as project entry points. Run them with `uv run`.

```bash
uv run audio-visualizer --audio audio.wav --output output/audio_visualizer.webm
uv run blink-animation --output output/blink_animation.webm
uv run pillar-choreography --audio audio.wav --output output/pillar_choreography.webm
uv run half-block-beats --audio audio.wav --output output/half_block_beats.webm
```

Debug helpers:

```bash
uv run debug-extract-layout --mask static/color-mask.png --color-mask static/color-mask.png --output static/layout.json
uv run debug-draw-layout --output output/layout_debug.webm
```

## Using as a library

You can write your own scripts using the provided abstractions.

```python
from pathlib import Path

from video_mapping import Canvas, Layout, VideoWriter
from video_mapping.constants import DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH

layout = Layout.default()
base = Canvas.transparent(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT)

# Light one pane in the top row (block 2, left half, row 0, col 1)
target = layout.top_row.blocks[2].left.pane_at(0, 1)

with VideoWriter(
    Path("output/custom_demo.webm"),
    width=base.width,
    height=base.height,
    fps=25,
    total_frames=75,
    transparent=True,
) as writer:
    for frame_idx in range(75):
        frame = base.copy()
        alpha = min(1.0, frame_idx / 20)
        frame.color_pane(target, (255, 80, 40), alpha=alpha)
        writer.write_canvas(frame)
```

## Contributing

This project uses `pre-commit` for linting, formatting, and type checking.

- Hooks include Ruff, basedpyright, and ty.
- Hook commands use `uv run` and `uvx`.

See `CONTRIBUTING.md` for contributor setup and commands.
