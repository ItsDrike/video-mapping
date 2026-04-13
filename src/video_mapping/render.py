"""Video rendering: stream raw RGB frames to ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Self

from video_mapping.types import U8Array

if TYPE_CHECKING:
    from video_mapping.canvas import Canvas


class VideoWriter:
    """Context manager that pipes raw RGB frames to ffmpeg.

    Usage::

        with VideoWriter(Path("output/out.mp4"), width=4096, height=606, fps=30) as writer:
            for frame in ...:
                canvas = base.copy()
                # ... draw on canvas ...
                if not writer.write_canvas(canvas):
                    break  # ffmpeg finished early (e.g. audio ended)
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        audio_path: Path | None = None,
        vf_filter: str | None = "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
        preset: str = "medium",
    ) -> None:
        """Create a VideoWriter.

        Args:
            output_path: Destination file (will be overwritten if it exists).
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Output frame rate.
            audio_path: Optional audio track to mux into the output.
            vf_filter: ffmpeg ``-vf`` filter string. The default pads dimensions to even
                numbers (required by yuv420p). Pass ``None`` to disable, or supply a
                custom string such as ``"pad=width=4096:height=606:x=0:y=0"``.
            preset: libx264 encoding preset (speed/quality trade-off).
        """
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._audio_path = audio_path
        self._vf_filter = vf_filter
        self._preset = preset
        self._proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> Self:
        cmd: list[str] = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(self._fps),
            "-i",
            "-",
        ]

        if self._audio_path is not None:
            cmd += ["-i", str(self._audio_path)]

        if self._vf_filter is not None:
            cmd += ["-vf", self._vf_filter]

        cmd += ["-c:v", "libx264", "-preset", self._preset, "-pix_fmt", "yuv420p"]

        if self._audio_path is not None:
            cmd += ["-c:a", "aac", "-shortest"]

        cmd.append(str(self._output_path))

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # noqa: S603
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._proc is not None:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            _ = self._proc.wait()
            self._proc = None

    def write_array(self, frame: U8Array) -> bool:
        """Write a raw frame array (height x width x 3, uint8).

        Returns ``False`` if the pipe is broken (ffmpeg finished early), in which
        case the caller should stop the render loop.
        """
        if self._proc is None or self._proc.stdin is None:
            msg = "VideoWriter must be used as a context manager"
            raise RuntimeError(msg)
        try:
            _ = self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            return False
        else:
            return True

    def write_canvas(self, canvas: Canvas) -> bool:
        """Convenience wrapper around :meth:`write_array` for a :class:`~video_mapping.canvas.Canvas`."""
        return self.write_array(canvas.to_array())
