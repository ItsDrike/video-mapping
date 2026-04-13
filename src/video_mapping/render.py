"""Video rendering: stream raw RGB/RGBA frames to ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Self

from video_mapping.types import U8Array

if TYPE_CHECKING:
    from video_mapping.canvas import Canvas


class VideoWriter:
    """Context manager that pipes raw RGB/RGBA frames to ffmpeg.

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
        preset: str | None = "medium",
        input_pix_fmt: str = "rgb24",
        output_codec: str = "libx264",
        output_pix_fmt: str = "yuv420p",
        audio_codec: str = "aac",
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
            preset: Optional encoder preset (commonly used by libx264).
            input_pix_fmt: ffmpeg pixel format of incoming raw frames.
            output_codec: ffmpeg video codec.
            output_pix_fmt: ffmpeg output pixel format.
            audio_codec: ffmpeg audio codec (when ``audio_path`` is provided).
        """
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._audio_path = audio_path
        self._vf_filter = vf_filter
        self._preset = preset
        self._input_pix_fmt = input_pix_fmt
        self._output_codec = output_codec
        self._output_pix_fmt = output_pix_fmt
        self._audio_codec = audio_codec
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
            self._input_pix_fmt,
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

        cmd += ["-c:v", self._output_codec]
        if self._preset is not None:
            cmd += ["-preset", self._preset]
        cmd += ["-pix_fmt", self._output_pix_fmt]

        if self._audio_path is not None:
            cmd += ["-c:a", self._audio_codec, "-shortest"]

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
        """Write a raw frame array (height x width x 3|4, uint8).

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
