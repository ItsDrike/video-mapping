"""Video rendering: stream raw RGB/RGBA frames to ffmpeg with tqdm progress."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import imageio_ffmpeg  # pyright: ignore[reportMissingTypeStubs]
from tqdm import tqdm

if TYPE_CHECKING:
    from video_mapping.canvas import Canvas
    from video_mapping.types import U8Array


class VideoWriter:
    """Context manager that pipes raw frames to ffmpeg and owns render progress."""

    def __init__(
        self,
        output_path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        total_frames: int | None = None,
        transparent: bool = True,
        audio_path: Path | None = None,
        audio_start_seconds: float = 0.0,
        audio_duration_seconds: float | None = None,
        progress_desc: str = "Render",
    ) -> None:
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._total_frames = total_frames
        self._transparent = transparent
        self._audio_path = audio_path
        self._audio_start_seconds = audio_start_seconds
        self._audio_duration_seconds = audio_duration_seconds
        self._progress_desc = progress_desc

        self._frames_written = 0
        self._temp_output_path: Path | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._progress: Any | None = None

    def __enter__(self) -> Self:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{self._output_path.stem}_",
            suffix=f".tmp{self._output_path.suffix}",
            dir=self._output_path.parent,
            delete=False,
        ) as tmp:
            self._temp_output_path = Path(tmp.name)

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        input_pix_fmt = "rgba" if self._transparent else "rgb24"
        output_pix_fmt = "yuva420p" if self._transparent else "yuv420p"

        cmd: list[str] = [
            ffmpeg_exe,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            input_pix_fmt,
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(self._fps),
            "-i",
            "-",
        ]

        if self._audio_path is not None:
            if self._audio_start_seconds > 0.0:
                cmd += ["-ss", f"{self._audio_start_seconds:.6f}"]
            if self._audio_duration_seconds is not None and self._audio_duration_seconds > 0.0:
                cmd += ["-t", f"{self._audio_duration_seconds:.6f}"]
            cmd += ["-i", str(self._audio_path)]

        cmd += ["-c:v", "libvpx-vp9", "-pix_fmt", output_pix_fmt]
        if self._audio_path is not None:
            cmd += ["-c:a", "libopus", "-shortest"]

        cmd.append(str(self._temp_output_path))

        self._progress = tqdm(total=self._total_frames, desc=self._progress_desc, unit="frame", dynamic_ncols=True)

        try:
            self._proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            self._cleanup_temp_file()
            self._close_progress()
            raise

        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._proc is not None:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            return_code = self._proc.wait()
            self._proc = None

            if self._temp_output_path is not None:
                if return_code == 0 and exc_info[0] is None:
                    _ = self._temp_output_path.replace(self._output_path)
                elif self._temp_output_path.exists():
                    self._temp_output_path.unlink()

        self._close_progress()
        self._temp_output_path = None

    def _cleanup_temp_file(self) -> None:
        if self._temp_output_path is not None and self._temp_output_path.exists():
            self._temp_output_path.unlink()
        self._temp_output_path = None

    def _close_progress(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None

    def write_array(self, frame: U8Array) -> bool:
        """Write a raw frame array (height x width x 3|4, uint8)."""
        if self._proc is None or self._proc.stdin is None:
            msg = "VideoWriter must be used as a context manager"
            raise RuntimeError(msg)
        try:
            _ = self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            return False
        else:
            self._frames_written += 1
            if self._progress is not None:
                _ = self._progress.update(1)
                _ = self._progress.set_postfix_str(
                    f"time={self._frames_written / self._fps:.1f}s",
                    refresh=False,
                )
            return True

    def write_canvas(self, canvas: Canvas) -> bool:
        """Write the current canvas contents as a frame."""
        return self.write_array(canvas.to_array())

    def log(self, message: str) -> None:
        """Write a message without breaking the progress bar."""
        if self._progress is not None:
            self._progress.write(message)
        else:
            print(message)
