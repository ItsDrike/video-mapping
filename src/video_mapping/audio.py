"""Audio processing: FFT, frequency bands, beat detection, smoothing."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from scipy.io import wavfile

from video_mapping.types import F32Array, U16Array


def load_audio(path: Path) -> tuple[int, F32Array]:
    """Load a WAV file and return (sample_rate, mono_float32_samples)."""
    sample_rate, raw = wavfile.read(path)
    audio = np.asarray(raw, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return sample_rate, audio


def compute_fft_frames(
    audio: F32Array,
    frame_size: int = 2048,
    hop_size: int = 1024,
) -> F32Array:
    """Compute the magnitude spectrum for overlapping Hann-windowed frames."""
    n_frames = (len(audio) - frame_size) // hop_size
    shape = (n_frames, frame_size)
    strides = (audio.strides[0] * hop_size, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    window = np.hanning(frame_size)
    frames = frames * window
    fft = np.fft.rfft(frames, axis=1)
    return np.abs(fft).astype(np.float32)


def map_to_bands_log(
    fft_frames: F32Array,
    sample_rate: int,
    num_bands: int,
) -> F32Array:
    """Collapse FFT bins into logarithmically-spaced frequency bands."""
    n_bins = fft_frames.shape[1]
    freqs = np.fft.rfftfreq((n_bins - 1) * 2, d=1 / sample_rate)

    edges = np.logspace(np.log10(20), np.log10(sample_rate / 2), num_bands + 1)
    bands = np.zeros((fft_frames.shape[0], num_bands), dtype=np.float32)

    for i in range(num_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if np.any(mask):
            bands[:, i] = fft_frames[:, mask].mean(axis=1)

    return bands


def apply_weighting(bands: F32Array) -> F32Array:
    """Boost high-frequency bands with a gentle geometric ramp."""
    weights = np.geomspace(0.7, 2.5, bands.shape[1]).astype(np.float32)
    return bands * weights


def enhance_transients(bands: F32Array) -> F32Array:
    """Sharpen attack transients by adding a fraction of the positive differential."""
    diff = np.diff(bands, axis=0, prepend=bands[:1])
    return (bands + 0.5 * np.maximum(diff, 0)).astype(np.float32)


def smooth_bands(
    bands: F32Array,
    attack: float = 0.5,
    decay: float = 0.85,
) -> F32Array:
    """Apply asymmetric smoothing: fast attack, slow decay."""
    smoothed = np.zeros_like(bands)
    smoothed[0] = bands[0]
    for i in range(1, len(bands)):
        rising = bands[i] > smoothed[i - 1]
        smoothed[i] = np.where(
            rising,
            attack * smoothed[i - 1] + (1.0 - attack) * bands[i],
            decay * smoothed[i - 1] + (1.0 - decay) * bands[i],
        )
    return smoothed.astype(np.float32)


def normalize_bands(bands: F32Array, height: int) -> U16Array:
    """Log-scale and normalize band energies into pixel heights."""
    bands = np.log10(bands + 1e-6)
    bands -= bands.min()
    bands /= bands.max() + 1e-6
    bands = np.power(bands, 1.8)
    return (bands * height).astype(np.uint16)


def detect_beats(bands: F32Array) -> F32Array:
    """Derive a per-frame beat strength from low-frequency transients."""
    low = bands[:, : bands.shape[1] // 3].mean(axis=1)
    low = (low - low.min()) / (low.max() + 1e-6)
    diff = np.diff(low, prepend=low[:1])
    beats = np.maximum(diff, 0)
    beats /= beats.max() + 1e-6
    return cast("F32Array", beats.astype(np.float32))


def smooth_beats(beats: F32Array, decay: float = 0.96) -> F32Array:
    """Apply one-pole envelope follower to beat signal (hold + decay)."""
    out = np.zeros_like(beats)
    for i in range(1, len(beats)):
        out[i] = max(beats[i], out[i - 1] * decay)
    return out


def process_audio(
    path: Path,
    num_bands: int,
    bar_height: int,
    frame_size: int = 2048,
    hop_size: int = 1024,
) -> tuple[int, U16Array, F32Array]:
    """Full pipeline: load → FFT → bands → enhance → normalize + beat detection.

    Returns (sample_rate, band_heights, beats) ready for use in a render loop.
    """
    sample_rate, audio = load_audio(path)

    fft_frames = compute_fft_frames(audio, frame_size=frame_size, hop_size=hop_size)
    bands = map_to_bands_log(fft_frames, sample_rate, num_bands)
    bands = apply_weighting(bands)
    bands = enhance_transients(bands)
    bands = smooth_bands(bands)

    beats = detect_beats(bands)
    beats = smooth_beats(beats)

    band_heights = normalize_bands(bands, height=bar_height)

    return sample_rate, band_heights, beats
