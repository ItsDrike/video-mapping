from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type RGBColor = tuple[int, int, int]
"""An RGB color as a (red, green, blue) tuple, each component in 0-255."""

type F32Array = NDArray[np.float32]
type U8Array = NDArray[np.uint8]
type U16Array = NDArray[np.uint16]
