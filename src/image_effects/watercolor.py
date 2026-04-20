"""Two watercolor filters.

- `watercolor_fast`: cv2.stylization (bilateral-based, < 1s).
- `watercolor_physics`: re-export of `watercolor.watercolorize_array`.

Inputs and outputs: uint8 RGB (H, W, 3).
"""

import cv2
import numpy as np
from watercolor import watercolorize_array as watercolor_physics  # noqa: F401


def watercolor_fast(image: np.ndarray, sigma_s: float = 60.0, sigma_r: float = 0.45) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out = cv2.stylization(bgr, sigma_s=sigma_s, sigma_r=sigma_r)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
