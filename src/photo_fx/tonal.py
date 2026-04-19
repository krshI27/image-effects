"""Tonal / contrast filters. Pure uint8 RGB → uint8 RGB.

Sources: average-abstraction/pages/4_manipulating.py.
"""

import cv2
import numpy as np
from skimage import exposure
from sklearn.cluster import KMeans


def _hsv_split(rgb: np.ndarray):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    return hsv, h, s, v


def _hsv_merge(h, s, v) -> np.ndarray:
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)


def contrast_stretch(image: np.ndarray, p_min: float = 2.0, p_max: float = 98.0) -> np.ndarray:
    _, h, s, v = _hsv_split(image)
    lo, hi = np.percentile(v, (p_min, p_max))
    scaled = exposure.rescale_intensity(v, in_range=(lo, hi)).astype(np.uint8)
    return _hsv_merge(h, s, scaled)


def hist_equalize(image: np.ndarray) -> np.ndarray:
    _, h, s, v = _hsv_split(image)
    return _hsv_merge(h, s, cv2.equalizeHist(v))


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    _, h, s, v = _hsv_split(image)
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return _hsv_merge(h, s, c.apply(v))


def oil_paint(
    image: np.ndarray, size: int = 7, dyn_ratio: float = 5.0, n_colors: int = 16
) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bilateral = cv2.bilateralFilter(bgr, size, dyn_ratio * 10, dyn_ratio * 10)
    smooth = cv2.medianBlur(bilateral, size if size % 2 == 1 else size + 1)
    h, w = smooth.shape[:2]
    flat = smooth.reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=n_colors, n_init=3, random_state=0).fit(flat)
    quantized = km.cluster_centers_[km.labels_].reshape(h, w, 3).astype(np.uint8)
    return cv2.cvtColor(quantized, cv2.COLOR_BGR2RGB)
