"""Sony VX1000 + MK1 fisheye look: four filters as pure uint8 RGB → uint8 RGB."""

import cv2
import numpy as np


def fisheye_2d(image: np.ndarray, strength: float = 0.9, scale: float = 0.98) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    xv, yv = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
    x, y = xv - cx, yv - cy
    r = np.sqrt(x * x + y * y)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    factor = 1.0 + strength * ((r / max_r) ** 2)
    map_x = (cx + x * factor * scale).astype(np.float32)
    map_y = (cy + y * factor * scale).astype(np.float32)
    return cv2.remap(
        image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )


def vignette(image: np.ndarray, amount: float = 0.4) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.linspace(-1.0, 1.0, w)
    y = np.linspace(-1.0, 1.0, h)
    xv, yv = np.meshgrid(x, y)
    mask = np.clip(1.0 - amount * np.power(xv ** 2 + yv ** 2, 1.2), 0.0, 1.0)
    return (image.astype(np.float32) * mask[..., np.newaxis]).astype(np.uint8)


def color_grade(
    image: np.ndarray, contrast: float = 1.2, saturation: float = 1.1, exposure: float = 0.0
) -> np.ndarray:
    n = image.astype(np.float32) / 255.0
    n = np.clip(n + exposure, 0.0, 1.0)
    n = np.clip((n - 0.5) * contrast + 0.5, 0.0, 1.0)
    hsv = cv2.cvtColor((n * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def grain(image: np.ndarray, amount: float = 0.05, seed: int | None = None) -> np.ndarray:
    if amount <= 0:
        return image
    if seed is None:
        noise = np.random.normal(0.0, 255.0 * amount, size=image.shape)
    else:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 255.0 * amount, size=image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
