# image-effects

Unified image-filter module. Pure functions operating on `numpy uint8 (H, W, 3)` arrays.
Merges filters that were previously duplicated across `vx1000-mk1`, `average-abstraction`, and `watercolor`.

## Modules

- `image_effects.vx1000` — fisheye_2d, vignette, color_grade, grain
- `image_effects.tonal` — contrast_stretch, hist_equalize, clahe, oil_paint
- `image_effects.watercolor` — watercolor_fast (cv2.stylization), watercolor_physics (pigment separation)

## Install

```bash
pip install -e .
```
