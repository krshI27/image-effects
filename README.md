# photo-fx

Unified image-filter module. Pure functions operating on `numpy uint8 (H, W, 3)` arrays.
Merges filters that were previously duplicated across `vx1000-mk1`, `average-abstraction`, and `watercolor`.

## Modules

- `photo_fx.vx1000` — fisheye_2d, vignette, color_grade, grain
- `photo_fx.tonal` — contrast_stretch, hist_equalize, clahe, oil_paint
- `photo_fx.watercolor` — watercolor_fast (cv2.stylization), watercolor_physics (pigment separation)

## Install

```bash
pip install -e .
```
