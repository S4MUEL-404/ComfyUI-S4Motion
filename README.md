## ComfyUI-S4Motion Nodes

A set of ComfyUI nodes for animating a foreground layer over a background, with unified timing and high-quality composition. It includes effectors for rotation, position, path-following position, scale, opacity, distortion, and masking, plus a core node that composes frames and exports APNG/WEBP and a frame sequence.

### Installation & Dependencies

- Environment: Python 3.10+ and ComfyUI
- Dependencies (core/optional):
  - Required: `numpy`, `Pillow`
  - Recommended: `opencv-python`, `scikit-image`, `scipy`
  - Optional: `bezier` (smoother easing curves)

Install example:

```bash
pip install numpy Pillow opencv-python scikit-image scipy bezier
```

On first run, the core node prints a dependency status summary. Missing packages will be suggested with an install command.

### Node Category

All nodes appear under category: `💀S4Motion`

---

## Core Node

### Motion Config (`py/motionConfig.py`)

- Purpose: Controls global timeline (duration/fps/loop/format), reads per-frame data from effectors, applies distortion/mask/rotation/scale/position/opacity to the foreground layer, and outputs an animation file and a frame sequence tensor.
- Inputs (required):
  - `layer_image: IMAGE` foreground image (single frame or sequence)
  - `background_image: IMAGE` background image (single frame or sequence)
  - `time: FLOAT` duration in seconds
  - `fps: INT` 1–60
  - `loop: BOOLEAN` loop playback
  - `format: webp | apng`
  - `rotation: FLOAT` base rotation in degrees
  - `position_x, position_y: FLOAT` base position (pixels)
  - `scale: FLOAT` base scale
  - `opacity: FLOAT` base opacity in [0, 1]
- Inputs (optional):
  - `rotation_effector: S4_ROTATION_EFFECTOR`
  - `position_effector: S4_POSITION_EFFECTOR` (supports parameter or path mode)
  - `scale_effector: S4_SCALE_EFFECTOR`
  - `opacity_effector: S4_OPACITY_EFFECTOR`
  - `distortion_effector: S4_DISTORTION_EFFECTOR`
  - `mask_effector: S4_MASK_EFFECTOR`
- Outputs:
  - `STRING` output file path (saved under repository `output/`)
  - `IMAGE` frame sequence (torch tensor; frames stacked on batch dimension, shape like `(F, H, W, C)` in [0, 1])
- Composition highlights:
  - Distortion is applied before rotation/scale; mask is applied to the foreground alpha before compositing.
  - Alignment: path mode uses center alignment; parameter mode uses top-left alignment.
  - Rotation/scale/opacity combine base value with effector values; sequences are padded to total frames as needed.
  - Coordinates can be negative; the layer can be partially out-of-bounds. High-quality resampling is used (with supersampling for small rotations on small images).
  - Saving: WEBP uses lossless. `loop=True` maps to loop=0 (infinite), otherwise loop=1.

---

## Effector Nodes

### Motion Rotation (`py/motionRotation.py`)

- Purpose: Provides per-frame rotation in degrees.
- Main inputs: `default_rotation`, `target_rotation`, `duration`, `delay`, `motion_curve`, `inverse`, `loop`
- Output: `S4_ROTATION_EFFECTOR` (callable `effector(total_frames, total_time) -> List[float]`)
- Notes: Supports `linear / ease_in / ease_out / ease_in_out`; when `bezier` is installed, cubic bezier easing is used for smoother motion.

### Motion Position (`py/motionPosition.py`)

- Purpose: Provides per-frame `(x, y)` positions.
- Main inputs: `default_x`, `default_y`, `target_x`, `target_y`, `duration`, `delay`, `motion_curve`, `inverse`, `loop`
- Output: `S4_POSITION_EFFECTOR` with `is_path_mode=False` (callable returns `List[(x, y)]`)

### Motion Position on Path (`py/motionPositionOnPath.py`)

- Purpose: Extracts a path from an input image and provides per-frame positions along that path.
- Main inputs: `path_image`, `duration`, `delay`, `motion_curve`, `inverse`, `loop`
- Output: `S4_POSITION_EFFECTOR` with `is_path_mode=True`
- Path extraction pipeline:
  - Convert to grayscale and normalize; threshold at 0.5 to select a darker path region.
  - Remove small objects and small holes; skeletonize; keep the largest component; order pixels into a continuous path via 8-connectivity.
  - Optionally smooth via splines and resample uniformly to motion frames; remap timing with easing.

### Motion Scale (`py/motionScale.py`)

- Purpose: Provides per-frame scale factors.
- Main inputs: `default_scale`, `target_scale`, `duration`, `delay`, `motion_curve`, `inverse`, `loop`
- Output: `S4_SCALE_EFFECTOR`
- Notes: Scale is clamped to a minimum of 0.01 to avoid invalid values.

### Motion Opacity (`py/motionOpacity.py`)

- Purpose: Provides per-frame opacity in [0, 1].
- Main inputs: `default_opacity`, `target_opacity`, `duration`, `delay`, `motion_curve`, `inverse`, `loop`
- Output: `S4_OPACITY_EFFECTOR`

### Motion Distortion (`py/motionDistortion.py`)

- Purpose: Provides per-frame distortion specs consumed by the core node.
- Main inputs:
  - Timeline: `mode` (`wave | swirl | radial`), `duration`, `delay`, `motion_curve`, `inverse`, `loop`
  - Wave: `wave_amplitude_*`, `wave_wavelength_*`, `wave_phase_*`, `wave_direction (x|y|xy)`
  - Swirl: `swirl_strength_*`, `swirl_radius_*`, `swirl_center_x/y_*` (centers are relative in [0, 1])
  - Radial: `radial_k_*`, `radial_center_x/y_*` (barrel/pincushion)
- Output: `S4_DISTORTION_EFFECTOR` (each frame emits `{mode, params}`)
- In core composition:
  - `wave` and `radial` use OpenCV `remap`;
  - `swirl` uses scikit-image `swirl`.
  - Distortion happens before rotation/scale.

### Motion Mask (`py/motionMask.py`)

- Purpose: Provides a base mask image plus per-frame transform specs for clipping/fade effects.
- Main inputs: `mask (MASK)`, `duration`, `delay`, `motion_curve`, `inverse`, `loop` and default/target for `x/y/rotation/scale/opacity`
- Output: `S4_MASK_EFFECTOR` (carries `mask_image` and per-frame `{x, y, rotation, scale, opacity}`)
- In core composition:
  - Convert base mask to L mode; per frame apply rotation/scale/translate and opacity to build a mask canvas.
  - Multiply with the layer alpha channel before alpha compositing.

---

## Basic Usage (ComfyUI workflow)

1. Place `Motion Config` and set `time / fps / loop / format` plus base `rotation / position / scale / opacity`.
2. Optionally add effectors: `Motion Rotation / Position / Position on Path / Scale / Opacity / Distortion / Mask`, and connect to corresponding inputs.
3. Connect `layer_image` and `background_image` (single image or sequence supported).
4. Run:
   - An animation file (WEBP/APNG) will be saved to `output/`.
   - A frame sequence tensor will be produced for downstream nodes.

### Tips & Notes

- Inputs support `PIL / numpy / torch` and are auto-converted; leading batch dims are removed when necessary.
- Effector outputs are padded to `total_frames`; `delay` is filled with the first value/spec.
- Path mode uses center alignment; parameter mode uses top-left alignment.
- Without `bezier`, mathematical easing functions are used as a fallback.

---

## Project Structure

```
py/
  motionConfig.py              # Core composition/export
  motionRotation.py            # Rotation effector
  motionPosition.py            # Position effector (parameter-based)
  motionPositionOnPath.py      # Position effector (path-based)
  motionScale.py               # Scale effector
  motionOpacity.py             # Opacity effector
  motionDistortion.py          # Distortion effector (wave/swirl/radial)
  motionMask.py                # Mask effector
```

If you need example workflows or demo assets, please describe your requirements and preferences.

