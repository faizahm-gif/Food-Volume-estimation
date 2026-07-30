# ThaliVolumePipeline

Estimates the real-world volume (in cc) of each food item on a plate/thali
from a single RGB photo, using:

- **Grounding DINO** + **SAM2** for food-item and plate/tray detection & segmentation
- **Depth Anything V2** (fine-tuned, metric depth) for a per-pixel depth map
- A **pinhole camera model** to convert pixel area -> real-world area, and
  depth difference -> food height, per item

This is a from-scratch restructure of the original Colab notebook into an
importable, script-runnable codebase. Running `demo.py` reproduces the
notebook's end-to-end result on one image, with no notebook required.

## Layout

```
thali_volume_pipeline/
├── config/
│   └── config.py          # all paths, thresholds, camera profiles in one place
├── src/
│   ├── segmentation.py     # DetectionDeduplicator, FoodPlateSegmentation, plate detection
│   ├── depth.py            # Depth Anything V2 model loading + inference
│   ├── camera.py           # manual camera intrinsics lookup (no EXIF/auto-detect)
│   ├── volume.py           # reference depth, per-item height/area/volume math
│   └── pipeline.py         # ThaliVolumePipeline: wires the above into one call
├── scripts/
│   └── setup_env.sh        # installs deps, clones sam2 + Depth-Anything-V2
├── models/checkpoints/      # put your dav2_nutrition5k_best.pth here
├── data/                    # put your input food image(s) here
├── outputs/                 # results are written here (depth map, masks, volumes.json)
├── demo.py                  # <-- run this
└── requirements.txt
```

## Setup

```bash
cd thali_volume_pipeline
bash scripts/setup_env.sh
```

This installs Python dependencies and clones `facebookresearch/sam2` and
`DepthAnything/Depth-Anything-V2` into the project root (needed because
`Depth-Anything-V2`'s `depth_anything_v2` package is imported directly, and
`sam2` is installed as an editable package).

## Required inputs (not included in this codebase)

1. **A food image** — place it under `data/`, e.g. `data/plate.jpeg`.
2. **A fine-tuned Depth Anything V2 checkpoint** (`dav2_nutrition5k_best.pth`)
   — place it under `models/checkpoints/`.
3. **A camera profile** for whatever device took the photo — add its
   `fx_px`, `fy_px`, and the reference resolution it was calibrated at to
   `CAMERA_INTRINSICS_OVERRIDES` in `config/config.py`. There is no EXIF or
   field-of-view fallback: an unregistered camera profile raises rather than
   guessing intrinsics.

Edit the paths/constants at the top of `config/config.py` (or pass CLI flags
to `demo.py`, see below) to point at your image, checkpoint, and camera
profile.

## Run

```bash
python demo.py \
  --image data/plate.jpeg \
  --checkpoint models/checkpoints/dav2_nutrition5k_best.pth \
  --camera-profile "google pixel 8" \
  --output-dir outputs
```

All flags are optional and fall back to the defaults in `config/config.py`.

### What it prints

- Detected food items and confidences
- Auto-detected plate/tray mask confidence
- Thali reference depth
- Per-item height / area / volume, and negative-pixel diagnostics
- Per-item and total volume in cc

### What it saves (to `--output-dir`, default `outputs/`)

- `depth_map.png`, `depth_map.npy`
- `segmentation_overlay.png` (detections + masks drawn on the original image)
- `masks/plate_mask.png`, `masks/<item_key>.png` — binary PNG masks
- `plate_mask.npy` — raw plate mask array
- `food_item_volumes_cc.json` — per-item + total volume in cc
  
