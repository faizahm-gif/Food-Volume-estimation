# ThaliVolumePipeline

Estimates the real-world volume (in cc) of each food item on a plate/thali
from a single RGB photo, using:

- **Grounding DINO** + **SAM2** for food-item and plate/tray detection & segmentation
- **Depth Anything V2** (fine-tuned, metric depth) for a per-pixel depth map
- A **pinhole camera model** to convert pixel area -> real-world area, and
  depth difference -> food height, per item

## Layout

```
thali_volume_pipeline/
├── config/
│   └── config.py         
├── src/
│   ├── segmentation.py     
│   ├── depth.py           
│   ├── camera.py          
│   ├── volume.py         
│   └── pipeline.py        
├── scripts/
│   └── setup_env.sh   
├── models/checkpoints/    
├── data/                  
├── outputs/                 
├── demo.py
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

## Run

If you don't want to run the code on your images, then run the following script 
```bash
python demo.py
```
If you want to run it on custom images, checkpoints or camera intrinstic values
```bash
python demo.py \
  --image data/plate.jpeg \
  --checkpoint models/checkpoints/dav2_nutrition5k_best.pth \
  --camera-profile "iphone" \
  --output-dir outputs
```

## Required inputs

1. **A food image** — place it under `data/`, e.g. `data/plate.jpeg`.
2. **Add model weights (you can download it from here) [Model weights](https://drive.google.com/file/d/1GBDcIkjbvt089516xQsQK_SYa0i4PMDL/view?usp=sharing)** 
   — place it under `models/checkpoints/`.
3. **A camera profile** for whatever device took the photo — add its
   `fx_px`, `fy_px`, and the reference resolution it was calibrated at to
   `CAMERA_INTRINSICS_OVERRIDES` in `config/config.py`. There is no EXIF or
   field-of-view fallback: an unregistered camera profile raises rather than
   guessing intrinsics.

Edit the paths/constants at the top of `config/config.py` (or pass CLI flags
to `demo.py`, see below) to point at your image, checkpoint, and camera
profile.





All flags are optional and fall back to the defaults in `config/config.py`.

### Output

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
  
