"""
Single place for every path/threshold/camera-profile default used by the
pipeline. `demo.py` CLI flags override these; nothing else in `src/` hardcodes
a path or magic number.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# External repos cloned by scripts/setup_env.sh — needed on sys.path for
# `from depth_anything_v2.dpt import DepthAnythingV2` to work.
DAV2_REPO_DIR = os.path.join(PROJECT_ROOT, "Depth-Anything-V2")
DAV2_METRIC_DEPTH_DIR = os.path.join(DAV2_REPO_DIR, "metric_depth")

# Default input image / checkpoint. Override with --image / --checkpoint on
# the demo.py CLI, or just change these two lines.
DEFAULT_IMAGE_PATH = os.path.join(DATA_DIR, "plate.jpeg")
DEFAULT_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "dav2_finetuned.pth")

# ---------------------------------------------------------------------------
# Depth Anything V2 model config
# ---------------------------------------------------------------------------
DAV2_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
DAV2_ENCODER = "vits"
DAV2_DATASET = "hypersim"
DAV2_CHECKPOINT_TRAINED_MAX_DEPTH = 0.31  # meters

# ---------------------------------------------------------------------------
# Segmentation thresholds
# ---------------------------------------------------------------------------
FOOD_CONFIDENCE_THRESHOLD = 0.25
FOOD_CUSTOM_PROMPT = None  # None -> use FoodPlateSegmentation's built-in prompt list
PLATE_MIN_CONFIDENCE = 0.15
MAX_DETECTIONS = 10

# ---------------------------------------------------------------------------
# Manual camera intrinsics only. No EXIF, no auto field-of-view fallback: if
# the camera used isn't a key below, get_camera_intrinsics() raises rather
# than guessing.
#
# Add one entry per camera/phone (from a one-time calibration, e.g. a
# checkerboard, or the phone's spec sheet), keyed by a name you choose.
# ---------------------------------------------------------------------------
CAMERA_INTRINSICS_OVERRIDES = {
    "iphone": {
        "fx_px": 930,
        "fy_px": 930,
        "reference_width_px": 960,
        "reference_height_px": 1280,
    },
}
DEFAULT_CAMERA_PROFILE_NAME = "iphone"

# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------
M3_TO_CC = 1_000_000.0
