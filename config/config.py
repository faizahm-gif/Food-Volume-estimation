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
DEFAULT_IMAGE_PATH = os.path.join(DATA_DIR, "sample.jpeg")
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
DAV2_CHECKPOINT_TRAINED_MAX_DEPTH = 0.4  # meters


FOOD_CONFIDENCE_THRESHOLD = 0.25
FOOD_CUSTOM_PROMPT = None  
PLATE_MIN_CONFIDENCE = 0.15
MAX_DETECTIONS = 10
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
# Pixel -> real-world area estimation method
# ---------------------------------------------------------------------------
# "plate_heuristic": scale is derived from a known real-world plate diameter
#     vs. the plate's measured pixel diameter (from its SAM2 mask). This is
#     now the default.
# "camera_intrinsics": scale is derived from fx/fy and per-pixel depth via
#     the pinhole camera model (the previous default). Kept available for
#     comparison/fallback.
AREA_ESTIMATION_METHOD = "plate_heuristic"

# Real-world diameter of the plate/thali used in your photos, in meters.
# Standard steel thali plates are typically ~0.26-0.30 m; measure your own
# plate and set this accordingly.
PLATE_DIAMETER_M = 0.26

# ---------------------------------------------------------------------------
# Plate shape (used only when AREA_ESTIMATION_METHOD == "plate_heuristic")
# ---------------------------------------------------------------------------
# "circular": scale derived from PLATE_DIAMETER_M, averaged isotropically
#     over the mask bbox's width and height.
# "rectangular": scale derived from PLATE_LENGTH_M and PLATE_WIDTH_M,
#     applied anisotropically to the mask bbox's width and height
#     (longer physical side paired with the longer bbox side).
PLATE_SHAPE = "circular"

# Real-world length/width of a rectangular plate/tray, in meters. Only used
# when PLATE_SHAPE == "rectangular". Order doesn't matter -- the longer of
# the two is automatically paired with the longer bbox pixel dimension.
PLATE_LENGTH_M = 0.355
PLATE_WIDTH_M = 0.255

M3_TO_CC = 1_000_000.0
