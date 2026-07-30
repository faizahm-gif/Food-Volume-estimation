
import sys

import cv2
import numpy as np
import torch

from config import config


def _ensure_dav2_on_path():
    if config.DAV2_METRIC_DEPTH_DIR not in sys.path:
        sys.path.insert(0, config.DAV2_METRIC_DEPTH_DIR)


def load_depth_model(
    checkpoint_path,
    encoder=config.DAV2_ENCODER,
    max_depth=config.DAV2_CHECKPOINT_TRAINED_MAX_DEPTH,
    device=None,
):
   
    _ensure_dav2_on_path()
    from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402 (path set above)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = config.DAV2_MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**{**model_config, "max_depth": max_depth})
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    return model, device


def infer_depth(model, image_path):
    """Run the depth model on an image path, returning an (H, W) float32
    metric depth map in meters."""
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Could not read image for depth inference: {image_path}")
    depth = model.infer_image(raw_img)
    return depth


def save_depth_outputs(depth, output_dir):
    """Save the depth map as both a colorized PNG and a raw .npy array."""
    import os

    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    png_path = os.path.join(output_dir, "depth_map.png")
    plt.figure(figsize=(8, 6))
    plt.imshow(depth, cmap="plasma")
    plt.colorbar(label="Depth (m)")
    plt.title("Depth Map")
    plt.axis("off")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    npy_path = os.path.join(output_dir, "depth_map.npy")
    np.save(npy_path, depth)

    return png_path, npy_path
