#!/usr/bin/env python3
"""
Run the end-to-end ThaliVolumePipeline on a single food image:
depth estimation -> food/plate segmentation -> per-item volume in cc.

Usage:
    python demo.py
    python demo.py --image data/plate.jpeg --checkpoint models/checkpoints/dav2_nutrition5k_best.pth \
        --camera-profile "google pixel 8" --output-dir outputs

All flags are optional; defaults come from config/config.py.
"""
import argparse
import sys

from config import config
from src.pipeline import ThaliVolumePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the thali food-volume pipeline on one image.")
    parser.add_argument("--image", default=config.DEFAULT_IMAGE_PATH, help="Path to the food photo.")
    parser.add_argument(
        "--checkpoint",
        default=config.DEFAULT_CHECKPOINT_PATH,
        help="Path to the fine-tuned Depth Anything V2 checkpoint (dav2_nutrition5k_best.pth).",
    )
    parser.add_argument(
        "--camera-profile",
        default=config.DEFAULT_CAMERA_PROFILE_NAME,
        help="Key into CAMERA_INTRINSICS_OVERRIDES in config/config.py.",
    )
    parser.add_argument(
        "--food-confidence",
        type=float,
        default=config.FOOD_CONFIDENCE_THRESHOLD,
        help="Grounding DINO confidence threshold for food-item detection.",
    )
    parser.add_argument(
        "--plate-confidence",
        type=float,
        default=config.PLATE_MIN_CONFIDENCE,
        help="Grounding DINO confidence threshold for plate/tray detection.",
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help="Where to save the depth map, masks, and food_item_volumes_cc.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = ThaliVolumePipeline(
        checkpoint_path=args.checkpoint,
        camera_profile_name=args.camera_profile,
        food_confidence_threshold=args.food_confidence,
        plate_min_confidence=args.plate_confidence,
    )

    try:
        results = pipeline.run(args.image)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    pipeline.save_results(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
