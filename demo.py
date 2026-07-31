
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
        help="Key into CAMERA_INTRINSICS_OVERRIDES in config/config.py. "
        "Only used when --area-method=camera_intrinsics.",
    )
    parser.add_argument(
        "--area-method",
        choices=["plate_heuristic", "camera_intrinsics"],
        default=config.AREA_ESTIMATION_METHOD,
        help="How to convert pixels to real-world area: 'plate_heuristic' "
        "(default) uses the plate's known diameter; 'camera_intrinsics' "
        "uses fx/fy and per-pixel depth.",
    )
    parser.add_argument(
        "--plate-diameter-m",
        type=float,
        default=config.PLATE_DIAMETER_M,
        help="Known real-world plate diameter in meters, used when "
        "--area-method=plate_heuristic and --plate-shape=circular.",
    )
    parser.add_argument(
        "--plate-shape",
        choices=["circular", "rectangular"],
        default=config.PLATE_SHAPE,
        help="Shape of the plate/tray, used when --area-method=plate_heuristic. "
        "'circular' (default) uses --plate-diameter-m. 'rectangular' uses "
        "--plate-length-m and --plate-width-m instead.",
    )
    parser.add_argument(
        "--plate-length-m",
        type=float,
        default=config.PLATE_LENGTH_M,
        help="Known real-world plate/tray length in meters, used when "
        "--area-method=plate_heuristic and --plate-shape=rectangular.",
    )
    parser.add_argument(
        "--plate-width-m",
        type=float,
        default=config.PLATE_WIDTH_M,
        help="Known real-world plate/tray width in meters, used when "
        "--area-method=plate_heuristic and --plate-shape=rectangular.",
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
        area_estimation_method=args.area_method,
        plate_diameter_m=args.plate_diameter_m,
        plate_shape=args.plate_shape,
        plate_length_m=args.plate_length_m,
        plate_width_m=args.plate_width_m,
    )

    try:
        results = pipeline.run(args.image)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

    pipeline.save_results(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
