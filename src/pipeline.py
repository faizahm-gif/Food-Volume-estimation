
import json
import os
import zipfile

import cv2
import numpy as np
from PIL import Image

from config import config
from src import camera, depth, plate_heuristics, volume
from src.segmentation import (
    FoodPlateSegmentation,
    detect_plate_mask,
    resize_mask_to,
    run_thali_segmentation,
)


class ThaliVolumePipeline:
    def __init__(
        self,
        checkpoint_path=config.DEFAULT_CHECKPOINT_PATH,
        camera_profile_name=config.DEFAULT_CAMERA_PROFILE_NAME,
        camera_intrinsics_overrides=None,
        food_confidence_threshold=config.FOOD_CONFIDENCE_THRESHOLD,
        food_custom_prompt=config.FOOD_CUSTOM_PROMPT,
        plate_min_confidence=config.PLATE_MIN_CONFIDENCE,
        max_detections=config.MAX_DETECTIONS,
        area_estimation_method=config.AREA_ESTIMATION_METHOD,
        plate_diameter_m=config.PLATE_DIAMETER_M,
        plate_shape=config.PLATE_SHAPE,
        plate_length_m=config.PLATE_LENGTH_M,
        plate_width_m=config.PLATE_WIDTH_M,
    ):
        self.checkpoint_path = checkpoint_path
        self.camera_profile_name = camera_profile_name
        self.camera_intrinsics_overrides = (
            camera_intrinsics_overrides or config.CAMERA_INTRINSICS_OVERRIDES
        )
        self.food_confidence_threshold = food_confidence_threshold
        self.food_custom_prompt = food_custom_prompt
        self.plate_min_confidence = plate_min_confidence
        self.max_detections = max_detections
        self.area_estimation_method = area_estimation_method
        self.plate_diameter_m = plate_diameter_m
        self.plate_shape = plate_shape
        self.plate_length_m = plate_length_m
        self.plate_width_m = plate_width_m

        if self.area_estimation_method not in ("plate_heuristic", "camera_intrinsics"):
            raise ValueError(
                f"Unknown area_estimation_method: {self.area_estimation_method!r}. "
                "Must be 'plate_heuristic' or 'camera_intrinsics'."
            )

        if self.plate_shape not in ("circular", "rectangular"):
            raise ValueError(
                f"Unknown plate_shape: {self.plate_shape!r}. "
                "Must be 'circular' or 'rectangular'."
            )

        self._depth_model = None
        self._device = None
        self._segmenter = None

    # -- lazy model loading -------------------------------------------------
    def _get_depth_model(self):
        if self._depth_model is None:
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found at: {self.checkpoint_path}\n"
                    "Point --checkpoint (or config.DEFAULT_CHECKPOINT_PATH) at your "
                    "fine-tuned dav2_nutrition5k_best.pth file."
                )
            self._depth_model, self._device = depth.load_depth_model(self.checkpoint_path)
        return self._depth_model

    def _get_segmenter(self):
        if self._segmenter is None:
            self._segmenter = FoodPlateSegmentation()
        return self._segmenter

    # -- main entry point ----------------------------------------------------
    def run(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found at: {image_path}\n"
                "Point --image (or config.DEFAULT_IMAGE_PATH) at your food photo."
            )

        # 1. Depth
        depth_model = self._get_depth_model()
        depth_map = depth.infer_depth(depth_model, image_path)
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f} m")

        # 2. Food-item detection + segmentation
        segmenter = self._get_segmenter()
        seg_result = run_thali_segmentation(
            segmenter,
            image_path,
            custom_prompt=self.food_custom_prompt,
            confidence_threshold=self.food_confidence_threshold,
        )
        detections = seg_result["detections"]

        print(f"Detected {len(detections)} food item(s):")
        for idx, label in seg_result["index_to_label"].items():
            print(f"  {idx}: {label}")

        if len(detections) == 0:
            raise ValueError(
                "No food items were detected/segmented. Try lowering "
                "food_confidence_threshold."
            )

        print("Composition:")
        for food_item, stats in seg_result["composition"].items():
            print(
                f"  {food_item}: {stats['percentage']:.1f}% of plate, "
                f"confidence {stats['confidence']:.3f}, area {stats['area_pixels']} px"
            )

        # 3. Plate/tray detection + segmentation (no convex-hull fallback)
        plate_mask, plate_confidence = detect_plate_mask(
            segmenter, seg_result["original_image"], min_confidence=self.plate_min_confidence
        )
        if plate_mask is None:
            raise ValueError(
                "No plate/tray detected via Grounding DINO + SAM2. Try lowering "
                "plate_min_confidence, adjust the plate prompt in "
                "src/segmentation.py, or use a photo where the plate/tray rim "
                "is clearly visible. (There is no convex-hull fallback -- the "
                "SAM2 plate mask is required.)"
            )
        plate_mask = resize_mask_to(plate_mask, depth_map.shape)
        print(f"Plate/thali mask (from SAM2) foreground pixels: {int(plate_mask.sum())}")

        # 4. Reference depth for the whole thali
        thali_reference_depth = volume.compute_thali_reference_depth(depth_map, plate_mask)
        print(
            f"Thali reference depth (median over {int(plate_mask.sum())} "
            f"plate-mask pixels): {thali_reference_depth:.4f} m"
        )

        # 5. Pixel -> real-world area map
        if self.area_estimation_method == "plate_heuristic":
            if self.plate_shape == "circular":
                meters_per_pixel, plate_diameter_px = plate_heuristics.compute_scale_from_plate(
                    plate_mask, self.plate_diameter_m
                )
                print(
                    f"Using circular plate-diameter heuristic: known plate diameter="
                    f"{self.plate_diameter_m * 100:.1f} cm, measured plate diameter="
                    f"{plate_diameter_px:.1f} px, scale={meters_per_pixel * 100:.5f} cm/px"
                )
                pixel_area_map = volume.compute_pixel_area_map_from_plate_scale(
                    depth_map, meters_per_pixel
                )
            else:  # "rectangular"
                (
                    meters_per_pixel_x,
                    meters_per_pixel_y,
                    bbox_width_px,
                    bbox_height_px,
                ) = plate_heuristics.compute_scale_from_plate_rectangular(
                    plate_mask, self.plate_length_m, self.plate_width_m
                )
                print(
                    f"Using rectangular plate heuristic: known plate length="
                    f"{self.plate_length_m * 100:.1f} cm, known plate width="
                    f"{self.plate_width_m * 100:.1f} cm, measured bbox="
                    f"{bbox_width_px:.1f}x{bbox_height_px:.1f} px, "
                    f"scale_x={meters_per_pixel_x * 100:.5f} cm/px, "
                    f"scale_y={meters_per_pixel_y * 100:.5f} cm/px"
                )
                pixel_area_map = volume.compute_pixel_area_map_from_plate_scale_rectangular(
                    depth_map, meters_per_pixel_x, meters_per_pixel_y
                )
        else:  # "camera_intrinsics"
            fx, fy = camera.get_camera_intrinsics(
                image_path, self.camera_profile_name, self.camera_intrinsics_overrides
            )
            print(f"Using manual camera profile: {self.camera_profile_name}")
            print(f"Focal length: Fx={fx:.2f} px, Fy={fy:.2f} px")
            pixel_area_map = volume.compute_pixel_area_map(depth_map, fx, fy)

        print(
            "Pixel area map range: "
            f"{pixel_area_map.min():.3e} - {pixel_area_map.max():.3e} m^2/px"
        )

        # 6. Per-item masks (resized to depth resolution), keyed "<i>_<label>"
        labels_list = (
            detections.data["labels"]
            if hasattr(detections, "data") and "labels" in detections.data
            else [f"item_{i}" for i in range(len(detections))]
        )
        item_masks = {}
        for i, mask in enumerate(detections.mask):
            m = resize_mask_to(mask.astype(bool), depth_map.shape)
            key = f"{i}_{labels_list[i]}"
            item_masks[key] = m

        # 7. Height, area, volume per item
        height_arrays, diagnostics = volume.compute_food_heights(
            depth_map, thali_reference_depth, item_masks
        )
        for key, diag in diagnostics.items():
            print(f"{key}:")
            if diag["n_negative"] > 0:
                print(
                    f"  {diag['n_negative']} pixels had a negative raw difference "
                    f"before abs() ({diag['pct_negative']:.1f}% of this item)"
                )
            print(f"  food height array sum: {float(height_arrays[key].sum()):.6f}")

        area_arrays = volume.compute_food_areas(pixel_area_map, item_masks)
        for key, area_arr in area_arrays.items():
            print(f"{key}: area mask sum = {float(area_arr.sum())} m^2")

        volumes_m3, volume_arrays = volume.compute_food_volumes(height_arrays, area_arrays)
        for key, v in volumes_m3.items():
            print(f"{key}: volume = {v:.8f} m^3")

        total_volume_m3 = float(sum(volumes_m3.values()))
        volumes_cc = {key: v * config.M3_TO_CC for key, v in volumes_m3.items()}
        total_volume_cc = total_volume_m3 * config.M3_TO_CC

        print("\nEstimated volume of each segmented food item:")
        for key, v in volumes_m3.items():
            print(f"  {key}: {v:.8f} m^3")
        print(f"\nTotal estimated food volume: {total_volume_m3:.8f} m^3")

        print("\nEstimated volume of each segmented food item (cc):")
        for key, v_cc in volumes_cc.items():
            print(f"  {key}: {v_cc:.2f} cc")
        print(f"\nTotal estimated food volume: {total_volume_cc:.2f} cc")

        return {
            "image_path": image_path,
            "original_image": seg_result["original_image"],
            "visualization": seg_result["visualization"],
            "depth_map": depth_map,
            "plate_mask": plate_mask,
            "item_masks": item_masks,
            "thali_reference_depth": thali_reference_depth,
            "composition": seg_result["composition"],
            "volumes_m3": volumes_m3,
            "volumes_cc": volumes_cc,
            "total_volume_m3": total_volume_m3,
            "total_volume_cc": total_volume_cc,
        }

    # -- persistence -----------------------------------------------------
    def save_results(self, results, output_dir=config.OUTPUT_DIR):
        """Save depth map, segmentation masks, and volumes.json.

        No histograms or side-by-side comparison figures are produced --
        only the depth map and segmentation masks, plus the numeric results.
        """
        os.makedirs(output_dir, exist_ok=True)

        depth.save_depth_outputs(results["depth_map"], output_dir)

        Image.fromarray(results["visualization"]).save(
            os.path.join(output_dir, "segmentation_overlay.png")
        )

        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        plate_mask = results["plate_mask"]
        np.save(os.path.join(output_dir, "plate_mask.npy"), plate_mask)
        Image.fromarray((plate_mask.astype("uint8") * 255), mode="L").save(
            os.path.join(masks_dir, "plate_mask.png")
        )

        for key, mask_arr in results["item_masks"].items():
            Image.fromarray((mask_arr.astype("uint8") * 255), mode="L").save(
                os.path.join(masks_dir, f"{key}.png")
            )

        with open(os.path.join(output_dir, "food_item_volumes_cc.json"), "w") as f:
            json.dump(
                {**results["volumes_cc"], "total_cc": results["total_volume_cc"]},
                f,
                indent=2,
            )

        zip_path = os.path.join(os.path.dirname(output_dir) or ".", "thali_pipeline_results.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for root, _, filenames in os.walk(output_dir):
                for fname in filenames:
                    filepath = os.path.join(root, fname)
                    arcname = os.path.relpath(filepath, output_dir)
                    zf.write(filepath, arcname=arcname)

        print(f"\nSaved results to: {output_dir}")
        print(f"Zip archive at: {zip_path}")

        return output_dir, zip_path
