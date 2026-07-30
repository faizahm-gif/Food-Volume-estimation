from io import BytesIO

import cv2
import numpy as np
import requests
import supervision as sv
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class DetectionDeduplicator:
    """Removes duplicate detections first by bounding-box IoU, then (if
    masks are available) by mask IoU, keeping the higher-confidence one."""

    @staticmethod
    def calculate_iou(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def mask_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    @classmethod
    def remove_duplicates(cls, detections, iou_threshold=0.5, use_masks=True):
        if len(detections) == 0:
            return detections

        keep_indices = []
        removed_indices = set()
        sorted_indices = np.argsort(-detections.confidence)

        for i in sorted_indices:
            if i in removed_indices:
                continue

            keep_indices.append(i)
            box_i = detections.xyxy[i]

            for j in sorted_indices:
                if j <= i or j in removed_indices:
                    continue

                box_j = detections.xyxy[j]
                iou = cls.calculate_iou(box_i, box_j)

                if iou > iou_threshold:
                    removed_indices.add(j)
                    continue

                if (
                    use_masks
                    and hasattr(detections, "mask")
                    and detections.mask is not None
                ):
                    mask_iou_val = cls.mask_iou(detections.mask[i], detections.mask[j])
                    if mask_iou_val > 0.65:
                        removed_indices.add(j)

        keep_indices = [idx for idx in keep_indices if idx not in removed_indices]

        if len(keep_indices) == 0:
            return detections

        filtered = sv.Detections(
            xyxy=detections.xyxy[keep_indices],
            confidence=detections.confidence[keep_indices],
            class_id=detections.class_id[keep_indices],
        )

        if hasattr(detections, "mask") and detections.mask is not None:
            filtered.mask = detections.mask[keep_indices]

        if hasattr(detections, "data") and detections.data:
            filtered.data = {}
            for key, values in detections.data.items():
                if isinstance(values, list):
                    filtered.data[key] = [values[i] for i in keep_indices]
                elif isinstance(values, np.ndarray):
                    filtered.data[key] = values[keep_indices]
                else:
                    filtered.data[key] = values

        print(f"   Deduplication: {len(detections)} -> {len(filtered)} detections")
        return filtered


class FoodPlateSegmentation:
    """Grounding DINO (open-vocab detection) + SAM2 (segmentation) for food
    items on a plate/thali."""

    def __init__(self, sam2_checkpoint=None, sam2_config=None, use_huggingface=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading Grounding DINO...")
        self.grounding_dino_processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        )
        self.grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(self.device)

        print("Loading SAM2...")
        if use_huggingface:
            try:
                print("Trying SAM2.1 (latest version)...")
                self.sam2_predictor = SAM2ImagePredictor.from_pretrained(
                    "facebook/sam2.1-hiera-large", device=self.device
                )
                print("Successfully loaded SAM2.1")
            except Exception as e:
                print(f"SAM2.1 failed: {e}")
                print("Falling back to SAM2.0...")
                try:
                    self.sam2_predictor = SAM2ImagePredictor.from_pretrained(
                        "facebook/sam2-hiera-large", device=self.device
                    )
                    print("Successfully loaded SAM2.0")
                except Exception as e2:
                    print(f"SAM2.0 also failed: {e2}")
                    raise Exception("Could not load any SAM2 model from HuggingFace")
        else:
            if sam2_checkpoint is None or sam2_config is None:
                raise ValueError(
                    "When use_huggingface=False, both sam2_checkpoint and "
                    "sam2_config must be provided"
                )

            self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # Generic terms plus common Indian thali items, so detection isn't
        # biased toward Western/fast-food vocabulary.
        self.food_prompts = [
            "rice", "steamed rice", "biryani rice", "jeera rice",
            "bread", "roti", "chapati", "naan", "paratha", "puri",
            "curry", "gravy", "dal", "sambar", "rasam", "kootu", "poriyal", "avial",
            "vegetable curry", "paneer curry",
            "meat", "chicken curry", "mutton curry", "fish curry", "egg curry", "biryani",
            "salad", "raita", "curd", "yogurt",
            "pickle", "achaar", "chutney", "papad", "papadum", "sauce",
            "payasam", "kheer", "sweet", "dessert", "halwa",
            "food", "thali item", "egg"
        ]

        print("Models loaded successfully!")

    def detect_food_items(self, image, text_prompt=None, confidence_threshold=0.3):
        if text_prompt is None:
            text_prompt = ". ".join(self.food_prompts) + "."

        inputs = self.grounding_dino_processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.grounding_dino_model(**inputs)

        results = self.grounding_dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            target_sizes=[image.size[::-1]],
        )

        detections = sv.Detections(
            xyxy=results[0]["boxes"].cpu().numpy(),
            confidence=results[0]["scores"].cpu().numpy(),
            class_id=np.arange(len(results[0]["boxes"])),
        )
        detections.data = {"labels": results[0]["labels"]}

        return detections

    def segment_with_sam2(self, image, detections):
        image_np = np.array(image) if isinstance(image, Image.Image) else image

        self.sam2_predictor.set_image(image_np)

        masks = []
        mask_qualities = []

        for box in detections.xyxy:
            mask, quality, _ = self.sam2_predictor.predict(
                point_coords=None, point_labels=None, box=box, multimask_output=False
            )
            masks.append(mask[0])
            mask_qualities.append(quality[0])

        if len(masks) > 0:
            detections.mask = np.array(masks)
            detections.data["mask_quality"] = np.array(mask_qualities)

        return masks, detections

    def process_food_plate(
        self,
        image_path_or_url,
        custom_prompt=None,
        confidence_threshold=0.20,
        enable_deduplication=True,
        max_detections=10,
    ):
        if image_path_or_url.startswith(("http://", "https://")):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")

        print(f"Processing image of size: {image.size}")

        print("Detecting food items...")
        detections = self.detect_food_items(image, custom_prompt, confidence_threshold)
        print(f"Found {len(detections)} food items")

        if len(detections) == 0:
            print("No food items detected. Try lowering the confidence threshold or adjusting the prompt.")
            return image, detections, np.array(image)

        if enable_deduplication:
            print("Removing duplicate detections...")
            detections = DetectionDeduplicator.remove_duplicates(
                detections, iou_threshold=0.5, use_masks=False
            )

            if len(detections) > max_detections:
                sorted_indices = np.argsort(-detections.confidence)[:max_detections]
                detections = sv.Detections(
                    xyxy=detections.xyxy[sorted_indices],
                    confidence=detections.confidence[sorted_indices],
                    class_id=detections.class_id[sorted_indices],
                )
                if hasattr(detections, "data"):
                    detections.data = {
                        k: [v[i] for i in sorted_indices] if isinstance(v, list) else v
                        for k, v in detections.data.items()
                    }
                print(f"Limited to top {max_detections} detections")

        print("Segmenting detected items...")
        masks, enhanced_detections = self.segment_with_sam2(image, detections)

        if enable_deduplication and hasattr(enhanced_detections, "mask"):
            print("Final mask-based deduplication...")
            enhanced_detections = DetectionDeduplicator.remove_duplicates(
                enhanced_detections, iou_threshold=0.6, use_masks=True
            )

        visualization = self.create_visualization(image, enhanced_detections)

        return image, enhanced_detections, visualization

    def create_visualization(self, image, detections):
        image_np = np.array(image)
        annotated_image = image_np.copy()

        box_annotator = sv.BoxAnnotator(thickness=2)
        try:
            mask_annotator = sv.MaskAnnotator(opacity=0.3)
        except TypeError:
            mask_annotator = sv.MaskAnnotator()

        if hasattr(sv, "LabelAnnotator"):
            try:
                label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
            except TypeError:
                label_annotator = sv.LabelAnnotator()
        else:
            label_annotator = None

        if hasattr(detections, "mask") and detections.mask is not None:
            detections.mask = np.array(
                [(m > 0.5) if m.dtype != bool else m for m in detections.mask]
            )
            annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)

        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)

        if "labels" in detections.data and label_annotator is not None:
            labels = [
                f"{label} ({conf:.2f})"
                for label, conf in zip(detections.data["labels"], detections.confidence)
            ]
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels
            )

        return annotated_image

    def analyze_food_composition(self, detections, image_shape):
        if not hasattr(detections, "mask") or detections.mask is None:
            return {}

        total_image_area = image_shape[0] * image_shape[1]
        composition = {}

        for i, (mask, label) in enumerate(
            zip(detections.mask, detections.data.get("labels", []))
        ):
            mask_area = np.sum(mask)
            percentage = (mask_area / total_image_area) * 100

            composition[label] = {
                "area_pixels": int(mask_area),
                "percentage": round(percentage, 2),
                "confidence": round(float(detections.confidence[i]), 3),
                "bbox": detections.xyxy[i].tolist(),
            }

        return composition


def build_indexed_mask(detections, image_shape):
    """Collapse per-item boolean masks into one uint8/16 label map (0 =
    background), plus a {index: label} lookup."""
    H, W = image_shape[:2]

    if not hasattr(detections, "mask") or detections.mask is None or len(detections) == 0:
        return np.zeros((H, W), dtype=np.uint8), {}

    n = len(detections)
    dtype = np.uint8 if n < 255 else np.uint16
    indexed_mask = np.zeros((H, W), dtype=dtype)
    index_to_label = {}

    labels = (
        detections.data.get("labels", [f"item_{i}" for i in range(n)])
        if hasattr(detections, "data")
        else [f"item_{i}" for i in range(n)]
    )

    order = np.argsort(detections.confidence)

    for rank, i in enumerate(order):
        idx = rank + 1
        mask_i = detections.mask[i].astype(bool)
        indexed_mask[mask_i] = idx
        index_to_label[idx] = labels[i]

    return indexed_mask, index_to_label


def run_thali_segmentation(
    segmenter,
    image_path_or_url,
    custom_prompt=None,
    confidence_threshold=0.25,
    enable_deduplication=True,
):
    """Detect + segment every food item on the plate, and return everything
    downstream steps need (detections, indexed mask, composition stats)."""
    original_image, detections, visualization = segmenter.process_food_plate(
        image_path_or_url,
        custom_prompt=custom_prompt,
        confidence_threshold=confidence_threshold,
        enable_deduplication=enable_deduplication,
    )

    image_shape = np.array(original_image).shape
    indexed_mask, index_to_label = build_indexed_mask(detections, image_shape)
    composition = segmenter.analyze_food_composition(detections, original_image.size)

    return {
        "original_image": original_image,
        "detections": detections,
        "visualization": visualization,
        "indexed_mask": indexed_mask,
        "index_to_label": index_to_label,
        "composition": composition,
    }


def detect_plate_mask(segmenter, image, min_confidence=0.15):
    """Auto-detect the plate/tray itself (not the food) via Grounding DINO +
    SAM2, keeping only the single highest-confidence detection.

    Returns (None, None) if no plate/tray is found -- there is intentionally
    no convex-hull or other geometric fallback; callers should raise.
    """
    plate_prompt = (
        "plate. thali plate. steel plate. round plate. dinner plate. "
        "serving tray. metal tray."
    )
    plate_detections = segmenter.detect_food_items(
        image, text_prompt=plate_prompt, confidence_threshold=min_confidence
    )

    if len(plate_detections) == 0:
        print("No plate detected via Grounding DINO.")
        return None, None

    best_idx = int(np.argmax(plate_detections.confidence))
    single = sv.Detections(
        xyxy=plate_detections.xyxy[best_idx : best_idx + 1],
        confidence=plate_detections.confidence[best_idx : best_idx + 1],
        class_id=plate_detections.class_id[best_idx : best_idx + 1],
    )
    single.data = {"labels": [plate_detections.data["labels"][best_idx]]}

    _, seg_detections = segmenter.segment_with_sam2(image, single)
    plate_mask = seg_detections.mask[0].astype(bool)
    label = seg_detections.data["labels"][0]
    confidence = float(seg_detections.confidence[0])
    print(f'Auto-detected plate: "{label}" (confidence {confidence:.3f})')
    return plate_mask, confidence


def resize_mask_to(mask, target_shape_hw):
    """Nearest-neighbor resize a boolean mask to (H, W) if it doesn't
    already match, matching the notebook's resize-on-mismatch behavior."""
    if mask.shape == target_shape_hw:
        return mask
    resized = cv2.resize(
        mask.astype("uint8"),
        (target_shape_hw[1], target_shape_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)
