import numpy as np


def compute_thali_reference_depth(depth_map, plate_mask):
    """Single reference depth for the whole thali: median depth over every
    pixel inside the plate/tray mask."""
    plate_depths = depth_map[plate_mask]
    if plate_depths.size == 0:
        raise ValueError("Plate mask is empty -- cannot compute a reference depth.")
    return float(np.median(plate_depths))


def compute_pixel_area_map(depth_map, fx, fy):
    """(H, W) map of real-world area (m^2) represented by each pixel."""
    return (depth_map ** 2) / (fx * fy)


def compute_food_heights(depth_map, thali_reference_depth, item_masks):
    """For each {key: bool mask}, compute a height array and report how many
    pixels needed the abs() cleanup.

    Returns (height_arrays: {key: np.ndarray}, diagnostics: {key: {"n_negative": int, "pct_negative": float}})
    """
    height_arrays = {}
    diagnostics = {}

    for key, item_mask in item_masks.items():
        height_array = (thali_reference_depth - depth_map) * item_mask

        n_negative = int((height_array < 0).sum())
        height_array = np.abs(height_array)

        item_px = int(item_mask.sum())
        pct_negative = (100 * n_negative / item_px) if item_px else 0.0

        height_arrays[key] = height_array
        diagnostics[key] = {"n_negative": n_negative, "pct_negative": pct_negative}

    return height_arrays, diagnostics


def compute_food_areas(pixel_area_map, item_masks):
    """{key: bool mask} -> {key: area array (m^2 per pixel, masked)}"""
    return {key: pixel_area_map * mask for key, mask in item_masks.items()}


def compute_food_volumes(height_arrays, area_arrays):
    """height_arrays, area_arrays (both {key: array}) -> per-item volume (m^3)
    plus the underlying volume arrays."""
    volumes = {}
    volume_arrays = {}

    for key in height_arrays:
        volume_array = height_arrays[key] * area_arrays[key]
        volume = float(volume_array.sum())
        assert volume >= 0, f"Unexpected negative volume for {key}"

        volume_arrays[key] = volume_array
        volumes[key] = volume

    return volumes, volume_arrays
