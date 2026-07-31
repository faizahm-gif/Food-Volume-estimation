
import numpy as np


def get_bbox_from_mask(plate_mask):
    """Axis-aligned bounding box [left, top, right, bottom] of a boolean
    mask's foreground pixels.

    This is the SAM2-mask equivalent of the reference
    WhiteXiezx/Food-Volume-Estimation repo's `mask2box()` /
    `get_bbox()`, which derived the same box from a labelme polygon by
    first rasterizing it into a mask and then taking np.argwhere. Since
    we already have a full mask from GSAM2 (no polygon rasterization
    needed), we go straight to np.argwhere.
    """
    ys, xs = np.where(plate_mask)
    if ys.size == 0:
        raise ValueError(
            "Plate mask is empty -- cannot compute a bounding box."
        )

    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max())
    bottom = int(ys.max())
    return [left, top, right, bottom]


def compute_plate_pixel_diameter(plate_mask):
    """Estimate the plate's diameter in pixels as the average of the
    plate mask's axis-aligned bounding-box width and height.

    This matches the reference WhiteXiezx/Food-Volume-Estimation repo's
    `get_scale()` exactly in spirit:

        bbox = get_bbox(points, h, w)
        diameter = (bbox[2]-bbox[0]+1 + bbox[3]-bbox[1]+1) / 2

    i.e. diameter = average of (bbox width, bbox height), computed here
    from the SAM2 plate mask instead of a labelme-clicked polygon.
    """
    left, top, right, bottom = get_bbox_from_mask(plate_mask)

    bbox_width = float(right - left + 1)
    bbox_height = float(bottom - top + 1)
    diameter_px = (bbox_width + bbox_height) / 2.0

    if diameter_px <= 0:
        raise ValueError("Estimated plate pixel diameter is non-positive.")

    return diameter_px


def compute_scale_from_plate(plate_mask, known_plate_diameter_m):
    """Meters-per-pixel scale factor (len_per_pix, in the reference repo's
    terminology), derived from a known real-world plate diameter and the
    plate's pixel diameter measured from its SAM2 mask's bounding box.

    This replaces camera-intrinsics-based scale recovery: instead of using
    fx/fy and per-pixel depth to work out real-world area, we anchor the
    scale directly to an object of known physical size that's already in
    the shot -- the plate/thali itself.

    Returns (meters_per_pixel, plate_diameter_px).
    """
    if known_plate_diameter_m is None or known_plate_diameter_m <= 0:
        raise ValueError(
            "known_plate_diameter_m must be a positive number (meters). "
            "Set PLATE_DIAMETER_M in config/config.py to your plate's real "
            "diameter, or pass --plate-diameter-m on the demo.py CLI."
        )

    diameter_px = compute_plate_pixel_diameter(plate_mask)
    meters_per_pixel = known_plate_diameter_m / diameter_px
    return meters_per_pixel, diameter_px


def compute_scale_from_plate_rectangular(
    plate_mask, known_plate_length_m, known_plate_width_m
):
    """Anisotropic (x, y) meters-per-pixel scale for a rectangular plate/tray,
    derived from the plate's known real-world length and width vs. the
    bounding-box width/height of its SAM2 mask.

    Unlike the circular case -- where bbox width and bbox height are just
    two noisy estimates of the *same* diameter and get averaged -- a
    rectangular plate genuinely has two different physical dimensions, so
    each bbox axis gets its own scale instead of being collapsed into one
    number.

    Pairing bbox axes to physical dimensions:
    We don't assume the plate's "length" necessarily lines up with the
    image's horizontal axis -- a thali can be photographed in either
    orientation. Instead we pair the *larger* physical dimension with the
    *larger* bbox-pixel dimension, and the smaller with the smaller. This
    is a heuristic (it assumes the plate's edges are roughly axis-aligned
    with the image, i.e. not photographed at a diagonal/rotated angle) but
    it makes the scale robust to portrait vs. landscape shots of the same
    plate.

    Returns (meters_per_pixel_x, meters_per_pixel_y, bbox_width_px, bbox_height_px).
    meters_per_pixel_x applies along the image's horizontal (column) axis,
    meters_per_pixel_y along the vertical (row) axis.
    """
    if known_plate_length_m is None or known_plate_length_m <= 0:
        raise ValueError(
            "known_plate_length_m must be a positive number (meters). "
            "Set PLATE_LENGTH_M in config/config.py to your plate's real "
            "length, or pass --plate-length-m on the demo.py CLI."
        )
    if known_plate_width_m is None or known_plate_width_m <= 0:
        raise ValueError(
            "known_plate_width_m must be a positive number (meters). "
            "Set PLATE_WIDTH_M in config/config.py to your plate's real "
            "width, or pass --plate-width-m on the demo.py CLI."
        )

    left, top, right, bottom = get_bbox_from_mask(plate_mask)
    bbox_width_px = float(right - left + 1)
    bbox_height_px = float(bottom - top + 1)

    # Pair the longer physical side with the longer pixel side, regardless
    # of which image axis (x or y) that happens to be.
    if bbox_width_px >= bbox_height_px:
        long_bbox_axis, short_bbox_axis = "x", "y"
    else:
        long_bbox_axis, short_bbox_axis = "y", "x"

    long_dim_m = max(known_plate_length_m, known_plate_width_m)
    short_dim_m = min(known_plate_length_m, known_plate_width_m)

    scale_per_axis = {}
    scale_per_axis[long_bbox_axis] = long_dim_m / (
        bbox_width_px if long_bbox_axis == "x" else bbox_height_px
    )
    scale_per_axis[short_bbox_axis] = short_dim_m / (
        bbox_width_px if short_bbox_axis == "x" else bbox_height_px
    )

    meters_per_pixel_x = scale_per_axis["x"]
    meters_per_pixel_y = scale_per_axis["y"]

    return meters_per_pixel_x, meters_per_pixel_y, bbox_width_px, bbox_height_px
