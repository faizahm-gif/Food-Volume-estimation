"""
Manually-calibrated camera intrinsics lookup.

There is intentionally no EXIF reading and no field-of-view fallback: if the
camera used for a photo isn't registered in config.CAMERA_INTRINSICS_OVERRIDES,
get_camera_intrinsics() raises rather than guessing focal length.
"""
from PIL import Image


def get_camera_intrinsics(image_path, profile_name, overrides):
    """Return (fx_px, fy_px) scaled to the actual image resolution, using a
    manually calibrated profile keyed by `profile_name` in `overrides`.

    Each profile entry gives fx_px/fy_px measured at a reference resolution;
    this scales them linearly to whatever resolution `image_path` actually is.
    """
    if profile_name is None or profile_name not in overrides:
        raise ValueError(
            "No manual camera profile selected. Add fx_px, fy_px, "
            "reference_width_px, reference_height_px for your camera to "
            "CAMERA_INTRINSICS_OVERRIDES in config/config.py, then set "
            "the profile name to match that key."
        )

    with Image.open(image_path) as image:
        width, height = image.size

    profile = overrides[profile_name]
    fx = float(profile["fx_px"]) * width / float(profile["reference_width_px"])
    fy = float(profile["fy_px"]) * height / float(profile["reference_height_px"])
    return fx, fy
