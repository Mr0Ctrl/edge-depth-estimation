"""
Synthetic aperture / Bokeh Depth-of-Field rendering engine.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def compute_focus_mask(
    depth_map: np.ndarray,
    focus_depth: float = 0.5,
    dof_width: float = 0.05,
    linear: bool = True,
) -> np.ndarray:
    """
    Computes a continuous blur weight map [0.0 = sharp, 1.0 = fully blurred]
    proportional to the distance from the focal plane across the entire depth range.

    In physical optics, the Circle of Confusion (CoC) diameter is directly proportional
    to the disparity distance |d - d_focus|. Rather than abruptly clamping at a short
    cutoff distance, this calculates a continuous gradient so farther objects receive
    proportionally stronger blur.

    Args:
        depth_map: (H, W) float array normalized in [0.0, 1.0].
        focus_depth: Target depth plane to keep in focus (0.0 to 1.0).
        dof_width: In-focus margin around focus_depth with zero blur.
        linear: If True, uses linear CoC scaling; if False, applies smooth cubic curve.

    Returns:
        blur_weight: (H, W) float32 array in [0.0, 1.0].
    """
    # Absolute distance from focal plane in normalized disparity space
    distance = np.abs(depth_map - focus_depth)

    # Values within dof_width remain completely sharp (weight = 0)
    excess_distance = np.maximum(0.0, distance - dof_width)

    # The maximum possible distance from focus_depth in [0, 1]
    max_possible_dist = max(focus_depth, 1.0 - focus_depth)
    denom = max(1e-4, max_possible_dist - dof_width)

    # Continuous proportional scaling across the entire scene depth
    t = np.clip(excess_distance / denom, 0.0, 1.0)

    if not linear:
        # Smoothstep curve for softer transition
        blur_weight = t * t * (3.0 - 2.0 * t)
    else:
        blur_weight = t

    return blur_weight.astype(np.float32)


def apply_synthetic_bokeh(
    image: np.ndarray,
    depth_map: np.ndarray,
    focus_depth: float = 0.5,
    blur_power: float = 0.6,
    dof_width: float = 0.05,
    max_kernel_size: int = 51,
    highlight_focus: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies synthetic depth-of-field blur to an image guided by its depth map.
    Blur intensity scales continuously with distance to the focal plane.

    Args:
        image: RGB input image (H, W, 3) as uint8.
        depth_map: Normalized depth map (H, W) float32 in [0.0, 1.0].
        focus_depth: Focal depth plane in [0.0, 1.0].
        blur_power: Intensity of blur effect in [0.0, 1.0].
        dof_width: Thickness of the in-focus band in [0.0, 0.5].
        max_kernel_size: Maximum blur kernel size (odd integer).
        highlight_focus: If True, tints in-focus region in green for visual feedback.

    Returns:
        bokeh_image: (H, W, 3) uint8 resulting image with bokeh blur.
        blur_weight: (H, W) float32 blur intensity map [0.0 to 1.0].
    """
    # 1. Continuous proportional blur weight map
    blur_weight = compute_focus_mask(depth_map, focus_depth=focus_depth, dof_width=dof_width)

    if blur_power <= 0.001:
        result = image.copy()
        if highlight_focus:
            in_focus = (1.0 - blur_weight)[:, :, np.newaxis]
            green_tint = np.array([0, 255, 0], dtype=np.float32)
            tinted = image.astype(np.float32) * 0.7 + green_tint * 0.3
            result = (result * (1.0 - in_focus * 0.4) + tinted * (in_focus * 0.4)).astype(np.uint8)
        return result, blur_weight

    # Effective continuous blur scale
    effective_blur = blur_weight * float(blur_power)

    # 2. Multi-tier pyramid Gaussian blur for smooth, continuous optical bokeh
    img_f32 = image.astype(np.float32)

    scaled_max_kernel = int(max_kernel_size * blur_power)
    if scaled_max_kernel % 2 == 0:
        scaled_max_kernel += 1
    scaled_max_kernel = max(3, scaled_max_kernel)

    sigma_max = scaled_max_kernel / 3.0

    # 7 fine-grained blur levels to eliminate stepped quantization and flat cutoffs
    num_levels = 7
    sigmas = [0.0] + [sigma_max * ((i / (num_levels - 1)) ** 1.2) for i in range(1, num_levels)]
    blurred_levels = [img_f32]

    for s in sigmas[1:]:
        k = int(s * 3) * 2 + 1
        blurred = cv2.GaussianBlur(img_f32, (k, k), sigmaX=s, sigmaY=s)
        blurred_levels.append(blurred)

    # 3. Piecewise linear interpolation across the blur pyramid based on continuous depth distance
    num_intervals = num_levels - 1
    scaled_factor = effective_blur * num_intervals  # in [0, num_intervals]

    result_f32 = np.zeros_like(img_f32)

    for i in range(num_intervals):
        lower_bound = float(i)
        upper_bound = float(i + 1)

        interval_mask = (scaled_factor >= lower_bound) & (scaled_factor <= upper_bound)
        if i == num_intervals - 1:
            interval_mask = scaled_factor >= lower_bound

        if np.any(interval_mask):
            local_t = np.clip(scaled_factor - lower_bound, 0.0, 1.0)
            local_t_3ch = local_t[:, :, np.newaxis]

            blended_segment = (
                blurred_levels[i] * (1.0 - local_t_3ch)
                + blurred_levels[i + 1] * local_t_3ch
            )
            mask_3ch = interval_mask[:, :, np.newaxis]
            result_f32 = np.where(mask_3ch, blended_segment, result_f32)

    bokeh_image = np.clip(result_f32, 0.0, 255.0).astype(np.uint8)

    # 4. Visual indicator for in-focus region if requested
    if highlight_focus:
        in_focus = (1.0 - blur_weight)[:, :, np.newaxis]
        green_tint = np.array([0, 255, 0], dtype=np.float32)
        tinted = bokeh_image.astype(np.float32) * 0.7 + green_tint * 0.3
        bokeh_image = np.clip(
            bokeh_image * (1.0 - in_focus * 0.35) + tinted * (in_focus * 0.35),
            0.0,
            255.0,
        ).astype(np.uint8)

    return bokeh_image, blur_weight
