"""
Synthetic aperture / Bokeh Depth-of-Field rendering engine.
Supports multiple blur falloff curves and fine-grained focal plane control.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

# Available blur curve types for UI and pipeline
CURVE_OPTIONS = [
    "Aggressive (Sharp Subject Pop)",
    "Ultra Pop (Instant Defocus)",
    "Linear (Direct Distance)",
    "Smoothstep (Natural S-Curve)",
    "Quadratic (Gentle Near Focus)",
    "Exponential (Deep Background)",
]


def apply_blur_curve(t: np.ndarray, curve_type: str = "Aggressive (Sharp Subject Pop)") -> np.ndarray:
    """
    Transforms normalized distance t in [0.0, 1.0] using the selected response curve.
    """
    key = curve_type.lower()

    if "ultra" in key:
        # 4th root: rapid initial defocus right outside the focal plane
        return np.power(t, 0.25)
    elif "aggressive" in key or "pop" in key or "root" in key:
        # Square root: steep initial slope makes in-focus subject pop sharply
        return np.sqrt(t)
    elif "smoothstep" in key or "s-curve" in key:
        # Hermite smoothstep 3t^2 - 2t^3
        return t * t * (3.0 - 2.0 * t)
    elif "quadratic" in key:
        # Quadratic t^2: slow defocus near focus, steep in distant background
        return t * t
    elif "exponential" in key:
        # Exponential curve
        exp_denom = np.exp(3.0) - 1.0
        return (np.exp(3.0 * t) - 1.0) / exp_denom
    else:
        # Default: linear
        return t


def compute_focus_mask(
    depth_map: np.ndarray,
    focus_depth: float = 0.5,
    dof_width: float = 0.02,
    curve_type: str = "Aggressive (Sharp Subject Pop)",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a continuous blur weight map [0.0 = sharp, 1.0 = fully blurred]
    and a strict in-focus boolean mask.

    Args:
        depth_map: (H, W) float array normalized in [0.0, 1.0].
        focus_depth: Target depth plane to keep in focus (0.0 to 1.0).
        dof_width: In-focus margin around focus_depth with zero blur.
        curve_type: Blur response curve name.

    Returns:
        blur_weight: (H, W) float32 array in [0.0, 1.0].
        exact_in_focus: (H, W) bool array, True strictly inside in-focus band.
    """
    # Absolute distance from focal plane in normalized disparity space
    distance = np.abs(depth_map - focus_depth)

    # Values within dof_width remain completely sharp (excess == 0)
    excess_distance = np.maximum(0.0, distance - dof_width)
    exact_in_focus = excess_distance <= 1e-4

    # The maximum possible distance from focus_depth in [0, 1]
    max_possible_dist = max(focus_depth, 1.0 - focus_depth)
    denom = max(1e-4, max_possible_dist - dof_width)

    # Continuous proportional distance
    t = np.clip(excess_distance / denom, 0.0, 1.0)

    # Apply selected curve
    blur_weight = apply_blur_curve(t, curve_type=curve_type)

    return blur_weight.astype(np.float32), exact_in_focus


def apply_synthetic_bokeh(
    image: np.ndarray,
    depth_map: np.ndarray,
    focus_depth: float = 0.5,
    blur_power: float = 0.6,
    dof_width: float = 0.02,
    curve_type: str = "Aggressive (Sharp Subject Pop)",
    max_kernel_size: int = 51,
    highlight_focus: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies synthetic depth-of-field blur to an image guided by its depth map.

    Args:
        image: RGB input image (H, W, 3) as uint8.
        depth_map: Normalized depth map (H, W) float32 in [0.0, 1.0].
        focus_depth: Focal depth plane in [0.0, 1.0].
        blur_power: Intensity of blur effect in [0.0, 1.0].
        dof_width: Thickness of the in-focus band in [0.0, 0.5].
        curve_type: Response curve ('Aggressive', 'Linear', 'Smoothstep', etc.).
        max_kernel_size: Maximum blur kernel size (odd integer).
        highlight_focus: If True, tints in-focus region in green for visual feedback.

    Returns:
        bokeh_image: (H, W, 3) uint8 resulting image with bokeh blur.
        blur_weight: (H, W) float32 blur intensity map [0.0 to 1.0].
    """
    # 1. Calculate blur weight and strict in-focus plane
    blur_weight, exact_in_focus = compute_focus_mask(
        depth_map,
        focus_depth=focus_depth,
        dof_width=dof_width,
        curve_type=curve_type,
    )

    if blur_power <= 0.001:
        result = image.copy()
        if highlight_focus:
            # Highlight strictly the exact in-focus plane
            in_focus_3ch = exact_in_focus[:, :, np.newaxis]
            green_tint = np.array([0, 255, 0], dtype=np.float32)
            tinted = image.astype(np.float32) * 0.4 + green_tint * 0.6
            result = np.where(in_focus_3ch, tinted, image).astype(np.uint8)
        return result, blur_weight

    # Effective continuous blur scale
    effective_blur = blur_weight * float(blur_power)

    # 2. Multi-tier pyramid Gaussian blur
    img_f32 = image.astype(np.float32)

    scaled_max_kernel = int(max_kernel_size * blur_power)
    if scaled_max_kernel % 2 == 0:
        scaled_max_kernel += 1
    scaled_max_kernel = max(3, scaled_max_kernel)

    sigma_max = scaled_max_kernel / 3.0

    num_levels = 7
    sigmas = [0.0] + [sigma_max * ((i / (num_levels - 1)) ** 1.2) for i in range(1, num_levels)]
    blurred_levels = [img_f32]

    for s in sigmas[1:]:
        k = int(s * 3) * 2 + 1
        blurred = cv2.GaussianBlur(img_f32, (k, k), sigmaX=s, sigmaY=s)
        blurred_levels.append(blurred)

    # 3. Piecewise linear interpolation across the blur pyramid
    num_intervals = num_levels - 1
    scaled_factor = effective_blur * num_intervals

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
    # Shows strictly the in-focus band without bleeding into soft out-of-focus areas
    if highlight_focus:
        in_focus_3ch = exact_in_focus[:, :, np.newaxis]
        green_tint = np.array([0, 255, 0], dtype=np.float32)
        tinted = bokeh_image.astype(np.float32) * 0.4 + green_tint * 0.6
        bokeh_image = np.where(in_focus_3ch, tinted, bokeh_image).astype(np.uint8)

    return bokeh_image, blur_weight
