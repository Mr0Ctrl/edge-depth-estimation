"""
Standalone CLI script to process an image and generate synthetic depth-of-field bokeh.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image

# Ensure local src directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth_estimator import DepthEstimator
from bokeh_blur import apply_synthetic_bokeh


def main():
    parser = argparse.ArgumentParser(description="Run Monocular Depth & Bokeh Pipeline on an Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--focus-depth", type=float, default=0.5, help="Focal plane depth in [0.0, 1.0]")
    parser.add_argument("--blur-power", type=float, default=0.7, help="Bokeh blur intensity in [0.0, 1.0]")
    parser.add_argument("--dof-width", type=float, default=0.02, help="Depth of field margin")
    parser.add_argument("--curve-type", type=str, default="Aggressive (Sharp Subject Pop)", help="Blur curve type")
    parser.add_argument("--depth-mode", type=str, default="Square Root Normalized (Linearized Distance)", help="Depth representation mode")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf", help="Model ID")
    parser.add_argument("--highlight-focus", action="store_true", help="Highlight the focal plane in green")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load image
    print(f"[Pipeline] Reading image: {args.image}")
    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)

    # 2. Estimate depth
    estimator = DepthEstimator(model_name=args.model)
    depth_norm, depth_colored = estimator.estimate_depth(image_np, depth_mode=args.depth_mode)

    # 3. Apply synthetic aperture bokeh
    print(f"[Pipeline] Applying bokeh (focus={args.focus_depth}, blur_power={args.blur_power})...")
    bokeh_image, blur_weights = apply_synthetic_bokeh(
        image=image_np,
        depth_map=depth_norm,
        focus_depth=args.focus_depth,
        blur_power=args.blur_power,
        dof_width=args.dof_width,
        curve_type=args.curve_type,
        highlight_focus=args.highlight_focus,
    )

    # 4. Save results
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    out_depth_path = os.path.join(args.output_dir, f"{base_name}_depth.png")
    out_bokeh_path = os.path.join(args.output_dir, f"{base_name}_bokeh_focus{args.focus_depth:.2f}.png")
    out_comp_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")

    # Save images (convert RGB to BGR for cv2)
    cv2.imwrite(out_depth_path, cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_bokeh_path, cv2.cvtColor(bokeh_image, cv2.COLOR_RGB2BGR))

    # Create side-by-side comparison: [Original | Depth | Bokeh]
    h, w, _ = image_np.shape
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    comparison[:, :w] = image_np
    comparison[:, w : 2 * w] = depth_colored
    comparison[:, 2 * w :] = bokeh_image
    cv2.imwrite(out_comp_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    print(f"[Pipeline] Results saved:")
    print(f"  - Depth map:   {out_depth_path}")
    print(f"  - Bokeh image: {out_bokeh_path}")
    print(f"  - Comparison:  {out_comp_path}")


if __name__ == "__main__":
    main()
