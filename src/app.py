"""
Gradio Interactive Application for EdgeDepth-Lite Prototype.
Simulates synthetic aperture / optical bokeh using Depth Anything.
"""

import os
import sys
import argparse
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import gradio as gr

# Ensure local src directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from depth_estimator import DepthEstimator
from bokeh_blur import apply_synthetic_bokeh, CURVE_OPTIONS

DEPTH_MODES = [
    "Square Root Normalized (Linearized Distance)",
    "Raw Disparity (Direct Model Output)",
]


class BokehApp:
    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        self.estimator = DepthEstimator(model_name=model_name)
        # In-memory cache for current active image and depth map to ensure instant slider updates
        self.cached_image: Optional[np.ndarray] = None
        self.cached_depth: Optional[np.ndarray] = None
        self.cached_depth_colored: Optional[np.ndarray] = None
        self.current_depth_mode: str = DEPTH_MODES[0]

    def process_new_image(
        self,
        image: Optional[np.ndarray],
        depth_mode: str = DEPTH_MODES[0],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Runs Depth Anything inference on newly uploaded image and caches the depth map.
        """
        if image is None:
            self.cached_image = None
            self.cached_depth = None
            self.cached_depth_colored = None
            return None, None, 0.5

        self.cached_image = image
        self.current_depth_mode = depth_mode
        depth_norm, depth_colored = self.estimator.estimate_depth(image, depth_mode=depth_mode)
        self.cached_depth = depth_norm
        self.cached_depth_colored = depth_colored

        default_focus = float(np.median(depth_norm))
        return depth_colored, default_focus

    def update_bokeh(
        self,
        image: Optional[np.ndarray],
        focus_depth: float,
        blur_power: float,
        dof_width: float,
        curve_type: str,
        depth_mode: str,
        highlight_focus: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Applies synthetic bokeh effect using cached depth map (instant slider response).
        """
        if image is None:
            return None, None

        # Check if depth needs to be calculated or re-linearized
        need_recompute = (
            self.cached_depth is None
            or self.cached_image is None
            or not np.array_equal(self.cached_image, image)
            or self.current_depth_mode != depth_mode
        )

        if need_recompute:
            self.cached_image = image
            self.current_depth_mode = depth_mode
            self.cached_depth, self.cached_depth_colored = self.estimator.estimate_depth(
                image, depth_mode=depth_mode
            )

        # Apply bokeh effect
        bokeh_result, _ = apply_synthetic_bokeh(
            image=self.cached_image,
            depth_map=self.cached_depth,
            focus_depth=focus_depth,
            blur_power=blur_power,
            dof_width=dof_width,
            curve_type=curve_type,
            highlight_focus=highlight_focus,
        )

        return bokeh_result, self.cached_depth_colored

    def on_click_focus(
        self,
        evt: gr.SelectData,
        image: Optional[np.ndarray],
        blur_power: float,
        dof_width: float,
        curve_type: str,
        depth_mode: str,
        highlight_focus: bool,
    ):
        """
        Click-to-focus: user clicks on any point on the image to set that object's depth plane in focus!
        """
        if self.cached_depth is None or image is None:
            return gr.update(), None, None

        # evt.index returns [x, y] coordinates
        x, y = evt.index[0], evt.index[1]
        h, w = self.cached_depth.shape[:2]
        x = min(max(0, x), w - 1)
        y = min(max(0, y), h - 1)

        clicked_depth = float(self.cached_depth[y, x])
        bokeh_result, _ = apply_synthetic_bokeh(
            image=self.cached_image,
            depth_map=self.cached_depth,
            focus_depth=clicked_depth,
            blur_power=blur_power,
            dof_width=dof_width,
            curve_type=curve_type,
            highlight_focus=highlight_focus,
        )

        return clicked_depth, bokeh_result, self.cached_depth_colored


def create_demo(app: BokehApp, sample_images_dir: str = "assets") -> gr.Blocks:
    samples = []
    if os.path.exists(sample_images_dir):
        for f in sorted(os.listdir(sample_images_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append([os.path.join(sample_images_dir, f)])

    with gr.Blocks(title="EdgeDepth-Lite | Synthetic Aperture Simulator") as demo:
        gr.Markdown(
            """
            # 📷 EdgeDepth-Lite: Synthetic Aperture & Bokeh Prototype
            Simulate real-time physical depth of field (optical bokeh) using **Depth Anything** monocular depth estimation.
            
            - **Click on the image** or adjust **Focus Depth** to change the focal plane.
            - Adjust **Blur Power** to increase defocus blur strength.
            - Toggle **Depth Representation** to view **Linearized Depth** (expanded background contrast) vs **Raw Disparity**.
            - Adjust **Depth of Field (DOF)** down to `0.000` for razor-sharp focal planes.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image (Upload or Click to Focus)",
                    type="numpy",
                )

                with gr.Accordion("⚙️ Focus & Aperture Controls", open=True):
                    focus_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.005,
                        label="🎯 Focus Depth (0.0 = Far, 1.0 = Near)",
                        info="Adjust focal plane position or click directly on the image",
                    )
                    blur_power_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.02,
                        label="🔮 Blur Power (Bokeh Strength)",
                        info="Simulates wider physical lens aperture",
                    )
                    dof_slider = gr.Slider(
                        minimum=0.000,
                        maximum=0.15,
                        value=0.015,
                        step=0.001,
                        label="📏 Depth of Field (DOF) Margin",
                        info="Thickness of the in-focus band (down to 0.000 for razor-thin)",
                    )
                    curve_dropdown = gr.Dropdown(
                        choices=CURVE_OPTIONS,
                        value=CURVE_OPTIONS[0],
                        label="📈 Blur Response Curve",
                        info="Controls how rapidly blur ramps up away from the focal plane",
                    )
                    depth_mode_dropdown = gr.Dropdown(
                        choices=DEPTH_MODES,
                        value=DEPTH_MODES[0],
                        label="🗺️ Depth Map Representation",
                        info="Linearize disparity to expand background contrast and depth perception",
                    )
                    highlight_checkbox = gr.Checkbox(
                        value=False,
                        label="🟢 Highlight In-Focus Region (Visual Aid)",
                        info="Overlays strict green tint on the active focal plane only",
                    )

            with gr.Column(scale=1):
                output_bokeh = gr.Image(
                    label="📸 Synthetic Aperture Result (Defocus Bokeh)",
                    type="numpy",
                )
                output_depth = gr.Image(
                    label="🗺️ Estimated Depth Map (Depth Anything)",
                    type="numpy",
                )

        if samples:
            gr.Examples(
                examples=samples,
                inputs=input_image,
                label="Sample Images",
            )

        # Dynamic events
        # 1. When image changes, compute depth map and update bokeh
        def on_image_change(img, focus, blur, dof, curve, d_mode, highlight):
            if img is None:
                return None, None, 0.5
            _, default_focus = app.process_new_image(img, depth_mode=d_mode)
            active_focus = focus if focus is not None else default_focus
            bokeh_img, depth_col = app.update_bokeh(img, active_focus, blur, dof, curve, d_mode, highlight)
            return bokeh_img, depth_col, active_focus

        all_controls = [
            input_image,
            focus_slider,
            blur_power_slider,
            dof_slider,
            curve_dropdown,
            depth_mode_dropdown,
            highlight_checkbox,
        ]
        slider_outputs = [output_bokeh, output_depth]

        input_image.change(
            fn=on_image_change,
            inputs=all_controls,
            outputs=[output_bokeh, output_depth, focus_slider],
        )

        # 2. Sliders & controls update bokeh instantly using cached depth
        focus_slider.change(fn=app.update_bokeh, inputs=all_controls, outputs=slider_outputs)
        blur_power_slider.change(fn=app.update_bokeh, inputs=all_controls, outputs=slider_outputs)
        dof_slider.change(fn=app.update_bokeh, inputs=all_controls, outputs=slider_outputs)
        curve_dropdown.change(fn=app.update_bokeh, inputs=all_controls, outputs=slider_outputs)
        depth_mode_dropdown.change(fn=app.update_bokeh, inputs=all_controls, outputs=slider_outputs)
        highlight_checkbox.change(fn=app.update_bokeh, inputs=all_controls, outputs=slider_outputs)

        # 3. Click-to-focus on input image
        input_image.select(
            fn=app.on_click_focus,
            inputs=[input_image, blur_power_slider, dof_slider, curve_dropdown, depth_mode_dropdown, highlight_checkbox],
            outputs=[focus_slider, output_bokeh, output_depth],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="EdgeDepth-Lite Synthetic Aperture Prototype")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf", help="HuggingFace model ID")
    parser.add_argument("--share", action="store_true", help="Enable public Gradio link")
    args = parser.parse_args()

    app = BokehApp(model_name=args.model)
    demo = create_demo(app)
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, theme=theme)


if __name__ == "__main__":
    main()
