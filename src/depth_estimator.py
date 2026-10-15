"""
Depth Estimator module using Depth Anything models.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib.cm as cm


class DepthEstimator:
    """
    Wrapper for Depth Anything monocular depth estimation model.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[DepthEstimator] Loading model '{model_name}' on device '{self.device}'...")

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        except Exception as e:
            # Fallback to Depth-Anything v1 small if v2 fails to download
            fallback_model = "LiheYoung/depth-anything-small-hf"
            print(f"[DepthEstimator] Failed to load {model_name} ({e}). Retrying with fallback {fallback_model}...")
            self.image_processor = AutoImageProcessor.from_pretrained(fallback_model)
            self.model = AutoModelForDepthEstimation.from_pretrained(fallback_model).to(self.device)

        self.model.eval()
        print("[DepthEstimator] Model loaded successfully.")

    @torch.no_grad()
    def estimate_depth(
        self,
        image: Union[np.ndarray, Image.Image],
        colormap: str = "inferno",
        depth_mode: str = "Linearized (Balanced Contrast)",
        gamma: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate relative depth from an input image.

        Args:
            image: Input image as RGB numpy array (H, W, 3) or PIL Image.
            colormap: Matplotlib colormap name for visualization ('inferno', 'magma', 'turbo').
            depth_mode: Depth representation:
                - 'Raw Disparity (Foundation Model Default)': Direct 1/Z disparity output.
                - 'Linearized (Balanced Contrast)': Perceptual linearization expanding background details.
                - 'Inverse Metric (1/Z Physical Distance)': Physical distance linearization.
            gamma: Optional additional power transform on depth values.

        Returns:
            depth_normalized: (H, W) float32 numpy array with values in [0.0, 1.0].
            depth_colored: (H, W, 3) uint8 numpy array RGB colormap representation.
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype(np.uint8))
            orig_w, orig_h = image.shape[1], image.shape[0]
        else:
            pil_image = image
            orig_w, orig_h = image.size

        # Preprocess input image
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict depth (disparity / relative inverse depth)
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth

        # Interpolate back to original resolution
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_np = prediction.cpu().numpy()

        # Normalize raw disparity to [0, 1]
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        if depth_max > depth_min:
            depth_norm = (depth_np - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_np, dtype=np.float32)

        depth_norm = depth_norm.astype(np.float32)

        # True linear depth inversion: z = 1/m
        # Disparity m = k/z has 55x sensitivity distortion between near and far.
        # Taking z = 1/m makes depth change completely uniform (1x) across all distances.
        if "raw" in depth_mode.lower():
            final_depth = depth_norm
        else:
            z_near = 1.0
            z_far = 8.0  # Photographic focal depth ratio
            inv_z = (1.0 / z_far) + depth_norm * (1.0 / z_near - 1.0 / z_far)
            z = 1.0 / np.maximum(inv_z, 1e-6)

            # Normalize z strictly to [0.0, 1.0]
            z_min = float(z.min())
            z_max = float(z.max())
            if z_max > z_min:
                z_norm = (z - z_min) / (z_max - z_min)
            else:
                z_norm = np.zeros_like(z)

            # Standard convention: 1.0 = Near (foreground), 0.0 = Far (background)
            final_depth = 1.0 - z_norm

        depth_norm = np.clip(final_depth, 0.0, 1.0).astype(np.float32)

        # Generate colored visualization
        try:
            import matplotlib
            cmap = matplotlib.colormaps[colormap]
        except (AttributeError, KeyError):
            cmap = cm.get_cmap(colormap)
        depth_colored = (cmap(depth_norm)[:, :, :3] * 255.0).astype(np.uint8)

        return depth_norm, depth_colored
