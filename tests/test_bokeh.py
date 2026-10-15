"""
Unit tests for bokeh blur module.
"""

import unittest
import numpy as np
import sys
import os

# Add src to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from bokeh_blur import compute_focus_mask, apply_synthetic_bokeh


class TestBokehBlur(unittest.TestCase):
    def setUp(self):
        # 100x100 synthetic test image and depth ramp (0.0 to 1.0)
        self.image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Create depth gradient from 0.0 (top) to 1.0 (bottom)
        self.depth_map = np.tile(np.linspace(0.0, 1.0, 100)[:, np.newaxis], (1, 100)).astype(np.float32)

    def test_focus_mask_bounds(self):
        focus = 0.5
        dof = 0.1
        mask = compute_focus_mask(self.depth_map, focus_depth=focus, dof_width=dof, falloff=0.2)

        self.assertEqual(mask.shape, (100, 100))
        self.assertGreaterEqual(mask.min(), 0.0)
        self.assertLessEqual(mask.max(), 1.0)

        # Regions inside [focus - dof, focus + dof] (0.4 to 0.6) should have weight == 0
        in_focus_slice = mask[42:58, :]
        np.testing.assert_allclose(in_focus_slice, 0.0, atol=1e-3)

        # Regions far from focus (e.g. 0.0 and 1.0) should have high blur weight ~ 1.0
        self.assertAlmostEqual(float(mask[0, 50]), 1.0, places=1)
        self.assertAlmostEqual(float(mask[99, 50]), 1.0, places=1)

    def test_zero_blur_power_preserves_image(self):
        bokeh, weight = apply_synthetic_bokeh(
            self.image,
            self.depth_map,
            focus_depth=0.5,
            blur_power=0.0,
        )
        np.testing.assert_array_equal(bokeh, self.image)

    def test_bokeh_blur_shape_and_dtype(self):
        bokeh, weight = apply_synthetic_bokeh(
            self.image,
            self.depth_map,
            focus_depth=0.5,
            blur_power=0.8,
        )
        self.assertEqual(bokeh.shape, self.image.shape)
        self.assertEqual(bokeh.dtype, np.uint8)
        self.assertEqual(weight.shape, (100, 100))


if __name__ == "__main__":
    unittest.main()
