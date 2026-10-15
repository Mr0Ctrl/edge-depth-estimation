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
        mask, in_focus = compute_focus_mask(self.depth_map, focus_depth=focus, dof_width=dof)

        self.assertEqual(mask.shape, (100, 100))
        self.assertGreaterEqual(mask.min(), 0.0)
        self.assertLessEqual(mask.max(), 1.0)

        # Regions inside [focus - dof, focus + dof] (0.4 to 0.6) should have weight == 0
        in_focus_slice = mask[42:58, :]
        np.testing.assert_allclose(in_focus_slice, 0.0, atol=1e-3)

        # Regions far from focus (0.0 and 1.0) should have maximum blur weight
        self.assertAlmostEqual(float(mask[0, 50]), 1.0, places=1)
        self.assertAlmostEqual(float(mask[99, 50]), 1.0, places=1)

    def test_blur_scales_proportionally_with_distance(self):
        # Blur should strictly increase as distance from focus increases (no flat plateau)
        focus = 0.5
        dof = 0.05
        mask, _ = compute_focus_mask(self.depth_map, focus_depth=focus, dof_width=dof)

        # Check weights at depths: 0.55 (just outside DOF), 0.70 (mid distance), 0.95 (far distance)
        # Depth values correspond to row index / 100
        w_near = float(mask[55, 50])
        w_mid = float(mask[70, 50])
        w_far = float(mask[95, 50])

        self.assertLess(w_near, w_mid)
        self.assertLess(w_mid, w_far)

    def test_curve_options_and_razor_thin_dof(self):
        exact_focus = float(self.depth_map[50, 50])
        mask_linear, in_focus_linear = compute_focus_mask(self.depth_map, focus_depth=exact_focus, dof_width=0.001, curve_type="Linear")
        mask_aggr, in_focus_aggr = compute_focus_mask(self.depth_map, focus_depth=exact_focus, dof_width=0.001, curve_type="Aggressive")

        # Just outside focal plane (e.g. row 60), aggressive curve should be much stronger than linear
        self.assertGreater(float(mask_aggr[60, 50]), float(mask_linear[60, 50]))

        # Razor-thin in_focus should only cover exact row 50, but not row 52
        self.assertTrue(in_focus_aggr[50, 50])
        self.assertFalse(in_focus_aggr[52, 50])

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
