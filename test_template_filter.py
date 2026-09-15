"""Pixel-domain step response and sampling-rate checks using real ORT sessions."""
import unittest
import numpy as np
from nanotrack import NanoTrackORT


class TemplateFilterTests(unittest.TestCase):
    def test_pixel_step_and_sample_rate(self):
        tracker = NanoTrackORT(template_fc_hz=1.0, debug=True)
        dark = np.full((160, 160, 3), (10, 20, 30), np.uint8)
        bright = np.full_like(dark, (110, 170, 230))
        results = []
        for rate in (30, 60):
            tracker.init(dark, (40, 40, 60, 60))
            box = tracker.bbox
            tracker.refresh_template(bright, dt=0)
            np.testing.assert_array_equal(tracker.template_pixels[0, :, 0, 0], [10, 20, 30])
            for _ in range(rate):
                tracker.refresh_template(bright, dt=1/rate)
            expected = np.array([110, 170, 230]) - np.exp(-2*np.pi)*np.array([100, 150, 200])
            np.testing.assert_allclose(tracker.template_pixels[0, :, 0, 0], expected, atol=5e-5)
            self.assertEqual(tracker.bbox, box)
            self.assertEqual(tracker.template_pixels.dtype, np.float32)
            results.append(tracker.template_pixels.copy())
        np.testing.assert_allclose(results[0], results[1], atol=5e-5)
        tracker.template_fc_hz = 0
        tracker.refresh_template(dark, dt=1/30)
        np.testing.assert_array_equal(tracker.template_pixels[0, :, 0, 0], [10, 20, 30])
        tracker.template_fc_hz = 1
        tracker.reseed(bright, (40, 40, 60, 60))
        np.testing.assert_array_equal(tracker.template_pixels[0, :, 0, 0], [110, 170, 230])
        for invalid in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                tracker.template_fc_hz = invalid
            with self.assertRaises(ValueError):
                tracker.update(bright, dt=invalid)


if __name__ == '__main__':
    unittest.main()
