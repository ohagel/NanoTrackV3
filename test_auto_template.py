"""Validate template refresh without geometry resets or session reconstruction."""
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from nanotrack import NanoTrackORT


class AutoTemplateTests(unittest.TestCase):
    def test_refresh_order_state_and_toggle(self):
        fixed = NanoTrackORT(debug=True)
        adaptive = fixed.new_target()
        cap = cv2.VideoCapture(str(Path(__file__).parent / "reference/girl_dance.mp4"))
        try:
            ok, first = cap.read()
            self.assertTrue(ok)
            for tracker in (fixed, adaptive):
                tracker.init(first, (440, 158, 197, 310))
            old = adaptive.template_features.copy()
            adaptive.auto_update_template = True
            ok, frame = cap.read()
            self.assertTrue(ok)
            expected = fixed.update(frame)
            with patch("nanotrack.ort.InferenceSession", side_effect=AssertionError("New session")):
                with patch.object(adaptive, "_features", wraps=adaptive._features) as infer:
                    actual = adaptive.update(frame)
                    self.assertEqual([c.args[0].shape for c in infer.call_args_list],
                                     [(1, 3, 255, 255), (1, 3, 127, 127)])
                self.assertEqual(actual, expected)
                np.testing.assert_array_equal(adaptive.center_pos, fixed.center_pos)
                np.testing.assert_array_equal(adaptive.search_image, fixed.search_image)
                self.assertFalse(np.array_equal(old, adaptive.template_features))
                fixed.refresh_template(frame)
                np.testing.assert_array_equal(adaptive.template_features, fixed.template_features)
                adaptive.auto_update_template = False
                frozen = adaptive.template_features.copy()
                with patch.object(adaptive, "_features", wraps=adaptive._features) as infer:
                    adaptive.update(frame)
                    self.assertEqual([c.args[0].shape for c in infer.call_args_list], [(1, 3, 255, 255)])
                np.testing.assert_array_equal(frozen, adaptive.template_features)
        finally:
            cap.release()


if __name__ == "__main__":
    unittest.main()
