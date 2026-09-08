"""Integration checks using the bundled video and real ONNX inference."""
from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import main
from nanotrack import NanoTrackORT


VIDEO = str(Path(__file__).resolve().parent / "reference/girl_dance.mp4")


class MultiObjectTests(unittest.TestCase):
    def test_independent_state_shared_sessions(self):
        engine = NanoTrackORT()
        cap = cv2.VideoCapture(VIDEO)
        try:
            ok, frame = cap.read()
            self.assertTrue(ok)
            with patch("nanotrack.ort.InferenceSession", side_effect=AssertionError("New session")):
                first, second = engine.new_target(), engine.new_target()
                first.init(frame, (200, 165, 185, 300))
                second.init(frame, (440, 158, 197, 310))
                self.assertIs(first.backbone, second.backbone)
                self.assertIs(first.head, second.head)
                self.assertFalse(np.shares_memory(first.template_features, second.template_features))
                template = second.template_features.copy()
                for _ in range(20):
                    ok, frame = cap.read()
                    self.assertTrue(ok)
                    self.assertTrue(first.update(frame)[0])
                    self.assertTrue(second.update(frame)[0])
                box = second.bbox
                first.reseed(frame, first.bbox)
                self.assertEqual(box, second.bbox)
                np.testing.assert_array_equal(template, second.template_features)
        finally:
            cap.release()

    def test_live_add_select_reseed_replace_remove(self):
        state = {"tick": 0, "mouse": None}
        created, initializations = [], []
        new_target, init = NanoTrackORT.new_target, NanoTrackORT.init

        def create(engine):
            target = new_target(engine)
            created.append(target)
            return target

        def initialize(target, frame, box):
            initializations.append((created.index(target), state["tick"]))
            return init(target, frame, box)

        def register(window, callback):
            state["mouse"] = callback

        def wait(delay):
            state["tick"] += 1
            tick = state["tick"]
            if tick in (1, 4, 9):
                state["mouse"](cv2.EVENT_LBUTTONDOWN, 440, 158, 0, None)
            if tick in (2, 5, 10):
                state["mouse"](cv2.EVENT_LBUTTONUP, 637, 468, 0, None)
            return {3: ord("a"), 6: 9, 7: ord("t"), 8: ord("r"),
                    11: ord("d"), 12: ord("q")}.get(tick, -1)

        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["main.py", "--source", VIDEO]))
            for name in ("namedWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(patch(f"main.cv2.{name}"))
            stack.enter_context(patch("main.cv2.getWindowProperty", return_value=1))
            stack.enter_context(patch("main.cv2.setMouseCallback", register))
            stack.enter_context(patch("main.cv2.waitKey", wait))
            stack.enter_context(patch.object(NanoTrackORT, "new_target", create))
            stack.enter_context(patch.object(NanoTrackORT, "init", initialize))
            main.main()
        self.assertEqual(len(created), 2)
        self.assertEqual(initializations, [(0, 2), (1, 5), (0, 7), (0, 10)])
        self.assertIs(created[0].backbone, created[1].backbone)


if __name__ == "__main__":
    unittest.main()
