"""Purdue evaluation lifecycle/annotation contract tests; no dataset mutation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from evaluate_purdue import SequenceTracker, box_metrics, clip_selection, discover, load_annotations


class FakeTracker:
    def __init__(self):
        self.initialized = self.updated = 0

    def init(self, frame, box):
        self.initialized += 1
        self.bbox = box

    def update(self, frame, *, dt):
        self.updated += 1
        return True, self.bbox, .5


class FakeEngine:
    def new_target(self):
        return FakeTracker()


class PurdueTests(unittest.TestCase):
    def test_births_and_reappearance_do_not_correct_predictions(self):
        sequence = SequenceTracker(FakeEngine())
        box = (10,20,30,40)
        self.assertEqual(sequence.step(None,1,{},1/30), {})
        result = sequence.step(None,2,{1:box},1/30)
        self.assertIsNone(result[1][2])
        sequence.step(None,3,{2:box},1/30)
        sequence.step(None,4,{},1/30)
        result = sequence.step(None,5,{1:(100,100,30,40)},1/30)
        self.assertEqual(result[1][1], box)
        self.assertEqual(sequence.births,{1:2,2:3})
        self.assertEqual(sequence.targets[1].initialized,1)
        self.assertEqual(sequence.targets[1].updated,3)

    def test_annotation_coordinates_and_selection(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'Clip_1_refined.txt'
            path.write_text('1,1,1237,159.75,13,11,1,-1,-1,-1\n3,2,20,30,4,5,0.5,-1,-1,-1\n')
            annotations = load_annotations(path)
            self.assertEqual(annotations[1][1],(1237,159.75,13,11))
            self.assertIn(2,annotations[3])
        self.assertEqual(clip_selection('3,1-2',[1,2,3]),[1,2,3])
        self.assertEqual(box_metrics((0,0,10,10),(0,0,10,10)),(1.0,0.0))
        self.assertEqual(box_metrics((0,0,10,10),(20,0,10,10)),(0.0,20.0))

    def test_partial_edge_box_is_preserved(self):
        sequence = SequenceTracker(FakeEngine())
        frame = np.zeros((100,100,3),np.uint8)
        sequence.step(frame,1,{1:(10,-30,20,10)},1/30)
        self.assertFalse(sequence.targets)
        box = (10,-8,20,20)
        result = sequence.step(frame,2,{1:box},1/30)
        self.assertEqual(result[1][1],box)
        self.assertEqual(sequence.births,{1:2})

    def test_recovery_preserves_failure_and_waits_for_patience(self):
        sequence = SequenceTracker(FakeEngine(),gt_reset=True,reset_patience=2)
        original, shifted = (10,10,10,10), (50,50,10,10)
        sequence.step(None,1,{1:original,2:original},1/30)
        sequence.step(None,2,{1:shifted,2:original},1/30)
        self.assertFalse(sequence.frame_resets)
        result = sequence.step(None,3,{1:shifted,2:original},1/30)
        self.assertEqual(result[1][1],original)  # failed box is still scored/displayed
        self.assertEqual(sequence.targets[1].bbox,shifted)  # next-frame state reset
        self.assertEqual(sequence.targets[1].initialized,2)
        self.assertEqual(sequence.targets[2].initialized,1)
        self.assertEqual(sequence.reset_events[0]['iou'],0)
        self.assertEqual(sequence.reset_events[0]['frames_since_seed'],2)
        self.assertEqual(sequence.births[1],1)
        sequence.step(None,4,{1:shifted},1/30)
        self.assertFalse(sequence.frame_resets)

    def test_center_error_and_annotation_gaps(self):
        sequence = SequenceTracker(FakeEngine(),gt_reset=True,reset_iou=0,
                                   reset_error_px=5,reset_patience=2)
        box, shifted = (0,0,20,20), (6,0,20,20)
        sequence.step(None,1,{1:box},1/30)
        sequence.step(None,2,{1:shifted},1/30)
        sequence.step(None,3,{},1/30)  # a gap breaks consecutive-failure count
        sequence.step(None,4,{1:shifted},1/30)
        self.assertFalse(sequence.reset_events)
        sequence.step(None,5,{1:shifted},1/30)
        self.assertEqual(len(sequence.reset_events),1)


if __name__ == '__main__':
    unittest.main()
