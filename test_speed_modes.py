import gc
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import onnxruntime as ort

from nanotrack import NanoTrackORT
from video_reader import VideoReader
from evaluate_purdue import SequenceTracker
from test_purdue import FakeEngine


class SpeedModeTests(unittest.TestCase):
    def test_annotation_gap_preserves_state_and_elapsed_time(self):
        seq = SequenceTracker(FakeEngine(), annotated_only=True)
        box = (10, 20, 30, 40)
        seq.step(None, 1, {1: box}, .1)
        self.assertEqual(seq.step(None, 2, {}, .1), {})
        self.assertEqual(seq.step(None, 3, {}, .1), {})
        target = seq.targets[1]
        with patch.object(target, 'update', wraps=target.update) as update:
            result = seq.step(None, 4, {1: (40, 50, 30, 40)}, .1)
            self.assertAlmostEqual(update.call_args.kwargs['dt'], .3)
        self.assertEqual(result[1][1], box)
        self.assertEqual(target.initialized, 1)

    def test_prefetch_order_eof_and_early_close(self):
        with TemporaryDirectory() as folder:
            path = str(Path(folder) / 'sample.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(12):
                writer.write(np.full((48, 64, 3), i * 15, np.uint8))
            writer.release()
            sequences = []
            for prefetch in (0, 2):
                reader = VideoReader(path, prefetch)
                frames = []
                timestamps = []
                try:
                    while True:
                        ok, frame = reader.read()
                        if not ok:
                            break
                        frames.append(frame)
                        timestamps.append(reader.get(cv2.CAP_PROP_POS_MSEC))
                    self.assertFalse(reader.read()[0])
                finally:
                    reader.release()
                sequences.append((frames, timestamps))
            self.assertEqual(len(sequences[0][0]), 12)
            self.assertEqual(sequences[0][1], sequences[1][1])
            for a, b in zip(sequences[0][0], sequences[1][0]):
                np.testing.assert_array_equal(a, b)
            reader = VideoReader(path, 1)
            reader.read()
            reader.release()
            self.assertFalse(reader.worker.is_alive())

    @unittest.skipUnless('CUDAExecutionProvider' in ort.get_available_providers(), 'CUDA required')
    def test_graph_replay_reseed_isolation_and_slot_reuse(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        engine = NanoTrackORT(providers=providers, cuda_graphs=True,
                              auto_update_template=True, template_fc_hz=.1)
        plain = NanoTrackORT(providers=providers, auto_update_template=True, template_fc_hz=.1)
        a, b = engine.new_target(), engine.new_target()
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        box = (100, 70, 40, 35)
        with patch('nanotrack.ort.InferenceSession', side_effect=AssertionError('New session')):
            a.init(frame, box)
            b.init(frame, (200, 140, 30, 30))
            plain.init(frame, box)
            saved = b.template_features.numpy().copy()
            for i in range(8):
                moved = np.roll(frame, i, axis=1)
                if i == 4:
                    a.reseed(moved, box)
                    plain.reseed(moved, box)
                actual = a.update(moved, dt=1/30)
                expected = plain.update(moved, dt=1/30)
                np.testing.assert_allclose(actual[1], expected[1], atol=1e-4, rtol=1e-5)
                self.assertAlmostEqual(actual[2], expected[2], places=5)
            np.testing.assert_array_equal(saved, b.template_features.numpy())
            slot = a._graph_slot
            del a
            gc.collect()
            replacement = engine.new_target()
            self.assertIs(replacement._graph_slot, slot)
            replacement.init(frame, box)
            plain.init(frame, box)
            np.testing.assert_allclose(replacement.update(frame, dt=1/30)[1],
                                       plain.update(frame, dt=1/30)[1], atol=1e-4, rtol=1e-5)
            self.assertEqual(len(engine._graph_pool.slots), 3)


if __name__ == '__main__':
    unittest.main()
