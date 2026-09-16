"""Transfer routing and real CUDA binding regression checks."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import onnxruntime as ort

from nanotrack import NanoTrackORT


class DeviceRoutingTests(unittest.TestCase):
    def test_precedence_device_id_and_mismatch(self):
        def session(providers, options=None):
            return Mock(get_providers=lambda: providers,
                        get_provider_options=lambda: options or {})
        cpu = session(['CPUExecutionProvider', 'CUDAExecutionProvider'])
        cuda = session(['CUDAExecutionProvider', 'CPUExecutionProvider'],
                       {'CUDAExecutionProvider': {'device_id': '2'}})
        self.assertEqual(NanoTrackORT._select_feature_device(cpu, cpu), ('cpu', 0))
        self.assertEqual(NanoTrackORT._select_feature_device(cuda, cuda), ('cuda', 2))
        other = session(['DmlExecutionProvider', 'CPUExecutionProvider'])
        self.assertEqual(NanoTrackORT._select_feature_device(other, other), ('cpu', 0))
        with self.assertRaises(ValueError):
            NanoTrackORT._select_feature_device(cpu, cuda)


@unittest.skipUnless('CUDAExecutionProvider' in ort.get_available_providers(),
                     'Requires CUDA-enabled ORT and working NVIDIA runtime')
class CudaTransferTests(unittest.TestCase):
    def test_parity_buffer_reuse_and_target_isolation(self):
        engine = NanoTrackORT(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.assertIsNot(engine.backbone, engine.template_backbone)
        rng = np.random.default_rng(42)
        features = []
        for side in (127, 255):
            pixels = rng.uniform(0, 255, (1, 3, side, side)).astype(np.float32)
            session = engine.template_backbone if side == 127 else engine.backbone
            expected = session.run([engine.output_name], {engine.input_name: pixels})[0]
            actual = engine._features(pixels)
            self.assertEqual(actual.device_name(), 'cuda')
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-5)
            features.append(actual)
        expected = engine.head.run([engine.cls_name, engine.loc_name],
                                   {engine.z_name: features[0].numpy(),
                                    engine.x_name: features[1].numpy()})
        for actual, reference in zip(engine._run_head(*features), expected):
            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)

        frame = rng.integers(0, 256, (400, 600, 3), dtype=np.uint8)
        first, second = engine.new_target(), engine.new_target()
        with patch('nanotrack.ort.InferenceSession', side_effect=AssertionError('New session')), \
             patch.object(engine.backbone, 'run', side_effect=AssertionError('Host feature path')), \
             patch.object(engine.template_backbone, 'run', side_effect=AssertionError('Host feature path')), \
             patch.object(engine.backbone, 'run_with_iobinding', wraps=engine.backbone.run_with_iobinding) as search_run, \
             patch.object(engine.template_backbone, 'run_with_iobinding', wraps=engine.template_backbone.run_with_iobinding) as template_run, \
             patch.object(engine.head, 'run', side_effect=AssertionError('Host feature path')):
            first.init(frame, (100, 100, 60, 40))
            second.init(frame, (300, 150, 80, 60))
            saved = second.template_features.numpy().copy()
            pointer = first.template_features.data_ptr()
            self.assertNotEqual(pointer, second.template_features.data_ptr())
            first.auto_update_template = True
            first.template_fc_hz = 0.1
            for _ in range(3):
                success, bbox, confidence = first.update(frame)
                self.assertTrue(success)
                self.assertTrue(np.isfinite(bbox).all())
                self.assertTrue(0 <= confidence <= 1)
            buffers = {k: (v[0].data_ptr(), v[1].data_ptr())
                       for k, v in first._backbone_buffers.items()}
            first.reseed(frame, (130, 100, 60, 40))
            first.update(frame)
            self.assertEqual(pointer, first.template_features.data_ptr())
            self.assertEqual(buffers, {k: (v[0].data_ptr(), v[1].data_ptr())
                                       for k, v in first._backbone_buffers.items()})
            np.testing.assert_array_equal(saved, second.template_features.numpy())
            self.assertIs(first.backbone, second.backbone)
            self.assertIs(first.template_backbone, second.template_backbone)
            self.assertIs(first.head, second.head)
            self.assertEqual(search_run.call_count, 4)
            self.assertEqual(template_run.call_count, 7)


if __name__ == '__main__':
    unittest.main()
