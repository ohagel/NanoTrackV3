"""NanoTrack V3 inference and tracking, ported from HonglinChu/SiamTrackers.

Algorithm/config source is pinned in models/provenance.json. No TrackerNano API.
"""
from time import perf_counter
from copy import copy
import weakref

import cv2
import numpy as np
import onnxruntime as ort


class _GraphPool:
    """Retain captured addresses and recycle slots after targets are discarded."""
    def __init__(self):
        self.slots, self.free = [], []

    def acquire(self):
        if self.free:
            return self.free.pop()
        options = ort.RunOptions()
        options.add_run_config_entry('gpu_graph_id', str(len(self.slots)))
        slot = dict(backbone={}, head=None, options=options)
        self.slots.append(slot)
        return slot

    def release(self, slot):
        self.free.append(slot)


class NanoTrackORT:
    TEMPLATE_SIZE = 127
    SEARCH_SIZE = 255
    SCORE_SIZE = 15
    STRIDE = 16
    CONTEXT = 0.5
    PENALTY_K = 0.138
    WINDOW_INFLUENCE = 0.455
    LR = 0.348

    def __init__(self, backbone_path="models/nanotrack_backbone_sim.onnx",
                 head_path="models/nanotrack_head_sim.onnx", *, providers=None,
                 threads=1, confidence_threshold=0.0, debug=False,
                 auto_update_template=False, template_fc_hz=0.0, cuda_graphs=False):
        if threads < 0 or not 0 <= confidence_threshold <= 1:
            raise ValueError("threads must be >= 0; confidence_threshold must be in [0, 1]")
        providers = ["CPUExecutionProvider"] if providers is None else list(providers)
        self.cuda_graphs = bool(cuda_graphs)
        if self.cuda_graphs:
            primary = providers[0] if providers else None
            if (primary if isinstance(primary, str) else primary[0] if primary else None) != 'CUDAExecutionProvider':
                raise ValueError('cuda_graphs requires CUDAExecutionProvider first')
            provider_options = {} if isinstance(primary, str) else dict(primary[1])
            provider_options['enable_cuda_graph'] = '1'
            providers[0] = ('CUDAExecutionProvider', provider_options)
        elif any(not isinstance(p, str) and str(p[1].get('enable_cuda_graph', '0')) == '1' for p in providers):
            raise ValueError('Use cuda_graphs=True to enable graph-safe buffer management')
        requested = set(p if isinstance(p, str) else p[0] for p in providers)
        unavailable = requested - set(ort.get_available_providers())
        if unavailable or not providers:
            hint = " Install requirements-gpu.txt after uninstalling the CPU-only onnxruntime package." if 'CUDAExecutionProvider' in unavailable else ''
            raise ValueError(f"Unavailable providers: {unavailable}; available: {ort.get_available_providers()}.{hint}")
        if 'CUDAExecutionProvider' in requested and hasattr(ort, 'preload_dlls'):
            # Load CUDA/cuDNN DLLs shipped by the NVIDIA pip packages on Windows.
            ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.backbone = ort.InferenceSession(str(backbone_path), options, providers=providers)
        self.head = ort.InferenceSession(str(head_path), options, providers=providers)
        for name, session in (("Backbone", self.backbone), ("Head", self.head)):
            active = session.get_providers()
            print(f"{name} execution providers: {active}")
            if 'CUDAExecutionProvider' in requested and 'CUDAExecutionProvider' not in active:
                raise RuntimeError(f"{name}: CUDA failed to initialize; refusing silent CPU fallback. "
                                   "Check the CUDA/cuDNN DLL error above and requirements-gpu.txt.")
        self.feature_device, self.device_id = self._select_feature_device(self.backbone, self.head)
        # CUDA convolution setup is expensive when a session alternates spatial
        # sizes. Keep one persistent session per crop size, shared by all targets.
        self.template_backbone = self.backbone
        if self.feature_device == 'cuda':
            self.template_backbone = ort.InferenceSession(str(backbone_path), options, providers=providers)
            self._select_feature_device(self.template_backbone, self.backbone)
            print(f"Template backbone execution providers: {self.template_backbone.get_providers()}")
        self._backbone_buffers = {}
        self._head_buffers = None
        self._graph_pool = _GraphPool() if self.cuda_graphs else None
        self._graph_slot = None
        if self.cuda_graphs:
            self._attach_graph_slot()
            print('CUDA Graphs: enabled (reusable per-target capture slots)')
        print(f"Feature transport: {self.feature_device}:{self.device_id} "
              f"({'resident I/O bindings' if self.feature_device != 'cpu' else 'host NumPy'})")
        self.debug = debug
        self.auto_update_template = auto_update_template
        self.template_fc_hz = template_fc_hz
        self.template_pixels = None
        self._last_update_time = self._last_template_time = None
        self.confidence_threshold = confidence_threshold
        self.template_image = self.search_image = None
        self.template_features = None
        self.last_tracking_ms = 0.0
        self.confidence = 0.0
        for label, session in (("Backbone", self.backbone), ("Head", self.head)):
            print(f"{label}:")
            for kind, values in (("inputs", session.get_inputs()), ("outputs", session.get_outputs())):
                print(f"  {kind}")
                for value in values:
                    print(f"    {value.name}: {value.shape} {value.type}")
        self._validate_interfaces()
        axis = (np.arange(self.SCORE_SIZE) - self.SCORE_SIZE // 2) * self.STRIDE
        x, y = np.meshgrid(axis, axis)
        self.points = np.stack((x.ravel(), y.ravel()), axis=1).astype(np.float32)
        self.window = np.outer(np.hanning(self.SCORE_SIZE), np.hanning(self.SCORE_SIZE)).ravel()

    def new_target(self):
        """Create uninitialized target state sharing these validated ORT sessions.

        Templates, geometry and debug crops remain independent. Use init() on the
        returned tracker. Sessions and read-only grid/window arrays are shared.
        """
        target = copy(self)
        # Sessions are shared; writable device buffers/bindings belong to each target.
        target._backbone_buffers = {}
        target._head_buffers = None
        if self.cuda_graphs:
            target._attach_graph_slot()
        target.template_features = None
        target.template_pixels = None
        target._last_update_time = target._last_template_time = None
        target.template_image = target.search_image = None
        target.confidence = target.last_tracking_ms = 0.0
        for name in ("center_pos", "size", "channel_average"):
            target.__dict__.pop(name, None)
        return target

    def _attach_graph_slot(self):
        self._graph_slot = self._graph_pool.acquire()
        self._backbone_buffers = self._graph_slot['backbone']
        self._head_buffers = self._graph_slot['head']
        # Neither the slot nor this callback owns the target itself.
        weakref.finalize(self, self._graph_pool.release, self._graph_slot)

    @staticmethod
    def _select_feature_device(backbone, head):
        """Honor active provider precedence and configured device IDs.

        CUDA/TensorRT use public CUDA OrtValues. Other EPs retain ORT-managed
        host transfers; do not incorrectly label DirectML/OpenVINO memory CUDA.
        """
        devices = []
        for session in (backbone, head):
            primary = session.get_providers()[0]
            if primary in ('CUDAExecutionProvider', 'TensorrtExecutionProvider'):
                options = session.get_provider_options().get(primary, {})
                devices.append(('cuda', int(options.get('device_id', 0))))
            else:
                devices.append(('cpu', 0))
        if devices[0] != devices[1]:
            raise ValueError(f"Backbone/head must use the same feature device: {devices}")
        return devices[0]

    @property
    def template_fc_hz(self):
        return self._template_fc_hz

    @template_fc_hz.setter
    def template_fc_hz(self, value):
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError("template_fc_hz must be finite and >= 0 (0 bypasses filtering)")
        self._template_fc_hz = value

    def _validate_interfaces(self):
        bi, bo = self.backbone.get_inputs(), self.backbone.get_outputs()
        hi, ho = self.head.get_inputs(), self.head.get_outputs()
        def matches(value, shape):
            return value.type == "tensor(float)" and len(value.shape) == len(shape) and all(
                expected is None or actual == expected for actual, expected in zip(value.shape, shape))
        if len(bi) != 1 or len(bo) != 1 or not matches(bi[0], [1, 3, None, None]) or not matches(bo[0], [1, 96, None, None]):
            raise ValueError("Expected V3 float32 NCHW backbone with 96 output channels")
        if any(isinstance(d, int) for d in bi[0].shape[2:]):
            raise ValueError("Backbone must accept BOTH 127 and 255 spatial sizes. Run prepare_models.py.")
        if len(hi) != 2 or len(ho) != 2:
            raise ValueError("Expected two head inputs and two outputs")
        def resolve(values, shape):
            found = [v.name for v in values if matches(v, shape)]
            if len(found) != 1:
                raise ValueError(f"Missing/ambiguous V3 tensor {shape}")
            return found[0]
        self.z_name = resolve(hi, [1, 96, 8, 8])
        self.x_name = resolve(hi, [1, 96, 16, 16])
        self.cls_name = resolve(ho, [1, 2, 15, 15])
        self.loc_name = resolve(ho, [1, 4, 15, 15])
        self.input_name, self.output_name = bi[0].name, bo[0].name
        features = []
        for side, expected in ((127, 8), (255, 16)):
            result = self._features(np.zeros((1, 3, side, side), np.float32))
            # A one-time startup readback validates actual tensors; updates never
            # copy backbone features to the host on the device-resident path.
            observed = result.numpy() if isinstance(result, ort.OrtValue) else result
            if observed.shape != (1, 96, expected, expected) or not np.isfinite(observed).all():
                raise ValueError(f"Invalid runtime backbone output for {side}: {observed.shape}")
            features.append(result)
        cls, loc = self._run_head(*features)
        if cls.shape != (1, 2, 15, 15) or loc.shape != (1, 4, 15, 15) or not np.isfinite(cls).all() or not np.isfinite(loc).all() or np.any(loc <= 0):
            raise ValueError("Head runtime outputs do not match V3 logits and positive LTRB distances")

    def _features(self, tensor):
        if self.feature_device == 'cpu':
            return self.backbone.run([self.output_name], {self.input_name: tensor})[0]
        shape = tuple(tensor.shape)
        if shape not in ((1,3,127,127), (1,3,255,255)):
            raise ValueError(f"Unsupported V3 backbone input shape: {shape}")
        session = self.template_backbone if shape[-1] == self.TEMPLATE_SIZE else self.backbone
        if shape not in self._backbone_buffers:
            side = 8 if shape[-1] == 127 else 16
            device_input = ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float32, self.feature_device, self.device_id)
            features = ort.OrtValue.ortvalue_from_shape_and_type((1,96,side,side), np.float32, self.feature_device, self.device_id)
            binding = session.io_binding()
            binding.bind_ortvalue_input(self.input_name, device_input)
            binding.bind_ortvalue_output(self.output_name, features)
            self._backbone_buffers[shape] = (device_input, features, binding)
        device_input, features, binding = self._backbone_buffers[shape]
        device_input.update_inplace(tensor)  # One crop upload; stable device allocation.
        binding.synchronize_inputs()
        if self.cuda_graphs:
            session.run_with_iobinding(binding, self._graph_slot['options'])
        else:
            session.run_with_iobinding(binding)
        binding.synchronize_outputs()  # Head may execute on a different session stream.
        return features

    def _run_head(self, template, search):
        if self.feature_device == 'cpu':
            return self.head.run([self.cls_name, self.loc_name],
                                 {self.z_name: template, self.x_name: search})
        if self._head_buffers is None:
            binding = self.head.io_binding()
            # Only final small logits/distances are needed on CPU for postprocessing.
            arrays = [np.empty((1,2,15,15), np.float32), np.empty((1,4,15,15), np.float32)]
            host_values = ([ort.OrtValue.ortvalue_from_shape_and_type(a.shape, np.float32, 'cuda', self.device_id)
                            for a in arrays] if self.cuda_graphs else
                           [ort.OrtValue.ortvalue_from_numpy(a) for a in arrays])
            for name, value in zip((self.cls_name, self.loc_name), host_values):
                binding.bind_ortvalue_output(name, value)
            self._head_buffers = binding, arrays, host_values
            if self.cuda_graphs:
                self._graph_slot['head'] = self._head_buffers
        binding, arrays, values = self._head_buffers
        binding.bind_ortvalue_input(self.z_name, template)
        binding.bind_ortvalue_input(self.x_name, search)
        binding.synchronize_inputs()
        if self.cuda_graphs:
            expected = (self._backbone_buffers[(1,3,127,127)][1], self._backbone_buffers[(1,3,255,255)][1])
            if (template.data_ptr(), search.data_ptr()) != tuple(v.data_ptr() for v in expected):
                raise ValueError('CUDA Graph head inputs must use this target\'s fixed feature buffers')
            self.head.run_with_iobinding(binding, self._graph_slot['options'])
        else:
            self.head.run_with_iobinding(binding)
        binding.synchronize_outputs()
        if self.cuda_graphs:
            for array, value in zip(arrays, values):
                np.copyto(array, value.numpy())
        return arrays  # Per-target reusable arrays; consumed before the next update.

    @staticmethod
    def _validate_frame(frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 10:
            raise ValueError("frame must be a uint8 BGR image at least 10x10")

    @staticmethod
    def _crop(frame, center, model_size, original_size, average):
        # Same rounding and uint8 mean-padding as upstream get_subwindow.
        side = max(1, int(original_size))
        x, y = np.floor(center - (side + 1) / 2 + 0.5).astype(int)
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + side), min(h, y + side)
        if x >= 0 and y >= 0 and x + side <= w and y + side <= h:
            patch = frame[y:y + side, x:x + side]
        else:
            # Allocate just the crop, rather than a padded copy of the whole frame.
            patch = np.empty((side, side, 3), np.uint8)
            patch[:] = average
            if x1 > x0 and y1 > y0:
                patch[y0-y:y1-y, x0-x:x1-x] = frame[y0:y1, x0:x1]
        if side != model_size:
            patch = cv2.resize(patch, (model_size, model_size), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(patch.transpose(2, 0, 1)[None], dtype=np.float32)
        return tensor, patch

    def init(self, frame, bbox):
        """Select a positive-area xywh ROI overlapping the image; compute template once."""
        self._validate_frame(frame)
        box = np.asarray(bbox, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("bbox must contain four finite xywh coordinates")
        x, y, w, h = box
        # Predicted upstream boxes may extend past the image; allow overlap for reseed.
        if w < 1 or h < 1 or w > frame.shape[1] or h > frame.shape[0] or x+w <= 0 or y+h <= 0 or x >= frame.shape[1] or y >= frame.shape[0]:
            raise ValueError("bbox must overlap the image and have valid positive dimensions")
        center = np.array([x + (w-1)/2, y + (h-1)/2])
        size = np.array([w, h])
        average = np.asarray(cv2.mean(frame)[:3])
        side = round(np.sqrt(np.prod(size + self.CONTEXT * size.sum())))
        crop, image = self._crop(frame, center, self.TEMPLATE_SIZE, side, average)
        features = self._features(crop)
        self.center_pos, self.size, self.channel_average = center, size, average
        self.template_features = features
        self.template_pixels = crop
        self._last_update_time = self._last_template_time = perf_counter()
        self.template_image = image if self.debug else None
        self.search_image = None
        self.confidence = 0.0

    def reseed(self, frame, bbox):
        """Reset target geometry and template; keep both inference sessions alive."""
        self.init(frame, bbox)

    def refresh_template(self, frame, *, dt=None):
        """Refresh pixels at current geometry, optionally low-pass filtering in time.

        dt is seconds between samples; omitted means monotonic wall time.
        Filtering is per BGR pixel on resized 127x127 crops, before the backbone.
        """
        self._validate_frame(frame)
        if self.template_features is None:
            raise RuntimeError("Call init() before refreshing the template")
        now = perf_counter()
        dt = now - self._last_template_time if dt is None else float(dt)
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and >= 0 seconds")
        average = np.asarray(cv2.mean(frame)[:3])
        side = round(np.sqrt(np.prod(self.size + self.CONTEXT * self.size.sum())))
        crop, image = self._crop(frame, self.center_pos, self.TEMPLATE_SIZE, side, average)
        if self.template_fc_hz > 0:
            alpha = float(-np.expm1(-2 * np.pi * self.template_fc_hz * dt))
            crop = self.template_pixels + np.float32(alpha) * (crop - self.template_pixels)
            if self.debug:
                image = np.rint(crop[0].transpose(1, 2, 0)).clip(0, 255).astype(np.uint8)
        features = self._features(crop)
        self.template_pixels = crop
        self._last_template_time = now
        self.template_features = features
        self.channel_average = average
        self.template_image = image if self.debug else None

    @property
    def bbox(self):
        if self.template_features is None:
            raise RuntimeError("Call init() first")
        return tuple(float(v) for v in np.concatenate((self.center_pos - self.size / 2, self.size)))

    def update(self, frame, *, dt=None):
        """Return (success, xywh, raw foreground probability at chosen candidate).

        Upstream has no lost-target test. Default success means a finite prediction;
        an optional confidence threshold flags low scores without changing tracking.
        """
        self._validate_frame(frame)
        if self.template_features is None:
            raise RuntimeError("Call init() before update()")
        start = perf_counter()
        dt = start - self._last_update_time if dt is None else float(dt)
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and >= 0 seconds")
        self._last_update_time = start
        side = np.sqrt(np.prod(self.size + self.CONTEXT * self.size.sum()))
        scale = self.TEMPLATE_SIZE / side
        crop, image = self._crop(frame, self.center_pos, self.SEARCH_SIZE,
                                 round(side * self.SEARCH_SIZE / self.TEMPLATE_SIZE), self.channel_average)
        if self.debug:
            self.search_image = image
        search = self._features(crop)
        cls, loc = self._run_head(self.template_features, search)
        if not np.isfinite(cls).all() or not np.isfinite(loc).all() or np.any(loc <= 0):
            self.last_tracking_ms = (perf_counter() - start) * 1000
            self.confidence = 0.0
            return False, self.bbox, 0.0
        logits = cls[0].reshape(2, -1)
        exps = np.exp(logits - logits.max(axis=0))
        scores = exps[1] / exps.sum(axis=0)
        left, top, right, bottom = loc[0].reshape(4, -1)
        x1, y1 = self.points[:, 0] - left, self.points[:, 1] - top
        x2, y2 = self.points[:, 0] + right, self.points[:, 1] + bottom
        boxes = np.stack(((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1))

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w+h) * 0.5
            return np.sqrt((w+pad) * (h+pad))

        scale_penalty = change(sz(boxes[2], boxes[3]) / sz(*(self.size * scale)))
        ratio_penalty = change((self.size[0]/self.size[1]) / (boxes[2]/boxes[3]))
        penalty = np.exp(-(ratio_penalty * scale_penalty - 1) * self.PENALTY_K)
        ranked = penalty * scores * (1-self.WINDOW_INFLUENCE) + self.window * self.WINDOW_INFLUENCE
        best = int(np.argmax(ranked))
        selected = boxes[:, best] / scale
        lr = penalty[best] * scores[best] * self.LR
        center = self.center_pos + selected[:2]
        size = self.size * (1-lr) + selected[2:] * lr
        # Faithful upstream clipping: center and size, NOT four corners.
        boundary = np.array(frame.shape[1::-1])
        self.center_pos = np.clip(center, 0, boundary)
        self.size = np.maximum(10, np.minimum(size, boundary))
        self.confidence = float(scores[best])
        if self.auto_update_template:
            # Use this frame's predicted location for the NEXT frame's search.
            # Deliberately no confidence gating, blending or update interval.
            self.refresh_template(frame, dt=dt)
        self.last_tracking_ms = (perf_counter() - start) * 1000
        return self.confidence >= self.confidence_threshold, self.bbox, self.confidence
