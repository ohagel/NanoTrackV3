"""Reproducible CPU validation against upstream source and local video.

Uses a tiny NumPy tensor adapter to run the original tracker without PyTorch.
It is only a test harness, never part of application inference.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from nanotrack import NanoTrackORT

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "validation_output"


class Tensor:
    def __init__(self, a): self.a = a
    def permute(self, *axes): return Tensor(self.a.transpose(axes))
    def contiguous(self): return Tensor(np.ascontiguousarray(self.a))
    def view(self, *shape): return Tensor(self.a.reshape(shape))
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a
    def __getitem__(self, key): return Tensor(self.a[key])
    def softmax(self, axis):
        a = np.exp(self.a - self.a.max(axis=axis, keepdims=True))
        return Tensor(a / a.sum(axis=axis, keepdims=True))


def original_tracker(tracker):
    cfg = SimpleNamespace(CUDA=False, POINT=SimpleNamespace(STRIDE=16), TRACK=SimpleNamespace(
        OUTPUT_SIZE=15, CONTEXT_AMOUNT=.5, EXEMPLAR_SIZE=127, INSTANCE_SIZE=255,
        PENALTY_K=.138, WINDOW_INFLUENCE=.455, LR=.348))
    def corner2center(box):
        x1, y1, x2, y2 = box
        return (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
    env = dict(np=np, cv2=cv2, cfg=cfg, torch=SimpleNamespace(from_numpy=Tensor), corner2center=corner2center)
    for name in ("nanotrack_tracker_base_tracker.py", "nanotrack_tracker_nano_tracker.py"):
        tree = ast.parse((ROOT / "reference" / name).read_text(encoding="utf-8-sig"))
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, name, "exec"), env)
    class Model:
        def eval(self): pass
        def template(self, crop): self.z = tracker._features(crop.a)
        def track(self, crop):
            cls, loc = tracker.head.run([tracker.cls_name, tracker.loc_name],
                {tracker.z_name: self.z, tracker.x_name: tracker._features(crop.a)})
            return dict(cls=Tensor(cls), loc=Tensor(loc))
    return env["NanoTracker"](Model())


def iou(a, b):
    a, b = np.array(a), np.array(b)
    size = np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = np.prod(size)
    return float(intersection / (np.prod(a[2:])+np.prod(b[2:])-intersection))


def main():
    OUT.mkdir(exist_ok=True)
    tracker = NanoTrackORT(debug=True)
    reference = original_tracker(tracker)
    cap = cv2.VideoCapture(str(ROOT / "reference/girl_dance.mp4"))
    ok, first = cap.read()
    assert ok
    initial = (440, 158, 197, 310)
    # Exact crop checks include padding, small sizes, fractional centers, corners.
    crops = 0
    for center in ((0., 0.), (855., 479.), (320.3, 212.7), (40., 80.)):
        for side in (17, 127, 255, 701):
            for model_size in (127, 255):
                actual, _ = tracker._crop(first, np.array(center), model_size, side, first.mean((0, 1)))
                expected = reference.get_subwindow(first, np.array(center), model_size, side, first.mean((0, 1))).a
                np.testing.assert_array_equal(actual, expected)
                crops += 1
    tracker.init(first, initial)
    reference.init(first, initial)
    errors, confidence, times, snapshots = [], [], [], []
    writer = cv2.VideoWriter(str(OUT / "dance_tracked.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (856, 480))
    assert writer.isOpened()
    for index in range(300):
        ok, frame = cap.read()
        if not ok: break
        success, box, score = tracker.update(frame)
        original = reference.track(frame)
        error = float(np.max(np.abs(np.array(box) - original["bbox"])))
        errors.append(error)
        np.testing.assert_allclose(box, original["bbox"], rtol=0, atol=2e-3)
        np.testing.assert_allclose(score, original["best_score"], atol=1e-5)
        assert success and 0 <= score <= 1
        confidence.append(score)
        times.append(tracker.last_tracking_ms)
        x,y,w,h = box
        cv2.rectangle(frame, (round(x),round(y)), (round(x+w),round(y+h)), (0,255,0), 2)
        cv2.putText(frame, f"frame {index+1} confidence {score:.3f}", (12,30), 0, .7, (0,255,255), 2)
        writer.write(frame)
        if index in (0,49,99,149,199,249,299):
            snapshots.append(cv2.resize(frame, (428,240)))
    writer.release()
    cap.release()
    while len(snapshots) % 2: snapshots.append(np.zeros_like(snapshots[0]))
    cv2.imwrite(str(OUT / "dance_contact_sheet.jpg"), np.vstack([np.hstack(snapshots[i:i+2]) for i in range(0,len(snapshots),2)]))
    # Reseed and normal update must never construct new sessions.
    ids = (id(tracker.backbone), id(tracker.head))
    old_template = tracker.template_features.copy()
    with patch("nanotrack.ort.InferenceSession", side_effect=AssertionError("Session recreated")):
        with patch.object(tracker, "_features", wraps=tracker._features) as infer:
            tracker.reseed(first, initial)
            tracker.update(first)
            assert [call.args[0].shape for call in infer.call_args_list] == [(1,3,127,127), (1,3,255,255)]
    assert ids == (id(tracker.backbone), id(tracker.head))
    # Same frame/ROI gives the same template; feature storage is replaced by reseed.
    np.testing.assert_array_equal(old_template, tracker.template_features)
    # Moving textured target with known xywh at multiple positions/scales.
    target = first[165:465, 445:630]
    results = []
    for origin, scale in (((20,20), .25), ((250,150), .45), ((500,260), .60)):
        frames, truth = [], []
        for index in range(70):
            factor = scale * (1 + .15*np.sin(index/20))
            w,h = round(185*factor), round(300*factor)
            x,y = origin[0]+index, origin[1]+round(8*np.sin(index/12))
            frame = np.full((480,856,3), 70, np.uint8)
            frame[y:y+h,x:x+w] = cv2.resize(target,(w,h))
            frames.append(frame)
            truth.append((x,y,w,h))
        tracker.init(frames[0], truth[0])
        overlaps = []
        for frame, box in zip(frames[1:], truth[1:]):
            _, predicted, _ = tracker.update(frame)
            overlaps.append(iou(predicted, box))
        assert np.mean(overlaps) > .65, overlaps
        present = tracker.confidence
        absent = [tracker.update(np.full_like(frame, 70))[2] for _ in range(10)]
        results.append(dict(origin=origin, scale=scale, mean_iou=float(np.mean(overlaps)),
                            min_iou=min(overlaps), present_confidence=present, absent_mean_confidence=float(np.mean(absent))))
    report = dict(frames=len(times), exact_crop_cases=crops, max_upstream_bbox_error_px=max(errors),
        confidence_min=min(confidence), confidence_mean=float(np.mean(confidence)), confidence_max=max(confidence),
        tracking_ms=float(np.mean(times[10:])), tracking_fps=1000/float(np.mean(times[10:])),
        reseed_sessions_preserved=True, controlled_motion=results,
        versions=dict(numpy=np.__version__, opencv=cv2.__version__))
    (OUT / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
