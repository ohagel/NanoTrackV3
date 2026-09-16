"""Minimal webcam/video ROI tracking application."""
import argparse
from collections import deque
from datetime import datetime, timezone
import math
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from nanotrack import NanoTrackORT
from recording import Recorder
from video_reader import VideoReader


class LiveROI:
    """Mouse-only selection state; the main loop keeps reading/displaying frames."""

    def __init__(self):
        self.active = False
        self.anchor = self.cursor = self.pending = None

    def begin(self):
        self.active = True
        self.anchor = self.cursor = self.pending = None

    def cancel(self):
        self.active = False
        self.anchor = self.cursor = self.pending = None

    def mouse(self, event, x, y, flags, param):
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.anchor = self.cursor = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.anchor is not None:
            self.cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.anchor is not None:
            self.cursor = (x, y)
            self.pending = (self.anchor, self.cursor)
            self.anchor = None

    def take(self, frame):
        if self.pending is None:
            return None
        (x0, y0), (x1, y1) = self.pending
        self.pending = None
        h, w = frame.shape[:2]
        left, right = sorted((max(0, min(x0, w)), max(0, min(x1, w))))
        top, bottom = sorted((max(0, min(y0, h)), max(0, min(y1, h))))
        if right - left < 1 or bottom - top < 1:
            return None
        self.cancel()
        return left, top, right - left, bottom - top

    def draw(self, image, label="Add target"):
        if self.active:
            cv2.putText(image, f"{label}: drag and release | Esc cancels", (12, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if self.anchor is not None:
                cv2.rectangle(image, self.anchor, self.cursor, (0, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index or video path")
    models = Path(__file__).resolve().parent / "models"
    parser.add_argument("--backbone", default=str(models / "nanotrack_backbone_sim.onnx"))
    parser.add_argument("--head", default=str(models / "nanotrack_head_sim.onnx"))
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    parser.add_argument("--cuda-graphs", action="store_true", help="GPU graph replay; requires CUDA first")
    parser.add_argument("--threads", type=int, default=1, help="ORT threads; 0 = runtime default")
    parser.add_argument("--prefetch", type=int, default=2, help="file decode queue; 0 disables; cameras stay live")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--auto-template", action="store_true", help="replace every target template after each prediction")
    parser.add_argument("--template-fc", type=float, default=2.0, help="template temporal low-pass cutoff in Hz; 0 bypasses (default 2)")
    parser.add_argument("--bbox", nargs=4, type=float, action="append", metavar=("X", "Y", "W", "H"), help="initial ROI; repeat for multiple targets")
    parser.add_argument("--headless", action="store_true", help="benchmark without windows; requires --bbox")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--trail-length", type=int, default=60, help="center positions per target; 0 disables trails")
    parser.add_argument("--record", action="store_true", help="start recording immediately (also works headless)")
    parser.add_argument("--record-dir", default="recordings", help="parent directory for recording sessions")
    parser.add_argument("--record-fps", type=float, help="video playback FPS; defaults to source FPS or 30")
    args = parser.parse_args()
    if args.prefetch < 0:
        parser.error('--prefetch must be nonnegative')
    if args.headless and args.bbox is None:
        parser.error("--headless requires --bbox")
    if args.trail_length < 0:
        parser.error("--trail-length must be >= 0")
    if not math.isfinite(args.template_fc) or args.template_fc < 0:
        parser.error("--template-fc must be finite and >= 0")
    if args.record_fps is not None and (not math.isfinite(args.record_fps) or args.record_fps <= 0):
        parser.error("--record-fps must be finite and > 0")
    engine = NanoTrackORT(args.backbone, args.head, providers=args.providers,
                           threads=args.threads, debug=args.debug, cuda_graphs=args.cuda_graphs)
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = VideoReader(source, args.prefetch) if isinstance(source, str) else cv2.VideoCapture(source)
    count, tracked, total_tracking, total_loop = 0, 0, 0.0, 0.0
    window = "NanoTrack V3 | a add | Tab next | r select | t reseed | u auto-template | d remove | v record | q quit"
    selection = LiveROI()
    targets = {}
    trails = {}
    recorder = None
    show_trails = True
    auto_template = args.auto_template
    template_fc = args.template_fc
    run_start = perf_counter()
    record_fps = args.record_fps or capture.get(cv2.CAP_PROP_FPS)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    video_dt = 1 / source_fps if isinstance(source, str) and math.isfinite(source_fps) and source_fps > 0 else None
    previous_capture_elapsed = None
    selected = None
    next_id = 1
    replace_id = None
    colors = ((0, 255, 0), (255, 180, 0), (0, 140, 255), (255, 0, 255), (0, 255, 255), (255, 160, 160))

    def add(frame, roi):
        nonlocal next_id, selected
        target = engine.new_target()
        target.init(frame, roi)
        targets[next_id] = target
        trails[next_id] = deque(maxlen=args.trail_length)
        selected = next_id
        next_id += 1

    def mouse(event, x, y, flags, param):
        nonlocal selected
        if selection.active:
            selection.mouse(event, x, y, flags, param)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Smallest containing box wins when targets overlap; Tab cycles all.
            hits = []
            for identity, target in targets.items():
                bx, by, bw, bh = target.bbox
                if bx <= x <= bx+bw and by <= y <= by+bh:
                    hits.append((bw*bh, identity))
            if hits:
                selected = min(hits)[1]

    try:
        ok, frame = capture.read()
        captured_utc = datetime.now(timezone.utc).isoformat()
        captured_elapsed = perf_counter() - run_start
        if not ok:
            raise RuntimeError(f"Cannot read source: {args.source}")
        if args.bbox:
            for roi in args.bbox:
                add(frame, roi)
        else:
            selection.begin()
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window, mouse)
        if args.record:
            recorder = Recorder(args.record_dir, frame.shape, record_fps, args.source)
        fps = 0.0
        while True:
            start = perf_counter()
            predictions = {}
            frame_dt = video_dt if video_dt is not None else (0.0 if previous_capture_elapsed is None else captured_elapsed - previous_capture_elapsed)
            previous_capture_elapsed = captured_elapsed
            frame_tracking_ms = 0.0
            for identity, target in targets.items():
                target.auto_update_template = auto_template
                target.template_fc_hz = template_fc
                predictions[identity] = target.update(frame, dt=frame_dt)
                success, box, _ = predictions[identity]
                if success:
                    trails[identity].append((round(box[0]+box[2]/2), round(box[1]+box[3]/2)))
                else:
                    trails[identity].clear()
                frame_tracking_ms += target.last_tracking_ms
            if targets:
                total_tracking += frame_tracking_ms
                tracked += 1
            count += 1
            if not args.headless or recorder is not None:
                display = frame.copy()  # Keep annotations out of r/t template crops.
                status = f"{len(targets)} targets | selected {selected or '-'} | {fps:.1f} FPS | {frame_tracking_ms:.1f} ms"
                for identity, (success, bbox, confidence) in predictions.items():
                    x, y, w, h = bbox
                    color = colors[(identity-1) % len(colors)] if success else (0, 0, 255)
                    if show_trails and len(trails[identity]) > 1:
                        cv2.polylines(display, [np.asarray(trails[identity], dtype=np.int32)],
                                      False, color, 2, cv2.LINE_AA)
                    cv2.rectangle(display, (round(x), round(y)), (round(x+w), round(y+h)),
                                  color, 3 if identity == selected else 1)
                    cv2.putText(display, f"{'*' if identity == selected else ''}ID {identity}: {confidence:.3f}",
                                (max(0, round(x)), max(75, min(frame.shape[0]-5, round(y)-8))),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
                cv2.putText(display, status,
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                filter_label = f"{template_fc:g} Hz" if template_fc else "bypass"
                cv2.putText(display, f"Auto-template: {'ON' if auto_template else 'OFF'} [u] | Fc: {filter_label} [ / ]",
                            (12, display.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (0, 165, 255) if auto_template else (220, 220, 220), 2)
                selection.draw(display, f"Replace ID {replace_id}" if replace_id is not None else "Add target")
                if recorder is not None:
                    cv2.putText(display, "REC", (max(0, display.shape[1]-65), 55),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                    source_ms = capture.get(cv2.CAP_PROP_POS_MSEC) if isinstance(source, str) else None
                    if source_ms is not None and not math.isfinite(source_ms):
                        source_ms = None
                    recorder.write(display, source_frame=count-1, captured_utc=captured_utc,
                                   elapsed_s=captured_elapsed, source_ms=source_ms,
                                   predictions=predictions, tracking_ms=frame_tracking_ms,
                                   auto_template=auto_template, template_fc_hz=template_fc,
                                   template_dt_s=frame_dt)
            if not args.headless:
                cv2.imshow(window, display)
                if args.debug and selected in targets:
                    target = targets[selected]
                    cv2.imshow("Template (BGR)", target.template_image)
                    cv2.imshow("Search (BGR)", target.search_image)
                    print(f"ID={selected} confidence={target.confidence:.4f} bbox={target.bbox}")
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("u"):
                    auto_template = not auto_template
                    print(f"Auto-template: {'ON' if auto_template else 'OFF'} (all targets)")
                elif key in (ord("["), ord("]")):
                    template_fc = max(0.0, round(template_fc + (-0.5 if key == ord("[") else 0.5), 3))
                    print(f"Template Fc: {template_fc:g} Hz (0 = bypass)")
                elif key == ord("v"):
                    if recorder is None:
                        recorder = Recorder(args.record_dir, frame.shape, record_fps, args.source)
                    else:
                        recorder.close()
                        recorder = None
                elif key == ord("l"):
                    show_trails = not show_trails
                elif key == ord("a"):
                    replace_id = None
                    selection.begin()
                elif key == ord("r"):
                    replace_id = selected
                    selection.begin()
                elif key in (27, ord("c")):
                    selection.cancel()
                    replace_id = None
                elif not selection.active:
                    if key == 9 and targets:
                        identities = list(targets)
                        selected = identities[(identities.index(selected)+1) % len(identities)]
                    elif key == ord("t") and selected in targets:
                        target = targets[selected]
                        target.reseed(frame, target.bbox)
                    elif key == ord("d") and selected in targets:
                        del targets[selected]
                        del trails[selected]
                        selected = next(iter(targets), None)
                        if args.debug and selected is None:
                            cv2.destroyWindow("Template (BGR)")
                            cv2.destroyWindow("Search (BGR)")
                # Mouse callbacks run in waitKey: seed on the clean frame that
                # was displayed when the button was released, before reading again.
                roi = selection.take(frame)
                if roi is not None:
                    if replace_id in targets:
                        targets[replace_id].init(frame, roi)
                        trails[replace_id].clear()
                        selected = replace_id
                    else:
                        add(frame, roi)
                    replace_id = None
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            finished = bool(args.max_frames and count >= args.max_frames)
            if not finished:
                ok, frame = capture.read()
                captured_utc = datetime.now(timezone.utc).isoformat()
                captured_elapsed = perf_counter() - run_start
            elapsed = perf_counter() - start
            total_loop += elapsed
            instantaneous = 1 / max(elapsed, 1e-9)
            fps = instantaneous if fps == 0 else 0.9*fps + 0.1*instantaneous
            if finished or not ok:
                break
        if tracked:
            print(f"{tracked} tracked frames | tracking {total_tracking/tracked:.2f} ms ({1000*tracked/total_tracking:.1f} FPS)"
                  f" | loop {count/max(total_loop, 1e-9):.1f} FPS | {len(targets)} targets remaining")
    finally:
        if recorder is not None:
            recorder.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
