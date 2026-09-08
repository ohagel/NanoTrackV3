"""Minimal webcam/video ROI tracking application."""
import argparse
from pathlib import Path
from time import perf_counter

import cv2

from nanotrack import NanoTrackORT


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

    def draw(self, image):
        if self.active:
            cv2.putText(image, "Drag target; release to track | Esc cancels", (12, 55),
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
    parser.add_argument("--threads", type=int, default=1, help="ORT threads; 0 = runtime default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("X", "Y", "W", "H"), help="optional initial ROI for repeatable tests")
    parser.add_argument("--headless", action="store_true", help="benchmark without windows; requires --bbox")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if args.headless and args.bbox is None:
        parser.error("--headless requires --bbox")
    tracker = NanoTrackORT(args.backbone, args.head, providers=args.providers,
                           threads=args.threads, debug=args.debug)
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    count, tracked, total_tracking, total_loop = 0, 0, 0.0, 0.0
    window = "NanoTrack V3 | q quit | r select | t reseed"
    selection = LiveROI()
    initialized = False

    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read source: {args.source}")
        if args.bbox:
            tracker.init(frame, args.bbox)
            initialized = True
        else:
            selection.begin()
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window, selection.mouse)
        fps = 0.0
        while True:
            start = perf_counter()
            if initialized:
                success, bbox, confidence = tracker.update(frame)
                total_tracking += tracker.last_tracking_ms
                tracked += 1
            count += 1
            if not args.headless:
                display = frame.copy()  # Keep annotations out of r/t template crops.
                status = f"{fps:.1f} FPS"
                if initialized:
                    x, y, w, h = bbox
                    cv2.rectangle(display, (round(x), round(y)), (round(x+w), round(y+h)),
                                  (0, 255, 0) if success else (0, 0, 255), 2)
                    status = f"confidence {confidence:.3f} | {status} | {tracker.last_tracking_ms:.1f} ms"
                cv2.putText(display, status,
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                selection.draw(display)
                cv2.imshow(window, display)
                if args.debug and initialized:
                    cv2.imshow("Template (BGR)", tracker.template_image)
                    cv2.imshow("Search (BGR)", tracker.search_image)
                    print(f"confidence={confidence:.4f} bbox={bbox}")
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    selection.begin()
                elif key in (27, ord("c")):
                    selection.cancel()
                elif key == ord("t") and initialized and not selection.active:
                    tracker.reseed(frame, bbox)
                # Mouse callbacks run in waitKey: seed on the clean frame that
                # was displayed when the button was released, before reading again.
                roi = selection.take(frame)
                if roi is not None:
                    tracker.init(frame, roi)
                    initialized = True
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            finished = bool(args.max_frames and count >= args.max_frames)
            if not finished:
                ok, frame = capture.read()
            elapsed = perf_counter() - start
            total_loop += elapsed
            instantaneous = 1 / max(elapsed, 1e-9)
            fps = instantaneous if fps == 0 else 0.9*fps + 0.1*instantaneous
            if finished or not ok:
                break
        if tracked:
            print(f"{tracked} tracked frames | tracking {total_tracking/tracked:.2f} ms ({1000*tracked/total_tracking:.1f} FPS)"
                  f" | loop {count/max(total_loop, 1e-9):.1f} FPS | final confidence {tracker.confidence:.3f}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
