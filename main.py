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
    parser.add_argument("--threads", type=int, default=1, help="ORT threads; 0 = runtime default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--bbox", nargs=4, type=float, action="append", metavar=("X", "Y", "W", "H"), help="initial ROI; repeat for multiple targets")
    parser.add_argument("--headless", action="store_true", help="benchmark without windows; requires --bbox")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if args.headless and args.bbox is None:
        parser.error("--headless requires --bbox")
    engine = NanoTrackORT(args.backbone, args.head, providers=args.providers,
                           threads=args.threads, debug=args.debug)
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    count, tracked, total_tracking, total_loop = 0, 0, 0.0, 0.0
    window = "NanoTrack V3 | a add | Tab next | r select | t reseed | d remove | q quit"
    selection = LiveROI()
    targets = {}
    selected = None
    next_id = 1
    replace_id = None
    colors = ((0, 255, 0), (255, 180, 0), (0, 140, 255), (255, 0, 255), (0, 255, 255), (255, 160, 160))

    def add(frame, roi):
        nonlocal next_id, selected
        target = engine.new_target()
        target.init(frame, roi)
        targets[next_id] = target
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
        fps = 0.0
        while True:
            start = perf_counter()
            predictions = {}
            frame_tracking_ms = 0.0
            for identity, target in targets.items():
                predictions[identity] = target.update(frame)
                frame_tracking_ms += target.last_tracking_ms
            if targets:
                total_tracking += frame_tracking_ms
                tracked += 1
            count += 1
            if not args.headless:
                display = frame.copy()  # Keep annotations out of r/t template crops.
                status = f"{len(targets)} targets | selected {selected or '-'} | {fps:.1f} FPS | {frame_tracking_ms:.1f} ms"
                for identity, (success, bbox, confidence) in predictions.items():
                    x, y, w, h = bbox
                    color = colors[(identity-1) % len(colors)] if success else (0, 0, 255)
                    cv2.rectangle(display, (round(x), round(y)), (round(x+w), round(y+h)),
                                  color, 3 if identity == selected else 1)
                    cv2.putText(display, f"{'*' if identity == selected else ''}ID {identity}: {confidence:.3f}",
                                (max(0, round(x)), max(75, min(frame.shape[0]-5, round(y)-8))),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
                cv2.putText(display, status,
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                selection.draw(display, f"Replace ID {replace_id}" if replace_id is not None else "Add target")
                cv2.imshow(window, display)
                if args.debug and selected in targets:
                    target = targets[selected]
                    cv2.imshow("Template (BGR)", target.template_image)
                    cv2.imshow("Search (BGR)", target.search_image)
                    print(f"ID={selected} confidence={target.confidence:.4f} bbox={target.bbox}")
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("a"):
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
                        selected = replace_id
                    else:
                        add(frame, roi)
                    replace_id = None
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
                  f" | loop {count/max(total_loop, 1e-9):.1f} FPS | {len(targets)} targets remaining")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
