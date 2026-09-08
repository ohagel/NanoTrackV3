"""Annotated MJPEG recording plus one JSON record per video frame."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from uuid import uuid4

import cv2


class Recorder:
    def __init__(self, directory, frame_shape, fps, source):
        self.size = (frame_shape[1], frame_shape[0])
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self.folder = Path(directory) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
        self.folder.mkdir(parents=True, exist_ok=False)
        self.writer = cv2.VideoWriter(str(self.folder / "video.avi"),
                                      cv2.VideoWriter_fourcc(*"MJPG"), self.fps, self.size)
        self.log = None
        self.frames = 0
        try:
            if not self.writer.isOpened():
                raise RuntimeError(f"Cannot open recording: {self.folder}")
            self.log = (self.folder / "tracking.jsonl").open("w", encoding="utf-8")
            (self.folder / "metadata.json").write_text(json.dumps({
                "source": source, "started_utc": datetime.now(timezone.utc).isoformat(),
                "video_fps": self.fps, "width": self.size[0], "height": self.size[1],
                "bbox_format": "xywh in original image pixels",
                "timing": "Video is constant-FPS; tracking.jsonl preserves capture receipt times and source position.",
            }, indent=2), encoding="utf-8")
        except Exception:
            self.close()
            raise
        print(f"Recording: {self.folder.resolve()}")

    def write(self, image, *, source_frame, captured_utc, elapsed_s, source_ms, predictions, tracking_ms):
        if (image.shape[1], image.shape[0]) != self.size:
            raise RuntimeError("Frame size changed during recording; stop and start a new recording")
        record = {"recording_frame": self.frames, "source_frame": source_frame,
                  "captured_utc": captured_utc, "elapsed_s": elapsed_s,
                  "source_ms": source_ms, "tracking_ms": tracking_ms,
                  "targets": [{"id": identity, "success": bool(success),
                               "bbox": list(box), "confidence": float(confidence)}
                              for identity, (success, box, confidence) in predictions.items()]}
        line = json.dumps(record, allow_nan=False)
        self.writer.write(image)
        self.log.write(line + "\n")
        self.frames += 1
        if self.frames % 30 == 0:
            self.log.flush()

    def close(self):
        self.writer.release()
        if self.log is not None:
            self.log.close()
        print(f"Recording saved: {self.folder.resolve()} ({self.frames} frames)")
