"""Integration check for real video encoding and per-frame tracking alignment."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import cv2
import main


class RecordingTest(unittest.TestCase):
    def test_video_and_tracking_alignment(self):
        with TemporaryDirectory() as folder:
            args = ["main.py", "--source", "reference/girl_dance.mp4", "--headless",
                    "--bbox", "200", "165", "185", "300",
                    "--bbox", "440", "158", "197", "310",
                    "--max-frames", "12", "--record", "--record-dir", folder]
            with patch.object(sys, "argv", args):
                main.main()
            session, = Path(folder).iterdir()
            records = [json.loads(line) for line in (session / "tracking.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 12)
            self.assertEqual([r["recording_frame"] for r in records], list(range(12)))
            self.assertEqual([r["source_frame"] for r in records], list(range(12)))
            self.assertTrue(all([t["id"] for t in r["targets"]] == [1, 2] for r in records))
            self.assertTrue(all(a["elapsed_s"] < b["elapsed_s"] for a,b in zip(records, records[1:])))
            cap = cv2.VideoCapture(str(session / "video.avi"))
            try:
                decoded = 0
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    self.assertEqual(image.shape, (480, 856, 3))
                    decoded += 1
                self.assertEqual(decoded, len(records))
            finally:
                cap.release()


if __name__ == "__main__":
    unittest.main()
