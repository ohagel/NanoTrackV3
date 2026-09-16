"""Bounded, lossless decode prefetch for video files (not live cameras)."""
from queue import Queue, Full
from threading import Event, Thread

import cv2


class VideoReader:
    def __init__(self, path, prefetch=2):
        if prefetch < 0:
            raise ValueError('prefetch must be nonnegative')
        self.capture = cv2.VideoCapture(str(path))
        self.queue = Queue(maxsize=prefetch) if prefetch else None
        self.stop = Event()
        self.worker = None
        self.finished = False
        self.position_ms = 0.0

    def isOpened(self):
        return self.capture.isOpened()

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.position_ms
        # Metadata is read before starting the worker.
        if self.worker is not None:
            raise RuntimeError('Read video metadata before the first frame')
        return self.capture.get(prop)

    def _decode(self):
        try:
            while not self.stop.is_set():
                item = self._read_frame()
                if not self._put(item) or not item[0]:
                    break
        except Exception as exc:
            self._put(exc)

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True
            except Full:
                pass
        return False

    def read(self):
        if self.finished:
            return False, None
        if self.queue is None:
            item = self._read_frame()
        else:
            if self.worker is None:
                self.worker = Thread(target=self._decode, daemon=True)
                self.worker.start()
            item = self.queue.get()
            if isinstance(item, Exception):
                self.finished = True
                raise item
        self.finished = not item[0]
        self.position_ms = item[2]
        return item[:2]

    def _read_frame(self):
        ok, frame = self.capture.read()
        return ok, frame, self.capture.get(cv2.CAP_PROP_POS_MSEC)

    def release(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join()
        self.capture.release()
        self.finished = True
