"""Random-access video frame reading for the timeline compositor.

Sequential reads (playback, export) decode forward with no seeking; jumps
backwards or far ahead seek to the previous keyframe and decode forward.
Readers are pooled per (file, size) so scrubbing and export reuse open decoders.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np

from ..imaging.color import srgb_to_linear

# uint8 sRGB -> float32 linear, as a lookup table (exact and ~20x faster than the math)
_LIN_LUT = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0).astype(np.float32)


def u8_to_linear(img: np.ndarray) -> np.ndarray:
    return _LIN_LUT[img]


class FrameReader:
    SEEK_AHEAD = 2.5  # seconds; jumps further than this seek instead of decoding through

    def __init__(self, path: Path, size: tuple[int, int] | None = None) -> None:
        self.path = path
        self.size = size  # (w, h) to scale to inside FFmpeg, or None for native
        self.lock = threading.Lock()
        self._container = av.open(str(path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        rate = self._stream.average_rate or self._stream.guessed_rate
        self.fps = float(rate) if rate else 30.0
        self.width = self._stream.codec_context.width
        self.height = self._stream.codec_context.height
        self._gen = None
        self._cur: av.VideoFrame | None = None
        self._cur_time = -1.0
        self._next: av.VideoFrame | None = None
        self._eof = False
        self._cache_key: float | None = None
        self._cache: np.ndarray | None = None

    def close(self) -> None:
        self._container.close()

    # ------------------------------------------------------------------ io
    def _seek(self, t: float) -> None:
        tb = self._stream.time_base
        self._container.seek(max(0, int(t / tb)) if tb else 0, stream=self._stream, backward=True, any_frame=False)
        self._gen = self._container.decode(self._stream)
        self._cur, self._next, self._eof, self._cur_time = None, None, False, -1.0

    def _decode_next(self) -> av.VideoFrame | None:
        if self._eof:
            return None
        if self._gen is None:
            self._gen = self._container.decode(self._stream)
        while True:
            try:
                frame = next(self._gen)
            except (StopIteration, av.EOFError):
                self._eof = True
                return None
            except av.FFmpegError:
                continue  # skip a corrupt packet rather than failing the render
            if frame.time is not None:
                return frame

    def frame_at(self, t: float) -> np.ndarray:
        """uint8 RGB frame shown at time t (seconds). Holds the last frame past the end."""
        t = max(0.0, t)
        tol = 0.25 / self.fps
        with self.lock:
            if self._cur is None or t < self._cur_time - tol or t > self._cur_time + self.SEEK_AHEAD:
                self._seek(t)
            while True:
                if self._next is None:
                    self._next = self._decode_next()
                if self._next is None:
                    break
                if self._cur is not None and self._next.time > t + tol:
                    break
                self._cur, self._cur_time, self._next = self._next, float(self._next.time), None
            if self._cur is None:
                raise ValueError(f"no frames in {self.path.name}")
            if self._cache_key != self._cur_time or self._cache is None:
                w, h = self.size or (None, None)
                self._cache = self._cur.to_ndarray(format="rgb24", width=w, height=h, interpolation="AREA")
                self._cache_key = self._cur_time
            return self._cache


class _ReaderPool:
    def __init__(self, size: int = 10) -> None:
        self._items: OrderedDict[tuple, FrameReader] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, path: Path, size: tuple[int, int] | None, channel: str = "") -> FrameReader:
        """`channel` separates independent consumers (e.g. export vs. scrubbing) so
        they don't fight over one decoder's position."""
        key = (str(path), path.stat().st_mtime_ns, size, channel)
        with self._lock:
            reader = self._items.get(key)
            if reader is not None:
                self._items.move_to_end(key)
                return reader
        reader = FrameReader(path, size)
        with self._lock:
            self._items[key] = reader
            while len(self._items) > self._size:
                _, old = self._items.popitem(last=False)
                old.close()
        return reader

    def drop_channel(self, channel: str) -> None:
        with self._lock:
            for key in [k for k in self._items if k[3] == channel]:
                self._items.pop(key).close()


READERS = _ReaderPool(64)
