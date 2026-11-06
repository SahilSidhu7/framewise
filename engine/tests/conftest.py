from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    path = tmp_path / "src" / "photo.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (200, 120, 40)).save(path)
    return path


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    """2 s, 1280x960 @ 25 fps with a stereo sine tone. Encoded with mpeg4,
    which every FFmpeg build has."""
    path = tmp_path / "src" / "clip.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    fps, seconds, rate = 25, 2, 48000
    with av.open(str(path), "w") as out:
        v = out.add_stream("mpeg4", rate=fps)
        v.width, v.height, v.pix_fmt = 1280, 960, "yuv420p"
        a = out.add_stream("aac", rate=rate, layout="stereo")
        for i in range(fps * seconds):
            img = np.full((960, 1280, 3), (i * 5) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            out.mux(v.encode(frame))
        out.mux(v.encode(None))
        samples = 1024
        t = np.arange(rate * seconds) / rate
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for start in range(0, len(tone) - samples, samples):
            chunk = tone[start : start + samples]
            frame = av.AudioFrame.from_ndarray(np.stack([chunk, chunk]), format="fltp", layout="stereo")
            frame.sample_rate = rate
            frame.pts = start
            out.mux(a.encode(frame))
        out.mux(a.encode(None))
    return path
