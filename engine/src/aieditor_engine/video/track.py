"""Follow a subject picked with one click through a video.

    track = track_subject(path, t=4.2, x=0.61, y=0.40)

The click is turned into a box (the subject's outline from imaging.subject at that frame),
then feature points inside the box are followed frame to frame with pyramidal Lucas-Kanade
optical flow, forwards to the end and backwards to the start. Points that fail a
forward-backward check are dropped and the box is re-seeded with fresh corners every few
samples, so the track survives turns and partial occlusion. No model download needed.

Output matches reframe.subject_track: {"t", "x", "conf", "method", "aspect"} plus "box".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from ..imaging.subject import mask_box, subject_mask
from .reframe import _smooth

SAMPLE_FPS = 8.0
WIDTH = 384
MAX_SAMPLES = 1600
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _frames(path: Path) -> tuple[list[float], list[np.ndarray], float]:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        w0, h0 = s.codec_context.width, s.codec_context.height
        dur = float(s.duration * s.time_base) if s.duration else float(c.duration or 0) / 1e6
        fps = min(SAMPLE_FPS, MAX_SAMPLES / max(dur, 1e-3))
        h = max(2, int(h0 * WIDTH / max(w0, 1)) // 2 * 2)
        times, frames, nxt = [], [], 0.0
        for f in c.decode(s):
            if f.time is None or f.time + 1e-6 < nxt:
                continue
            nxt = f.time + 1 / fps
            times.append(float(f.time))
            frames.append(f.to_ndarray(format="rgb24", width=WIDTH, height=h))
    return times, frames, w0 / max(h0, 1)


def _seed(grey: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    h, w = grey.shape
    x0, y0, x1, y1 = (box * [w, h, w, h]).astype(int)
    m = np.zeros_like(grey)
    m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
    pts = cv2.goodFeaturesToTrack(grey, maxCorners=80, qualityLevel=0.01, minDistance=4, mask=m, blockSize=5)
    return pts


def _step(a: np.ndarray, b: np.ndarray, pts: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Move box from frame a to b. Returns (new box, surviving points in b, confidence)."""
    h, w = a.shape
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, **LK)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, **LK)
    fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
    if good.sum() < 5:
        return box, None, 0.0
    p0, p1 = pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)
    d = np.median(p1 - p0, axis=0)
    scale = 1.0
    if len(p0) >= 6:
        i, j = np.triu_indices(len(p0), 1)
        d0 = np.linalg.norm(p0[i] - p0[j], axis=1)
        d1 = np.linalg.norm(p1[i] - p1[j], axis=1)
        ok = d0 > 4
        if ok.any():
            scale = float(np.clip(np.median(d1[ok] / d0[ok]), 0.9, 1.1))
    cx, cy = (box[0] + box[2]) / 2 + d[0] / w, (box[1] + box[3]) / 2 + d[1] / h
    hw, hh = (box[2] - box[0]) / 2 * scale, (box[3] - box[1]) / 2 * scale
    nb = np.clip(np.array([cx - hw, cy - hh, cx + hw, cy + hh]), 0, 1)
    return nb, p1.reshape(-1, 1, 2).astype(np.float32), float(good.mean())


def _run(greys: list[np.ndarray], start: int, box: np.ndarray, direction: int) -> dict[int, tuple[np.ndarray, float]]:
    out: dict[int, tuple[np.ndarray, float]] = {}
    pts = _seed(greys[start], box)
    i = start
    while 0 <= i + direction < len(greys):
        j = i + direction
        if pts is None or len(pts) < 8 or (abs(j - start) % 6 == 0):
            fresh = _seed(greys[i], box)
            pts = fresh if fresh is not None else pts
        if pts is None:
            out[j] = (box, 0.0)
        else:
            box, pts, conf = _step(greys[i], greys[j], pts, box)
            out[j] = (box, conf)
        i = j
    return out


def track_subject(path: Path, t: float, x: float, y: float) -> dict[str, Any]:
    times, frames, aspect = _frames(Path(path))
    if not frames:
        raise ValueError("could not read any frames from the video")
    k = int(np.argmin([abs(tt - t) for tt in times]))
    mask = subject_mask(frames[k], (x, y), feather=0.2)
    b = mask_box(mask)
    box = np.array(b if b else (x - 0.1, y - 0.15, x + 0.1, y + 0.15), dtype=np.float64)
    greys = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    boxes: dict[int, tuple[np.ndarray, float]] = {k: (box, 1.0)}
    boxes.update(_run(greys, k, box, +1))
    boxes.update(_run(greys, k, box, -1))
    order = sorted(boxes)
    bx = np.array([boxes[i][0] for i in order])
    conf = np.array([boxes[i][1] for i in order], dtype=np.float32)
    cx = ((bx[:, 0] + bx[:, 2]) / 2).astype(np.float32)
    return {
        "t": [round(times[i], 3) for i in order],
        "x": [round(float(v), 4) for v in _smooth(cx, np.maximum(conf, 0.05))],
        "conf": [round(float(c), 3) for c in conf],
        "box": [[round(float(v), 4) for v in b] for b in bx],
        "method": "manual", "aspect": round(aspect, 4), "pick": {"t": t, "x": x, "y": y},
    }
