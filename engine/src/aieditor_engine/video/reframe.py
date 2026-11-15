"""Auto-reframe: follow the subject when a wide video is cropped to a taller shape.

subject_track() samples the video a few times per second and estimates where
the subject is horizontally (0 = left edge, 1 = right edge). With no detector
model installed it uses what people look at: things that move, skin tones,
and the centre. If a YuNet face model is present (models/face_detection_yunet*.onnx)
faces take priority.

reframe_keys() turns the track into a handful of `x` keyframes for a clip:
the camera holds still while the subject stays near the middle and pans
smoothly when they move away, like a human operator.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

SAMPLE_FPS = 3.0
SMALL_W = 160
DEADBAND = 0.07  # the subject may drift this far (fraction of source width) before the crop follows


def _face_model() -> Path | None:
    for base in (os.environ.get("AIEDITOR_MODELS"), os.path.join(os.environ.get("LOCALAPPDATA", ""), "AIEditor", "models")):
        if base and Path(base).is_dir():
            hits = sorted(Path(base).glob("face_detection_yunet*.onnx"))
            if hits:
                return hits[0]
    return None


def subject_track(path: Path) -> dict[str, Any]:
    """{"t": [...], "x": [...], "conf": [...], "method": ..., "aspect": w/h} sampled at SAMPLE_FPS."""
    times: list[float] = []
    xs: list[float] = []
    conf: list[float] = []
    model = _face_model()
    detector = None
    prev = None
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        w0, h0 = stream.codec_context.width, stream.codec_context.height
        sw = SMALL_W
        sh = max(2, int(h0 * sw / max(w0, 1)) // 2 * 2)
        if model is not None and hasattr(cv2, "FaceDetectorYN"):
            try:
                detector = cv2.FaceDetectorYN.create(str(model), "", (sw * 2, sh * 2), 0.6)
            except cv2.error:
                detector = None
        centre = np.exp(-0.5 * ((np.linspace(0, 1, sw) - 0.5) / 0.28) ** 2).astype(np.float32)
        next_t = 0.0
        for frame in c.decode(stream):
            t = frame.time
            if t is None or t + 1e-6 < next_t:
                continue
            next_t = t + 1 / SAMPLE_FPS
            x, cf = None, 0.0
            if detector is not None:
                big = frame.to_ndarray(format="bgr24", width=sw * 2, height=sh * 2)
                _, faces = detector.detect(big)
                if faces is not None and len(faces):
                    f = max(faces, key=lambda r: r[2] * r[3])
                    x, cf = float((f[0] + f[2] / 2) / (sw * 2)), 1.0
            rgb = frame.to_ndarray(format="rgb24", width=sw, height=sh)
            grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if x is None:
                ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
                skin = ((ycc[..., 1] > 135) & (ycc[..., 1] < 178) & (ycc[..., 2] > 85) & (ycc[..., 2] < 135) & (ycc[..., 0] > 50)).astype(np.float32)
                motion = np.zeros_like(grey) if prev is None else cv2.GaussianBlur(np.abs(grey - prev), (0, 0), 3) / 255
                face = _face_blob(skin, motion)
                if face is not None:
                    x, cf = face
            if x is None:
                # columns: movement counts most, then skin; the centre prior breaks ties
                col = motion.sum(axis=0) / max(float(motion.sum()), 1e-3) * min(1.0, float(motion.mean()) * 60) * 1.4
                col = col + skin.sum(axis=0) / max(float(skin.sum()), 1.0) * min(1.0, float(skin.mean()) * 12)
                strength = float(col.sum())
                col = col + centre / centre.sum() * 0.35
                col = np.convolve(col, np.ones(9) / 9, mode="same") ** 2
                x = float((col * np.arange(sw)).sum() / max(float(col.sum()), 1e-9) / (sw - 1))
                cf = min(1.0, strength / 1.5)
            prev = grey
            times.append(round(float(t), 3))
            xs.append(x)
            conf.append(round(cf, 3))
    smooth = _smooth(np.asarray(xs, dtype=np.float32), np.asarray(conf, dtype=np.float32))
    return {
        "t": times, "x": [round(float(v), 4) for v in smooth], "conf": conf,
        "method": "face" if detector is not None else "saliency", "aspect": round(w0 / max(h0, 1), 4),
    }


def _face_blob(skin: np.ndarray, motion: np.ndarray) -> tuple[float, float] | None:
    """Centre x of the most face-like skin region: compact, roughly face-shaped,
    in the upper part of the frame, with some texture/motion (wood or sand that
    happens to be skin-coloured is usually long, flat or still)."""
    h, w = skin.shape
    mask = cv2.morphologyEx((skin > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best, best_score = None, 0.0
    for i in range(1, n):
        x0, y0, bw, bh, area = stats[i]
        if area < 0.004 * w * h or area > 0.25 * w * h:
            continue
        fill = area / max(1, bw * bh)  # faces fill their box; curved sculptures and limbs don't
        aspect = bh / max(1, bw)
        shape = np.exp(-0.5 * ((aspect - 1.25) / 0.45) ** 2)  # faces are a bit taller than wide
        cy = cents[i][1] / h
        height = 1.0 if cy < 0.55 else max(0.0, 1.0 - (cy - 0.55) / 0.35)  # heads sit high, hands low
        alive = 0.5 + min(1.0, float(motion[labels == i].mean()) * 40)
        score = float(area / (w * h) * 20) ** 0.5 * fill * shape * height * alive
        if score > best_score:
            best, best_score = i, score
    if best is None or best_score < 0.08:
        return None
    return float(cents[best][0] / (w - 1)), float(min(1.0, best_score))


def _smooth(x: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Median to drop outliers, then a confidence-weighted zero-phase smoothing."""
    if x.size < 3:
        return x
    k = 5
    pad = np.pad(x, k // 2, mode="edge")
    med = np.array([np.median(pad[i : i + k]) for i in range(x.size)], dtype=np.float32)
    w = np.clip(conf, 0.15, 1.0)
    kernel = np.hanning(9).astype(np.float32)
    kernel /= kernel.sum()
    num = np.convolve(np.pad(med * w, 4, mode="edge"), kernel, mode="valid")
    den = np.convolve(np.pad(w, 4, mode="edge"), kernel, mode="valid")
    return (num / np.maximum(den, 1e-6)).astype(np.float32)


def _camera_path(t: np.ndarray, x: np.ndarray) -> list[tuple[float, float]]:
    """Hold until the subject leaves the deadband, then pan to it: [(time, x)] at the turning points."""
    cam = float(x[0])
    keys: list[tuple[float, float]] = [(float(t[0]), cam)]
    moving = False
    for i in range(1, x.size):
        if abs(float(x[i]) - cam) > DEADBAND:
            if not moving:  # start of a pan: pin where the camera was
                if keys[-1][0] < float(t[i - 1]):
                    keys.append((float(t[i - 1]), cam))
                moving = True
            cam = float(x[i])
            keys.append((float(t[i]), cam))
        else:
            moving = False
    if keys[-1][0] < float(t[-1]):
        keys.append((float(t[-1]), cam))
    # drop the middle key of any straight run
    out = [keys[0]]
    for i in range(1, len(keys) - 1):
        (t0, x0), (t1, x1), (t2, x2) = out[-1], keys[i], keys[i + 1]
        if abs(x1 - (x0 + (x2 - x0) * (t1 - t0) / max(t2 - t0, 1e-6))) > 0.01:
            out.append(keys[i])
    if len(keys) > 1:
        out.append(keys[-1])
    return out


def reframe_keys(track: dict[str, Any] | None, a: float, b: float, out_aspect: float, speed: float = 1.0) -> list[tuple[float, float]]:
    """`x` keyframes [(seconds after clip start, x offset)] that keep the subject
    centred while source range [a, b] fills a canvas of out_aspect (w/h).
    Empty when the source is not wider than the canvas or nothing was tracked."""
    if not track or not track.get("t"):
        return []
    ratio = float(track.get("aspect", 16 / 9)) / out_aspect  # filled picture width / canvas width
    if ratio <= 1.03:
        return []
    t = np.asarray(track["t"], dtype=np.float32)
    x = np.asarray(track["x"], dtype=np.float32)
    grid = np.arange(a, b + 1e-6, 1 / SAMPLE_FPS, dtype=np.float32)
    if grid.size < 2:
        grid = np.array([a, b], dtype=np.float32)
    path = _camera_path(grid, np.interp(grid, t, x))
    limit = (ratio - 1) / 2
    keys = [(round(max(0.0, (tt - a) / speed), 2), round(float(np.clip((0.5 - xx) * ratio, -limit, limit)), 4)) for tt, xx in path]
    if all(abs(v - keys[0][1]) < 0.004 for _, v in keys):
        return [(0.0, keys[0][1])] if abs(keys[0][1]) > 0.004 else []
    return keys


def x_param(keys: list[tuple[float, float]]) -> str:
    """Keyframes as a Video DSL parameter (" x=..." or "")."""
    if not keys:
        return ""
    if len(keys) == 1:
        return f" x={keys[0][1]:g}"
    return " x=" + ",".join(f"{t:g}:{v:g}" for t, v in keys)
