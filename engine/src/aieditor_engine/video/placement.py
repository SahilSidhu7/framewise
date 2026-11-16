"""Caption placement that keeps captions off faces and off text already burned into the video.

Before export, every caption whose position is not set by hand is checked against a few
frames of the finished picture (captions off): faces (YuNet when the model is installed,
otherwise the head of the largest skin-coloured regions) and high-contrast text bands such as an interview's
name title. The caption goes to the candidate position that covers the least of them,
preferring the style's own position and the previous caption's position so captions do not
jump around.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from ..core.project import Caption, Project
from .compose import RenderSource, render_frame
from .reframe import _face_model
from .text import STYLES, caption_box

log = logging.getLogger(__name__)

SAMPLE_W = 640  # small burned-in text (name titles) needs this much detail
Box = tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the frame

CANDIDATES = {  # portrait?: positions to try after the style's own
    True: ("center_low", "bottom", "high"),
    False: ("bottom", "lower_third", "top"),
}
FACE_WEIGHT = 3.0
TEXT_WEIGHT = 2.0
MOVE_COST = 0.12  # per step away from the style's own position
SWITCH_COST = 0.08  # changing position from the previous caption


# ---------------------------------------------------------------- detectors
_yunet: Any = None
_yunet_size: tuple[int, int] | None = None


def _yunet_faces(rgb: np.ndarray) -> list[Box] | None:
    global _yunet, _yunet_size
    model = _face_model()
    if model is None or not hasattr(cv2, "FaceDetectorYN"):
        return None
    h, w = rgb.shape[:2]
    try:
        if _yunet is None:
            _yunet = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.6)
            _yunet_size = (w, h)
        elif _yunet_size != (w, h):
            _yunet.setInputSize((w, h))
            _yunet_size = (w, h)
        _, faces = _yunet.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    except cv2.error:
        return None
    out = []
    for f in faces if faces is not None else []:
        x, y, fw, fh = (float(v) for v in f[:4])
        out.append((x / w, y / h, (x + fw) / w, (y + fh) / h))
    return out


def _skin_faces(rgb: np.ndarray) -> list[Box]:
    """Face-sized skin blobs. Crude, but finds the talking head in most interviews and vlogs."""
    h, w = rgb.shape[:2]
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    skin = ((ycc[..., 1] > 135) & (ycc[..., 1] < 175) & (ycc[..., 2] > 85) & (ycc[..., 2] < 130) & (ycc[..., 0] > 60))
    mask = cv2.morphologyEx(skin.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in sorted(range(1, n), key=lambda k: -stats[k][4])[:3]:
        x, y, bw, bh, area = stats[i]
        if not 0.004 <= area / (w * h) <= 0.7:
            continue
        head = _head(labels[y : y + bh, x : x + bw] == i)
        if head is None:
            continue
        hx0, hy0, hx1, hy1 = head
        if (y + hy0) / h > 0.62:  # a skin blob that starts this low is hands or arms, not a head
            continue
        out.append(((x + hx0) / w, (y + hy0) / h, (x + hx1) / w, (y + hy1) / h))
    return out


def _head(comp: np.ndarray) -> tuple[int, int, int, int] | None:
    """The head inside a skin region that may also hold the neck, chest or hands: start at the top
    (hair is not skin), take the typical width of the upper rows, a face is about 1.35x as tall."""
    widths = comp.sum(axis=1)
    rows = np.nonzero(widths)[0]
    if rows.size < 4:
        return None
    top = int(rows[0])
    upper = widths[top : top + max(4, int(rows.size * 0.35))]
    upper = upper[upper > 0]
    head_w = float(np.percentile(upper, 80))
    if head_w < 4:
        return None
    bottom = min(int(rows[-1]) + 1, top + int(round(head_w * 1.35)))
    ys, xs = np.nonzero(comp[top:bottom])
    if xs.size < 12:
        return None
    x0, x1 = int(np.percentile(xs, 3)), int(np.percentile(xs, 97)) + 1
    if (bottom - top) / max(x1 - x0, 1) > 3.0 or ys.size / max((bottom - top) * (x1 - x0), 1) < 0.3:
        return None
    return x0, top, x1, bottom


def faces(rgb: np.ndarray) -> list[Box]:
    """Face boxes, grown downwards to cover the mouth and chin."""
    found = _yunet_faces(rgb)
    if found is None:
        found = _skin_faces(rgb)
    grown = []
    for x0, y0, x1, y1 in found:
        fw, fh = x1 - x0, y1 - y0
        grown.append((max(0.0, x0 - fw * 0.1), max(0.0, y0 - fh * 0.1), min(1.0, x1 + fw * 0.1), min(1.0, y1 + fh * 0.2)))
    return grown


def text_mask(rgb: np.ndarray) -> np.ndarray:
    """uint8 mask of text-like areas: thin strokes much lighter (or darker) than their surroundings,
    with sharp edges, lined up in wide short bands (titles, name tags, subtitles, watermarks)."""
    h, w = rgb.shape[:2]
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad = cv2.morphologyEx(grey, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    k = max(5, (w // 40) | 1)
    local = grey.astype(np.int16) - cv2.blur(grey, (k, k)).astype(np.int16)
    stroke = (((local > 40) | (local < -40)) & (grad > 50)).astype(np.uint8) * 255
    joined = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 60), 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    out = np.zeros((h, w), np.uint8)
    for i in range(1, n):
        x, y, bw_, bh, area = stats[i]
        if bw_ < w * 0.06 or not h * 0.012 <= bh <= h * 0.12 or bw_ / max(bh, 1) < 2.5:
            continue
        if area / max(bw_ * bh, 1) < 0.3 or float(stroke[y : y + bh, x : x + bw_].mean()) / 255 < 0.1:
            continue
        out[y : y + bh, x : x + bw_] = 255
    return out


def _boxes(mask: np.ndarray) -> list[Box]:
    h, w = mask.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [(x / w, y / h, (x + bw) / w, (y + bh) / h) for x, y, bw, bh, _ in stats[1:n]]


# ---------------------------------------------------------------- placement
def _overlap(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def obstacles(frames: list[np.ndarray]) -> list[tuple[Box, float]]:
    """Faces and text-like bands in any of the sampled frames (a title may fade out mid-caption)."""
    out: list[tuple[Box, float]] = []
    for f in frames:
        out += [(b, FACE_WEIGHT) for b in faces(f)]
        out += [(b, TEXT_WEIGHT) for b in _boxes(text_mask(f))]
    return out


def choose(cap: Caption, w: int, h: int, obs: list[tuple[Box, float]], previous: str | None) -> tuple[str, float]:
    """Best position for one caption and its cost (0 = nothing covered)."""
    own = STYLES.get(cap.style, STYLES["clean"]).position
    order = [own] + [p for p in CANDIDATES[h > w] if p != own]
    best = (own, float("inf"))
    for rank, pos in enumerate(order):
        x0, y0, x1, y1 = caption_box(cap.text, cap.style, pos, w, h)
        box = (x0 / w, y0 / h, x1 / w, y1 / h)
        area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        covered = sum(wt * _overlap(box, b) / area for b, wt in obs)
        cost = covered + rank * MOVE_COST + (SWITCH_COST if previous and pos != previous else 0.0)
        if cost < best[1]:
            best = (pos, cost)
    return best


def place_captions(project: Project, src: RenderSource, *, sample_w: int = SAMPLE_W) -> dict[str, Any]:
    """Set `position` on captions left on auto. Returns a summary {moved, checked}."""
    caps = [c for c in project.captions if c.kind == "caption" and (c.position is None or c.auto_position)]
    if not caps or project.settings.letterbox and project.settings.width >= project.settings.height:
        return {"checked": 0, "moved": 0}
    w0, h0 = project.settings.width, project.settings.height
    sw = min(sample_w, w0)
    sh = max(2, int(round(h0 * sw / w0)) // 2 * 2)
    moved = 0
    previous: str | None = None
    frame_cache: dict[float, np.ndarray] = {}

    def frame(t: float) -> np.ndarray:
        key = round(t, 2)
        if key not in frame_cache:
            frame_cache[key] = render_frame(project, t, sw, sh, src, captions=False)
        return frame_cache[key]

    for cap in caps:
        span = cap.end - cap.start
        fractions = (0.05, 0.35, 0.65, 0.95) if span > 1.5 else (0.2, 0.8) if span > 0.6 else (0.5,)
        times = [cap.start + span * f for f in fractions]
        try:
            obs = obstacles([frame(t) for t in times])
        except Exception:  # noqa: BLE001 - placement must never break an export
            log.exception("caption placement failed at %.2fs", cap.start)
            continue
        pos, cost = choose(cap, w0, h0, obs, previous)
        log.debug("caption %.2fs %r -> %s (cost %.2f, %d obstacles)", cap.start, cap.text[:24], pos, cost, len(obs))
        own = STYLES.get(cap.style, STYLES["clean"]).position
        cap.position = None if pos == own else pos  # type: ignore[assignment]
        cap.auto_position = cap.position is not None
        moved += cap.position is not None
        previous = pos
    return {"checked": len(caps), "moved": moved}
