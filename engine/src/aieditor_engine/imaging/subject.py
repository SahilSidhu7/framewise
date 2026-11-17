"""Find the main subject of a picture, automatically or from one click.

    mask = subject_mask(rgb_u8)                 # auto: best guess of the subject
    mask = subject_mask(rgb_u8, (0.52, 0.40))   # the thing under this point (x, y as 0..1)

Returns a float32 mask (0..1, soft edge) the size of the input. Works without any model
download: a prior built from what usually makes a subject (in focus, different from the
borders, near faces, near the centre) seeds OpenCV's GrabCut, which then follows the real
colour edges. Results are cached by image content, so preview re-renders are free.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import cv2
import numpy as np

WORK = 420  # working size (long side) for the segmentation
_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_lock = threading.Lock()


def parse_point(value: str | None) -> tuple[float, float] | None:
    """"auto" / None -> None; "x,y" -> (x, y) clamped to 0..1."""
    if value is None or str(value).strip().lower() in ("", "auto"):
        return None
    x, y = (float(v) for v in str(value).split(","))
    return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))


def _small(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = WORK / max(h, w)
    if s >= 1:
        return np.ascontiguousarray(rgb)
    return cv2.resize(rgb, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def prior(rgb: np.ndarray) -> np.ndarray:
    """0..1 'how subject-like' map: sharpness, difference from the border colours, centre, faces."""
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(3, (min(h, w) // 40) | 1)
    focus = _norm(cv2.GaussianBlur(np.abs(cv2.Laplacian(grey, cv2.CV_32F, ksize=3)), (0, 0), k))
    b = max(2, min(h, w) // 25)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    n = min(6, max(2, len(border) // 50))
    _, _, centers = cv2.kmeans(border, n, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    dist = np.min(np.linalg.norm(lab.reshape(-1, 1, 3) - centers.reshape(1, -1, 3), axis=2), axis=1).reshape(h, w)
    distinct = _norm(cv2.GaussianBlur(dist, (0, 0), k))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    centre = np.exp(-(((xx / w - 0.5) / 0.32) ** 2 + ((yy / h - 0.48) / 0.36) ** 2))
    score = 0.42 * distinct + 0.33 * focus + 0.25 * centre
    try:
        # only a real face detector (YuNet, when installed): skin-colour guesses fire on bread, wood and fabric
        from ..video.placement import _yunet_faces

        for x0, y0, x1, y1 in _yunet_faces(rgb) or []:
            fw, fh = (x1 - x0) * w, (y1 - y0) * h
            # a head usually comes with shoulders and a body below it
            bx0, bx1 = int(max(0, x0 * w - fw)), int(min(w, x1 * w + fw))
            by0, by1 = int(max(0, y0 * h)), int(min(h, y1 * h + fh * 3))
            score[by0:by1, bx0:bx1] = np.maximum(score[by0:by1, bx0:bx1], 0.75)
            score[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 1.0
    except Exception:  # noqa: BLE001 - faces are a bonus, never required
        pass
    return np.clip(score, 0, 1).astype(np.float32)


def _segment(rgb: np.ndarray, point: tuple[float, float] | None) -> np.ndarray:
    h, w = rgb.shape[:2]
    score = prior(rgb)
    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    if point is None:
        gc[score > np.percentile(score, 72)] = cv2.GC_PR_FGD
        gc[score > np.percentile(score, 96)] = cv2.GC_FGD
    else:
        px, py = int(point[0] * (w - 1)), int(point[1] * (h - 1))
        r = int(min(h, w) * 0.22)
        near = np.zeros((h, w), np.uint8)
        cv2.circle(near, (px, py), r, 1, -1)
        gc[(near > 0) & (score > np.percentile(score, 45))] = cv2.GC_PR_FGD
        cv2.circle(gc, (px, py), max(2, int(min(h, w) * 0.025)), cv2.GC_FGD, -1)
    b = max(1, min(h, w) // 60)
    border = np.zeros((h, w), bool)
    border[:b] = border[-b:] = True
    border[:, :b] = border[:, -b:] = True
    gc[border & (gc != cv2.GC_FGD)] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return _fallback(h, w, point)
    fg = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, cent = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return _fallback(h, w, point)
    if point is not None:
        keep = labels[int(point[1] * (h - 1)), int(point[0] * (w - 1))]
        if keep == 0:  # clicked just outside the cut: take the nearest piece
            d = [np.hypot(cent[i][0] - point[0] * w, cent[i][1] - point[1] * h) for i in range(1, n)]
            keep = 1 + int(np.argmin(d))
        keep_set = {keep}
    else:  # the biggest piece plus any other large piece (two people, a person and their dog)
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = areas.max()
        keep_set = {i + 1 for i, a in enumerate(areas) if a >= big * 0.35}
    mask = np.isin(labels, list(keep_set)).astype(np.uint8)
    frac = mask.mean()
    if frac < 0.01 or frac > 0.92:
        return _fallback(h, w, point)
    # fill holes (e.g. a dark shirt inside a person)
    flood = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)  # 2 = background reachable from outside
    mask = (flood[1:-1, 1:-1] != 2).astype(np.uint8)
    return mask.astype(np.float32)


def _fallback(h: int, w: int, point: tuple[float, float] | None) -> np.ndarray:
    cx, cy = point if point else (0.5, 0.5)
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (int(cx * w), int(cy * h)), (int(w * 0.22), int(h * 0.32)), 0, 0, 360, 1, -1)
    return m.astype(np.float32)


MODEL_WORK = 1024  # the segmentation model sees 1024 px, so keep that much detail for its mask


def _model_mask(rgb_u8: np.ndarray, point: tuple[float, float] | None) -> np.ndarray | None:
    """Soft mask from the optional BiRefNet model, limited to the clicked (or the main) subject.
    None when the model isn't installed, sees no subject, or the click is on something it didn't pick."""
    from .segment import subject_probability

    h, w = rgb_u8.shape[:2]
    s = min(1.0, MODEL_WORK / max(h, w))
    img = rgb_u8 if s >= 1 else cv2.resize(rgb_u8, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    prob = subject_probability(img)
    if prob is None:
        return None
    hard = (prob > 0.5).astype(np.uint8)
    if hard.mean() < 0.004:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard)
    ih, iw = hard.shape
    if point is not None:
        px, py = int(point[0] * (iw - 1)), int(point[1] * (ih - 1))
        r = max(2, int(min(ih, iw) * 0.015))
        hits = labels[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1]
        hits = hits[hits > 0]
        if hits.size == 0:
            return None  # clicked something the model doesn't see as a subject: GrabCut takes over
        keep = {int(np.bincount(hits).argmax())}
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = {i + 1 for i, a in enumerate(areas) if a >= areas.max() * 0.35}
    region = np.isin(labels, list(keep)).astype(np.uint8)
    region = cv2.dilate(region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return (prob * region).astype(np.float32)


def subject_mask(rgb_u8: np.ndarray, point: tuple[float, float] | None = None, feather: float = 0.5) -> np.ndarray:
    """Soft subject mask at the input's size (float32 0..1). rgb_u8: HxWx3 uint8 display-referred.
    Uses the BiRefNet model when installed (hair-accurate edges), otherwise GrabCut."""
    model = _model_mask(rgb_u8, point)
    if model is not None:
        h, w = rgb_u8.shape[:2]
        m = cv2.resize(model, (w, h), interpolation=cv2.INTER_LINEAR)
        sigma = min(h, w) * 0.0015 * feather * 2  # the model's edge is already soft
        return np.clip(cv2.GaussianBlur(m, (0, 0), sigma) if sigma >= 0.5 else m, 0, 1).astype(np.float32)
    small = _small(rgb_u8)
    key = hashlib.blake2b(small.tobytes(), digest_size=16).hexdigest() + f"|{point}"
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
    if hit is None:
        hit = _segment(small, point)
        with _lock:
            _cache[key] = hit
            while len(_cache) > 24:
                _cache.popitem(last=False)
    h, w = rgb_u8.shape[:2]
    m = cv2.resize(hit, (w, h), interpolation=cv2.INTER_LINEAR)
    sigma = max(0.5, min(h, w) * 0.006 * (0.2 + feather * 1.6))
    return np.clip(cv2.GaussianBlur(m, (0, 0), sigma), 0, 1).astype(np.float32)


def mask_box(mask: np.ndarray, thresh: float = 0.5) -> tuple[float, float, float, float] | None:
    """Normalised (x0, y0, x1, y1) bounding box of a mask, or None if empty."""
    ys, xs = np.nonzero(mask > thresh)
    if xs.size == 0:
        return None
    h, w = mask.shape[:2]
    return float(xs.min() / w), float(ys.min() / h), float((xs.max() + 1) / w), float((ys.max() + 1) / h)
