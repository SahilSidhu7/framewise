"""Text and shape layers: drawn into the layer's own space (the canvas, like a
solid) as a colour plus alpha, then placed, masked and blended like any layer.

Sizes are fractions of the canvas, so a small preview and the full-size
export draw the same thing. The same raster feeds every compositor: the CPU
one directly, the GPU export through its CPU-layer path, and the live WebGL
preview as a PNG (/layers/{clip_id}/raster).
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..core.project import Clip, ShapeStyle, TextStyle
from .layers import hex_to_linear
from .text import _font

# colour (linear, straight), alpha, bounds (x0, y0, x1, y1 in canvas px)
Raster = tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]


class _Cache:
    def __init__(self, size: int = 24) -> None:
        self._items: OrderedDict[tuple, Raster] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, key: tuple, make) -> Raster:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        value = make()
        for arr in value[:2]:
            arr.flags.writeable = False
        with self._lock:
            self._items[key] = value
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return value


_CACHE = _Cache()


def _mix(fill: np.ndarray | None, stroke: np.ndarray, inner: np.ndarray, outer: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill inside `inner`, outline in `outer - inner`. Returns straight colour, alpha."""
    band = np.clip(outer - inner, 0, 1)
    if fill is None:
        rgb = np.empty((h, w, 3), np.float32)
        rgb[:] = stroke
        return rgb, band
    alpha = np.maximum(outer, inner)
    a = np.maximum(alpha, 1e-6)[..., None]
    rgb = (fill * inner[..., None] + stroke * band[..., None]) / a
    rgb[alpha <= 0] = fill  # colour under transparent pixels: no dark fringes when filtered
    return rgb.astype(np.float32), alpha


def _text(st: TextStyle, w: int, h: int) -> Raster:
    px = max(1, round(st.size * min(w, h)))
    font = _font(st.font, st.weight, px)
    spacing = (st.line_height - 1.0) * px
    sw = round(st.stroke_width * px)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    kw = {"font": font, "spacing": spacing, "align": st.align}
    bb = probe.multiline_textbbox((0, 0), st.text, stroke_width=sw, **kw)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    ox, oy = (w - tw) / 2 - bb[0], (h - th) / 2 - bb[1]

    def draw(stroke: int) -> np.ndarray:
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).multiline_text((ox, oy), st.text, fill=255, stroke_width=stroke, stroke_fill=255, **kw)
        return np.asarray(im, dtype=np.float32) / 255.0

    inner = draw(0)
    if sw > 0:
        rgb, alpha = _mix(hex_to_linear(st.color), hex_to_linear(st.stroke_color), inner, draw(sw), h, w)
    else:
        rgb, alpha = _mix(hex_to_linear(st.color), hex_to_linear(st.color), inner, inner, h, w)
    return rgb, alpha, ((w - tw) / 2, (h - th) / 2, (w + tw) / 2, (h + th) / 2)


def _shape_cover(kind: str, hw: float, hh: float, r: float, w: int, h: int) -> np.ndarray:
    """Antialiased coverage of a centred rectangle/ellipse (supersampled over its box only)."""
    cov = np.zeros((h, w), np.float32)
    if hw <= 0 or hh <= 0:
        return cov
    cx, cy = w / 2, h / 2
    x0, y0 = max(0, int(cx - hw) - 1), max(0, int(cy - hh) - 1)
    x1, y1 = min(w, int(np.ceil(cx + hw)) + 1), min(h, int(np.ceil(cy + hh)) + 1)
    if x1 <= x0 or y1 <= y0:
        return cov
    area = (x1 - x0) * (y1 - y0)
    ss = 4 if area < 4e6 else 2 if area < 16e6 else 1
    im = Image.new("L", ((x1 - x0) * ss, (y1 - y0) * ss), 0)
    box = [(cx - hw - x0) * ss, (cy - hh - y0) * ss, (cx + hw - x0) * ss - 1, (cy + hh - y0) * ss - 1]
    d = ImageDraw.Draw(im)
    if kind == "ellipse":
        d.ellipse(box, fill=255)
    else:
        d.rounded_rectangle(box, radius=min(r, hw, hh) * ss, fill=255)
    small = np.asarray(im, dtype=np.float32) / 255.0
    if ss > 1:
        small = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    cov[y0:y1, x0:x1] = small
    return cov


def _shape(st: ShapeStyle, w: int, h: int) -> Raster:
    short = min(w, h)
    hw, hh, r = st.width * w / 2, st.height * h / 2, st.radius * short
    sw = st.stroke_width * short
    outer = _shape_cover(st.kind, hw, hh, r, w, h)
    inner = _shape_cover(st.kind, hw - sw, hh - sw, max(0.0, r - sw), w, h) if sw > 0 else outer
    fill = hex_to_linear(st.fill) if st.fill else None
    rgb, alpha = _mix(fill, hex_to_linear(st.stroke_color), inner, outer, h, w)
    return rgb, alpha, (w / 2 - hw, h / 2 - hh, w / 2 + hw, h / 2 + hh)


def _raster(clip: Clip, w: int, h: int) -> Raster:
    if clip.text is not None:
        style = clip.text
        key = ("text", style.model_dump_json(), w, h)
        return _CACHE.get(key, lambda: _text(style, w, h))
    shape = clip.shape or ShapeStyle()
    key = ("shape", shape.model_dump_json(), w, h)
    return _CACHE.get(key, lambda: _shape(shape, w, h))


def vector_raster(clip: Clip, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(straight linear colour (h, w, 3), alpha (h, w)) of a text or shape layer, in layer space."""
    rgb, alpha, _ = _raster(clip, w, h)
    return rgb, alpha


def vector_bounds(clip: Clip, w: int, h: int) -> tuple[float, float, float, float]:
    """The drawn box in layer space, as fractions of the canvas (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = _raster(clip, w, h)[2]
    return x0 / w, y0 / h, x1 / w, y1 / h
