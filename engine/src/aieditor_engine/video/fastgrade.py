"""Fast finishing for video frames.

The photo engine runs every op on every pixel in float. That is fine for one
photo but too slow for 30 frames a second, so for video:

1. The leading run of per-pixel color ops in the grade (adjust without clarity,
   curves, hsl, color_grade, black_white, fade, lut) is evaluated once on a
   65^3 sRGB lattice, upsampled to a full 256^3 table and then applied to each
   frame with a single lookup.
2. Spatial ops after it (vignette, grain, bloom...) still run per frame.
3. Linear -> sRGB encoding uses a table plus a cached dither texture.

Preview frames and exports both go through this module, so they match.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from ..imaging.color import linear_to_srgb, srgb_to_linear
from ..imaging.render import apply_flat, flatten, make_context
from .decode import u8_to_linear

POINTWISE = {"curves", "hsl", "color_grade", "black_white", "fade", "lut"}
LATTICE = 65

_ENC_N = 16384
_ENC = (linear_to_srgb(np.linspace(0, 1, _ENC_N, dtype=np.float32)) * 255).astype(np.float32)


def _bakeable(name: str, params: dict[str, Any]) -> bool:
    if name == "adjust":
        return not params.get("clarity")
    return name in POINTWISE


class _Noise:
    """Triangular dither, generated once per frame size; per-frame offset."""

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, h: int, w: int, seed: int) -> np.ndarray:
        key = (h, w)
        with self._lock:
            tex = self._cache.get(key)
            if tex is None:
                rng = np.random.default_rng(12345)
                shape = (h + 32, w + 32, 3)
                tex = (rng.random(shape, dtype=np.float32) - rng.random(shape, dtype=np.float32)) + 0.5
                self._cache[key] = tex
                while len(self._cache) > 3:
                    self._cache.popitem(last=False)
        oy, ox = (seed * 7) % 32, (seed * 13) % 32
        return tex[oy : oy + h, ox : ox + w]

    def full(self, h: int, w: int) -> np.ndarray:
        """The whole (h + 32, w + 32, 3) texture the per-frame windows come from."""
        self.get(h, w, 0)
        with self._lock:
            return self._cache[(h, w)]


NOISE = _Noise()


def encode_u8(img_linear: np.ndarray, seed: int = 0) -> np.ndarray:
    """Linear float -> dithered 8-bit sRGB (same result family as render.to_display, ~5x faster)."""
    idx = np.clip(img_linear, 0.0, 1.0)
    idx *= _ENC_N - 1
    idx = idx.astype(np.int32)
    out = _ENC[idx]
    out += NOISE.get(*img_linear.shape[:2], seed)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def _upsample(table: np.ndarray, n_out: int = 256) -> np.ndarray:
    """Exact trilinear upsampling of a lattice (separable linear interpolation)."""
    n_in = table.shape[0]
    x = np.linspace(0, n_in - 1, n_out)
    i0 = np.clip(np.floor(x).astype(int), 0, n_in - 2)
    f = (x - i0).astype(np.float32)
    w = np.zeros((n_out, n_in), np.float32)
    w[np.arange(n_out), i0] = 1 - f
    w[np.arange(n_out), i0 + 1] += f
    out = np.tensordot(w, table, axes=([1], [0]))  # axis r
    out = np.tensordot(w, out, axes=([1], [1])).transpose(1, 0, 2, 3)  # axis g
    out = np.tensordot(w, out, axes=([1], [2])).transpose(1, 2, 0, 3)  # axis b
    return out


def bake_lattice(flat: list[tuple[str, dict[str, Any]]]) -> np.ndarray:
    """The grade on a LATTICE^3 grid of sRGB inputs: (r, g, b, 3) sRGB outputs in 0..1."""
    grid = np.linspace(0, 1, LATTICE, dtype=np.float32)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
    lattice = np.stack([r, g, b], axis=-1).reshape(LATTICE, LATTICE * LATTICE, 3)
    ctx = make_context(0)
    ctx.extra["pointwise"] = True
    graded = apply_flat(srgb_to_linear(lattice), flat, ctx)
    return np.clip(linear_to_srgb(np.clip(graded, 0, None)), 0, 1).reshape(LATTICE, LATTICE, LATTICE, 3)


def bake(flat: list[tuple[str, dict[str, Any]]], lattice: np.ndarray | None = None) -> np.ndarray:
    """(256**3, 3) uint8 table indexed by r<<16 | g<<8 | b (sRGB in, sRGB out)."""
    enc = lattice if lattice is not None else bake_lattice(flat)
    full = _upsample(enc)
    return np.clip(full * 255 + 0.5, 0, 255).astype(np.uint8).reshape(-1, 3)


class GradePlan:
    def __init__(self, grade_ops: list[dict[str, Any]]) -> None:
        flat = flatten(grade_ops)
        split = 0
        while split < len(flat) and _bakeable(*flat[split]):
            split += 1
        self.baked = flat[:split]
        self.rest = flat[split:]
        self.lattice = bake_lattice(self.baked) if self.baked else None  # also sent to the app's live preview
        self.lut = bake(self.baked, self.lattice) if self.baked else None

    def finish(self, canvas: np.ndarray, seed: int) -> np.ndarray:
        """Linear canvas -> graded 8-bit sRGB frame."""
        if self.lut is None and not self.rest:
            return encode_u8(canvas, seed)
        if self.lut is not None:
            u = encode_u8(canvas, seed)
            idx = u[..., 0].astype(np.int32)
            idx <<= 8
            idx |= u[..., 1]
            idx <<= 8
            idx |= u[..., 2]
            u = self.lut[idx]
            if not self.rest:
                return u
            canvas = u8_to_linear(u)
        canvas = apply_flat(canvas, self.rest, make_context(seed))
        return encode_u8(canvas, seed)


class _PlanCache:
    def __init__(self, size: int = 4) -> None:
        self._items: OrderedDict[str, GradePlan] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()
        self._building: dict[str, threading.Event] = {}

    def get(self, grade_ops: list[dict[str, Any]]) -> GradePlan:
        key = hashlib.sha1(json.dumps(grade_ops, sort_keys=True, default=str).encode()).hexdigest()
        while True:
            with self._lock:
                if key in self._items:
                    self._items.move_to_end(key)
                    return self._items[key]
                event = self._building.get(key)
                if event is None:  # we build it; other threads wait
                    event = self._building[key] = threading.Event()
                    break
            event.wait()
        try:
            plan = GradePlan(grade_ops)
            with self._lock:
                self._items[key] = plan
                while len(self._items) > self._size:
                    self._items.popitem(last=False)
            return plan
        finally:
            with self._lock:
                self._building.pop(key, None)
            event.set()


PLANS = _PlanCache()
