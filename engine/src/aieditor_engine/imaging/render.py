"""Photo renderer: decode -> linear float -> op stack -> encode/export.

One code path serves preview and export; previews just start from a
downscaled source. Spatial op params are resolution-relative, so both match.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

from .color import linear_to_srgb, luminance, srgb_to_linear
from .lut import load_cube
from .ops import OPS, RenderContext, detect_horizon_angle
from .recipes import RECIPE_DIR, expand

LUT_DIR = RECIPE_DIR / "luts"


# ------------------------------------------------------------------ decode


def _read_cv2(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)  # handles non-ASCII Windows paths
    img = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)  # applies EXIF orientation
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    return img.astype(np.float32) / scale


def _read_pil(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im, dtype=np.float32) / 255.0


def load_linear(path: Path, max_side: int | None = None) -> np.ndarray:
    srgb = _read_cv2(path)
    if srgb is None:
        srgb = _read_pil(path)
    lin = srgb_to_linear(srgb)
    if max_side and max(lin.shape[:2]) > max_side:
        s = max_side / max(lin.shape[:2])
        lin = cv2.resize(lin, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)  # area-average in linear light
    return np.ascontiguousarray(lin, dtype=np.float32)


class _SourceCache:
    """Decoded, downscaled sources; decoding a 24 MP JPEG costs ~0.3 s."""

    def __init__(self, size: int = 12) -> None:
        self._items: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, path: Path, max_side: int | None) -> np.ndarray:
        key = (str(path), path.stat().st_mtime_ns, max_side)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        img = load_linear(path, max_side)
        with self._lock:
            self._items[key] = img
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return img


SOURCES = _SourceCache()


# ------------------------------------------------------------------- stack


def _load_lut(name: str) -> np.ndarray:
    path = (LUT_DIR / f"{name}.cube").resolve()
    if not path.is_relative_to(LUT_DIR.resolve()) or not path.is_file():
        raise ValueError(f"unknown LUT: {name}")
    return load_cube(path)


def make_context(seed: int = 0) -> RenderContext:
    return RenderContext(seed_base=seed, expand_look=expand, load_lut=_load_lut)


def flatten(ops: Iterable[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Enabled ops with looks expanded into their steps."""
    flat: list[tuple[str, dict[str, Any]]] = []
    for item in ops:
        if not item.get("enabled", True):
            continue
        name, params = item["op"], item.get("params", {})
        if name == "look":
            flat.extend(expand(params["name"], float(params.get("strength", 1.0))))
        else:
            flat.append((name, params))
    return flat


def apply_stack(img: np.ndarray, ops: Iterable[dict[str, Any]], ctx: RenderContext | None = None) -> np.ndarray:
    return apply_flat(img, flatten(ops), ctx)


def apply_flat(img: np.ndarray, flat: list[tuple[str, dict[str, Any]]], ctx: RenderContext | None = None) -> np.ndarray:
    ctx = ctx or make_context()
    domain = "linear"
    for name, params in flat:
        spec = OPS[name]
        p = spec.params_model.model_validate(params)
        if spec.domain != domain:
            img = linear_to_srgb(img) if spec.domain == "display" else srgb_to_linear(img)
            domain = spec.domain
        img = spec.fn(img, p, ctx).astype(np.float32, copy=False)
        if domain == "linear":
            img = np.maximum(img, 0.0)
    if domain == "display":
        img = srgb_to_linear(img)
    return img


# ------------------------------------------------------------------ encode


def to_display(img_linear: np.ndarray, bit_depth: Literal[8, 16] = 8, seed: int = 0) -> np.ndarray:
    """sRGB encode with triangular dither (prevents banding in skies/gradients)."""
    d = np.clip(linear_to_srgb(img_linear), 0, 1)
    if bit_depth == 16:
        return np.round(d * 65535).astype(np.uint16)
    rng = np.random.default_rng(seed)
    noise = (rng.random(d.shape, dtype=np.float32) - rng.random(d.shape, dtype=np.float32)) / 255.0
    return np.round(np.clip(d + noise, 0, 1) * 255).astype(np.uint8)


def encode_jpeg(img_linear: np.ndarray, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(to_display(img_linear)).save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


class _PreviewCache:
    def __init__(self, size: int = 96) -> None:
        self._items: OrderedDict[str, bytes] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get_or_render(self, key: str, render) -> bytes:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        data = render()
        with self._lock:
            self._items[key] = data
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return data


PREVIEWS = _PreviewCache()


def render_preview(path: Path, ops: list[dict[str, Any]], max_side: int = 1600, quality: int = 86) -> bytes:
    key_src = json.dumps([str(path), path.stat().st_mtime_ns, max_side, quality, ops], sort_keys=True, default=str)
    key = hashlib.sha1(key_src.encode()).hexdigest()
    return PREVIEWS.get_or_render(key, lambda: encode_jpeg(apply_stack(SOURCES.get(path, max_side), ops), quality))


ExportFormat = Literal["jpeg", "png", "tiff", "webp"]


def export_image(
    path: Path,
    ops: list[dict[str, Any]],
    out_path: Path,
    fmt: ExportFormat = "jpeg",
    quality: int = 95,
    bit_depth: Literal[8, 16] = 8,
    long_edge: int | None = None,
) -> dict[str, Any]:
    """Full-resolution render from the original file."""
    img = apply_stack(load_linear(path), ops)
    if long_edge and max(img.shape[:2]) > long_edge:
        s = long_edge / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if bit_depth == 16 and fmt not in ("png", "tiff"):
        raise ValueError("16-bit export needs png or tiff")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = to_display(img, bit_depth)
    if bit_depth == 16:
        ok, buf = cv2.imencode(".png" if fmt == "png" else ".tiff", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("encode failed")
        out_path.write_bytes(buf.tobytes())
    else:
        pil = Image.fromarray(pixels)
        if fmt == "jpeg":
            pil.save(out_path, "JPEG", quality=quality, subsampling=0 if quality >= 90 else 2, optimize=True)
        elif fmt == "webp":
            pil.save(out_path, "WEBP", quality=quality)
        elif fmt == "png":
            pil.save(out_path, "PNG", optimize=True)
        else:
            pil.save(out_path, "TIFF", compression="tiff_lzw")
    h, w = img.shape[:2]
    return {"path": str(out_path), "width": w, "height": h, "format": fmt, "bit_depth": bit_depth}


# ----------------------------------------------------------------- analyze


def analyze(img_linear: np.ndarray) -> dict[str, Any]:
    """L1 measurements for the AI's media brief (cheap, exact numbers)."""
    small = img_linear
    if max(small.shape[:2]) > 512:
        s = 512 / max(small.shape[:2])
        small = cv2.resize(small, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    d = np.clip(linear_to_srgb(small), 0, 1)
    lum = luminance(d)
    p1, p50, p99 = (float(v) for v in np.percentile(lum, [1, 50, 99]))
    sat = d.max(axis=-1) - d.min(axis=-1)
    mean = d.reshape(-1, 3).mean(axis=0)
    warmth = float(mean[0] - mean[2])
    tint = float(mean[1] - (mean[0] + mean[2]) / 2)
    grey8 = (lum * 255).astype(np.uint8)
    sharpness = float(cv2.Laplacian(grey8, cv2.CV_64F).var())

    pixels = (d.reshape(-1, 3) * 255).astype(np.float32)
    k = 5
    _, labels, centers = cv2.kmeans(
        pixels, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.ravel(), minlength=k)
    palette = [
        {"hex": "#%02x%02x%02x" % tuple(int(c) for c in centers[i]), "share": round(float(counts[i] / counts.sum()), 3)}
        for i in np.argsort(-counts)
    ]

    notes = []
    hot = lum > 0.985
    clipped = float(hot.mean())
    # Product/food shots on white: the clipped area is a neutral backdrop, not a mistake.
    white_bg = clipped > 0.02 and float((sat[hot] < 0.05).mean()) > 0.9 and float(((lum > 0.9) & (sat < 0.05)).mean()) > 0.15
    # suggested exposure: bring the brightest tones (p99) up to ~0.9 display, in linear stops
    ev_fix = float(np.log2(srgb_to_linear(np.float32(0.9)) / max(float(srgb_to_linear(np.float32(max(p99, 0.02)))), 1e-4)))
    # Night / blue-hour / low-key shots: dark overall, but real light sources (lamps, windows, a
    # lit subject) reach near white. Lifting exposure there ruins the mood, so it is not a mistake.
    p999 = float(np.percentile(lum, 99.9))
    if 0.18 <= p50 < 0.35 and p99 >= 0.6 and p999 > 0.84 and p999 - p50 > 0.5:
        notes.append("low-key / night scene with lights (keep it dark, do not lift exposure)")
    elif p50 < 0.28 or p99 < 0.68:
        notes.append(f"underexposed (try exposure +{min(2.0, max(0.3, ev_fix)):.1f})")
    elif white_bg:
        notes.append("white background (keep whites bright, do not darken)")
    elif clipped > 0.02 or (p50 > 0.82 and p1 > 0.3):
        notes.append("overexposed")
    elif p50 > 0.68:
        notes.append("bright / high-key (keep it bright)")
    elif p99 < 0.85 and ev_fix > 0.25:
        notes.append(f"slightly dim (try exposure +{min(1.0, ev_fix):.1f})")
    if clipped > 0.02 and not white_bg:
        notes.append("clipped highlights")
    if float((lum < 0.015).mean()) > 0.05:
        notes.append("crushed shadows")
    if p99 - p1 < 0.45 and p50 <= 0.68:
        notes.append("flat / low contrast")
    elif p99 - p1 > 0.95:
        notes.append("high contrast")
    if float(sat.mean()) < 0.06:
        notes.append("nearly monochrome")
    elif float(sat.mean()) > 0.35:
        notes.append("very saturated")
    temp_fix = int(round(min(40, max(8, abs(warmth) * 130))))
    if warmth > 0.08:
        notes.append(f"warm cast (try temperature -{temp_fix})")
    elif warmth < -0.08:
        notes.append(f"cool cast (try temperature +{temp_fix})")
    if abs(tint) > 0.05:
        notes.append("green cast" if tint > 0 else "magenta cast")
    if sharpness < 60:
        notes.append("soft / blurry")
    angle = detect_horizon_angle(img_linear)  # needs >= ~800 px of detail; `small` is too coarse
    if abs(angle) >= 0.8:
        notes.append(f"tilted horizon ({angle:+.1f} deg)")

    h, w = img_linear.shape[:2]
    return {
        "size": [w, h],
        "orientation": "landscape" if w > h * 1.05 else "portrait" if h > w * 1.05 else "square",
        "brightness_median": round(p50, 3),
        "black_point_p1": round(p1, 3),
        "white_point_p99": round(p99, 3),
        "highlights_clipped_pct": round(float((lum > 0.985).mean()) * 100, 2),
        "shadows_crushed_pct": round(float((lum < 0.015).mean()) * 100, 2),
        "saturation_mean": round(float(sat.mean()), 3),
        "warmth": round(warmth, 3),
        "tint": round(tint, 3),
        "sharpness": round(sharpness, 1),
        "horizon_tilt_deg": round(angle, 2),
        "palette": palette,
        "notes": notes,
    }
