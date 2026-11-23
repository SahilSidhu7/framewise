"""Layer maths shared by the timeline and (later) the photo editor: blend modes,
mask rasterising and track mattes.

Pictures are linear light, float32, premultiplied by their alpha. The base
(what is already on the canvas) is treated as opaque.

Blend modes follow Photoshop / the W3C compositing spec. add, screen and
multiply are computed in linear light (physically right for light and
shadow, and what this engine always did); every other mode works on
display-encoded values, as Photoshop and After Effects do by default, so the
results look the way people expect from those apps.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..core.project import Mask
from ..imaging.color import linear_to_srgb, srgb_to_linear

Alpha = np.ndarray | float
LINEAR_MODES = ("add", "screen", "multiply")
_LUMA = np.array([0.3, 0.59, 0.11], dtype=np.float32)  # W3C non-separable blend weights


# ------------------------------------------------------------------- blend


def _lum(c: np.ndarray) -> np.ndarray:
    return c @ _LUMA


def _clip_color(c: np.ndarray) -> np.ndarray:
    lum = _lum(c)[..., None]
    lo = c.min(axis=-1, keepdims=True)
    hi = c.max(axis=-1, keepdims=True)
    out = np.where(lo < 0, lum + (c - lum) * lum / np.maximum(lum - lo, 1e-6), c)
    out = np.where(hi > 1, lum + (out - lum) * (1 - lum) / np.maximum(hi - lum, 1e-6), out)
    return out


def _set_lum(c: np.ndarray, lum: np.ndarray) -> np.ndarray:
    return _clip_color(c + (lum - _lum(c))[..., None])


def _sat(c: np.ndarray) -> np.ndarray:
    return c.max(axis=-1) - c.min(axis=-1)


def _set_sat(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    lo = c.min(axis=-1, keepdims=True)
    hi = c.max(axis=-1, keepdims=True)
    rng = hi - lo
    return np.where(rng > 1e-6, (c - lo) * s[..., None] / np.maximum(rng, 1e-6), 0.0)


def blend_rgb(cb: np.ndarray, cs: np.ndarray, mode: str) -> np.ndarray:
    """B(base, source) for straight (not premultiplied) colours in 0..1."""
    if mode == "normal":
        return cs
    if mode == "darken":
        return np.minimum(cb, cs)
    if mode == "multiply":
        return cb * cs
    if mode == "color_burn":
        return np.where(cb >= 1, 1.0, np.where(cs <= 0, 0.0, 1 - np.minimum(1, (1 - cb) / np.maximum(cs, 1e-6))))
    if mode == "linear_burn":
        return np.maximum(0, cb + cs - 1)
    if mode == "darker_color":
        return np.where((_lum(cs) < _lum(cb))[..., None], cs, cb)
    if mode == "lighten":
        return np.maximum(cb, cs)
    if mode == "screen":
        return cb + cs - cb * cs
    if mode == "color_dodge":
        return np.where(cb <= 0, 0.0, np.where(cs >= 1, 1.0, np.minimum(1, cb / np.maximum(1 - cs, 1e-6))))
    if mode == "add":
        return cb + cs
    if mode == "lighter_color":
        return np.where((_lum(cs) > _lum(cb))[..., None], cs, cb)
    if mode == "overlay":
        return blend_rgb(cs, cb, "hard_light")
    if mode == "soft_light":
        d = np.where(cb <= 0.25, ((16 * cb - 12) * cb + 4) * cb, np.sqrt(np.maximum(cb, 0)))
        return np.where(cs <= 0.5, cb - (1 - 2 * cs) * cb * (1 - cb), cb + (2 * cs - 1) * (d - cb))
    if mode == "hard_light":
        return np.where(cs <= 0.5, cb * 2 * cs, 1 - (1 - cb) * (1 - (2 * cs - 1)))
    if mode == "vivid_light":
        burn = blend_rgb(cb, np.clip(2 * cs, 0, 1), "color_burn")
        dodge = blend_rgb(cb, np.clip(2 * cs - 1, 0, 1), "color_dodge")
        return np.where(cs <= 0.5, burn, dodge)
    if mode == "linear_light":
        return np.clip(cb + 2 * cs - 1, 0, 1)
    if mode == "pin_light":
        return np.where(cs <= 0.5, np.minimum(cb, 2 * cs), np.maximum(cb, 2 * cs - 1))
    if mode == "hard_mix":
        return (cb + cs >= 1).astype(np.float32)
    if mode == "difference":
        return np.abs(cb - cs)
    if mode == "exclusion":
        return cb + cs - 2 * cb * cs
    if mode == "subtract":
        return np.maximum(0, cb - cs)
    if mode == "divide":
        return np.where(cs <= 0, np.where(cb > 0, 1.0, 0.0), np.minimum(1, cb / np.maximum(cs, 1e-6)))
    if mode == "hue":
        return _set_lum(_set_sat(cs, _sat(cb)), _lum(cb))
    if mode == "saturation":
        return _set_lum(_set_sat(cb, _sat(cs)), _lum(cb))
    if mode == "color":
        return _set_lum(cs, _lum(cb))
    if mode == "luminosity":
        return _set_lum(cb, _lum(cs))
    raise ValueError(f"unknown blend mode: {mode}")


def composite(canvas: np.ndarray | None, img: np.ndarray, alpha: Alpha, blend: str) -> np.ndarray:
    """Put one layer (premultiplied picture + alpha) on top of the canvas."""
    if canvas is None:
        if blend in ("normal", "add", "screen", "lighten", "lighter_color", "difference", "exclusion"):
            return img  # over black these all leave just the layer
        canvas = np.zeros_like(img)
    a3: Any = alpha[..., None] if isinstance(alpha, np.ndarray) else alpha
    if blend == "normal":
        if isinstance(alpha, float) and alpha >= 1.0:
            return img
        return canvas * (1.0 - a3) + img
    if blend == "add":
        return canvas + img
    if blend == "screen":
        return canvas + img - np.clip(canvas, 0, 1) * np.clip(img, 0, 1)
    if blend == "multiply":
        return canvas * (1.0 - a3) + canvas * img
    # display-referred modes: un-premultiply, blend encoded values, re-encode
    cb = linear_to_srgb(np.clip(canvas, 0, 1))
    cs = linear_to_srgb(np.clip(img / np.maximum(a3, 1e-4), 0, 1))
    mixed = srgb_to_linear(np.clip(blend_rgb(cb, cs, blend), 0, 1).astype(np.float32))
    return (canvas * (1.0 - a3) + mixed * a3).astype(np.float32)


# ------------------------------------------------------------------- masks


def _bezier_points(mask: Mask, sw: int, sh: int, m: np.ndarray) -> np.ndarray:
    """The closed path sampled densely, in canvas pixels (float)."""
    pts = mask.points
    out: list[np.ndarray] = []
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        p0 = np.array([a.x, a.y])
        p1 = p0 + [a.out_x, a.out_y]
        p3 = np.array([b.x, b.y])
        p2 = p3 + [b.in_x, b.in_y]
        straight = not (a.out_x or a.out_y or b.in_x or b.in_y)
        n = 1 if straight else 24
        s = np.linspace(0, 1, n, endpoint=False)[:, None]
        seg = ((1 - s) ** 3) * p0 + 3 * ((1 - s) ** 2) * s * p1 + 3 * (1 - s) * s * s * p2 + s**3 * p3
        out.append(seg)
    path = np.vstack(out) * [sw, sh]  # layer space -> source pixels
    return path @ m[:, :2].T + m[:, 2]  # -> canvas pixels


def mask_shape(mask: Mask, sw: int, sh: int, m: np.ndarray, w: int, h: int) -> np.ndarray:
    """One mask as a (h, w) coverage in 0..1: antialiased, expanded and feathered."""
    short = min(w, h)
    path = _bezier_points(mask, sw, sh, m)
    shape = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(shape, [np.round(path * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)
    cov = shape.astype(np.float32) / 255.0
    grow = mask.expansion * short
    if abs(grow) >= 0.5:
        r = int(round(abs(grow)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        cov = cv2.dilate(cov, kernel) if grow > 0 else cv2.erode(cov, kernel)
    if mask.feather > 0:
        sigma = max(0.5, mask.feather * short / 2.0)
        cov = cv2.GaussianBlur(cov, (0, 0), sigma)
    return cov


def masks_coverage(masks: list[Mask], sw: int, sh: int, m: np.ndarray, w: int, h: int) -> np.ndarray | None:
    """All of a layer's masks combined (After Effects rules), or None if it has none."""
    active = [k for k in masks if k.enabled]
    if not active:
        return None
    acc = np.ones((h, w), np.float32) if active[0].mode in ("subtract", "intersect") else np.zeros((h, w), np.float32)
    for k in active:
        cov = mask_shape(k, sw, sh, m, w, h)
        if k.inverted:
            cov = 1.0 - cov
        cov *= k.opacity
        if k.mode == "add":
            acc = acc + cov - acc * cov
        elif k.mode == "subtract":
            acc = acc * (1.0 - cov)
        elif k.mode == "intersect":
            acc = acc * cov
        else:  # difference
            acc = np.abs(acc - cov)
    return acc


# ------------------------------------------------------------------- matte


def matte_value(img: np.ndarray, alpha: Alpha, mode: str, h: int, w: int) -> Alpha:
    """How much of the layer below a matte layer shows (0..1 per pixel)."""
    if mode.startswith("alpha"):
        v: Alpha = alpha
    else:
        lum = linear_to_srgb(np.clip(img, 0, 1)) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        v = lum.astype(np.float32)  # img is premultiplied, so this is already luma x alpha
    if mode.endswith("inverted"):
        v = 1.0 - v
    return v


def multiply_alpha(img: np.ndarray, alpha: Alpha, v: Alpha, h: int, w: int) -> tuple[np.ndarray, Alpha]:
    if isinstance(v, float):
        return img * v, alpha * v
    a = (alpha if isinstance(alpha, np.ndarray) else np.full((h, w), alpha, np.float32)) * v
    return img * v[..., None], a


def hex_to_linear(color: str) -> np.ndarray:
    rgb = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0
    return srgb_to_linear(rgb)
