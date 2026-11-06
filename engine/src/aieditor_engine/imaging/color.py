"""Color math on float32 HxWx3 arrays (vectorized numpy)."""

from __future__ import annotations

import cv2
import numpy as np

# Rec.709 / sRGB luminance weights (linear light)
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    return img @ LUMA


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """H in [0,1), S and V in [0,1]. Uses OpenCV's float path (fast)."""
    flat = np.ascontiguousarray(np.clip(rgb, 0, 1).reshape(-1, 1, 3), dtype=np.float32)
    hsv = cv2.cvtColor(flat, cv2.COLOR_RGB2HSV).reshape(rgb.shape)
    hsv[..., 0] /= 360.0
    return hsv


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    tmp = np.array(hsv, dtype=np.float32, copy=True)
    tmp[..., 0] = (tmp[..., 0] % 1.0) * 360.0
    flat = np.ascontiguousarray(tmp.reshape(-1, 1, 3))
    return cv2.cvtColor(flat, cv2.COLOR_HSV2RGB).reshape(hsv.shape)


def hue_to_rgb(hue_deg: float) -> np.ndarray:
    """Fully saturated unit color for a hue in degrees, normalized to luminance 0 offset."""
    rgb = hsv_to_rgb(np.array([[hue_deg / 360.0, 1.0, 1.0]], dtype=np.float32))[0]
    return rgb


def kelvin_shift(temperature: float, tint: float) -> np.ndarray:
    """Per-channel gains for a creative white-balance shift.

    temperature/tint in [-100, 100]; positive = warmer / more magenta.
    Gains are luminance-normalized so overall brightness stays put.
    """
    t = temperature / 100.0
    m = tint / 100.0
    gains = np.array([1.0 + 0.30 * t + 0.05 * m, 1.0 - 0.10 * m, 1.0 - 0.30 * t + 0.05 * m], dtype=np.float32)
    gains = np.clip(gains, 0.05, None)
    return gains / float(gains @ LUMA)


def monotone_curve(xs: np.ndarray, ys: np.ndarray, n: int = 1024) -> np.ndarray:
    """Fritsch-Carlson monotone cubic through control points; returns an n-entry LUT on [0,1]."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    grid = np.linspace(0, 1, n)
    if len(xs) == 1:
        return np.full(n, ys[0], dtype=np.float32)
    h = np.diff(xs)
    h[h == 0] = 1e-6
    delta = np.diff(ys) / h
    m = np.zeros_like(ys)
    m[0], m[-1] = delta[0], delta[-1]
    for k in range(1, len(xs) - 1):
        if delta[k - 1] * delta[k] <= 0:
            m[k] = 0
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    idx = np.clip(np.searchsorted(xs, grid) - 1, 0, len(xs) - 2)
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    hh = x1 - x0
    t = np.clip((grid - x0) / hh, 0, 1)
    t2, t3 = t * t, t * t * t
    out = (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * hh * m[idx] + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * hh * m[idx + 1]
    out = np.where(grid < xs[0], ys[0], np.where(grid > xs[-1], ys[-1], out))
    return np.clip(out, 0, 1).astype(np.float32)


def apply_lut1d(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
    n = len(lut) - 1
    pos = np.clip(x, 0, 1) * n
    i = np.floor(pos).astype(np.int32)
    i = np.clip(i, 0, n - 1)
    f = pos - i
    return (lut[i] * (1 - f) + lut[i + 1] * f).astype(np.float32)
