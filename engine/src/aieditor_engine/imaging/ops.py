"""Photo operations.

Every op = Pydantic params model + pure function (image, params, ctx) -> image.
Images are float32 HxWx3. Each op declares the domain it wants:
  "linear"  - scene-linear light (physically meaningful: exposure, blur, bloom)
  "display" - sRGB-encoded (perceptual: contrast, curves, grain, textures)
The renderer converts lazily between domains.

Spatial sizes (radius, grain size...) are in *relative pixels*: units of
1/1000 of the image's short side. Preview and full-res export therefore look
the same.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, field_validator

from .color import (
    LUMA,
    apply_lut1d,
    hsv_to_rgb,
    hue_to_rgb,
    kelvin_shift,
    linear_to_srgb,
    luminance,
    monotone_curve,
    rgb_to_hsv,
    smoothstep,
    srgb_to_linear,
)

Domain = Literal["linear", "display"]


@dataclass
class RenderContext:
    seed_base: int = 0
    expand_look: Callable[[str, float], list[tuple[str, dict[str, Any]]]] | None = None
    load_lut: Callable[[str], np.ndarray] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def px(img: np.ndarray, rel: float) -> float:
        return rel * min(img.shape[:2]) / 1000.0


@dataclass(frozen=True)
class OpSpec:
    name: str
    params_model: type[BaseModel]
    fn: Callable[[np.ndarray, Any, RenderContext], np.ndarray]
    domain: Domain
    description: str
    category: str
    positional: tuple[str, ...]
    geometry: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "positional": list(self.positional),
            "params": self.params_model.model_json_schema(),
        }


OPS: dict[str, OpSpec] = {}


def op(
    name: str,
    params: type[BaseModel],
    domain: Domain,
    description: str,
    category: str = "color",
    positional: tuple[str, ...] = (),
    geometry: bool = False,
):
    def deco(fn):
        OPS[name] = OpSpec(name, params, fn, domain, description, category, positional, geometry)
        return fn

    return deco


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.3:
        return img
    if sigma <= 8:
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    # Large blur: do it on a downscaled copy (result is smooth anyway), ~10-30x faster.
    h, w = img.shape[:2]
    f = sigma / 4.0
    small = cv2.resize(img, (max(1, int(w / f)), max(1, int(h / f))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=4.0, sigmaY=4.0, borderType=cv2.BORDER_REFLECT)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _hex(color: str) -> np.ndarray:
    c = color.lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", c):
        raise ValueError(f"bad color: {color}")
    srgb = np.array([int(c[i : i + 2], 16) / 255 for i in (0, 2, 4)], dtype=np.float32)
    return srgb_to_linear(srgb)


def _rng(ctx: RenderContext, seed: int) -> np.random.Generator:
    return np.random.default_rng(ctx.seed_base * 7919 + seed)


def _with_luma(img: np.ndarray, new_l: np.ndarray, old_l: np.ndarray) -> np.ndarray:
    """Scale RGB so luminance becomes new_l while keeping hue/saturation."""
    ratio = new_l / np.maximum(old_l, 1e-5)
    return img * ratio[..., None]


Pct = Field(0.0, ge=-100, le=100)

# ----------------------------------------------------------------- adjust


class AdjustParams(BaseModel):
    exposure: float = Field(0.0, ge=-5, le=5, description="EV stops")
    contrast: float = Pct
    highlights: float = Pct
    shadows: float = Pct
    whites: float = Pct
    blacks: float = Pct
    temperature: float = Field(0.0, ge=-100, le=100, description="+ warmer, - cooler")
    tint: float = Field(0.0, ge=-100, le=100, description="+ magenta, - green")
    vibrance: float = Pct
    saturation: float = Pct
    clarity: float = Pct


@op("adjust", AdjustParams, "linear", "Basic tone and color: exposure, contrast, highlights, shadows, whites, blacks, temperature, tint, vibrance, saturation, clarity.", "tone")
def adjust(img: np.ndarray, p: AdjustParams, ctx: RenderContext) -> np.ndarray:
    if p.exposure:
        img = img * np.float32(2.0**p.exposure)
    if p.temperature or p.tint:
        img = img * kelvin_shift(p.temperature, p.tint)

    d = linear_to_srgb(img)
    lum = luminance(d)

    if p.whites or p.blacks:
        b = -p.blacks / 100 * 0.08
        w = 1 - p.whites / 100 * 0.08
        d = (d - b) / (w - b)
        lum = luminance(d)

    if p.highlights or p.shadows:
        base = _gauss(lum, ctx.px(img, 8))  # soft mask avoids posterized edges
        mh = smoothstep(0.45, 1.0, base)
        ms = 1 - smoothstep(0.0, 0.55, base)
        new_l = lum + (p.highlights / 100) * 0.30 * mh * (1 - lum if p.highlights > 0 else lum) \
                    + (p.shadows / 100) * 0.35 * ms * (1 - lum if p.shadows > 0 else lum)
        d = _with_luma(d, np.clip(new_l, 0, None), lum)
        lum = luminance(d)

    if p.clarity:
        base = _gauss(lum, ctx.px(img, 25))
        detail = lum - base
        mid = 1 - np.abs(2 * np.clip(lum, 0, 1) - 1) ** 2
        new_l = lum + (p.clarity / 100) * 1.2 * detail * mid
        d = _with_luma(d, np.clip(new_l, 0, None), lum)
        lum = luminance(d)

    if p.contrast:
        c = p.contrast / 100
        x = np.clip(lum, 0, 1)
        s_curve = x * x * (3 - 2 * x)
        flat = 0.5 + (x - 0.5) * 0.55
        new_l = x + c * (s_curve - x) * 1.4 if c > 0 else x + (-c) * (flat - x)
        d = _with_luma(d, np.clip(new_l, 0, None) + (lum - x), lum)
        lum = luminance(d)

    if p.vibrance or p.saturation:
        sat_now = d.max(axis=-1) - d.min(axis=-1)
        amt = np.full_like(lum, p.saturation / 100)
        if p.vibrance:
            amt = amt + (p.vibrance / 100) * (1 - np.clip(sat_now, 0, 1)) ** 2
        d = lum[..., None] + (d - lum[..., None]) * (1 + amt)[..., None]

    return srgb_to_linear(np.clip(d, 0, None))


# ------------------------------------------------------------------ curves


def _parse_points(v: Any) -> list[tuple[float, float]] | None:
    """Accept [[x,y],...] or "x:y,x:y" (0..1 or 0..255)."""
    if v is None or v == "":
        return None
    if isinstance(v, str):
        pts = []
        for part in v.replace(" ", "").split(","):
            x, y = part.split(":")
            pts.append((float(x), float(y)))
    else:
        pts = [(float(a), float(b)) for a, b in v]
    if any(x > 1 or y > 1 for x, y in pts):
        pts = [(x / 255, y / 255) for x, y in pts]
    if len(pts) < 2:
        raise ValueError("curve needs at least 2 points")
    return pts


class CurvesParams(BaseModel):
    rgb: list[tuple[float, float]] | None = Field(None, description='master curve, e.g. "0:0,0.25:0.2,0.75:0.8,1:1"')
    r: list[tuple[float, float]] | None = None
    g: list[tuple[float, float]] | None = None
    b: list[tuple[float, float]] | None = None

    _pts = field_validator("rgb", "r", "g", "b", mode="before")(lambda cls, v: _parse_points(v))


@op("curves", CurvesParams, "display", "Tone curves (master and per channel), points as x:y pairs.", "tone", ("rgb",))
def curves(img: np.ndarray, p: CurvesParams, ctx: RenderContext) -> np.ndarray:
    out = img.copy()
    if p.rgb:
        lut = monotone_curve(*zip(*p.rgb))
        out = apply_lut1d(out, lut)
    for i, pts in enumerate((p.r, p.g, p.b)):
        if pts:
            out[..., i] = apply_lut1d(out[..., i], monotone_curve(*zip(*pts)))
    return out


# --------------------------------------------------------------------- hsl

BANDS = {"red": 0, "orange": 30, "yellow": 60, "green": 120, "aqua": 180, "blue": 225, "purple": 270, "magenta": 315}


class HslParams(BaseModel):
    band: Literal["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
    hue: float = Pct
    sat: float = Pct
    lum: float = Pct


@op("hsl", HslParams, "display", "Shift hue/saturation/luminance of one color band (e.g. make blues deeper, greens muted).", "color", ("band",))
def hsl(img: np.ndarray, p: HslParams, ctx: RenderContext) -> np.ndarray:
    hsv = rgb_to_hsv(np.clip(img, 0, 1))
    center = BANDS[p.band] / 360
    dist = np.abs(((hsv[..., 0] - center) + 0.5) % 1.0 - 0.5) * 360
    w = np.clip(1 - dist / 45, 0, 1)
    w = w * w * (3 - 2 * w) * smoothstep(0.02, 0.2, hsv[..., 1])
    hsv[..., 0] = hsv[..., 0] + w * (p.hue / 100) * (30 / 360)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + w * p.sat / 100), 0, 1)
    hsv[..., 2] = hsv[..., 2] * (1 + w * p.lum / 100 * 0.5)
    return hsv_to_rgb(hsv)


# ------------------------------------------------------------- color grade


class GradeParams(BaseModel):
    shadows_hue: float = Field(220, ge=0, le=360)
    shadows_sat: float = Field(0, ge=0, le=100)
    midtones_hue: float = Field(40, ge=0, le=360)
    midtones_sat: float = Field(0, ge=0, le=100)
    highlights_hue: float = Field(40, ge=0, le=360)
    highlights_sat: float = Field(0, ge=0, le=100)
    balance: float = Field(0, ge=-100, le=100, description="+ favors highlights")


@op("color_grade", GradeParams, "display", "Split-tone / three-way grade: tint shadows, midtones and highlights (hue in degrees).", "color")
def color_grade(img: np.ndarray, p: GradeParams, ctx: RenderContext) -> np.ndarray:
    lum = np.clip(luminance(img), 0, 1)
    pivot = 0.5 - p.balance / 100 * 0.25
    ms = 1 - smoothstep(0, pivot, lum)
    mh = smoothstep(pivot, 1, lum)
    mm = np.clip(1 - ms - mh, 0, 1)
    out = img
    for hue, sat, mask in ((p.shadows_hue, p.shadows_sat, ms), (p.midtones_hue, p.midtones_sat, mm), (p.highlights_hue, p.highlights_sat, mh)):
        if sat <= 0:
            continue
        tint = hue_to_rgb(hue)
        tint = tint - float(tint @ LUMA)  # zero-luminance color offset
        out = out + mask[..., None] * tint * (sat / 100) * 0.18
    return out


# ---------------------------------------------------------- black & white


class BWParams(BaseModel):
    amount: float = Field(1.0, ge=0, le=1)
    red: float = Field(0, ge=-100, le=100)
    green: float = Field(0, ge=-100, le=100)
    blue: float = Field(0, ge=-100, le=100)


@op("black_white", BWParams, "linear", "Convert to monochrome with a channel mixer (amount 0..1 for partial desaturation).", "color", ("amount",))
def black_white(img: np.ndarray, p: BWParams, ctx: RenderContext) -> np.ndarray:
    w = LUMA * np.array([1 + p.red / 100, 1 + p.green / 100, 1 + p.blue / 100], dtype=np.float32)
    w = w / max(float(w.sum()), 1e-3)
    grey = (img @ w)[..., None]
    return img + (grey - img) * p.amount


# --------------------------------------------------------------------- LUT


class LutParams(BaseModel):
    name: str
    intensity: float = Field(1.0, ge=0, le=1)


@op("lut", LutParams, "display", "Apply a 3D .cube LUT by name.", "color", ("name",))
def lut(img: np.ndarray, p: LutParams, ctx: RenderContext) -> np.ndarray:
    if ctx.load_lut is None:
        raise ValueError("LUTs unavailable")
    from .lut import apply_lut3d

    graded = apply_lut3d(np.clip(img, 0, 1), ctx.load_lut(p.name))
    return img + (graded - img) * p.intensity


# -------------------------------------------------------------------- fade


class FadeParams(BaseModel):
    amount: float = Field(0.3, ge=0, le=1, description="matte / lifted blacks")


@op("fade", FadeParams, "display", "Matte film fade: lift blacks and soften highlights.", "tone", ("amount",))
def fade(img: np.ndarray, p: FadeParams, ctx: RenderContext) -> np.ndarray:
    lift, roll = 0.12 * p.amount, 0.06 * p.amount
    return lift + img * (1 - lift - roll)


# ---------------------------------------------------------------- detail


class SharpenParams(BaseModel):
    amount: float = Field(40, ge=0, le=150)
    radius: float = Field(1.0, ge=0.3, le=5, description="relative px")


@op("sharpen", SharpenParams, "display", "Unsharp-mask sharpening on luminance.", "detail", ("amount",))
def sharpen(img: np.ndarray, p: SharpenParams, ctx: RenderContext) -> np.ndarray:
    lum = luminance(img)
    detail = lum - _gauss(lum, max(0.5, ctx.px(img, p.radius)))
    return _with_luma(img, np.clip(lum + detail * p.amount / 100 * 1.5, 0, None), lum)


class DenoiseParams(BaseModel):
    strength: float = Field(30, ge=0, le=100)


@op("denoise", DenoiseParams, "display", "Edge-preserving noise reduction.", "detail", ("strength",))
def denoise(img: np.ndarray, p: DenoiseParams, ctx: RenderContext) -> np.ndarray:
    if p.strength <= 0:
        return img
    d = int(max(3, min(15, ctx.px(img, 3))))
    return cv2.bilateralFilter(np.ascontiguousarray(img, dtype=np.float32), d, p.strength / 100 * 0.12, d * 2.0)


class DehazeParams(BaseModel):
    amount: float = Field(30, ge=-100, le=100)


@op("dehaze", DehazeParams, "display", "Remove (or add, if negative) atmospheric haze.", "tone", ("amount",))
def dehaze(img: np.ndarray, p: DehazeParams, ctx: RenderContext) -> np.ndarray:
    a = p.amount / 100
    k = max(3, int(ctx.px(img, 15)) | 1)
    dark = cv2.erode(img.min(axis=-1), cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    dark = _gauss(dark, k)
    atmos = float(np.percentile(luminance(img), 99.5))
    atmos = max(atmos, 0.2)
    if a >= 0:
        t = np.clip(1 - 0.85 * a * dark / atmos, 0.25, 1)[..., None]
        return (img - atmos) / t + atmos
    return img + (atmos * 0.9 - img) * (-a) * 0.45


class BlurParams(BaseModel):
    radius: float = Field(5, ge=0, le=200, description="relative px")


@op("blur", BlurParams, "linear", "Gaussian blur of the whole image.", "detail", ("radius",))
def blur(img: np.ndarray, p: BlurParams, ctx: RenderContext) -> np.ndarray:
    return _gauss(img, ctx.px(img, p.radius))


# ---------------------------------------------------------------- effects


class GrainParams(BaseModel):
    amount: float = Field(0.3, ge=0, le=1)
    size: float = Field(1.2, ge=0.5, le=6, description="relative px")
    seed: int = 1


@op("grain", GrainParams, "display", "Film grain (luminance, strongest in midtones).", "effects", ("amount",))
def grain(img: np.ndarray, p: GrainParams, ctx: RenderContext) -> np.ndarray:
    if p.amount <= 0:
        return img
    h, w = img.shape[:2]
    cell = max(1.0, ctx.px(img, p.size))
    gh, gw = max(2, int(h / cell)), max(2, int(w / cell))
    noise = _rng(ctx, p.seed).standard_normal((gh, gw)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    lum = np.clip(luminance(img), 0, 1)
    strength = p.amount * 0.12 * (0.35 + 2.6 * lum * (1 - lum))
    return img + (noise * strength)[..., None]


class VignetteParams(BaseModel):
    amount: float = Field(0.35, ge=-1, le=1, description="+ darkens edges, - lightens")
    midpoint: float = Field(0.5, ge=0, le=1)
    feather: float = Field(0.5, ge=0.05, le=1)


@op("vignette", VignetteParams, "linear", "Darken (or lighten) the corners.", "effects", ("amount",))
def vignette(img: np.ndarray, p: VignetteParams, ctx: RenderContext) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = np.ogrid[-1 : 1 : complex(0, h), -1 : 1 : complex(0, w)]
    aspect = w / h
    r = np.sqrt((x * min(aspect, 1)) ** 2 + (y / max(aspect, 1)) ** 2) / math.sqrt(2) * 1.4
    start = 0.25 + 0.6 * p.midpoint
    mask = smoothstep(start - p.feather * 0.5, start + p.feather * 0.6, r).astype(np.float32)
    gain = 1 - p.amount * 0.85 * mask if p.amount >= 0 else 1 + (-p.amount) * 0.6 * mask
    return img * gain[..., None]


class BloomParams(BaseModel):
    amount: float = Field(0.3, ge=0, le=1)
    threshold: float = Field(0.7, ge=0, le=1, description="display brightness where glow starts")
    radius: float = Field(30, ge=1, le=200, description="relative px")
    tint: str = Field("#ffffff", description="glow color; reddish = film halation")


@op("bloom", BloomParams, "linear", "Glow around bright areas (dreamy look, or red-tinted film halation).", "effects", ("amount",))
def bloom(img: np.ndarray, p: BloomParams, ctx: RenderContext) -> np.ndarray:
    thr = float(srgb_to_linear(np.float32(p.threshold)))
    bright = np.clip(img - thr, 0, None)
    glow = _gauss(bright, ctx.px(img, p.radius))
    return img + glow * _hex(p.tint) * (p.amount * 2.0)


class CAParams(BaseModel):
    amount: float = Field(0.3, ge=0, le=1)


@op("chromatic_aberration", CAParams, "linear", "Lens color fringing toward the edges.", "effects", ("amount",))
def chromatic_aberration(img: np.ndarray, p: CAParams, ctx: RenderContext) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for ch, s in ((0, 1 + p.amount * 0.004), (2, 1 - p.amount * 0.004)):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
        out[..., ch] = cv2.warpAffine(img[..., ch], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


class LightLeakParams(BaseModel):
    amount: float = Field(0.4, ge=0, le=1)
    position: Literal["left", "right", "top", "bottom", "top-left", "top-right", "bottom-left", "bottom-right"] = "top-right"
    color: str = "#ff7a3d"


@op("light_leak", LightLeakParams, "linear", "Warm film light leak glowing in from an edge.", "effects", ("amount",))
def light_leak(img: np.ndarray, p: LightLeakParams, ctx: RenderContext) -> np.ndarray:
    h, w = img.shape[:2]
    cx = {"left": 0, "right": 1, "top": 0.5, "bottom": 0.5}
    cy = {"left": 0.5, "right": 0.5, "top": 0, "bottom": 1}
    parts = p.position.split("-")
    px_ = 0.5 if len(parts) == 1 and parts[0] in ("top", "bottom") else (1.0 if "right" in parts else 0.0)
    py_ = 0.5 if len(parts) == 1 and parts[0] in ("left", "right") else (1.0 if "bottom" in parts else 0.0)
    if len(parts) == 1:
        px_, py_ = cx[parts[0]], cy[parts[0]]
    y, x = np.ogrid[0:1:complex(0, h), 0:1:complex(0, w)]
    d = np.sqrt(((x - px_) * (w / max(w, h))) ** 2 + ((y - py_) * (h / max(w, h))) ** 2)
    mask = np.exp(-(d / 0.45) ** 2).astype(np.float32)
    return img + mask[..., None] * _hex(p.color) * p.amount * 1.2


class DustParams(BaseModel):
    amount: float = Field(0.3, ge=0, le=1)
    seed: int = 3


@op("dust_scratches", DustParams, "display", "Old-film dust specks and scratches.", "effects", ("amount",))
def dust_scratches(img: np.ndarray, p: DustParams, ctx: RenderContext) -> np.ndarray:
    h, w = img.shape[:2]
    rng = _rng(ctx, p.seed)
    overlay = np.zeros((h, w), np.float32)
    unit = min(h, w) / 1000
    n = int(p.amount * 140)
    for _ in range(n):
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        r = max(1, int(rng.uniform(0.6, 3.2) * unit))
        cv2.circle(overlay, (cx, cy), r, float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0)), -1, lineType=cv2.LINE_AA)
    for _ in range(int(p.amount * 6)):
        x0 = int(rng.uniform(0, w))
        drift = int(rng.uniform(-8, 8) * unit)
        cv2.line(overlay, (x0, 0), (x0 + drift, h), float(rng.uniform(0.3, 0.8)), max(1, int(unit)), lineType=cv2.LINE_AA)
    overlay = _gauss(overlay, max(0.5, unit * 0.6))
    return img + overlay[..., None] * 0.35 * p.amount


class TextureParams(BaseModel):
    amount: float = Field(0.3, ge=0, le=1)
    kind: Literal["paper", "film"] = "paper"
    seed: int = 5


@op("texture", TextureParams, "display", "Aged paper or worn film texture overlay.", "effects", ("amount",))
def texture(img: np.ndarray, p: TextureParams, ctx: RenderContext) -> np.ndarray:
    h, w = img.shape[:2]
    rng = _rng(ctx, p.seed)
    coarse = cv2.resize(rng.standard_normal((12, 12)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    fine_cell = max(1.0, ctx.px(img, 2.5 if p.kind == "paper" else 1.0))
    fine = cv2.resize(
        rng.standard_normal((max(2, int(h / fine_cell)), max(2, int(w / fine_cell)))).astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )
    tex = coarse * 0.6 + fine * 0.4
    mult = 1 + tex * 0.09 * p.amount
    out = img * mult[..., None]
    if p.kind == "paper":  # warm, slightly yellowed base
        out = out * np.array([1.0, 0.985, 0.94], np.float32) ** (p.amount * 2)
    return out


# --------------------------------------------------------------- geometry


class CropParams(BaseModel):
    aspect: str | None = Field(None, description='"1:1", "4:5", "16:9", "9:16", "3:2"…')
    x: float = Field(0, ge=0, le=1)
    y: float = Field(0, ge=0, le=1)
    w: float = Field(1, gt=0, le=1)
    h: float = Field(1, gt=0, le=1)
    focus_x: float = Field(0.5, ge=0, le=1, description="keep this point centered when cropping to aspect")
    focus_y: float = Field(0.5, ge=0, le=1)

    @field_validator("aspect")
    @classmethod
    def _aspect(cls, v: str | None) -> str | None:
        if v is None or v == "original":
            return None
        if not re.fullmatch(r"\d+(\.\d+)?:\d+(\.\d+)?", v):
            raise ValueError('aspect must look like "4:5"')
        return v


@op("crop", CropParams, "linear", "Crop to an aspect ratio (centered on a focus point) or to a normalized box.", "geometry", ("aspect",), geometry=True)
def crop(img: np.ndarray, p: CropParams, ctx: RenderContext) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = int(p.x * w), int(p.y * h)
    cw, ch = max(1, int(min(p.w, 1 - p.x) * w)), max(1, int(min(p.h, 1 - p.y) * h))
    if p.aspect:
        a, b = (float(s) for s in p.aspect.split(":"))
        target = a / b
        if cw / ch > target:
            nw = max(1, int(round(ch * target)))
            x0 = x0 + int(np.clip(p.focus_x * cw - nw / 2, 0, cw - nw))
            cw = nw
        else:
            nh = max(1, int(round(cw / target)))
            y0 = y0 + int(np.clip(p.focus_y * ch - nh / 2, 0, ch - nh))
            ch = nh
    return img[y0 : y0 + ch, x0 : x0 + cw].copy()


class RotateParams(BaseModel):
    angle: float = Field(0, ge=-180, le=180, description="degrees, + counter-clockwise")
    fill: Literal["crop", "black"] = "crop"


def _rotate(img: np.ndarray, angle: float, fill: str) -> np.ndarray:
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    if fill == "crop" and abs(angle) <= 45:
        # zoom so no empty corners remain (largest inscribed rect, same aspect)
        a = math.radians(abs(angle))
        scale = math.cos(a) + max(w / h, h / w) * math.sin(a)
    else:
        scale = 1.0
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)


@op("rotate", RotateParams, "linear", "Rotate by an angle (auto-crops empty corners).", "geometry", ("angle",), geometry=True)
def rotate(img: np.ndarray, p: RotateParams, ctx: RenderContext) -> np.ndarray:
    if p.angle in (90, -90, 180, -180) :
        k = {90: 1, -90: 3, 180: 2, -180: 2}[int(p.angle)]
        return np.ascontiguousarray(np.rot90(img, k))
    return _rotate(img, p.angle, p.fill)


def detect_horizon_angle(img: np.ndarray) -> float:
    """Angle (deg) that levels the dominant near-horizontal lines; 0 if unsure."""
    # Fixed working size -> same answer for preview and full-res export.
    s = 800 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    grey = (np.clip(linear_to_srgb(luminance(small)), 0, 1) * 255).astype(np.uint8)
    grey = cv2.GaussianBlur(grey, (5, 5), 1.2)
    med = float(np.median(grey))
    edges = cv2.Canny(grey, max(10, 0.4 * med), max(30, 1.0 * med))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1440, threshold=50, minLineLength=small.shape[1] // 5, maxLineGap=10)
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # OpenCV 4: (N,1,4); 5: (N,4)
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(ang) <= 12:
            angles.append(ang)
            weights.append(math.hypot(x2 - x1, y2 - y1))
    if sum(weights) < small.shape[1] * 0.3:  # not enough horizontal structure to trust
        return 0.0
    order = np.argsort(angles)
    cum = np.cumsum(np.array(weights)[order])
    median = float(np.array(angles)[order][np.searchsorted(cum, cum[-1] / 2)])
    return median if abs(median) >= 0.2 else 0.0


class StraightenParams(BaseModel):
    pass


@op("straighten", StraightenParams, "linear", "Auto-level a tilted horizon.", "geometry", geometry=True)
def straighten(img: np.ndarray, p: StraightenParams, ctx: RenderContext) -> np.ndarray:
    return _rotate(img, detect_horizon_angle(img), "crop")


class FlipParams(BaseModel):
    horizontal: bool = True
    vertical: bool = False


@op("flip", FlipParams, "linear", "Mirror the image.", "geometry", geometry=True)
def flip(img: np.ndarray, p: FlipParams, ctx: RenderContext) -> np.ndarray:
    if p.horizontal:
        img = img[:, ::-1]
    if p.vertical:
        img = img[::-1]
    return np.ascontiguousarray(img)


# ------------------------------------------------------------------- look


class LookParams(BaseModel):
    name: str
    strength: float = Field(1.0, ge=0, le=1.5)


@op("look", LookParams, "linear", "Apply a named look (recipe) at a strength; 1 = as designed.", "looks", ("name",))
def look(img: np.ndarray, p: LookParams, ctx: RenderContext) -> np.ndarray:
    raise RuntimeError("look is expanded by the renderer")  # handled in render.apply_stack


def validate_op(name: str, params: dict[str, Any]) -> BaseModel:
    spec = OPS.get(name)
    if spec is None:
        raise ValueError(f"unknown op: {name}")
    return spec.params_model.model_validate(params)
