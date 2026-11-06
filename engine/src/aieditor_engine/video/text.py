"""Caption and title rendering (Pillow, display space, on the final 8-bit frame).

Sizes are relative to the frame's short side, so a 360p preview and a 4K
export look identical. Bundled fonts are OFL (see fonts/*-OFL.txt).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path(__file__).resolve().parents[1] / "fonts"
FONTS = {
    "sans": "Inter[opsz,wght].ttf",
    "display": "Montserrat[wght].ttf",
    "serif": "PlayfairDisplay[wght].ttf",
}


@dataclass(frozen=True)
class Style:
    family: str
    weight: int
    size: float  # fraction of the short side
    position: str
    color: tuple[int, int, int] = (255, 255, 255)
    stroke: float = 0.0  # fraction of font size
    shadow: bool = False
    box: bool = False
    upper: bool = False
    tracking: float = 0.0  # extra letter spacing, fraction of font size
    max_chars: int = 42
    fade: float = 0.12  # seconds
    pop: bool = False  # quick scale-in


STYLES: dict[str, Style] = {
    "clean": Style("sans", 650, 0.052, "bottom", stroke=0.09, shadow=True),
    "bold_pop": Style("display", 850, 0.074, "center_low", stroke=0.13, upper=True, max_chars=22, pop=True, fade=0.05),
    "cinematic": Style("serif", 600, 0.044, "lower_third", stroke=0.06, shadow=True, tracking=0.05, fade=0.3),
    "boxed": Style("sans", 600, 0.046, "bottom", box=True),
    "title": Style("serif", 700, 0.11, "center", shadow=True, max_chars=26, fade=0.5, tracking=0.02),
    "minimal": Style("sans", 400, 0.036, "bottom", shadow=True, fade=0.2),
}
STYLE_INFO = {
    "clean": "White subtitles with an outline. Readable on anything.",
    "bold_pop": "Big bold uppercase words that pop in. Reels / TikTok.",
    "cinematic": "Elegant serif, lower third, soft shadow.",
    "boxed": "White text on a dark box. Maximum legibility.",
    "title": "Large centered title.",
    "minimal": "Small, light and unobtrusive.",
}



@lru_cache(maxsize=64)
def _font(family: str, weight: int, px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = FONT_DIR / FONTS.get(family, FONTS["sans"])
    try:
        font = ImageFont.truetype(str(path), px)
        axes = font.get_variation_axes()
        values = []
        for axis in axes:
            name = axis["name"].decode() if isinstance(axis["name"], bytes) else str(axis["name"])
            if name.lower().startswith("weight"):
                values.append(max(axis["minimum"], min(axis["maximum"], weight)))
            elif name.lower().startswith("optical"):
                values.append(max(axis["minimum"], min(axis["maximum"], px)))
            else:
                values.append(axis["default"])
        if values:
            font.set_variation_by_axes(values)
        return font
    except OSError:
        return ImageFont.load_default(px)


def _wrap(text: str, max_chars: int) -> list[str]:
    lines = textwrap.wrap(" ".join(text.split()), max_chars) or [text]
    if len(lines) > 2:  # playbook: max two lines; squeeze into two balanced lines
        words = text.split()
        half = max(1, len(words) // 2)
        lines = [" ".join(words[:half]), " ".join(words[half:])]
    return lines


def _anchor_y(position: str, h: int, block_h: int, portrait: bool) -> int:
    if position == "top":
        return int(h * 0.08)
    if position == "center":
        return (h - block_h) // 2
    if position == "center_low":
        return int(h * (0.62 if portrait else 0.6)) - block_h // 2
    if position == "lower_third":
        return int(h * 0.80) - block_h // 2
    # bottom: stay above platform UI on vertical video
    return int(h * (0.76 if portrait else 0.9)) - block_h


def draw_text(frame: np.ndarray, text: str, style_name: str, position: str | None, age: float, remaining: float) -> None:
    """Draw one caption onto a uint8 RGB frame in place. age/remaining (seconds)
    drive fade and pop animations."""
    st = STYLES.get(style_name, STYLES["clean"])
    h, w = frame.shape[:2]
    short = min(w, h)
    portrait = h > w
    alpha = 1.0
    if st.fade > 0:
        alpha = min(1.0, max(0.0, age / st.fade), max(0.0, remaining / st.fade))
    if alpha <= 0.01:
        return
    scale = 1.0
    if st.pop and age < 0.12:
        scale = 0.82 + 0.18 * (age / 0.12)

    body = text.upper() if st.upper else text
    px = max(8, int(st.size * short * scale))
    font = _font(st.family, st.weight, px)
    lines = _wrap(body, st.max_chars)
    max_w = int(w * 0.88)
    # shrink to fit width
    for _ in range(6):
        widths = [_line_width(font, ln, px, st.tracking) for ln in lines]
        if max(widths) <= max_w or px <= 10:
            break
        px = max(10, int(px * max_w / max(widths) * 0.98))
        font = _font(st.family, st.weight, px)
    widths = [_line_width(font, ln, px, st.tracking) for ln in lines]
    line_h = int(px * 1.22)
    block_h = line_h * len(lines)
    y0 = _anchor_y(position or st.position, h, block_h, portrait)

    pad = int(px * 0.6)
    bx0 = max(0, (w - max(widths)) // 2 - pad)
    by0 = max(0, y0 - pad // 2)
    bx1 = min(w, (w + max(widths)) // 2 + pad)
    by1 = min(h, y0 + block_h + pad // 2)
    if bx1 <= bx0 or by1 <= by0:
        return
    # draw on a small RGBA layer covering the text block, then alpha-blend
    layer = Image.new("RGBA", (bx1 - bx0, by1 - by0), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if st.box:
        draw.rounded_rectangle((0, 0, layer.width - 1, layer.height - 1), radius=int(px * 0.3), fill=(0, 0, 0, 170))
    stroke_w = int(round(st.stroke * px))
    text_layer = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(text_layer)
    for i, (ln, lw) in enumerate(zip(lines, widths, strict=True)):
        x = (w - lw) // 2 - bx0
        y = y0 + i * line_h - by0
        _draw_line(tdraw, (x, y), ln, font, px, st, stroke_w)
    if st.shadow:
        shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        shadow.putalpha(text_layer.getchannel("A").point(lambda v: int(v * 0.7)))
        shadow = shadow.filter(ImageFilter.GaussianBlur(max(1, px * 0.08)))
        off = max(1, int(px * 0.05))
        layer.alpha_composite(shadow, (off, off))
    layer.alpha_composite(text_layer)

    rgba = np.asarray(layer, dtype=np.float32) / 255.0
    a = rgba[..., 3:4] * alpha
    region = frame[by0:by1, bx0:bx1].astype(np.float32)
    frame[by0:by1, bx0:bx1] = np.clip(region * (1 - a) + rgba[..., :3] * 255.0 * a + 0.5, 0, 255).astype(np.uint8)


def _line_width(font, text: str, px: int, tracking: float) -> int:
    base = font.getlength(text)
    return int(base + max(0, len(text) - 1) * tracking * px)


def _draw_line(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, px: int, st: Style, stroke_w: int) -> None:
    fill = (*st.color, 255)
    stroke = (0, 0, 0, 255)
    if st.tracking <= 0:
        draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)
        return
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke)
        x += font.getlength(ch) + st.tracking * px
