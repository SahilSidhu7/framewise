"""Run the real AI photo pipeline on sample images and save before/after sheets.

Usage (from engine/):  .venv/Scripts/python ../demo/make_photo_demos.py
"""

from __future__ import annotations

import shutil
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "engine" / "src"))

from aieditor_engine.agent.planner import plan  # noqa: E402
from aieditor_engine.dsl import compile_program  # noqa: E402
from aieditor_engine.imaging.render import SOURCES, analyze, apply_stack, load_linear, to_display  # noqa: E402

OUT = ROOT / "output"
WALL = Path(r"C:\Windows\Web")
FONT = ROOT.parent / "engine" / "src" / "aieditor_engine" / "fonts" / "Inter[opsz,wght].ttf"

CASES = [
    (WALL / "Wallpaper/ThemeC/img28.jpg", "make it cinematic, like a movie"),
    (WALL / "Wallpaper/ThemeC/img30.jpg", "give it an ancient look"),
    (WALL / "Screen/img102.jpg", "warm golden hour, a bit dreamy"),
    (WALL / "Wallpaper/ThemeC/img31.jpg", "black and white, dramatic"),
    (WALL / "Wallpaper/ThemeC/img29.jpg", "vintage 70s film with grain"),
    ("degraded", "fix it"),
]

# Real camera photos (Wikimedia Commons, see demo/stock/CREDITS.txt); used when a stock dir is given.
STOCK_CASES = [
    ("portrait.jpg", "make it look like a magazine portrait"),
    ("desert_portrait.jpg", "warm golden hour, a bit dreamy"),
    ("food.jpg", "make the food look delicious"),
    ("pastry.jpg", "moody dark aesthetic"),
    ("portrait.jpg", "black and white, dramatic"),
    ("degraded", "fix it"),
]
# Style set (demo/stock/photos, see photo_credits.json there): one photo per look.
STYLE_CASES = [
    ("lake_fog.jpg", "dreamy soft pastel morning"),
    ("old_street.jpg", "vintage 70s film with grain"),
    ("coffee.jpg", "moody dark aesthetic"),
    ("lighthouse.jpg", "black and white, dramatic"),
    ("dunes.jpg", "warm golden hour, make the colours pop"),
]
DEGRADE_FROM = WALL / "Wallpaper/ThemeC/img29.jpg"


def degraded_source(dst: Path) -> Path:
    """A 'bad phone photo': underexposed, blue cast, tilted 3 degrees."""
    img = cv2.imread(str(DEGRADE_FROM))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.08)
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    img = img.astype(np.float32)
    img *= np.array([1.15, 0.95, 0.75], np.float32)  # BGR: more blue, less red
    img = 255 * (img / 255) ** 1.6 * 0.8  # dark and flat
    cv2.imwrite(str(dst), np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return dst


def label_font(px: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT), px)
    f.set_variation_by_axes([24, 600])
    return f


def sheet(before: np.ndarray, after: np.ndarray, prompt: str, program: str, rationale: str, dst: Path) -> None:
    h = 900
    def fit(a: np.ndarray) -> Image.Image:
        im = Image.fromarray(a)
        return im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
    b, a = fit(before), fit(after)
    pad, top, bottom = 24, 90, 150
    W = b.width + a.width + pad * 3
    canvas = Image.new("RGB", (W, h + top + bottom), (18, 19, 22))
    canvas.paste(b, (pad, top))
    canvas.paste(a, (pad * 2 + b.width, top))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 24), f"Request: “{prompt}”", font=label_font(40), fill=(240, 240, 240))
    d.text((pad + 10, top + 10), "BEFORE", font=label_font(30), fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    d.text((pad * 2 + b.width + 10, top + 10), "AFTER (AI edit)", font=label_font(30), fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    lines = [f"Why: {rationale}"] + textwrap.wrap("Edit program: " + program.replace("\n", "  |  "), 150)
    for i, line in enumerate(lines[:4]):
        d.text((pad, top + h + 16 + i * 32), line, font=label_font(24), fill=(170, 175, 185))
    canvas.save(dst, quality=90)


def main() -> None:
    global DEGRADE_FROM
    cases, prefix = CASES, "photo"
    if len(sys.argv) > 1:  # stock dir of real photos
        stock = Path(sys.argv[1])
        if (stock / "lake_fog.jpg").exists():
            cases, prefix = [(stock / s, p) for s, p in STYLE_CASES], "style"
        else:
            cases = [(s if s == "degraded" else stock / s, p) for s, p in STOCK_CASES]
            DEGRADE_FROM, prefix = stock / "food.jpg", "real"
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "work"
    work.mkdir(exist_ok=True)
    results = []
    for i, (src, prompt) in enumerate(cases, start=1):
        if src == "degraded":
            src = degraded_source(work / "bad_phone_photo.jpg")
        t = time.time()
        analysis = analyze(SOURCES.get(src, 1024))
        res = plan(prompt, analysis, "", 3)
        v = res["variations"][0]
        ops = compile_program(v["program"])
        img = load_linear(src, 2400)
        before = to_display(img)
        after = to_display(apply_stack(img, ops))
        name = f"{prefix}_{i}_{prompt.split(',')[0].replace(' ', '_')[:28]}.jpg"
        sheet(before, after, prompt, v["program"], v["rationale"] or "-", OUT / name)
        if prefix == "style":  # plain before/after pair for the website slider
            for tag, arr in (("before", before), ("after", after)):
                im = Image.fromarray(arr)
                im.thumbnail((1600, 1600), Image.LANCZOS)
                im.save(OUT / f"{name[:-4]}_{tag}.jpg", quality=86)
        dt = time.time() - t
        results.append((name, prompt, v["source"], v["program"].replace("\n", " | "), analysis["notes"], dt))
        print(f"{name}: [{v['source']}] {v['program'].replace(chr(10), ' | ')}  notes={analysis['notes']}  {dt:.1f}s")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
