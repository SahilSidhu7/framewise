"""Copy demo output into the website: styled photo pairs, the travel film's web copy, credits.

Usage (from engine/):  .venv/Scripts/python ../demo/build_site_examples.py
Needs demo/output from make_photo_demos.py (stock/photos) and make_video_demos.py (stock/travel).
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import av
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SITE = ROOT.parent / "site"
STOCK = ROOT / "stock"
PAGES_LIMIT = 25 * 1024 * 1024  # Cloudflare Pages per-file limit

PHOTOS = [  # (output prefix, stock key, request)
    ("style_5_warm_golden_hour", "dunes", "warm golden hour, make the colours pop"),
    ("style_2_vintage_70s_film_with_grain", "old_street", "vintage 70s film with grain"),
    ("style_3_moody_dark_aesthetic", "coffee", "moody dark aesthetic"),
    ("style_4_black_and_white", "lighthouse", "black and white, dramatic"),
    ("style_1_dreamy_soft_pastel_morning", "lake_fog", "dreamy soft pastel morning"),
]


def photos() -> tuple[str, list[str]]:
    dst = SITE / "img" / "examples"
    dst.mkdir(parents=True, exist_ok=True)
    credits = json.loads((STOCK / "photos" / "photo_credits.json").read_text(encoding="utf-8"))
    figs, notes = [], []
    for prefix, key, ask in PHOTOS:
        for tag in ("before", "after"):
            im = Image.open(OUT / f"{prefix}_{tag}.jpg").convert("RGB")
            im.thumbnail((1400, 1400), Image.LANCZOS)
            im.save(dst / f"{key}_{tag}.jpg", quality=82, optimize=True, progressive=True)
        w, h = im.size
        c = credits[key]
        alt = html.escape(re.sub(r"\.jpg$", "", c["title"]))
        figs.append(
            f'      <figure>\n        <div class="ba" style="aspect-ratio:{w}/{h}">\n'
            f'          <img src="/img/examples/{key}_before.jpg" width="{w}" height="{h}" alt="{alt}, original" loading="lazy">\n'
            f'          <img class="after" src="/img/examples/{key}_after.jpg" width="{w}" height="{h}" alt="{alt}, edited by Framewise" loading="lazy">\n'
            f'          <span class="rule"></span><span class="tag l">Before</span><span class="tag r">After</span>\n'
            f'          <input type="range" min="0" max="100" value="50" aria-label="Compare original and edited: {html.escape(ask)}">\n'
            f'        </div>\n        <figcaption><q>{html.escape(ask)}</q></figcaption>\n      </figure>'
        )
        if c["license"] != "CC0":
            notes.append(f'"{html.escape(alt)}" by {html.escape(c["author"])} ({c["license"]})')
    return "\n".join(figs), notes


def film() -> tuple[str, list[str]]:
    src = OUT / "travel_4k.mp4"
    media = SITE / "media"
    media.mkdir(exist_ok=True)
    dst = media / "wide_open_1080p.mp4"
    with av.open(str(src)) as c:
        dur = float(c.duration) / 1e6
        for frame in c.decode(video=0):  # poster: the title card
            if frame.time >= 1.6:
                im = frame.to_image()
                break
    im.thumbnail((1920, 1080), Image.LANCZOS)
    (SITE / "img" / "examples").mkdir(parents=True, exist_ok=True)
    im.save(SITE / "img" / "examples" / "wide_open.jpg", quality=80, optimize=True, progressive=True)
    kbps = int((PAGES_LIMIT * 0.9 * 8 / dur - 160_000) / 1000)  # video budget under the per-file limit
    kbps = min(kbps, 8000)
    found = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not found:  # WinGet's link in PATH is not always visible to Python; look in its package folder
        pkgs = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        found = next((str(p) for p in pkgs.glob("*FFmpeg*/*/bin/ffmpeg.exe")), None)
    if not found:
        sys.exit("ffmpeg is needed for the web copy")
    ffmpeg = os.path.realpath(found)  # WinGet installs a link that only resolves for some shells
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(src), "-vf", "scale=1920:-2:flags=lanczos",
                    "-c:v", "libx264", "-preset", "slow", "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.5)}k",
                    "-bufsize", f"{kbps * 2}k", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", "160k", str(dst)], check=True)
    size = dst.stat().st_size
    if size > PAGES_LIMIT:
        sys.exit(f"web copy is {size / 1e6:.1f} MB, over the Pages limit")
    print(f"web copy: {size / 1e6:.1f} MB at {kbps} kbps, {dur:.1f} s")
    facts = f"{dur:.0f} s · rendered at 3840×2160 · web copy 1080p"
    credits = json.loads((STOCK / "travel" / "credits.json").read_text(encoding="utf-8"))
    used = {p.stem for p in (STOCK / "travel").iterdir()}
    notes = [f'"{html.escape(re.sub(r"[.](webm|mp3)$", "", c["title"]))}" by {html.escape(c["author"])} ({c["license"]})'
             for k, c in credits.items() if k in used and c["license"] != "CC0"]
    return facts, notes


def main() -> None:
    page = SITE / "index.html"
    text = page.read_text(encoding="utf-8")
    figs, photo_notes = photos()
    facts, film_notes = film()
    text = text.replace("@@STYLE_FIGURES@@", figs).replace("@@FILM_FACTS@@", facts)
    credit = ("<p class=\"credits\">Examples: footage and music " + ", ".join(film_notes) + "; photos " +
              ", ".join(photo_notes) + " and four public-domain (CC0) photos; via Wikimedia Commons.</p>")
    text = text.replace("  </div>\n</footer>", f"    {credit}\n  </div>\n</footer>", 1)
    page.write_text(text, encoding="utf-8")
    print("site updated")


if __name__ == "__main__":
    main()
