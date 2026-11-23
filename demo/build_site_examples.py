"""Copy demo output into the website: styled before/after photo pairs and their credits.

Usage (from engine/):  .venv/Scripts/python ../demo/build_site_examples.py
Needs demo/output from make_photo_demos.py (stock/photos).
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SITE = ROOT.parent / "site"
STOCK = ROOT / "stock"

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


def main() -> None:
    page = SITE / "index.html"
    text = page.read_text(encoding="utf-8")
    figs, photo_notes = photos()
    text = text.replace("@@STYLE_FIGURES@@", figs)
    credit = ("<p class=\"credits\">Examples: photos " + ", ".join(photo_notes) +
              " and four public-domain (CC0) photos; via Wikimedia Commons.</p>")
    text = text.replace("  </div>\n</footer>", f"    {credit}\n  </div>\n</footer>", 1)
    page.write_text(text, encoding="utf-8")
    print("site updated")


if __name__ == "__main__":
    main()
