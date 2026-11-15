"""Phase C demo: one long video in, ranked vertical reels out.

Usage (from engine/):  .venv/Scripts/python ../demo/make_reel_demos.py ../demo/stock [video file name]
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "engine" / "src"))

from aieditor_engine.agent.options import OutputOptions  # noqa: E402
from aieditor_engine.agent.reels import make_reels  # noqa: E402
from aieditor_engine.core.project import ProjectKind  # noqa: E402
from aieditor_engine.session import Session  # noqa: E402
from aieditor_engine.video.compose import RenderSource, output_size, render_frame  # noqa: E402
from aieditor_engine.video.export import ExportSpec, export_video  # noqa: E402

OUT = ROOT / "output"
FONT = ROOT.parent / "engine" / "src" / "aieditor_engine" / "fonts" / "Inter[opsz,wght].ttf"


def font(px: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT), px)
    f.set_variation_by_axes([24, 600])
    return f


def main() -> None:
    stock = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else "interview.webm"
    files = [stock / name] + ([stock / "music_upbeat.mp3"] if (stock / "music_upbeat.mp3").exists() else [])
    root = OUT / "work" / "reels"
    shutil.rmtree(root, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    s = Session.create(root, "reels", ProjectKind.VIDEO)
    s.import_media(files)
    s.wait_for_jobs()
    t = time.time()
    ctx = s.video_context(transcribe_speech=True)
    res = make_reels(ctx, count=2, output=OutputOptions(platform="tiktok", length=15), subject_of=s.subject_track, styles_dir=s.styles_dir)
    print(f"planned {len(res['variations'])} reels in {time.time() - t:.1f}s (tracking: {res['tracking']})")
    strips = []
    for i, v in enumerate(res["variations"], start=1):
        print(f"\n#{i} score {v['score']}  {v['title']!r}: {v['rationale']}\n   {v['info']}\n{v['program']}")
        s.bus.execute("timeline.apply_program", {"program": v["program"], "transcripts": s.transcripts_for_timeline() or None}, source="ai")
        p = s.bus.project.model_copy(deep=True)
        dst = OUT / f"reel_{i}.mp4"
        w, h = output_size(p, 1280)
        t = time.time()
        r = export_video(p, root, ExportSpec(dst, w, h, p.settings.fps, "standard"))
        print(f"   exported {r['width']}x{r['height']} {r['duration']}s in {time.time() - t:.1f}s")
        times = [round(p.duration * k / 4 + 0.6, 1) for k in range(4)]
        sw, sh = output_size(p, 480)
        frames = [Image.fromarray(render_frame(p, tt, sw, sh, RenderSource(root, use_proxy=True))) for tt in times]
        strip = Image.new("RGB", (len(frames) * (sw + 10) + 10, sh + 70), (18, 19, 22))
        d = ImageDraw.Draw(strip)
        d.text((10, 14), f"Reel {i}: {v['title']}  ({v['info']['duration']}s, {v['info']['silence_removed']}s of pauses cut)", font=font(22), fill=(235, 235, 235))
        for k, f in enumerate(frames):
            strip.paste(f, (10 + k * (sw + 10), 58))
        strips.append(strip)
    sheet = Image.new("RGB", (max(x.width for x in strips), sum(x.height for x in strips)), (18, 19, 22))
    y = 0
    for x in strips:
        sheet.paste(x, (0, y))
        y += x.height
    sheet.save(OUT / "reels_frames.jpg", quality=88)
    s.close()
    shutil.rmtree(OUT / "work", ignore_errors=True)


if __name__ == "__main__":
    main()
