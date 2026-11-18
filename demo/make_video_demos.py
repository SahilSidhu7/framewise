"""Run the real AI video pipeline on sample media and export before/after MP4s.

Before = the media simply joined in import order (what you'd get with no editing).
After  = the AI director's first variation, exported at full quality.
Usage (from engine/):  .venv/Scripts/python ../demo/make_video_demos.py <test media dir>
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "engine" / "src"))

from aieditor_engine.agent.video_planner import plan_video  # noqa: E402
from aieditor_engine.core.project import ProjectKind  # noqa: E402
from aieditor_engine.session import Session  # noqa: E402
from aieditor_engine.video.compose import RenderSource, output_size, render_frame  # noqa: E402
from aieditor_engine.video.export import ExportSpec, export_video  # noqa: E402
from aieditor_engine.video.vdsl import apply_video_program  # noqa: E402

OUT = ROOT / "output"
ONLY: str | None = None  # optional: render one case by name
WALL = Path(r"C:\Windows\Web")
STILLS = [WALL / "Wallpaper/ThemeC/img28.jpg", WALL / "Screen/img102.jpg", WALL / "Wallpaper/ThemeC/img29.jpg",
          WALL / "Wallpaper/ThemeC/img31.jpg", WALL / "Wallpaper/ThemeC/img30.jpg"]
FONT = ROOT.parent / "engine" / "src" / "aieditor_engine" / "fonts" / "Inter[opsz,wght].ttf"


def font(px: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT), px)
    f.set_variation_by_axes([24, 600])
    return f


def strip(project, root: Path, times: list[float], title: str, dst: Path) -> None:
    w, h = output_size(project, 480)
    frames = [render_frame(project, t, w, h, RenderSource(root, use_proxy=True)) for t in times]
    pad = 12
    W = len(frames) * (w + pad) + pad
    img = Image.new("RGB", (W, h + 70), (18, 19, 22))
    d = ImageDraw.Draw(img)
    d.text((pad, 14), title, font=font(30), fill=(235, 235, 235))
    for i, (f, t) in enumerate(zip(frames, times, strict=True)):
        img.paste(Image.fromarray(f), (pad + i * (w + pad), 58))
        d.text((pad + i * (w + pad) + 8, 64), f"{t:.1f}s", font=font(20), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    img.save(dst, quality=88)


def run_case(name: str, files: list[Path], prompt: str) -> None:
    if ONLY and ONLY != name:
        return
    root = OUT / "work" / name
    shutil.rmtree(root, ignore_errors=True)
    s = Session.create(root, name, ProjectKind.VIDEO)
    s.import_media(files)
    s.wait_for_jobs()
    n_visual = sum(1 for f in files if f.suffix.lower() not in (".m4a", ".mp3", ".wav", ".ogg"))

    # BEFORE: everything joined, untouched
    before = s.bus.project.model_copy(deep=True)
    apply_video_program(before, "\n".join(f"clip {i}" for i in range(1, n_visual + 1)))
    # AFTER: the AI
    t = time.time()
    res = plan_video(prompt, s.video_context(transcribe_speech=True), 3)
    v = res["variations"][0]
    s.bus.execute("timeline.apply_program", {"program": v["program"], "transcripts": s.transcripts_for_timeline() or None}, source="ai")
    after = s.bus.project.model_copy(deep=True)
    plan_s = time.time() - t
    print(f"\n[{name}] '{prompt}' ({v['source']}, planned in {plan_s:.1f}s): {v['rationale']}\n{v['program']}")

    for label, proj in (("before", before), ("after", after)):
        dst = OUT / f"video_{name}_{label}.mp4"
        t = time.time()
        w, h = output_size(proj, 1920)
        r = export_video(proj, root, ExportSpec(dst, w, h, proj.settings.fps, "standard"))
        print(f"  {label}: {r['width']}x{r['height']} {r['duration']}s via {r['encoder']}, rendered in {time.time() - t:.1f}s")
        times = [round(proj.duration * k / 5 + 0.4, 1) for k in range(5)]
        strip(proj, root, times, f"{label.upper()}  ·  {prompt if label == 'after' else 'clips joined, no editing'}", OUT / f"video_{name}_{label}_frames.jpg")
    (OUT / f"video_{name}_program.txt").write_text(f"Request: {prompt}\nWhy: {v['rationale']}\n\n{v['program']}\n", encoding="utf-8")
    s.close()


def travel_4k(files: list[Path], prompt: str) -> None:
    """One AI edit, exported at 3840x2160, plus a small 1080p copy for the website."""
    root = OUT / "work" / "travel"
    shutil.rmtree(root, ignore_errors=True)
    s = Session.create(root, "travel", ProjectKind.VIDEO)
    s.import_media(files)
    s.wait_for_jobs()
    t = time.time()
    res = plan_video(prompt, s.video_context(transcribe_speech=False), 3)
    v = res["variations"][0]
    s.bus.execute("timeline.apply_program", {"program": v["program"]}, source="ai")
    proj = s.bus.project.model_copy(deep=True)
    print(f"[travel] planned in {time.time() - t:.1f}s ({v['source']}): {v['rationale']}\n{v['program']}", flush=True)
    dst = OUT / "travel_4k.mp4"
    t = time.time()
    r = export_video(proj, root, ExportSpec(dst, 3840, 2160, proj.settings.fps, "high"))
    print(f"  4K: {r['width']}x{r['height']} {r['duration']}s via {r['encoder']}, rendered in {time.time() - t:.0f}s", flush=True)
    times = [round(proj.duration * k / 6 + 0.5, 1) for k in range(6)]
    strip(proj, root, times, f"AFTER  ·  {prompt}", OUT / "travel_frames.jpg")
    (OUT / "travel_program.txt").write_text(f"Request: {prompt}\nWhy: {v['rationale']}\n\n{v['program']}\n", encoding="utf-8")
    s.close()


def main() -> None:
    media = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    global ONLY
    ONLY = sys.argv[2] if len(sys.argv) > 2 else None
    if (media / "credits.json").exists():  # 4K travel footage (demo/stock/travel)
        clips = sorted(p for p in media.glob("*.webm"))
        music = next(media.glob("music.*"))
        travel_4k(clips + [music], 'cinematic travel film titled "Wide Open", slow and epic, with music')
    elif (media / "CREDITS.txt").exists():  # real stock footage (demo/stock)
        clips = ["beach_drone.webm", "waves_vertical.webm", "city_timelapse.webm", "viaduct_timelapse.webm", "mountain_snow.webm"]
        run_case("real_reel", [media / c for c in clips] + [media / "music_upbeat.mp3"], "make an energetic travel reel with music")
        run_case("real_cinematic", [media / c for c in ("mountain_snow.webm", "beach_drone.webm", "interview.webm", "city_timelapse.webm")],
                 'cinematic video with subtitles titled "Out There"')
    else:
        run_case("reel", STILLS + [media / "song.m4a"], "make an energetic reel with music")
        run_case("cinematic", [STILLS[0], STILLS[1], media / "talking.mp4", STILLS[2], STILLS[4]],
                 'cinematic travel video with subtitles titled "Lakes of the North"')
    shutil.rmtree(OUT / "work", ignore_errors=True)


if __name__ == "__main__":
    main()
