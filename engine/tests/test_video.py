from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aieditor_engine.api.server import create_app
from aieditor_engine.core.bus import CommandBus
from aieditor_engine.core.commands import CommandError
from aieditor_engine.core.project import (
    Clip,
    MediaItem,
    MediaKind,
    Project,
    ProjectKind,
    Track,
)
from aieditor_engine.session import Session
from aieditor_engine.video.analysis import analyze_music, analyze_video, best_window
from aieditor_engine.video.audio import loudness_lufs, mix_timeline, soft_limit
from aieditor_engine.video.captions import group_words
from aieditor_engine.video.compose import RenderSource, render_frame
from aieditor_engine.video.export import ExportSpec, export_video
from aieditor_engine.video.fastgrade import GradePlan
from aieditor_engine.video.vdsl import (
    VideoDslError,
    apply_video_program,
    timeline_to_dsl,
)

TOKEN = "t" * 32


def _video(path: Path, color: tuple[int, int, int], seconds: float = 2.0, fps: int = 25, tone: float | None = 440.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 320, 240
    with av.open(str(path), "w") as out:
        v = out.add_stream("mpeg4", rate=fps)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        a = out.add_stream("aac", rate=48000, layout="stereo") if tone else None
        for i in range(int(fps * seconds)):
            img = np.zeros((h, w, 3), np.uint8)
            img[:] = color
            img[:, (i * 4) % w : (i * 4) % w + 20] = 255  # moving bar = motion
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts, f.time_base = i, Fraction(1, fps)
            out.mux(v.encode(f))
        out.mux(v.encode(None))
        if a is not None:
            t = np.arange(int(48000 * seconds)) / 48000
            sig = (0.2 * np.sin(2 * np.pi * tone * t)).astype(np.float32)
            for s in range(0, len(sig) - 1024, 1024):
                ch = sig[s : s + 1024]
                fr = av.AudioFrame.from_ndarray(np.stack([ch, ch]), format="fltp", layout="stereo")
                fr.sample_rate, fr.pts = 48000, s
                out.mux(a.encode(fr))
            out.mux(a.encode(None))
    return path


def _music(path: Path, bpm: float = 120, seconds: float = 12) -> Path:
    sr = 48000
    y = np.zeros(int(sr * seconds), np.float32)
    beat = 60 / bpm
    for k in range(int(seconds / beat)):
        s = int(k * beat * sr)
        n = int(0.1 * sr)
        env = np.exp(-np.arange(n) / (0.02 * sr))
        y[s : s + n] += (0.8 * np.sin(2 * np.pi * 60 * np.arange(n) / sr) * env)[: len(y) - s]
    with av.open(str(path), "w") as out:
        a = out.add_stream("aac", rate=sr, layout="stereo")
        for s in range(0, len(y) - 1024, 1024):
            ch = y[s : s + 1024]
            fr = av.AudioFrame.from_ndarray(np.stack([ch, ch]), format="fltp", layout="stereo")
            fr.sample_rate, fr.pts = sr, s
            out.mux(a.encode(fr))
        out.mux(a.encode(None))
    return path


@pytest.fixture
def clips(tmp_path: Path) -> dict[str, Path]:
    return {
        "red": _video(tmp_path / "m" / "red.mp4", (200, 40, 40)),
        "blue": _video(tmp_path / "m" / "blue.mp4", (40, 40, 200), tone=660),
        "song": _music(tmp_path / "m" / "song.m4a"),
    }


def _project(clips: dict[str, Path]) -> Project:
    p = Project(name="v", kind=ProjectKind.VIDEO)
    for key in ("red", "blue"):
        p.media[key] = MediaItem(id=key, name=key, path=str(clips[key]), kind=MediaKind.VIDEO, width=320, height=240, duration=2.0, fps=25, has_audio=True)
    p.media["song"] = MediaItem(id="song", name="song", path=str(clips["song"]), kind=MediaKind.AUDIO, duration=12.0, has_audio=True)
    return p


# ------------------------------------------------------------------ commands


def test_split_speed_transition_and_captions(clips):
    p = _project(clips)
    p.tracks.append(Track(id="v", clips=[Clip(id="a", media_id="red", start=0, in_point=0, out_point=2), Clip(id="b", media_id="blue", start=2, in_point=0, out_point=2)]))
    bus = CommandBus(p)
    right = bus.execute("clip.split", {"clip_id": "a", "at": 1.0})["clip_id"]
    _, r = bus.project.find_clip(right)
    assert (r.start, r.in_point, r.out_point) == (1.0, 1.0, 2.0)
    with pytest.raises(CommandError):
        bus.execute("clip.split", {"clip_id": "a", "at": 0.01})
    bus.execute("clip.set", {"clip_id": right, "speed": 0.5})  # 1 s -> 2 s, ripples "b"
    assert bus.project.find_clip("b")[1].start == pytest.approx(3.0)
    bus.execute("clip.set_transition", {"clip_id": "b", "kind": "crossfade", "duration": 0.6})
    assert bus.project.find_clip("b")[1].transition.kind == "crossfade"
    bus.execute("caption.add", {"start": 0, "end": 1, "text": "hi"})
    bus.execute("timeline.apply_look", {"name": "vivid_pop", "strength": 0.8})
    assert bus.project.grade[0].op == "look"
    bus.execute("clip.ripple_delete", {"clip_id": "a"})
    assert bus.project.tracks[0].clips[0].start == 0.0
    assert bus.undo() and bus.undo()


def test_trim_head_keeps_tail(clips):
    p = _project(clips)
    p.tracks.append(Track(id="v", clips=[Clip(id="a", media_id="red", start=1, in_point=0, out_point=2)]))
    bus = CommandBus(p)
    bus.execute("clip.trim", {"clip_id": "a", "in_point": 0.5, "start": 1.5})
    c = bus.project.find_clip("a")[1]
    assert (c.start, c.end) == (1.5, 3.0)


# ----------------------------------------------------------------------- DSL


def test_video_dsl_roundtrip(clips):
    p = _project(clips)
    prog = """# test
aspect 9:16
clip 1 0.2-1.5
clip 2 0-1 transition=crossfade:0.4 speed=0.5 motion=zoom_in volume=0.3
music 3 volume=0.7 fade=1
look teal_orange_cinematic s=0.8
grain 0.1
letterbox 2.39
fade_in 0.5
title "Hello there" at=0 dur=2 style=title
"""
    res = apply_video_program(p, prog)
    assert res["clips"] == 2
    assert (p.settings.width, p.settings.height) == (1080, 1920)
    v = p.tracks[0].clips
    assert v[1].start == pytest.approx(1.3) and v[1].speed == 0.5 and v[1].transition.duration == 0.4
    music = next(t for t in p.tracks if t.kind == "audio").clips[0]
    assert music.end == pytest.approx(p.tracks[0].clips[-1].end)  # music trimmed to the picture
    assert [o.op for o in p.grade] == ["look", "grain"]
    text = timeline_to_dsl(p)
    q = _project(clips)
    apply_video_program(q, text)
    assert timeline_to_dsl(q) == text


@pytest.mark.parametrize("bad", ["clip 9", "clip 3", "music 1", "aspect 7:3", "crop 1:1", "wobble 3", "clip 1 2-1", "transition sparkle"])
def test_video_dsl_errors(clips, bad):
    with pytest.raises(VideoDslError):
        apply_video_program(_project(clips), bad)


def test_program_command_is_one_undo_step(clips):
    bus = CommandBus(_project(clips))
    bus.execute("timeline.apply_program", {"program": "clip 1\nclip 2\nlook noir"}, source="ai")
    assert len(bus.project.tracks[0].clips) == 2
    bus.undo()
    assert all(not t.clips for t in bus.project.tracks)
    with pytest.raises(CommandError):
        bus.execute("timeline.apply_program", {"program": "clip 1\nclip 99"})
    assert all(not t.clips for t in bus.project.tracks)  # failed programs change nothing


# ------------------------------------------------------------------- render


def _sequence(clips) -> Project:
    p = _project(clips)
    apply_video_program(p, "aspect 16:9\nclip 1\nclip 2 transition=crossfade:1\nlook vivid_pop s=0.8\nvignette 0.3\nfade_in 0.5")
    return p


def test_render_frames_and_transition(clips, tmp_path):
    p = _sequence(clips)
    src = RenderSource(tmp_path)
    red = render_frame(p, 0.6, 160, 90, src)
    blue = render_frame(p, 3.0, 160, 90, src)
    mid = render_frame(p, 2.0, 160, 90, src)
    assert red.shape == (90, 160, 3) and red.dtype == np.uint8
    assert red[45, 10, 0] > red[45, 10, 2] and blue[45, 10, 2] > blue[45, 10, 0]
    assert red[45, 10, 2] < mid[45, 10, 2] < blue[45, 10, 2]  # crossfade blends
    assert render_frame(p, 0.0, 160, 90, src).mean() < 5  # fades in from black


def test_baked_grade_matches_direct(clips):
    from aieditor_engine.imaging.render import apply_flat, flatten, make_context
    from aieditor_engine.video.fastgrade import encode_u8

    rng = np.random.default_rng(0)
    img = rng.random((64, 64, 3), dtype=np.float32) ** 2.2
    grade = [{"op": "look", "params": {"name": "film_portra", "strength": 1.0}}]
    fast = GradePlan(grade).finish(img, 0).astype(np.float32)
    ctx = make_context(0)
    ctx.extra["pointwise"] = True
    ref = encode_u8(apply_flat(img, flatten(grade), ctx), 0).astype(np.float32)
    assert np.abs(fast - ref).mean() < 2.5


def test_export_with_audio(clips, tmp_path):
    p = _sequence(clips)
    apply_video_program(p, "music 3 volume=0.5", replace=False)
    out = tmp_path / "out.mp4"
    seen = []
    res = export_video(p, tmp_path, ExportSpec(out, 320, 180, 25, "small", workers=2), seen.append)
    assert out.is_file() and res["audio"] and res["frames"] == 100
    assert seen[-1] == pytest.approx(1.0)
    with av.open(str(out)) as c:
        assert c.streams.video[0].codec_context.width == 320
        assert len(c.streams.audio) == 1


def test_export_cancel_leaves_no_file(clips, tmp_path):
    from aieditor_engine.video.export import Cancelled

    out = tmp_path / "c.mp4"
    with pytest.raises(Cancelled):
        export_video(_sequence(clips), tmp_path, ExportSpec(out, 320, 180, 25, "small", workers=1), cancelled=lambda: True)
    assert not out.exists() and not list(tmp_path.glob("*.part*"))


def test_mix_loudness_and_limit(clips):
    p = _sequence(clips)
    apply_video_program(p, "music 3 volume=1", replace=False)
    mix = mix_timeline(p)
    assert mix.shape[0] == 2 and mix.shape[1] == pytest.approx(p.duration * 48000, abs=10)
    assert np.abs(mix).max() <= 0.9
    assert loudness_lufs(mix) == pytest.approx(-14, abs=1.5)
    x = np.full((2, 100), 3.0, np.float32)
    assert np.abs(soft_limit(x)).max() <= 0.9


# ----------------------------------------------------------------- analysis


def test_analysis_and_beats(clips):
    a = analyze_video(clips["red"])
    assert a["duration"] == pytest.approx(2.0, abs=0.3) and len(a["seconds"]) == 2
    m = analyze_music(clips["song"])
    assert m["bpm"] == pytest.approx(120, abs=2)
    assert np.diff(m["beats"]).mean() == pytest.approx(0.5, abs=0.02)
    assert best_window([0.1, 0.1, 0.9, 0.9, 0.1], 2.0) == (2.0, 4.0)


def test_caption_grouping():
    words = [{"start": i * 0.3, "end": i * 0.3 + 0.25, "text": w} for i, w in enumerate("hello there this is a test. and another sentence here".split())]
    caps = group_words(words, "clean")
    assert caps[0][2].endswith("test.")  # break after sentence end
    short = group_words(words, "bold_pop")
    assert all(len(t) <= 16 + 10 for _, _, t in short) and len(short) > len(caps)
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(short, short[1:]))  # never overlap


# ---------------------------------------------------------------------- API


def test_video_api_flow(clips, tmp_path):
    app = create_app(TOKEN)
    h = {"X-Engine-Token": TOKEN}
    with TestClient(app) as c:
        assert c.post("/projects", json={"path": str(tmp_path / "proj"), "name": "V", "kind": "video"}, headers=h).status_code == 200
        r = c.post("/media/import", json={"paths": [str(clips["red"]), str(clips["blue"]), str(clips["song"])]}, headers=h).json()
        assert len(r["imported"]) == 3
        session: Session = app.state.engine.session
        session.wait_for_jobs(60)
        prog = "aspect 9:16\nclip 1 0-2\nclip 2 0-2 transition=crossfade:0.5\nmusic 3 volume=0.8"
        f = c.get("/timeline/frame", params={"t": 1, "max": 200, "program": prog}, headers=h)
        assert f.status_code == 200 and f.headers["content-type"] == "image/jpeg"
        assert c.post("/ai/video/apply", json={"program": prog}, headers=h).status_code == 200
        assert c.get("/timeline/dsl", headers=h).json()["duration"] > 0
        # prompt options, reels and "edit like this"
        opts = c.get("/ai/options", headers=h).json()
        assert {s["id"] for s in opts["video"]["styles"]} >= {"cinematic", "energetic_reel"}
        assert c.post("/ai/video", json={"prompt": "x"}, headers=h).status_code in (404, 405)  # the prompt director is gone
        reels = c.post("/ai/reels", json={"count": 2, "length": 5, "output": {"platform": "tiktok"}}, headers=h).json()
        assert reels["aspect"] == "9:16" and reels["variations"] and "aspect 9:16" in reels["variations"][0]["program"]
        assert c.post("/ai/video/apply", json={"program": reels["variations"][0]["program"]}, headers=h).status_code == 200
        st = c.post("/ai/styles/from_video", json={"media_number": 1, "name": "Red test"}, headers=h).json()
        assert st["id"] == "red_test" and any(s["id"] == "red_test" for s in c.get("/ai/options", headers=h).json()["video"]["styles"])
        assert c.post("/ai/styles/from_video", json={"media_number": 3, "name": "song"}, headers=h).status_code == 422
        assert c.get("/timeline/frame", params={"t": 0.5, "max": 200, "grade": "look noir"}, headers=h).status_code == 200
        bad = c.get("/timeline/frame", params={"t": 0, "grade": "look nope"}, headers=h)
        assert bad.status_code == 422
        out = tmp_path / "export.mp4"
        job = c.post("/timeline/export", json={"path": str(out), "preset": "hd_720p", "quality": "small"}, headers=h).json()["job"]
        for _ in range(600):
            state = {j["id"]: j for j in c.get("/jobs", headers=h).json()}[job]
            if state["state"] in ("done", "failed"):
                break
            time.sleep(0.1)
        assert state["state"] == "done", state
        assert out.is_file() and state["result"]["width"] == 1280
        pv = c.post("/timeline/preview", json={"max_side": 320}, headers=h).json()
        assert pv["job"] or pv["url"]
        for _ in range(600):  # let the background render finish before the app shuts down
            if all(j["state"] in ("done", "failed", "cancelled") for j in c.get("/jobs", headers=h).json()):
                break
            time.sleep(0.1)


def test_syllabic_ratio_separates_speech_from_ambience() -> None:
    from aieditor_engine.video.analysis import _syllabic_ratio

    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(sr * 8) / sr
    noise = rng.standard_normal(t.size).astype(np.float32) * 0.1
    tone = np.sin(2 * np.pi * 900 * t).astype(np.float32)
    # voice-band carrier switched on and off at ~4 syllables per second, over noise
    syllables = (np.sin(2 * np.pi * 4 * t) > 0.2).astype(np.float32)
    speechy = 0.3 * tone * syllables + 0.02 * noise
    surf = 0.3 * noise * (1 + 0.3 * np.sin(2 * np.pi * 0.15 * t)).astype(np.float32)  # slow swell
    assert _syllabic_ratio(speechy, sr) > 0.43 > _syllabic_ratio(surf, sr)


def test_captions_move_into_letterbox_bar() -> None:
    from aieditor_engine.video.text import draw_text

    h, w = 1080, 1920
    bar = round((h - w / 2.39) / 2)
    frame = np.zeros((h, w, 3), np.uint8)
    draw_text(frame, "a short line of dialogue for the bar", "cinematic", None, 1.0, 1.0, bar=bar)
    rows = np.nonzero(frame.max(axis=(1, 2)) > 0)[0]
    assert rows.min() >= h - bar and rows.max() < h  # entirely inside the bottom bar


# -------------------------------------------------------------------- layers


def _lin(u8: np.ndarray) -> np.ndarray:
    from aieditor_engine.imaging.color import srgb_to_linear

    return srgb_to_linear(u8.astype(np.float32) / 255)


def _srgb(lin: np.ndarray) -> np.ndarray:
    from aieditor_engine.imaging.color import linear_to_srgb

    return linear_to_srgb(np.clip(lin, 0, 1).astype(np.float32)) * 255


W, H = 160, 120  # 4:3 like the test clips, so a clip covers the canvas exactly
FAR = (60, 150)  # a pixel the moving white bar never reaches at the sampled times


def _layered(clips, overlay: str) -> Project:
    p = _project(clips)
    apply_video_program(p, f"aspect 4:3\nclip 1\noverlay 2 0-2 at=0 {overlay}")
    return p


def _px(p: Project, tmp_path: Path, t: float = 1.0, at: tuple[int, int] = FAR) -> np.ndarray:
    return render_frame(p, t, W, H, RenderSource(tmp_path))[at].astype(np.float32)


def _is_red(v: np.ndarray) -> bool:
    return bool(v[0] > v[2] + 40)


def _is_blue(v: np.ndarray) -> bool:
    return bool(v[2] > v[0] + 40)


def test_layer_opacity_and_blend_modes_match_the_math(clips, tmp_path):
    base = _project(clips)
    apply_video_program(base, "aspect 4:3\nclip 1")
    top = _project(clips)
    apply_video_program(top, "aspect 4:3\nclip 2")
    r, b = _lin(_px(base, tmp_path)), _lin(_px(top, tmp_path))
    assert r[0] > r[2] and b[2] > b[0]
    expected = {
        "opacity=1": b,
        "opacity=0.5": 0.5 * b + 0.5 * r,
        "opacity=0": r,
        "blend=add": r + b,
        "blend=screen": r + b - r * b,
        "blend=multiply": r * b,
        "blend=add opacity=0.5": r + 0.5 * b,
    }
    for params, lin in expected.items():
        got = _px(_layered(clips, params), tmp_path)
        assert np.abs(got - _srgb(lin)).max() <= 3, (params, got, _srgb(lin))
    # overlay is a contrast blend: the result differs from the base and stays in range
    ov = _px(_layered(clips, "blend=overlay"), tmp_path)
    assert np.abs(ov - _srgb(r)).max() > 3 and ov.min() >= 0 and ov.max() <= 255


def test_layer_transform_crop_and_hidden_track(clips, tmp_path):
    base = _project(clips)
    apply_video_program(base, "aspect 4:3\nclip 1")
    assert _is_red(_px(base, tmp_path, at=(60, 100)))
    pip = _layered(clips, "scale=0.5")  # blue picture-in-picture in the middle: x 40-120, y 30-90
    assert _is_blue(_px(pip, tmp_path, at=(60, 100)))
    assert _is_red(_px(pip, tmp_path, at=(10, 10))) and _is_red(_px(pip, tmp_path, at=(60, 150)))
    moved = _layered(clips, "scale=0.5 x=0.25 y=-0.25")  # centre moves to x=120, y=30
    assert _is_blue(_px(moved, tmp_path, at=(30, 140)))
    assert _is_red(_px(moved, tmp_path, at=(60, 30))) and _is_red(_px(moved, tmp_path, at=(100, 140)))
    cropped = _layered(clips, "crop=0.5,0,0,0")  # left half of the layer removed
    assert _is_red(_px(cropped, tmp_path, at=(100, 20))) and _is_blue(_px(cropped, tmp_path, at=(100, 150)))
    turned = _layered(clips, "scale=0.5 rotate=90")  # the 80x60 box becomes 60 wide and 80 tall
    assert _is_blue(_px(turned, tmp_path, at=(25, 80))) and _is_red(_px(turned, tmp_path, at=(60, 45)))
    pip.tracks[-1].hidden = True
    assert _is_red(_px(pip, tmp_path, at=(60, 100)))


def test_keyframes_animate_layer_and_volume(clips, tmp_path):
    p = _layered(clips, "opacity=0:0,1:1")
    ov = next(t for t in p.tracks if t.kind == "overlay").clips[0]
    assert ov.value("opacity", 0.0) == 0 and ov.value("opacity", 0.5) == pytest.approx(0.5) and ov.value("opacity", 1.7) == 1
    start, mid, end = (_px(p, tmp_path, t) for t in (0.04, 0.5, 1.5))
    assert _is_red(start) and _is_blue(end)
    assert start[2] < mid[2] < end[2] and start[0] > mid[0] > end[0]

    grow = _layered(clips, "scale=0.2>1")
    k = next(t for t in grow.tracks if t.kind == "overlay").clips[0]
    assert k.value("scale", 0) == pytest.approx(0.2) and k.value("scale", 2) == pytest.approx(1.0) and 0.2 < k.value("scale", 1) < 1
    assert _is_red(_px(grow, tmp_path, 0.1, (10, 150))) and _is_blue(_px(grow, tmp_path, 1.95, (10, 150)))

    # volume automation: silent at the start, full after one second
    q = _project(clips)
    apply_video_program(q, "clip 1 volume=0:0,1:1")
    q.settings.normalize_audio = False
    mix = mix_timeline(q)

    def rms(a: float, b: float) -> float:
        return float(np.sqrt((mix[:, int(a * 48000) : int(b * 48000)] ** 2).mean()))

    assert rms(0.02, 0.12) < 0.25 * rms(1.3, 1.8)
    q.tracks[0].muted = True
    assert not mix_timeline(q).any()


LAYER_LINES = (
    "clip 1 0-2 scale=0:1,2:1.15",
    "clip 2 0-1.5 transition=crossfade:0.5",
    "overlay 2 0-1 at=0.5 volume=0.5 x=0.3 y=-0.3 scale=0.3 rotate=10 opacity=0.8 blend=screen crop=0.1,0,0.1,0",
    "overlay 1 0.5-1.5 at=1 track=2 fade=0.2 mute opacity=0:0,0.5:1",
)


def test_layer_dsl_roundtrip_and_errors(clips):
    p = _project(clips)
    apply_video_program(p, "aspect 16:9\n" + "\n".join(LAYER_LINES))
    kinds = [t.kind for t in p.tracks]
    assert kinds.count("overlay") == 2 and kinds.index("video") < kinds.index("overlay")
    dsl = timeline_to_dsl(p)
    assert dsl.splitlines()[1:5] == list(LAYER_LINES)
    q = _project(clips)
    apply_video_program(q, dsl)
    assert timeline_to_dsl(q) == dsl
    for bad in ("overlay 2 at=0 blend=glow", "overlay 2 at=0 crop=0.6,0,0.6,0", "overlay 2 at=0 opacity=2",
                "overlay 2 0-2 at=0\noverlay 1 0-2 at=1", "clip 1 scale=0"):
        with pytest.raises(VideoDslError):
            apply_video_program(_project(clips), bad)


def test_layer_commands_undo_and_markers(clips):
    bus = CommandBus(_project(clips))
    bus.execute("timeline.apply_program", {"program": "clip 1\noverlay 2 0-2 at=0"})
    ov_track = next(t for t in bus.project.tracks if t.kind == "overlay")
    cid = ov_track.clips[0].id
    bus.execute("clip.set_layer", {"clip_id": cid, "scale": 0.4, "x": 0.2, "opacity": 0.7, "blend": "add", "crop_left": 0.1})
    c = bus.project.find_clip(cid)[1]
    assert (c.transform.scale, c.transform.x, c.opacity, c.blend, c.crop.left) == (0.4, 0.2, 0.7, "add", 0.1) and c.is_layered
    keys = [{"t": 1, "value": 90}, {"t": 0, "value": 0, "ease": "linear"}]
    bus.execute("clip.set_keyframes", {"clip_id": cid, "param": "rotation", "keys": keys})
    c = bus.project.find_clip(cid)[1]
    assert [k.t for k in c.keyframes["rotation"]] == [0, 1] and c.value("rotation", 0.5) == pytest.approx(45)
    with pytest.raises(CommandError):
        bus.execute("clip.set_keyframes", {"clip_id": cid, "param": "speed", "keys": [{"t": 0, "value": 1}]})
    with pytest.raises(CommandError):
        bus.execute("clip.set_keyframes", {"clip_id": cid, "param": "opacity", "keys": [{"t": 0, "value": 3}]})
    bus.execute("clip.set_keyframes", {"clip_id": cid, "param": "rotation", "keys": []})
    assert "rotation" not in bus.project.find_clip(cid)[1].keyframes
    bus.undo()
    assert "rotation" in bus.project.find_clip(cid)[1].keyframes

    bus.execute("track.set", {"track_id": ov_track.id, "name": "Logo", "hidden": True, "locked": True, "index": 0})
    assert bus.project.tracks[0].name == "Logo" and bus.project.tracks[0].hidden
    m = bus.execute("marker.add", {"t": 1.5, "name": "drop"})
    assert bus.project.markers[0].name == "drop"
    bus.execute("marker.remove", {"marker_id": m["marker_id"]})
    assert not bus.project.markers
    removed = bus.execute("track.remove", {"track_id": ov_track.id})
    assert removed["clips_removed"] == 1 and all(t.kind != "overlay" for t in bus.project.tracks)


def test_v1_project_migrates_to_layers():
    from aieditor_engine.core.project import SCHEMA_VERSION
    from aieditor_engine.core.store import migrate

    clip = {"id": "c1", "media_id": "m", "start": 0, "in_point": 0, "out_point": 2, "motion": "zoom_in"}
    raw = {"schema_version": 1, "name": "old", "kind": "video", "tracks": [{"id": "t1", "kind": "video", "clips": [clip]}]}
    p = Project.model_validate(migrate(raw))
    c = p.tracks[0].clips[0]
    assert p.schema_version == SCHEMA_VERSION == 3 and not c.is_layered and c.blend == "normal" and c.opacity == 1 and p.markers == []


def test_export_parallelism_fits_memory(monkeypatch):
    from aieditor_engine.imaging import segment
    from aieditor_engine.video.export import _parallelism

    monkeypatch.setattr(segment, "_free_gb", lambda: 16.0)
    w1080, a1080 = _parallelism(1920, 1080, 30, 8)
    w4k, a4k = _parallelism(3840, 2160, 30, 8)
    assert w1080 >= 4 and w4k < w1080  # 4K gets fewer threads
    frame = 3840 * 2160 * 3
    peak_gb = (w4k * 12 * frame * 4 + a4k * 30 * frame) / 2**30
    assert peak_gb <= 8.0  # half of what is free
    monkeypatch.setattr(segment, "_free_gb", lambda: 1.0)
    assert _parallelism(3840, 2160, 30, 8) == (1, 1)  # low memory: still renders, one at a time


def test_live_preview_routes(clips, tmp_path):
    app = create_app(TOKEN)
    h = {"X-Engine-Token": TOKEN}
    with TestClient(app) as c:
        c.post("/projects", json={"path": str(tmp_path / "live"), "name": "L", "kind": "video"}, headers=h)
        c.post("/media/import", json={"paths": [str(clips["red"]), str(clips["song"])]}, headers=h)
        session: Session = app.state.engine.session
        session.wait_for_jobs(60)
        assert c.get("/timeline/grade_lattice", headers=h).status_code == 404  # no grade yet
        prog = "clip 1 0-2\nmusic 2\nlook teal_orange_cinematic\ngrain 0.2"
        assert c.post("/ai/video/apply", json={"program": prog}, headers=h).status_code == 200
        info = c.get("/timeline/live", headers=h).json()
        assert info["lattice"] and info["lattice_size"] == 65
        assert [p["op"] for p in info["post"]] == ["vignette", "grain"] and info["missing"] == []
        lat = c.get("/timeline/grade_lattice", headers=h)
        assert lat.status_code == 200 and len(lat.content) == 65**3 * 3
        mid = next(iter(session.bus.project.media))
        src = c.get(f"/media/{mid}/source", params={"token": TOKEN})
        assert src.status_code == 200 and len(src.content) == clips["red"].stat().st_size
        assert c.get("/media/nope/source", headers=h).status_code == 404
