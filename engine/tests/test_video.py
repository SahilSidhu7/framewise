from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aieditor_engine.agent.video_planner import (
    MediaInfo,
    VideoContext,
    plan_video,
    video_hints,
)
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


# ------------------------------------------------------------------ planner


def _ctx(clips, speech=False) -> VideoContext:
    media = []
    for n, key in enumerate(("red", "blue"), start=1):
        a = analyze_video(clips[key])
        tr = {"words": [{"start": 0.2, "end": 0.6, "text": "hello"}, {"start": 0.6, "end": 1.0, "text": "world."}, {"start": 1.0, "end": 1.5, "text": "bye"}],
              "segments": [{"start": 0.2, "end": 1.0, "text": "hello world."}, {"start": 1.0, "end": 1.5, "text": "bye"}]} if speech and n == 1 else None
        media.append(MediaInfo(n, key, key, "video", 2.0, "landscape", a, {"speech_likely": speech and n == 1}, tr))
    media.append(MediaInfo(3, "song", "song", "audio", 12.0, music=analyze_music(clips["song"])))
    return VideoContext(media)


def test_hints():
    ctx = VideoContext([MediaInfo(1, "a", "a", "video", 5, "portrait"), MediaInfo(2, "s", "s", "audio", 30)])
    h = video_hints("make an energetic tiktok with music", ctx)
    assert (h.aspect, h.pace, h.transition, h.music) == ("9:16", "fast", "none", 2)
    h = video_hints('cinematic wedding film titled "Anna & Tom"', ctx)
    assert (h.pace, h.letterbox, h.title) == ("slow", 2.39, "Anna & Tom")
    assert video_hints("no music please", ctx).music is None


@pytest.mark.parametrize("prompt", ["make an energetic reel with music", "cinematic video with captions", "combine the clips"])
def test_rule_planner_programs_apply(clips, prompt):
    ctx = _ctx(clips, speech=True)
    res = plan_video(prompt, ctx, 3, use_llm=False)
    assert 1 <= len(res["variations"]) <= 3
    for v in res["variations"]:
        p = _project(clips)
        out = apply_video_program(p, v["program"], transcripts={"red": ctx.media[0].transcript})
        assert out["clips"] >= 1
    if "captions" in prompt:
        assert "captions" in res["variations"][0]["program"]
    if "reel" in prompt:
        prog = res["variations"][0]["program"]
        assert "aspect 9:16" in prog and "music 3" in prog and "transition=" not in prog


def test_fast_edit_never_repeats_same_clip_back_to_back(clips):
    ctx = _ctx(clips)
    prog = plan_video("fast montage with music", ctx, 1, use_llm=False)["variations"][0]["program"]
    order = [ln.split()[1] for ln in prog.splitlines() if ln.startswith("clip ")]
    assert len(order) >= 2
    assert all(a != b for a, b in zip(order, order[1:])) or len(set(order)) == 1


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
        plan = c.post("/ai/video", json={"prompt": "energetic reel with music"}, headers=h).json()
        assert plan["variations"]
        prog = plan["variations"][0]["program"]
        f = c.get("/timeline/frame", params={"t": 1, "max": 200, "program": prog}, headers=h)
        assert f.status_code == 200 and f.headers["content-type"] == "image/jpeg"
        assert c.post("/ai/video/apply", json={"program": prog}, headers=h).status_code == 200
        assert c.get("/timeline/dsl", headers=h).json()["duration"] > 0
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
