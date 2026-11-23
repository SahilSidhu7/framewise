"""Phase C: prompt options, highlights, silence/filler removal, auto-reframe, reels."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from aieditor_engine.agent.cleanup import keep_ranges, saved_seconds
from aieditor_engine.agent.highlights import find_moments, sentences, text_score
from aieditor_engine.agent.options import (
    OptionError,
    OutputOptions,
    apply_photo_options,
    catalog,
    style_from_reference,
    video_styles,
)
from aieditor_engine.agent.planner import Hints, plan
from aieditor_engine.agent.reels import make_reels
from aieditor_engine.agent.media_context import MediaInfo, VideoContext
from aieditor_engine.core.project import MediaItem, MediaKind, Project, ProjectKind
from aieditor_engine.video.reframe import _camera_path, reframe_keys, subject_track, x_param
from aieditor_engine.video.vdsl import apply_video_program

NEUTRAL = {"size": [1200, 800], "orientation": "landscape", "brightness_median": 0.5, "saturation_mean": 0.2,
           "warmth": 0.0, "palette": [{"hex": "#808080", "share": 1.0}], "notes": []}


def _words(text: str, start: float = 0.0, gap_after: dict[int, float] | None = None) -> list[dict]:
    out, t = [], start
    for i, w in enumerate(text.split()):
        out.append({"start": round(t, 3), "end": round(t + 0.3, 3), "text": w})
        t += 0.35 + (gap_after or {}).get(i, 0.0)
    return out


TALK = (
    "um so yeah we got here early. "
    "Here's the secret nobody tells you about travel? "
    "You never need more than one bag. "
    "and then we walked to the hotel. "
    "The view was incredible and I finally understood why people love this city! "
    "okay. "
    "What would you do with 3 free days here? "
    "Most people get this wrong but the answer is simple."
)


def _talk_media(n: int = 1) -> MediaInfo:
    words = _words(TALK)
    duration = words[-1]["end"] + 1.0
    secs = int(duration) + 1
    return MediaInfo(n, f"talk{n}", "talk.mp4", "video", duration, "landscape",
                     {"seconds": [0.5] * secs}, {"speech_likely": True, "energy": [0.5] * secs},
                     {"words": words, "segments": []})


def _broll(n: int, dur: float = 12.0) -> MediaInfo:
    return MediaInfo(n, f"b{n}", f"b{n}.mp4", "video", dur, "landscape", {"seconds": [0.4 + 0.05 * (i % 5) for i in range(int(dur))]})


# ------------------------------------------------------------------ options


def test_catalog_and_presets_are_valid():
    c = catalog()
    ids = {s["id"] for s in c["video"]["styles"]}
    assert {"cinematic", "energetic_reel", "vlog", "sports_highlights", "tutorial"} <= ids
    assert all(s["looks"] for s in c["video"]["styles"])  # every look name in the YAML exists
    assert {o["id"] for o in c["video"]["outputs"]} >= {"tiktok", "youtube", "instagram_reel", "square_post"}
    assert {s["id"] for s in c["photo"]["styles"]} >= {"natural", "food", "moody", "black_white"}


def test_photo_options():
    h = Hints(None, None, False, 1.0, ("adjust",), " edit this ")
    natural = apply_photo_options(h, "natural", None)
    assert natural.looks == () and natural.fix_only
    food = apply_photo_options(h, "food", OutputOptions(platform="story"))
    assert food.looks[0] == "vivid_pop" and food.aspect == "9:16"
    r = plan("edit this", NEUTRAL, "", 2, style="black_white", output=OutputOptions(platform="instagram_post"))
    assert all("crop" in v["program"] and ("bw_classic" in v["program"] or "noir" in v["program"]) for v in r["variations"])


def test_style_from_reference_video():
    fast = style_from_reference({"duration": 30.0, "cuts": [i * 1.2 for i in range(1, 25)], "colorfulness": 0.6, "brightness": 0.55, "size": [1080, 1920]}, "Hype")
    assert fast["pace"] == "fast" and fast["aspect"] == "9:16" and fast["looks"][0] == "vivid_pop" and fast["transition"] == "none"
    slow = style_from_reference({"duration": 60.0, "cuts": [10.0, 25.0, 40.0], "colorfulness": 0.03, "brightness": 0.4, "size": [1920, 1080]}, "Mono")
    assert slow["pace"] == "slow" and slow["looks"][0] == "bw_classic" and slow["aspect"] == "16:9"


def test_saved_style_is_listed(tmp_path):
    from aieditor_engine.agent.options import save_style

    key = save_style(tmp_path, "My Hype Style!", style_from_reference({"duration": 20, "cuts": list(range(1, 15)), "colorfulness": 0.5, "brightness": 0.5}, "x"))
    assert key == "my_hype_style" and video_styles(tmp_path)[key]["custom"]


# --------------------------------------------------------------- highlights


def test_text_score_prefers_hooks_over_filler():
    hook, why = text_score("Here's the secret nobody tells you about travel?")
    assert hook > 0.7 and "asks a question" in why
    assert text_score("um")[0] == 0.0
    assert text_score("and then we walked to the hotel.")[0] < 0.35
    assert text_score("What would you do with 3 free days here?")[0] > hook - 0.15


def test_sentences_split_on_punctuation_and_pauses():
    words = _words("one two three four five six", gap_after={2: 1.2})
    s = sentences({"words": words})
    assert [x["text"] for x in s] == ["one two three", "four five six"]
    assert len(sentences({"words": _words(TALK)})) == 8


def test_moments_start_and_end_on_sentences_and_skip_filler():
    m = _talk_media()
    moments = find_moments(m, length=6.0, count=3, use_llm=False)
    assert moments and moments[0].score >= moments[-1].score
    starts = {round(s["start"], 2) for s in sentences(m.transcript)}
    ends = {round(s["end"], 2) for s in sentences(m.transcript)}
    for mo in moments:
        assert any(abs(mo.start + 0.2 - s) < 0.01 or mo.start == 0.0 for s in starts)
        assert any(abs(min(m.duration, e + 0.35) - mo.end) < 0.01 for e in ends)
        assert not mo.text.lower().startswith(("um", "and", "okay"))
    # no two moments overlap by more than a fifth
    for i, a in enumerate(moments):
        for b in moments[i + 1:]:
            assert min(a.end, b.end) - max(a.start, b.start) <= 0.2 * (a.end - a.start)
    assert moments[0].text.startswith(("Here's", "What", "The view", "Most"))


def test_moments_without_speech_use_picture_and_sound():
    m = MediaInfo(1, "v", "v.mp4", "video", 60.0, "landscape", {"seconds": [0.3] * 40 + [0.9] * 10 + [0.3] * 10},
                  {"energy": [0.2] * 60})
    best = find_moments(m, length=8.0, count=2, use_llm=False)[0]
    assert 34 <= best.start <= 46 and best.end - best.start == pytest.approx(8.0)


# ----------------------------------------------------------------- cleanup


def test_keep_ranges_cuts_long_pauses_and_fillers():
    words = _words("hello there um friends", gap_after={1: 1.5})
    ranges = keep_ranges({"words": words}, 0.0, 10.0)
    assert len(ranges) == 2
    um = words[2]
    assert all(not (a <= (um["start"] + um["end"]) / 2 <= b) for a, b in ranges)
    assert saved_seconds(ranges, 0.0, words[-1]["end"]) > 1.0
    assert keep_ranges(None, 2.0, 5.0) == [(2.0, 5.0)]


# ----------------------------------------------------------------- reframe


def _moving_dot_video(path: Path, seconds: float = 4.0, fps: int = 15) -> Path:
    w, h = 320, 180
    with av.open(str(path), "w") as out:
        v = out.add_stream("mpeg4", rate=fps)
        v.width, v.height, v.pix_fmt = w, h, "yuv420p"
        n = int(seconds * fps)
        for i in range(n):
            img = np.full((h, w, 3), 40, np.uint8)
            cx = 40 if i < n / 2 else 270  # subject jumps from the left to the right half-way
            wob = int(6 * np.sin(i))  # it keeps moving a little, like a person talking
            img[60 + wob : 130 + wob, cx - 20 : cx + 20] = (200, 150, 120)  # skin-ish block
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts, f.time_base = i, Fraction(1, fps)
            out.mux(v.encode(f))
        out.mux(v.encode(None))
    return path


def test_subject_track_follows_the_moving_subject(tmp_path):
    tr = subject_track(_moving_dot_video(tmp_path / "dot.mp4"))
    t, x = np.asarray(tr["t"]), np.asarray(tr["x"])
    assert tr["method"] in ("saliency", "face") and tr["aspect"] == pytest.approx(16 / 9, abs=0.01)
    assert x[(t > 0.4) & (t < 1.5)].mean() < 0.4 and x[t > 2.8].mean() > 0.6


def test_reframe_keys_hold_then_pan_and_stay_in_bounds():
    track = {"t": [i / 3 for i in range(30)], "x": [0.2] * 15 + [0.8] * 15, "aspect": 16 / 9}
    keys = reframe_keys(track, 0.0, 9.6, 9 / 16)
    ratio = (16 / 9) / (9 / 16)
    limit = (ratio - 1) / 2
    assert keys[0][1] > 0 and keys[-1][1] < 0  # subject left -> picture shifted right, then the other way
    assert all(abs(v) <= limit + 1e-6 for _, v in keys) and len(keys) <= 6
    assert reframe_keys(track, 0.0, 9.6, 16 / 9) == []  # same shape: nothing to do
    centred = {"t": [0, 1, 2, 3], "x": [0.5, 0.52, 0.49, 0.5], "aspect": 16 / 9}
    assert reframe_keys(centred, 0, 3, 9 / 16) == []
    assert x_param([(0.0, 0.3)]) == " x=0.3" and x_param([(0.0, 0.3), (2.0, -0.1)]) == " x=0:0.3,2:-0.1"
    path = _camera_path(np.arange(10.0), np.array([0.5, 0.51, 0.52, 0.5, 0.53, 0.9, 0.9, 0.9, 0.9, 0.9]))
    assert path[0][1] == 0.5 and path[-1][1] == pytest.approx(0.9)  # holds through jitter, then follows


# -------------------------------------------------------------------- reels


def test_reels_from_one_long_video_apply_cleanly():
    song = MediaInfo(3, "song", "song.mp3", "audio", 120.0, music={"bpm": 100, "beats": []})
    talk = _talk_media(1)
    track = {"t": [i / 3 for i in range(80)], "x": [0.3] * 80, "aspect": 16 / 9}
    ctx = VideoContext([talk, _broll(2), song])
    r = make_reels(ctx, count=2, length=8, output=OutputOptions(platform="tiktok"), use_llm=False, subject_of=lambda mid: track)
    assert r["aspect"] == "9:16" and r["media_number"] == 1 and r["export"] == "reel_1080" and len(r["variations"]) == 2
    p = Project(name="r", kind=ProjectKind.VIDEO)
    p.media["talk1"] = MediaItem(id="talk1", name="talk", path="talk.mp4", kind=MediaKind.VIDEO, width=1920, height=1080, duration=talk.duration, has_audio=True)
    p.media["b2"] = MediaItem(id="b2", name="b", path="b.mp4", kind=MediaKind.VIDEO, width=1920, height=1080, duration=12.0)
    p.media["song"] = MediaItem(id="song", name="song", path="song.mp3", kind=MediaKind.AUDIO, duration=120.0)
    for v in r["variations"]:
        prog = v["program"]
        assert "aspect 9:16" in prog and "captions bold_pop" in prog and "music 3 volume=0.18" in prog and " x=" in prog
        assert v["info"]["reframed"] and v["moment"]["end"] > v["moment"]["start"]
        res = apply_video_program(p, prog, transcripts={"talk1": talk.transcript})
        assert res["clips"] == v["info"]["pieces"] and res["captions"] > 0
        clip = p.tracks[0].clips[0]
        assert clip.transform.x > 0  # subject on the left third: picture moved right
    no_caps = make_reels(ctx, count=1, length=8, captions=False, style="vlog", use_llm=False)
    assert "captions" not in no_caps["variations"][0]["program"]
    with pytest.raises(ValueError):
        make_reels(VideoContext([song]), use_llm=False)
