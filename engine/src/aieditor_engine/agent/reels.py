"""One long video in, N ready-to-post short clips out.

Each reel is a Video DSL program (so it is previewable, editable and one undo
step like every other AI edit):
    the best moment (highlights) -> silences and fillers cut (cleanup)
    -> subject kept in frame for the target shape (reframe)
    -> look, captions and quiet music from the chosen style.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..video.reframe import reframe_keys, x_param
from ..video.vdsl import ASPECTS
from .cleanup import keep_ranges, saved_seconds
from .highlights import Moment, find_moments
from .options import OptionError, OutputOptions, outputs, video_styles
from .video_planner import SAFE_VIDEO_LOOKS, MediaInfo, VideoContext

DEFAULT_STYLE = "energetic_reel"
PUNCH_IN = 1.07  # alternate tightened pieces between normal and slightly zoomed, so jump cuts look intended


def pick_source(ctx: VideoContext, media_number: int | None) -> MediaInfo:
    videos = [m for m in ctx.visual if m.kind == "video"]
    if not videos:
        raise ValueError("import a video first")
    if media_number is not None:
        for m in videos:
            if m.number == media_number:
                return m
        raise ValueError(f"#{media_number} is not a video in this project")
    # the longest video with speech, else the longest video
    return max(videos, key=lambda m: (m.speech, m.duration))


def reel_program(
    m: MediaInfo, moment: Moment, *, aspect: str, look: str | None, strength: float, captions: str | None,
    music: int | None, tighten: bool, subject: dict[str, Any] | None, progress_bar: bool = False,
    end_card: str | None = None,
) -> tuple[str, dict[str, Any]]:
    w, h = ASPECTS[aspect]
    ranges = keep_ranges(m.transcript, moment.start, moment.end) if tighten and m.speech else [(moment.start, moment.end)]
    lines = [f"# {moment.title}: {moment.reason}", f"aspect {aspect}"]
    for i, (a, b) in enumerate(ranges):
        keys = reframe_keys(subject, a, b, w / h)
        if b - a < 3.0 and len(keys) > 1:  # too short to pan inside: hold on the subject's average spot
            keys = [(0.0, round(sum(v for _, v in keys) / len(keys), 4))]
        punch = f" scale={PUNCH_IN:g}" if len(ranges) > 1 and i % 2 == 1 else ""
        vol = " volume=1" if not music else ""
        lines.append(f"clip {m.number} {a:.2f}-{b:.2f}{x_param(keys)}{punch}{vol}".rstrip())
    if music:
        lines.append(f"music {music} volume={0.18 if m.speech else 0.8:g} fade=0.8")
    if look:
        lines.append(f"look {look} s={strength:g}")
    lines.append("fade_out 0.4")
    if captions and m.speech:
        lines.append(f"captions {captions}")
    if progress_bar:
        lines.append("progress_bar top")
    if end_card:
        text = end_card.replace('"', "'")[:120]
        lines.append(f'end_card "{text}" dur=1.5')
    duration = sum(b - a for a, b in ranges)
    info = {
        "pieces": len(ranges), "duration": round(duration, 2), "silence_removed": round(saved_seconds(ranges, moment.start, moment.end), 2),
        "reframed": bool(subject) and any(" x=" in ln for ln in lines),
    }
    return "\n".join(lines), info


def make_reels(
    ctx: VideoContext, *, media_number: int | None = None, count: int = 3, length: float | None = None,
    style: str | None = None, output: OutputOptions | None = None, captions: bool | None = None, tighten: bool = True,
    use_llm: bool = True, subject_of: Callable[[str], dict[str, Any] | None] | None = None, styles_dir: Path | None = None,
    progress_bar: bool = True, end_card: str | None = None,
) -> dict[str, Any]:
    """Ranked reel programs for one source video. Nothing is applied."""
    styles = video_styles(styles_dir)
    key = style if style and style != "auto" else DEFAULT_STYLE
    if key not in styles:
        raise OptionError(f"unknown style `{key}`; choose one of {', '.join(styles)}")
    st = styles[key]
    plat: dict[str, Any] = {}
    if output and output.platform:
        plats = outputs()["video"]
        if output.platform not in plats:
            raise OptionError(f"unknown output `{output.platform}`; choose one of {', '.join(plats)}")
        plat = plats[output.platform]
    aspect = plat.get("aspect") or st["aspect"] or "9:16"
    want = (output.length if output else None) or length or st["length"] or plat.get("default_length") or 30.0
    if plat.get("max_length"):
        want = min(float(want), float(plat["max_length"]))
    count = (output.versions if output and output.versions else count) or 3
    m = pick_source(ctx, media_number)

    cap_on = captions if captions is not None else (output.captions if output and output.captions is not None else True)
    cap_style = (st["captions"] or "bold_pop") if cap_on else None
    music_on = output.music if output and output.music is not None else st["music"] != "off"
    music = ctx.songs[0].number if (music_on and ctx.songs) else None
    look = st["looks"][0] if st["looks"] else SAFE_VIDEO_LOOKS[0]

    moments = find_moments(m, float(want), count, use_llm=use_llm)
    w, h = ASPECTS[aspect]
    needs_reframe = m.orientation == "landscape" and w / h < 1.5
    subject = subject_of(m.media_id) if (needs_reframe and subject_of) else None
    variations = []
    for mo in moments:
        program, info = reel_program(m, mo, aspect=aspect, look=look, strength=0.75, captions=cap_style, music=music,
                                     tighten=tighten, subject=subject, progress_bar=progress_bar, end_card=end_card)
        variations.append({
            "program": program, "title": mo.title, "rationale": mo.reason, "score": round(mo.score, 3), "source": "highlights",
            "moment": mo.to_json(), "info": info, "warnings": [],
        })
    return {
        "variations": variations, "media_number": m.number, "media_id": m.media_id, "aspect": aspect, "style": key,
        "export": plat.get("export"), "tracking": (subject or {}).get("method"),
    }
