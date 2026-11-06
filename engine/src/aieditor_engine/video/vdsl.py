"""Video Edit DSL: the compact program EditLM writes for videos.

    # travel reel: fast cuts on the beat, vivid look, bold captions
    aspect 9:16
    clip 1 2.0-3.5
    clip 3 0.0-1.5 transition=crossfade:0.3
    clip 2 4.1-5.0 speed=0.5 motion=zoom_in
    music 4 volume=0.8 fade=1.5
    look vivid_pop s=0.8
    grain 0.15
    letterbox 2.39
    fade_in 0.5
    fade_out 1
    title "Bali 2026" at=0 dur=2.5 style=title
    captions bold_pop

Media are referenced by their 1-based position in the media bin (or by id).
Any photo Edit DSL statement (look, adjust, grain, vignette...) becomes part of
the grade of the whole video. `captions STYLE` turns transcripts into captions.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from ..core.project import (
    Caption,
    Clip,
    EditOp,
    MediaKind,
    Project,
    Track,
    Transition,
    new_id,
)
from ..dsl import DslError, _fmt, _split_pipes, parse_statement
from ..imaging.ops import OPS
from .captions import captions_from_transcripts

ASPECTS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350), "4:3": (1440, 1080), "21:9": (2560, 1080)}
TRANSITIONS = ("crossfade", "fade_black", "fade_white", "wipe_left", "wipe_right", "slide_left", "slide_up", "zoom", "blur")
MOTIONS = ("none", "zoom_in", "zoom_out", "pan_left", "pan_right")
CAPTION_STYLES = ("clean", "bold_pop", "cinematic", "boxed", "title", "minimal")
VIDEO_STATEMENTS = ("aspect", "clip", "music", "letterbox", "fade_in", "fade_out", "title", "text", "captions", "transition", "fps")
STILL_SECONDS = 3.0


class VideoDslError(ValueError):
    pass


def _statements(text: str) -> list[str]:
    out = []
    for line in text.replace("\r", "").split("\n"):
        for part in _split_pipes(line):
            part = part.strip()
            if part and not part.startswith("#"):
                out.append(part)
    return out


def _tokens(stmt: str) -> list[str]:
    try:
        toks = shlex.split(stmt, posix=True)
    except ValueError as exc:
        raise VideoDslError(f"`{stmt}`: {exc}") from exc
    for i, tok in enumerate(toks):
        if tok.startswith("#"):
            return toks[:i]
    return toks


def _kv(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    pos, kv = [], {}
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.lower()] = v
        else:
            pos.append(tok)
    return pos, kv


def _num(raw: str, what: str, lo: float | None = None, hi: float | None = None) -> float:
    try:
        v = float(raw)
    except ValueError as exc:
        raise VideoDslError(f"{what}: `{raw}` is not a number") from exc
    if lo is not None and v < lo or hi is not None and v > hi:
        raise VideoDslError(f"{what} must be between {lo} and {hi} (got {v:g})")
    return v


def media_order(p: Project) -> list[str]:
    return list(p.media)


def _media(p: Project, ref: str) -> str:
    if ref in p.media:
        return ref
    ref = ref.lstrip("#")
    if ref.isdigit():
        ids = media_order(p)
        i = int(ref) - 1
        if 0 <= i < len(ids):
            return ids[i]
    raise VideoDslError(f"unknown media `{ref}` (use its number in the media bin)")


def _range(raw: str) -> tuple[float, float]:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", raw)
    if not m:
        raise VideoDslError(f"bad range `{raw}` (use start-end in seconds, e.g. 1.5-4)")
    a, b = float(m.group(1)), float(m.group(2))
    if b <= a:
        raise VideoDslError(f"range `{raw}`: end must be after start")
    return a, b


def _track(p: Project, kind: str, name: str) -> Track:
    for t in p.tracks:
        if t.kind == kind and (kind != "audio" or t.name == name):
            return t
    t = Track(id=new_id("trk"), kind=kind, name=name)  # type: ignore[arg-type]
    p.tracks.append(t)
    return t


def apply_video_program(
    p: Project, text: str, *, replace: bool = True, transcripts: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Mutate the project. Raises VideoDslError on the first bad statement
    (the bus runs commands on a copy, so a failed program changes nothing)."""
    stmts = _statements(text)
    if not stmts:
        raise VideoDslError("empty program")
    video = _track(p, "video", "Video 1")
    music_track: Track | None = None
    if replace:
        for t in p.tracks:
            if t.kind != "audio" or t.name == "Music":
                t.clips.clear()
        p.grade = []
        p.captions = []
        p.settings.letterbox = None
        p.settings.fade_in = p.settings.fade_out = 0.0
    cursor = max((c.end for c in video.clips), default=0.0)
    grade_lines: list[str] = []
    pending_captions: str | None = None
    all_transition: tuple[str, float] | None = None
    added_clips: list[Clip] = []

    for i, stmt in enumerate(stmts, start=1):
        toks = _tokens(stmt)
        if not toks:
            continue
        name = toks[0].lower()
        pos, kv = _kv(toks[1:])
        where = f"statement {i} `{stmt}`"
        try:
            if name == "aspect":
                ratio = (pos[0] if pos else kv.get("ratio", "")).strip()
                if ratio not in ASPECTS:
                    raise VideoDslError(f"aspect must be one of {', '.join(ASPECTS)}")
                p.settings.width, p.settings.height = ASPECTS[ratio]
            elif name == "fps":
                p.settings.fps = _num(pos[0] if pos else "", "fps", 12, 120)
            elif name == "clip":
                if not pos:
                    raise VideoDslError("clip needs a media number")
                media_id = _media(p, pos[0])
                item = p.media[media_id]
                if item.kind == MediaKind.AUDIO:
                    raise VideoDslError(f"#{pos[0]} is audio; use `music`")
                length = item.duration or STILL_SECONDS
                if len(pos) > 1:
                    a, b = _range(pos[1])
                elif item.kind == MediaKind.IMAGE:
                    a, b = 0.0, _num(kv.get("dur", str(STILL_SECONDS)), "dur", 0.2, 60)
                else:
                    a, b = 0.0, length
                if item.kind != MediaKind.IMAGE:
                    b = min(b, length)
                    if b - a < 0.1:
                        raise VideoDslError(f"range is outside the media ({length:.2f}s long)")
                speed = _num(kv.get("speed", "1"), "speed", 0.1, 10)
                clip = Clip(id=new_id("clp"), media_id=media_id, start=cursor, in_point=a, out_point=b, speed=speed)
                if "transition" in kv:
                    kind, _, dur = kv["transition"].partition(":")
                    if kind not in TRANSITIONS:
                        raise VideoDslError(f"transition must be one of {', '.join(TRANSITIONS)}")
                    clip.transition = Transition(kind=kind, duration=min(_num(dur or "0.5", "transition duration", 0.05, 5), clip.duration))  # type: ignore[arg-type]
                if "motion" in kv:
                    if kv["motion"] not in MOTIONS:
                        raise VideoDslError(f"motion must be one of {', '.join(MOTIONS)}")
                    clip.motion = kv["motion"]  # type: ignore[assignment]
                if "volume" in kv:
                    clip.volume = _num(kv["volume"], "volume", 0, 4)
                if "mute" in pos[1:] or kv.get("mute") in ("true", "1", "yes"):
                    clip.muted = True
                if "fit" in kv:
                    clip.fit = "fit" if kv["fit"] == "fit" else "fill"
                video.clips.append(clip)
                added_clips.append(clip)
                cursor = clip.end
            elif name == "music":
                if not pos:
                    raise VideoDslError("music needs a media number")
                media_id = _media(p, pos[0])
                item = p.media[media_id]
                if item.kind != MediaKind.AUDIO:
                    raise VideoDslError(f"#{pos[0]} is not an audio file")
                music_track = music_track or _track(p, "audio", "Music")
                offset = _num(kv.get("from", "0"), "from", 0)
                fade = _num(kv.get("fade", "1.5"), "fade", 0, 10)
                clip = Clip(
                    id=new_id("clp"), media_id=media_id, start=0.0, in_point=offset,
                    out_point=max(offset + 0.5, item.duration or offset + 0.5),
                    volume=_num(kv.get("volume", "0.8"), "volume", 0, 4), fade_in=min(fade, 1.0), fade_out=fade,
                )
                music_track.clips[:] = [clip]
            elif name == "letterbox":
                v = pos[0] if pos else "2.39"
                p.settings.letterbox = None if v in ("off", "none", "0") else _num(v, "letterbox", 1.0, 3.0)
            elif name == "fade_in":
                p.settings.fade_in = _num(pos[0] if pos else "1", "fade_in", 0, 10)
            elif name == "fade_out":
                p.settings.fade_out = _num(pos[0] if pos else "1", "fade_out", 0, 10)
            elif name in ("title", "text"):
                if not pos:
                    raise VideoDslError(f"{name} needs quoted text")
                style = kv.get("style", "title" if name == "title" else "clean")
                if style not in CAPTION_STYLES:
                    raise VideoDslError(f"style must be one of {', '.join(CAPTION_STYLES)}")
                at = _num(kv.get("at", "0"), "at", 0)
                dur = _num(kv.get("dur", "3"), "dur", 0.3, 60)
                position = kv.get("pos")
                if position not in (None, "bottom", "center", "top", "lower_third"):
                    raise VideoDslError("pos must be bottom, center, top or lower_third")
                p.captions.append(Caption(id=new_id("cap"), start=at, end=at + dur, text=" ".join(pos)[:300],
                                          style=style, position=position, kind="title" if name == "title" else "caption"))  # type: ignore[arg-type]
            elif name == "captions":
                style = pos[0] if pos else kv.get("style", "clean")
                if style not in CAPTION_STYLES:
                    raise VideoDslError(f"captions style must be one of {', '.join(CAPTION_STYLES)}")
                pending_captions = style
            elif name == "transition":
                kind = pos[0] if pos else "crossfade"
                if kind not in TRANSITIONS + ("none",):
                    raise VideoDslError(f"transition must be one of none, {', '.join(TRANSITIONS)}")
                all_transition = (kind, _num(pos[1] if len(pos) > 1 else kv.get("dur", "0.5"), "duration", 0.05, 5))
            elif name in OPS or parse_statement_known(name):
                grade_lines.append(stmt)
            else:
                raise VideoDslError(f"unknown statement `{name}`")
        except VideoDslError as exc:
            raise VideoDslError(f"{where}: {exc}") from None

    if grade_lines:
        ops = []
        for j, line in enumerate(grade_lines, start=1):
            try:
                op_name, params = parse_statement(line, j)
            except DslError as exc:
                raise VideoDslError(str(exc)) from None
            if OPS[op_name].geometry:
                raise VideoDslError(f"`{op_name}` is a photo crop/rotate; use `aspect` for videos")
            ops.append(EditOp(id=new_id("op"), op=op_name, params=params))
        if replace:
            p.grade = ops
        else:
            if any(o.op == "look" for o in ops):
                p.grade = [o for o in p.grade if o.op != "look"]
            p.grade.extend(ops)
    video.clips.sort(key=lambda c: c.start)
    if all_transition and video.clips:
        kind, dur = all_transition
        first = video.clips[0]
        for clip in added_clips or video.clips:
            if clip is not first:
                clip.transition = None if kind == "none" else Transition(kind=kind, duration=min(dur, clip.duration))  # type: ignore[arg-type]

    # music never runs past the picture
    end = max((c.end for c in video.clips), default=0.0)
    for t in p.tracks:
        if t.kind == "audio" and t.name == "Music":
            for c in t.clips:
                if end > 0 and c.end > end:
                    c.out_point = c.in_point + (end - c.start) * c.speed
    if pending_captions:
        caps = captions_from_transcripts(p, transcripts or {}, pending_captions)
        p.captions = [c for c in p.captions if c.kind != "caption"] + caps
    p.captions.sort(key=lambda c: c.start)
    return {
        "clips": len(video.clips),
        "duration": round(p.duration, 3),
        "captions": sum(1 for c in p.captions if c.kind == "caption"),
        "warnings": ["no transcript available; captions skipped"] if pending_captions and not transcripts else [],
    }


def parse_statement_known(name: str) -> bool:
    from ..dsl import ALIASES

    return name in ALIASES


# ---------------------------------------------------------------- decompile


def timeline_to_dsl(p: Project) -> str:
    """Current timeline as a program (shown to users; fed back to the model)."""
    ids = media_order(p)
    num = {mid: i + 1 for i, mid in enumerate(ids)}
    lines = []
    w, h = p.settings.width, p.settings.height
    ratio = next((r for r, (aw, ah) in ASPECTS.items() if aw * h == ah * w), None)
    if ratio:
        lines.append(f"aspect {ratio}")
    for t in p.tracks:
        if t.kind == "video":
            for c in t.clips:
                parts = [f"clip {num.get(c.media_id, c.media_id)} {_fmt(c.in_point)}-{_fmt(c.out_point)}"]
                if abs(c.speed - 1) > 1e-6:
                    parts.append(f"speed={_fmt(c.speed)}")
                if c.transition:
                    parts.append(f"transition={c.transition.kind}:{_fmt(c.transition.duration)}")
                if c.motion != "none":
                    parts.append(f"motion={c.motion}")
                if c.muted:
                    parts.append("mute")
                elif abs(c.volume - 1) > 1e-6:
                    parts.append(f"volume={_fmt(c.volume)}")
                lines.append(" ".join(parts))
            break
    for t in p.tracks:
        if t.kind == "audio" and t.name == "Music":
            for c in t.clips[:1]:
                lines.append(f"music {num.get(c.media_id, c.media_id)} volume={_fmt(c.volume)} fade={_fmt(c.fade_out)}")
    from ..dsl import to_dsl

    grade = to_dsl([o.model_dump(mode="json") for o in p.grade])
    if grade:
        lines.extend(grade.splitlines())
    s = p.settings
    if s.letterbox:
        lines.append(f"letterbox {_fmt(s.letterbox)}")
    if s.fade_in:
        lines.append(f"fade_in {_fmt(s.fade_in)}")
    if s.fade_out:
        lines.append(f"fade_out {_fmt(s.fade_out)}")
    for c in p.captions:
        if c.kind == "title":
            text = c.text.replace('"', "'")
            lines.append(f'title "{text}" at={_fmt(c.start)} dur={_fmt(c.end - c.start)} style={c.style}')
    caps = [c for c in p.captions if c.kind == "caption"]
    if caps:
        lines.append(f"captions {caps[0].style}")
    return "\n".join(lines)
