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
    progress_bar top
    end_card "Follow for part 2" dur=1.5

Layers: a clip can be placed and animated, and `overlay` puts media on a track
above the main video at a timeline time:

    clip 2 0-6 scale=1>1.15                         # slow push-in over the clip
    overlay 5 0-4 at=2 scale=0.3 x=0.32 y=-0.3      # picture-in-picture, top right
    overlay 6 at=1 blend=screen opacity=0.8         # light leak / effect layer
    overlay 7 at=0 opacity=0:0,0.5:1,3:1,3.5:0      # keyframes as time:value
    overlay 3 at=0 mask=ellipse:0.5,0.5,0.3,0.4 feather=0.05   # soft oval vignette cut-out
    solid #101820 at=0 dur=5 opacity=0.4 blend=multiply          # colour layer

x / y are fractions of the canvas (0 = centre), rotate is in degrees, crop is
left,top,right,bottom fractions, anchor=x,y moves the pivot for scale and
rotation. blend takes any Photoshop mode (multiply, screen, overlay, soft_light,
color_dodge, difference, hue, luminosity, ...). mask= is rect:l,t,r,b,
ellipse:cx,cy,rx,ry or path:x,y;x,y;... in layer fractions (a leading ! inverts
it), with feather= and expand= as fractions of the canvas's short side. Any of x y scale rotate opacity volume takes
a number, `from>to` (animated across the clip) or `t:v,t:v,...` keyframes.

Media are referenced by their 1-based position in the media bin (or by id).
Any photo Edit DSL statement (look, adjust, grain, vignette...) becomes part of
the grade of the whole video. `captions STYLE` turns transcripts into captions.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from ..core.project import (
    BLEND_MODES,
    Caption,
    Clip,
    Crop,
    EditOp,
    Keyframe,
    Mask,
    MaskPoint,
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
BLENDS = BLEND_MODES
VIDEO_STATEMENTS = ("aspect", "clip", "overlay", "music", "letterbox", "fade_in", "fade_out", "title", "text", "captions", "transition", "fps",
                    "progress_bar", "end_card", "solid")
POSITIONS = ("bottom", "center", "top", "lower_third", "center_low", "high")
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
        if tok.startswith("#") and not re.fullmatch(r"#[0-9a-fA-F]{6}", tok):  # a colour is not a comment
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


_LAYER_LIMITS = {"x": (-4.0, 4.0), "y": (-4.0, 4.0), "scale": (0.01, 20.0), "rotate": (-3600.0, 3600.0), "opacity": (0.0, 1.0), "volume": (0.0, 4.0)}


def _animated(raw: str, name: str, duration: float) -> float | list[Keyframe]:
    """`0.5` -> number; `1>1.2` -> two keyframes across the clip; `0:0,1:1` -> keyframes."""
    lo, hi = _LAYER_LIMITS[name]
    if ">" in raw:
        a, _, b = raw.partition(">")
        return [Keyframe(t=0.0, value=_num(a, name, lo, hi)), Keyframe(t=max(0.01, duration), value=_num(b, name, lo, hi))]
    if ":" in raw:
        keys = []
        for part in raw.split(","):
            t, _, v = part.partition(":")
            keys.append(Keyframe(t=_num(t, f"{name} keyframe time", 0), value=_num(v, name, lo, hi)))
        return sorted(keys, key=lambda k: k.t)
    return _num(raw, name, lo, hi)


def _apply_layer(clip: Clip, kv: dict[str, str]) -> None:
    """x= y= scale= rotate= opacity= blend= crop= (and animated volume) on a clip."""
    for name in ("x", "y", "scale", "rotate", "opacity", "volume"):
        if name not in kv:
            continue
        value = _animated(kv[name], name, clip.duration)
        param = "rotation" if name == "rotate" else name
        if isinstance(value, list):
            clip.keyframes[param] = value
        elif param in ("opacity", "volume"):
            setattr(clip, param, value)
        else:
            setattr(clip.transform, param, value)
    if "blend" in kv:
        if kv["blend"] not in BLENDS:
            raise VideoDslError(f"blend must be one of {', '.join(BLENDS)}")
        clip.blend = kv["blend"]  # type: ignore[assignment]
    if "anchor" in kv:
        parts = kv["anchor"].split(",")
        if len(parts) != 2:
            raise VideoDslError("anchor needs x,y (fractions from the centre, e.g. -0.3,0.2)")
        clip.transform.anchor_x, clip.transform.anchor_y = (_num(x, "anchor", -4, 4) for x in parts)
    if "mask" in kv:
        clip.masks = [_mask(kv["mask"], kv)]
    elif "feather" in kv or "expand" in kv:
        raise VideoDslError("feather= and expand= need a mask=")
    if "crop" in kv:
        parts = kv["crop"].split(",")
        if len(parts) != 4:
            raise VideoDslError("crop needs left,top,right,bottom (fractions, e.g. 0.1,0,0.1,0)")
        left, top, right, bottom = (_num(x, "crop", 0, 0.95) for x in parts)
        if left + right >= 0.98 or top + bottom >= 0.98:
            raise VideoDslError("crop removes the whole picture")
        clip.crop = Crop(left=left, top=top, right=right, bottom=bottom)


def _mask(raw: str, kv: dict[str, str]) -> Mask:
    inverted = raw.startswith("!")
    kind, _, args = raw.lstrip("!").partition(":")
    extra = {"inverted": inverted, "feather": _num(kv.get("feather", "0"), "feather", 0, 0.5),
             "expansion": _num(kv.get("expand", "0"), "expand", -0.5, 0.5)}
    try:
        if kind == "rect":
            left, top, right, bottom = (float(v) for v in args.split(","))
            if right <= left or bottom <= top:
                raise VideoDslError("mask=rect needs left<right and top<bottom")
            return Mask.rect(left, top, right, bottom, **extra)
        if kind == "ellipse":
            cx, cy, rx, ry = (float(v) for v in args.split(","))
            if rx <= 0 or ry <= 0:
                raise VideoDslError("mask=ellipse needs positive radii")
            return Mask.ellipse(cx, cy, rx, ry, **extra)
        if kind == "path":
            pts = []
            for part in args.split(";"):
                v = [float(x) for x in part.split(",")]
                if len(v) not in (2, 6):
                    raise ValueError
                pts.append(MaskPoint(x=v[0], y=v[1]) if len(v) == 2 else
                           MaskPoint(x=v[0], y=v[1], in_x=v[2], in_y=v[3], out_x=v[4], out_y=v[5]))
            return Mask(points=pts, **extra)
    except (ValueError, TypeError) as exc:
        raise VideoDslError(f"bad mask `{raw}`: {exc}" if str(exc) else f"bad mask `{raw}`") from exc
    raise VideoDslError("mask must be rect:l,t,r,b, ellipse:cx,cy,rx,ry or path:x,y;x,y;...")


def _mask_dsl(m: Mask) -> list[str]:
    pts = ";".join(
        ",".join(_fmt(v) for v in ((p.x, p.y) if not (p.in_x or p.in_y or p.out_x or p.out_y)
                                   else (p.x, p.y, p.in_x, p.in_y, p.out_x, p.out_y)))
        for p in m.points
    )
    parts = [f"mask={'!' if m.inverted else ''}path:{pts}"]
    if m.feather:
        parts.append(f"feather={_fmt(m.feather)}")
    if m.expansion:
        parts.append(f"expand={_fmt(m.expansion)}")
    return parts


def _layer_dsl(c: Clip) -> list[str]:
    """Inverse of _apply_layer (volume is written by the caller)."""
    parts = []
    for param, name, default in (("x", "x", 0.0), ("y", "y", 0.0), ("scale", "scale", 1.0), ("rotation", "rotate", 0.0), ("opacity", "opacity", 1.0)):
        keys = c.keyframes.get(param)
        if keys:
            parts.append(f"{name}=" + ",".join(f"{_fmt(k.t)}:{_fmt(k.value)}" for k in keys))
            continue
        value = c.opacity if param == "opacity" else getattr(c.transform, param)
        if abs(value - default) > 1e-9:
            parts.append(f"{name}={_fmt(value)}")
    if c.blend != "normal":
        parts.append(f"blend={c.blend}")
    if c.transform.anchor_x or c.transform.anchor_y:
        parts.append(f"anchor={_fmt(c.transform.anchor_x)},{_fmt(c.transform.anchor_y)}")
    masks = [m for m in c.masks if m.enabled]
    if masks:
        parts += _mask_dsl(masks[0])  # the DSL carries one mask; more are kept by the editor
    if not c.crop.is_empty:
        parts.append("crop=" + ",".join(_fmt(v) for v in (c.crop.left, c.crop.top, c.crop.right, c.crop.bottom)))
    return parts


def _track(p: Project, kind: str, name: str) -> Track:
    for t in p.tracks:
        if t.kind == kind and (kind == "video" or t.name == name):
            return t
    t = Track(id=new_id("trk"), kind=kind, name=name)  # type: ignore[arg-type]
    p.tracks.append(t)  # list order is the stacking order: later tracks are drawn on top
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
            elif name in ("clip", "overlay"):
                if not pos:
                    raise VideoDslError(f"{name} needs a media number")
                media_id = _media(p, pos[0])
                item = p.media[media_id]
                if item.kind == MediaKind.AUDIO:
                    raise VideoDslError(f"#{pos[0]} is audio; use `music`")
                length = item.duration or STILL_SECONDS
                if len(pos) > 1 and pos[1] != "mute":
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
                at = _num(kv["at"], "at", 0) if name == "overlay" and "at" in kv else (0.0 if name == "overlay" else cursor)
                clip = Clip(id=new_id("clp"), media_id=media_id, start=at, in_point=a, out_point=b, speed=speed)
                if "transition" in kv:
                    kind, _, dur = kv["transition"].partition(":")
                    if kind not in TRANSITIONS:
                        raise VideoDslError(f"transition must be one of {', '.join(TRANSITIONS)}")
                    clip.transition = Transition(kind=kind, duration=min(_num(dur or "0.5", "transition duration", 0.05, 5), clip.duration))  # type: ignore[arg-type]
                if "motion" in kv:
                    if kv["motion"] not in MOTIONS:
                        raise VideoDslError(f"motion must be one of {', '.join(MOTIONS)}")
                    clip.motion = kv["motion"]  # type: ignore[assignment]
                if "mute" in pos[1:] or kv.get("mute") in ("true", "1", "yes"):
                    clip.muted = True
                if "fit" in kv:
                    clip.fit = "fit" if kv["fit"] == "fit" else "fill"
                if "fade" in kv:
                    clip.fade_in = clip.fade_out = min(_num(kv["fade"], "fade", 0, 30), clip.duration / 2)
                _apply_layer(clip, kv)
                if name == "overlay":
                    layer = _num(kv.get("track", "1"), "track", 1, 8)
                    target = _track(p, "overlay", f"Overlay {int(layer)}")
                    if any(c.start < clip.end and clip.start < c.end for c in target.clips):
                        raise VideoDslError(f"overlaps another clip on overlay track {int(layer)}; use track={int(layer) + 1}")
                    target.clips.append(clip)
                    target.clips.sort(key=lambda c: c.start)
                else:
                    video.clips.append(clip)
                    added_clips.append(clip)
                    cursor = clip.end
            elif name == "solid":
                color = (pos[0] if pos else kv.get("color", "")).lower()
                if not re.fullmatch(r"#[0-9a-f]{6}", color):
                    raise VideoDslError("solid needs a colour like #1a2b3c")
                at = _num(kv.get("at", "0"), "at", 0)
                dur = _num(kv.get("dur", "5"), "dur", 0.1, 36000)
                clip = Clip(id=new_id("clp"), source="solid", color=color, start=at, in_point=0.0, out_point=dur, name="Solid")
                if "fade" in kv:
                    clip.fade_in = clip.fade_out = min(_num(kv["fade"], "fade", 0, 30), dur / 2)
                _apply_layer(clip, kv)
                layer = _num(kv.get("track", "1"), "track", 1, 8)
                target = _track(p, "overlay", f"Overlay {int(layer)}")
                if any(c.start < clip.end and clip.start < c.end for c in target.clips):
                    raise VideoDslError(f"overlaps another clip on overlay track {int(layer)}; use track={int(layer) + 1}")
                target.clips.append(clip)
                target.clips.sort(key=lambda c: c.start)
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
                if position is not None and position not in POSITIONS:
                    raise VideoDslError(f"pos must be one of {', '.join(POSITIONS)}")
                p.captions.append(Caption(id=new_id("cap"), start=at, end=at + dur, text=" ".join(pos)[:300],
                                          style=style, position=position, kind="title" if name == "title" else "caption"))  # type: ignore[arg-type]
            elif name == "captions":
                style = pos[0] if pos else kv.get("style", "clean")
                if style not in CAPTION_STYLES:
                    raise VideoDslError(f"captions style must be one of {', '.join(CAPTION_STYLES)}")
                pending_captions = style
            elif name == "progress_bar":
                v = pos[0] if pos else "top"
                if v not in ("top", "bottom", "off", "none"):
                    raise VideoDslError("progress_bar must be top, bottom or off")
                p.settings.progress_bar = "off" if v == "none" else v  # type: ignore[assignment]
            elif name == "end_card":
                text = " ".join(pos).strip()
                if text.lower() in ("", "off", "none"):
                    p.settings.end_card = None
                else:
                    p.settings.end_card = text[:120]
                    p.settings.end_card_seconds = _num(kv.get("dur", "1.5"), "dur", 0.5, 10)
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
            if OPS[op_name].category == "subject":
                raise VideoDslError("`subject` edits are for photos; in a video, pick the subject in the viewer and use Follow subject")
            ops.append(EditOp(id=new_id("op"), op=op_name, params=params))
        if replace:
            p.grade = ops
        else:
            if any(o.op == "look" for o in ops):
                p.grade = [o for o in p.grade if o.op != "look"]
            p.grade.extend(ops)
    video.clips.sort(key=lambda c: c.start)
    # stacking order: main video at the bottom, overlay tracks by number, audio last
    def _stack(t: Track) -> tuple[int, int]:
        n = t.name.rsplit(" ", 1)[-1]
        return ({"video": 0, "overlay": 1, "audio": 2}[t.kind], int(n) if t.kind == "overlay" and n.isdigit() else 0)

    p.tracks.sort(key=_stack)
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


def _volume_dsl(c: Clip) -> list[str]:
    if c.muted:
        return ["mute"]
    keys = c.keyframes.get("volume")
    if keys:
        return ["volume=" + ",".join(f"{_fmt(k.t)}:{_fmt(k.value)}" for k in keys)]
    return [f"volume={_fmt(c.volume)}"] if abs(c.volume - 1) > 1e-6 else []


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
                parts += _volume_dsl(c) + _layer_dsl(c)
                lines.append(" ".join(parts))
            break
    for t in p.tracks:
        if t.kind != "overlay":
            continue
        layer = t.name.rsplit(" ", 1)[-1] if t.name.startswith("Overlay ") else "1"
        for c in t.clips:
            if c.source == "adjustment":
                continue  # edited in the app; the DSL has no adjustment layers yet
            if c.source == "solid":
                parts = [f"solid {c.color} at={_fmt(c.start)} dur={_fmt(c.duration)}"]
            else:
                parts = [f"overlay {num.get(c.media_id, c.media_id)} {_fmt(c.in_point)}-{_fmt(c.out_point)} at={_fmt(c.start)}"]
            if layer != "1":
                parts.append(f"track={layer}")
            if abs(c.speed - 1) > 1e-6:
                parts.append(f"speed={_fmt(c.speed)}")
            if c.fit != "fill":
                parts.append(f"fit={c.fit}")
            if c.fade_in and abs(c.fade_in - c.fade_out) < 1e-6:
                parts.append(f"fade={_fmt(c.fade_in)}")
            parts += _volume_dsl(c) + _layer_dsl(c)
            lines.append(" ".join(parts))
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
            line = f'title "{text}" at={_fmt(c.start)} dur={_fmt(c.end - c.start)} style={c.style}'
            lines.append(line + (f" pos={c.position}" if c.position else ""))
    caps = [c for c in p.captions if c.kind == "caption"]
    if caps:
        lines.append(f"captions {caps[0].style}")
    if s.progress_bar != "off":
        lines.append(f"progress_bar {s.progress_bar}")
    if s.end_card:
        text = s.end_card.replace('"', "'")
        lines.append(f'end_card "{text}" dur={_fmt(s.end_card_seconds)}')
    return "\n".join(lines)
