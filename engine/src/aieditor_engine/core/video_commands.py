"""Video timeline commands: split, speed, transitions, per-clip looks, grade,
captions and the AI's whole-program command. Each is one undo step."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..dsl import DslError, compile_program
from .commands import CommandError, command
from .photo_commands import _validated
from .project import (
    KEYABLE,
    BlendMode,
    Caption,
    CaptionPosition,
    CaptionStyle,
    Clip,
    EditOp,
    Keyframe,
    LayerSource,
    Marker,
    Mask,
    MediaKind,
    MatteMode,
    Motion,
    Project,
    Track,
    Transition,
    TransitionKind,
    new_id,
)

MIN_CLIP = 0.1  # seconds


def _ripple(track: Track, after: float, delta: float, skip: Clip | None = None) -> None:
    """Shift every clip on the track starting at/after `after` by delta seconds."""
    for c in track.clips:
        if c is not skip and c.start >= after - 1e-6:
            c.start = max(0.0, c.start + delta)


# ------------------------------------------------------------------- clips


class SplitParams(BaseModel):
    clip_id: str
    at: float = Field(description="Timeline time of the cut, inside the clip")


@command("clip.split", SplitParams, "Cut a clip in two at a timeline time.", "timeline")
def clip_split(p: Project, a: SplitParams) -> dict:
    track, clip = p.find_clip(a.clip_id)
    if not (clip.start + MIN_CLIP <= a.at <= clip.end - MIN_CLIP):
        raise CommandError("split point must be inside the clip")
    cut_src = clip.source_time(a.at)
    right = clip.model_copy(deep=True)
    right.id = new_id("clp")
    right.start = a.at
    right.in_point = cut_src
    right.transition = None
    right.fade_in = 0.0
    clip.out_point = cut_src
    clip.fade_out = 0.0
    track.clips.append(right)
    track.clips.sort(key=lambda c: c.start)
    return {"clip_id": right.id}


class ClipSetParams(BaseModel):
    clip_id: str
    speed: float | None = Field(None, ge=0.1, le=10)
    volume: float | None = Field(None, ge=0, le=4)
    muted: bool | None = None
    fade_in: float | None = Field(None, ge=0, le=30)
    fade_out: float | None = Field(None, ge=0, le=30)
    fit: Literal["fill", "fit"] | None = None
    motion: Motion | None = None
    ripple: bool = Field(True, description="When speed changes the length, shift later clips")


@command("clip.set", ClipSetParams, "Change a clip's speed, volume, fades, framing or motion.", "timeline")
def clip_set(p: Project, a: ClipSetParams) -> None:
    track, clip = p.find_clip(a.clip_id)
    old_end = clip.end
    for key in ("speed", "volume", "muted", "fade_in", "fade_out", "fit", "motion"):
        value = getattr(a, key)
        if value is not None:
            setattr(clip, key, value)
    if a.ripple and abs(clip.end - old_end) > 1e-6:
        _ripple(track, old_end, clip.end - old_end, skip=clip)


class TransitionParams(BaseModel):
    clip_id: str = Field(description="The clip the transition leads INTO")
    kind: TransitionKind | Literal["none"] = "crossfade"
    duration: float = Field(0.5, gt=0, le=5)


@command("clip.set_transition", TransitionParams, "Set or remove the transition into a clip.", "timeline")
def clip_set_transition(p: Project, a: TransitionParams) -> None:
    _, clip = p.find_clip(a.clip_id)
    if a.kind == "none":
        clip.transition = None
        return
    max_d = max(0.1, min(clip.duration, 5.0))
    clip.transition = Transition(kind=a.kind, duration=min(a.duration, max_d))


class TransitionAllParams(BaseModel):
    track_id: str | None = None
    kind: TransitionKind | Literal["none"] = "crossfade"
    duration: float = Field(0.5, gt=0, le=5)


@command("timeline.set_transitions", TransitionAllParams, "Apply one transition type to every cut on a track.", "timeline")
def set_all_transitions(p: Project, a: TransitionAllParams) -> dict:
    tracks = [p.find_track(a.track_id)] if a.track_id else [t for t in p.tracks if t.kind == "video"][:1]
    n = 0
    for track in tracks:
        for clip in track.clips[1:]:
            clip.transition = None if a.kind == "none" else Transition(kind=a.kind, duration=min(a.duration, clip.duration))
            n += 1
    return {"cuts": n}


class RippleDeleteParams(BaseModel):
    clip_id: str


@command("clip.ripple_delete", RippleDeleteParams, "Delete a clip and close the gap.", "timeline")
def clip_ripple_delete(p: Project, a: RippleDeleteParams) -> None:
    track, clip = p.find_clip(a.clip_id)
    track.clips.remove(clip)
    _ripple(track, clip.end, -clip.duration)


class ClipProgramParams(BaseModel):
    clip_id: str
    program: str = Field(description="Edit DSL (color/effect ops; geometry ops are ignored on video)")


@command("clip.apply_program", ClipProgramParams, "Replace a clip's color/effect stack with an Edit DSL program.", "timeline")
def clip_apply_program(p: Project, a: ClipProgramParams) -> dict:
    _, clip = p.find_clip(a.clip_id)
    clip.ops = _compile_ops(a.program)
    return {"ops": len(clip.ops)}


class ClipLookParams(BaseModel):
    clip_id: str
    name: str | None = Field(None, description="Look name; null removes the clip's look")
    strength: float = Field(1.0, ge=0, le=1.5)


@command("clip.apply_look", ClipLookParams, "Set (or clear) a look on one clip.", "timeline")
def clip_apply_look(p: Project, a: ClipLookParams) -> None:
    _, clip = p.find_clip(a.clip_id)
    clip.ops = [o for o in clip.ops if o.op != "look"]
    if a.name:
        clip.ops.append(EditOp(id=new_id("op"), op="look", params=_validated("look", {"name": a.name, "strength": a.strength})))


def _compile_ops(program: str) -> list[EditOp]:
    try:
        ops = compile_program(program)
    except DslError as exc:
        raise CommandError(str(exc)) from exc
    return [EditOp(id=new_id("op"), op=o["op"], params=o["params"]) for o in ops]


# ------------------------------------------------------------------- grade


class GradeProgramParams(BaseModel):
    program: str = Field(description="Edit DSL applied to the whole video, e.g. `look teal_orange_cinematic s=0.8 | grain 0.2`")


@command("timeline.apply_grade", GradeProgramParams, "Set the color grade / look of the whole video.", "timeline")
def timeline_apply_grade(p: Project, a: GradeProgramParams) -> dict:
    p.grade = _compile_ops(a.program)
    return {"ops": len(p.grade)}


class GradeLookParams(BaseModel):
    name: str | None = None
    strength: float = Field(1.0, ge=0, le=1.5)


@command("timeline.apply_look", GradeLookParams, "Set (or clear) the look of the whole video, keeping other grade ops.", "timeline")
def timeline_apply_look(p: Project, a: GradeLookParams) -> None:
    p.grade = [o for o in p.grade if o.op != "look"]
    if a.name:
        p.grade.insert(0, EditOp(id=new_id("op"), op="look", params=_validated("look", {"name": a.name, "strength": a.strength})))


class GradeOpSetParams(BaseModel):
    op_id: str
    params: dict[str, Any] | None = None
    enabled: bool | None = None


@command("timeline.set_grade_op", GradeOpSetParams, "Change or toggle one op of the video grade.", "timeline")
def timeline_set_grade_op(p: Project, a: GradeOpSetParams) -> None:
    item = next((o for o in p.grade if o.id == a.op_id), None)
    if item is None:
        raise CommandError(f"op not found: {a.op_id}")
    if a.params is not None:
        item.params = _validated(item.op, {**item.params, **a.params})
    if a.enabled is not None:
        item.enabled = a.enabled


class GradeOpAddParams(BaseModel):
    op: str
    params: dict[str, Any] = Field(default_factory=dict)


@command("timeline.add_grade_op", GradeOpAddParams, "Add an op to the video grade.", "timeline")
def timeline_add_grade_op(p: Project, a: GradeOpAddParams) -> dict:
    item = EditOp(id=new_id("op"), op=a.op, params=_validated(a.op, a.params))
    p.grade.append(item)
    return {"op_id": item.id}


class GradeOpRemoveParams(BaseModel):
    op_id: str


@command("timeline.remove_grade_op", GradeOpRemoveParams, "Remove an op from the video grade.", "timeline")
def timeline_remove_grade_op(p: Project, a: GradeOpRemoveParams) -> None:
    before = len(p.grade)
    p.grade = [o for o in p.grade if o.id != a.op_id]
    if len(p.grade) == before:
        raise CommandError(f"op not found: {a.op_id}")


# ---------------------------------------------------------------- captions


class CaptionAddParams(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1, max_length=300)
    style: CaptionStyle = "clean"
    position: CaptionPosition | None = None
    kind: Literal["caption", "title"] = "caption"


@command("caption.add", CaptionAddParams, "Add timed text (subtitle or title).", "text")
def caption_add(p: Project, a: CaptionAddParams) -> dict:
    if a.end <= a.start:
        raise CommandError("end must be after start")
    cap = Caption(id=new_id("cap"), **a.model_dump())
    p.captions.append(cap)
    p.captions.sort(key=lambda c: c.start)
    return {"caption_id": cap.id}


class CaptionSetParams(BaseModel):
    caption_id: str
    start: float | None = Field(None, ge=0)
    end: float | None = Field(None, gt=0)
    text: str | None = Field(None, min_length=1, max_length=300)
    style: CaptionStyle | None = None
    position: CaptionPosition | None = None


@command("caption.set", CaptionSetParams, "Edit a caption's text, timing or style.", "text")
def caption_set(p: Project, a: CaptionSetParams) -> None:
    cap = next((c for c in p.captions if c.id == a.caption_id), None)
    if cap is None:
        raise CommandError(f"caption not found: {a.caption_id}")
    for key in ("start", "end", "text", "style"):
        value = getattr(a, key)
        if value is not None:
            setattr(cap, key, value)
    if "position" in a.model_fields_set:  # null = back to auto (the style's place, moved off faces on export)
        cap.position = a.position
        cap.auto_position = False
    if a.text is not None and cap.words and len(cap.words) != len(a.text.split()):
        cap.words = None  # word timings no longer match the edited text
    if cap.end <= cap.start:
        raise CommandError("end must be after start")
    p.captions.sort(key=lambda c: c.start)


class CaptionRemoveParams(BaseModel):
    caption_id: str


@command("caption.remove", CaptionRemoveParams, "Delete a caption.", "text")
def caption_remove(p: Project, a: CaptionRemoveParams) -> None:
    before = len(p.captions)
    p.captions = [c for c in p.captions if c.id != a.caption_id]
    if len(p.captions) == before:
        raise CommandError(f"caption not found: {a.caption_id}")


class CaptionStyleAllParams(BaseModel):
    style: CaptionStyle
    kind: Literal["caption", "title"] = "caption"


@command("caption.style_all", CaptionStyleAllParams, "Restyle every caption (or every title).", "text")
def caption_style_all(p: Project, a: CaptionStyleAllParams) -> dict:
    n = 0
    for c in p.captions:
        if c.kind == a.kind:
            c.style = a.style
            n += 1
    return {"changed": n}


class CaptionReplaceParams(BaseModel):
    captions: list[CaptionAddParams] = Field(max_length=5000)
    kind: Literal["caption", "title"] = "caption"


@command("caption.replace_all", CaptionReplaceParams, "Replace all captions of a kind (e.g. after auto-transcription).", "text")
def caption_replace_all(p: Project, a: CaptionReplaceParams) -> dict:
    keep = [c for c in p.captions if c.kind != a.kind]
    new = [Caption(id=new_id("cap"), **{**c.model_dump(), "kind": a.kind}) for c in a.captions if c.end > c.start]
    p.captions = sorted(keep + new, key=lambda c: c.start)
    return {"captions": len(new)}


# ------------------------------------------------------------- AI program


class VideoProgramParams(BaseModel):
    program: str = Field(description="Video Edit DSL (see video/vdsl.py)")
    mode: Literal["replace", "append"] = "replace"
    transcripts: dict[str, dict[str, Any]] | None = Field(None, description="media_id -> transcript, for `captions`")


@command("timeline.apply_program", VideoProgramParams, "Build/modify the timeline from a Video Edit DSL program (one undo step). Used by the AI.", "timeline")
def timeline_apply_program(p: Project, a: VideoProgramParams) -> dict:
    from ..video.vdsl import (  # local: avoids an import cycle
        VideoDslError,
        apply_video_program,
    )

    try:
        return apply_video_program(p, a.program, replace=a.mode == "replace", transcripts=a.transcripts)
    except VideoDslError as exc:
        raise CommandError(str(exc)) from exc


# ------------------------------------------------------------------ layers


class ClipLayerParams(BaseModel):
    clip_id: str
    x: float | None = Field(None, ge=-4, le=4, description="Offset right, fraction of canvas width")
    y: float | None = Field(None, ge=-4, le=4, description="Offset down, fraction of canvas height")
    scale: float | None = Field(None, gt=0, le=20)
    rotation: float | None = Field(None, ge=-3600, le=3600, description="Degrees, clockwise")
    anchor_x: float | None = Field(None, ge=-4, le=4, description="Pivot right of the canvas centre, fraction of width")
    anchor_y: float | None = Field(None, ge=-4, le=4, description="Pivot below the canvas centre, fraction of height")
    opacity: float | None = Field(None, ge=0, le=1)
    blend: BlendMode | None = None
    name: str | None = Field(None, max_length=80)
    color: str | None = Field(None, pattern=r"^#[0-9a-fA-F]{6}$", description="Solid layers")
    crop_left: float | None = Field(None, ge=0, le=0.95)
    crop_top: float | None = Field(None, ge=0, le=0.95)
    crop_right: float | None = Field(None, ge=0, le=0.95)
    crop_bottom: float | None = Field(None, ge=0, le=0.95)


@command("clip.set_layer", ClipLayerParams, "Place a clip as a layer: position, size, rotation, opacity, blend mode, crop.", "timeline")
def clip_set_layer(p: Project, a: ClipLayerParams) -> None:
    _, clip = p.find_clip(a.clip_id)
    for key in ("x", "y", "scale", "rotation", "anchor_x", "anchor_y"):
        value = getattr(a, key)
        if value is not None:
            setattr(clip.transform, key, value)
    for key in ("opacity", "blend", "name", "color"):
        value = getattr(a, key)
        if value is not None:
            setattr(clip, key, value)
    for side in ("left", "top", "right", "bottom"):
        value = getattr(a, f"crop_{side}")
        if value is not None:
            setattr(clip.crop, side, value)
    if clip.crop.left + clip.crop.right >= 0.98 or clip.crop.top + clip.crop.bottom >= 0.98:
        raise CommandError("crop removes the whole picture")


class KeyframesParams(BaseModel):
    clip_id: str
    param: str = Field(description="One of: " + ", ".join(KEYABLE))
    keys: list[Keyframe] = Field(description="Empty list removes the animation")


@command("clip.set_keyframes", KeyframesParams, "Animate a clip parameter over time (replaces its keyframes).", "timeline")
def clip_set_keyframes(p: Project, a: KeyframesParams) -> None:
    _, clip = p.find_clip(a.clip_id)
    if a.param not in KEYABLE:
        raise CommandError(f"cannot animate `{a.param}`; choose one of {', '.join(KEYABLE)}")
    if not a.keys:
        clip.keyframes.pop(a.param, None)
        return
    limits = {"opacity": (0.0, 1.0), "volume": (0.0, 4.0), "scale": (0.01, 20.0), "x": (-4.0, 4.0), "y": (-4.0, 4.0),
              "anchor_x": (-4.0, 4.0), "anchor_y": (-4.0, 4.0)}
    lo, hi = limits.get(a.param, (-3600.0, 3600.0))
    keys = sorted(a.keys, key=lambda k: k.t)
    for k in keys:
        if not lo <= k.value <= hi:
            raise CommandError(f"{a.param} must be between {lo:g} and {hi:g}")
        if k.ease == "bezier" and k.bezier is not None and not (0 <= k.bezier[0] <= 1 and 0 <= k.bezier[2] <= 1):
            raise CommandError("bezier handles need x1 and x2 between 0 and 1")
    clip.keyframes[a.param] = keys


class MasksParams(BaseModel):
    clip_id: str
    masks: list[Mask] = Field(description="The layer's masks, top to bottom (replaces them); empty removes all")


@command("clip.set_masks", MasksParams, "Cut a layer with bezier masks (add, subtract, intersect, feather, expand).", "timeline")
def clip_set_masks(p: Project, a: MasksParams) -> None:
    _, clip = p.find_clip(a.clip_id)
    if len(a.masks) > 32:
        raise CommandError("at most 32 masks per layer")
    clip.masks = [m.model_copy(deep=True) for m in a.masks]


class LayerAddParams(BaseModel):
    source: LayerSource = Field(description="solid (a colour), adjustment (effects on everything below) or media (photo documents: an image)")
    media_id: str = Field("", description="source=media: the image to place")
    doc: str | None = Field(None, description="Photo media_id: add the layer to that photo's layer stack instead of the timeline")
    start: float = Field(0.0, ge=0)
    duration: float = Field(5.0, gt=0, le=36000)
    color: str = Field("#000000", pattern=r"^#[0-9a-fA-F]{6}$")
    name: str = Field("", max_length=80)
    track_id: str | None = Field(None, description="Default: a new overlay track on top")


@command("layer.add", LayerAddParams, "Add a solid colour or adjustment layer on its own track (or a given one).", "timeline")
def layer_add(p: Project, a: LayerAddParams) -> dict:
    if a.doc is not None:
        doc = p.media.get(a.doc)
        if doc is None or doc.kind != MediaKind.IMAGE:
            raise CommandError(f"photo not found: {a.doc}")
    if a.source == "media":
        item = p.media.get(a.media_id)
        if a.doc is None:
            raise CommandError("use clip.add for media on the timeline")
        if item is None or item.kind != MediaKind.IMAGE:
            raise CommandError(f"image not found: {a.media_id}")
    if a.track_id:
        track = p.find_track(a.track_id)
        if track.kind == "audio":
            raise CommandError("layers go on a video or overlay track")
        if any(c.start < a.start + a.duration and a.start < c.end for c in track.clips):
            raise CommandError("overlaps another clip on that track")
    else:
        default = {"adjustment": "Adjustment", "solid": "Solid", "media": p.media[a.media_id].name if a.source == "media" else ""}
        track = Track(id=new_id("trk"), kind="overlay", name=(a.name or default[a.source])[:40])
        (p.layers.setdefault(a.doc, []) if a.doc is not None else p.tracks).append(track)
    if a.source == "media":
        # a placed image starts fitted inside the photo at half size
        clip = Clip(id=new_id("clp"), media_id=a.media_id, start=a.start, in_point=0.0, out_point=a.duration, fit="fit", name=a.name)
        clip.transform.scale = 0.5
    else:
        clip = Clip(id=new_id("clp"), source=a.source, start=a.start, in_point=0.0, out_point=a.duration,
                    color=a.color, name=a.name or ("Adjustment layer" if a.source == "adjustment" else "Solid"))
    track.clips.append(clip)
    track.clips.sort(key=lambda c: c.start)
    return {"clip_id": clip.id, "track_id": track.id}


class TrackSetParams(BaseModel):
    track_id: str
    name: str | None = Field(None, max_length=40)
    muted: bool | None = None
    hidden: bool | None = None
    locked: bool | None = None
    matte: MatteMode | Literal["none"] | None = Field(None, description="Cut this track by the track above it; none removes it")
    index: int | None = Field(None, ge=0, description="New position in the stack (0 = bottom)")


@command("track.set", TrackSetParams, "Rename, mute, hide, lock, matte or reorder a track.", "timeline")
def track_set(p: Project, a: TrackSetParams) -> None:
    track = p.find_track(a.track_id)
    for key in ("name", "muted", "hidden", "locked"):
        value = getattr(a, key)
        if value is not None:
            setattr(track, key, value)
    if a.matte is not None:
        track.matte = None if a.matte == "none" else a.matte
    if a.index is not None:
        stack = p.stack_of(track)
        stack.remove(track)
        stack.insert(min(a.index, len(stack)), track)


class TrackRemoveParams(BaseModel):
    track_id: str


@command("track.remove", TrackRemoveParams, "Delete a track and the clips on it.", "timeline")
def track_remove(p: Project, a: TrackRemoveParams) -> dict:
    track = p.find_track(a.track_id)
    p.stack_of(track).remove(track)
    return {"clips_removed": len(track.clips)}


class MarkerAddParams(BaseModel):
    t: float = Field(ge=0)
    name: str = Field("", max_length=80)
    color: str = Field("#f5c542", pattern=r"^#[0-9a-fA-F]{6}$")


@command("marker.add", MarkerAddParams, "Drop a named marker on the timeline.", "timeline")
def marker_add(p: Project, a: MarkerAddParams) -> dict:
    m = Marker(id=new_id("mrk"), t=a.t, name=a.name, color=a.color)
    p.markers.append(m)
    p.markers.sort(key=lambda x: x.t)
    return {"marker_id": m.id}


class MarkerRemoveParams(BaseModel):
    marker_id: str


@command("marker.remove", MarkerRemoveParams, "Delete a marker.", "timeline")
def marker_remove(p: Project, a: MarkerRemoveParams) -> None:
    before = len(p.markers)
    p.markers = [m for m in p.markers if m.id != a.marker_id]
    if len(p.markers) == before:
        raise CommandError("marker not found")
