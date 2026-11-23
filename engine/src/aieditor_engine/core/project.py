"""Declarative project document.

The project is the single source of truth. Commands mutate it; the renderer
reads it. Everything here must stay JSON-serializable so it can be saved,
diffed, logged as training data and replayed.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 3  # v2: layered clips (transform, opacity, blend, crop, keyframes), track flags, markers
# v3: Photoshop blend modes, masks, track mattes, anchor point, bezier easing, solid and adjustment layers


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ProjectKind(StrEnum):
    PHOTO = "photo"
    VIDEO = "video"
    COLLAGE = "collage"


class MediaKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ProxyStatus(StrEnum):
    NONE = "none"  # not needed (images, audio)
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class SubjectPick(BaseModel):
    """The subject a person clicked on: position (0..1 of width/height) and, for video, the time."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    t: float = Field(0.0, ge=0)


class MediaItem(BaseModel):
    id: str
    name: str
    path: str  # absolute path of the original; originals are never modified
    kind: MediaKind
    width: int | None = None
    height: int | None = None
    duration: float | None = None  # seconds
    fps: float | None = None
    has_audio: bool = False
    thumb_path: str | None = None  # relative to project dir
    proxy_path: str | None = None  # relative to project dir
    proxy_status: ProxyStatus = ProxyStatus.NONE
    subject: SubjectPick | None = Field(None, description="Chosen subject (video: followed through the clip); None = automatic")


class EditOp(BaseModel):
    """One non-destructive operation in a photo's edit stack."""

    id: str
    op: str
    params: dict = Field(default_factory=dict)
    enabled: bool = True


TransitionKind = Literal["crossfade", "fade_black", "fade_white", "wipe_left", "wipe_right", "slide_left", "slide_up", "zoom", "blur"]
Motion = Literal["none", "zoom_in", "zoom_out", "pan_left", "pan_right"]


class Transition(BaseModel):
    """Transition INTO a clip from the previous clip on the same track. It is
    centred on the cut and does not change the timeline layout: missing source
    frames on either side are held (freeze frame)."""

    kind: TransitionKind = "crossfade"
    duration: float = Field(0.5, gt=0, le=5)


# The Photoshop / After Effects set. add = linear dodge.
BlendMode = Literal[
    "normal",
    "darken", "multiply", "color_burn", "linear_burn", "darker_color",
    "lighten", "screen", "color_dodge", "add", "lighter_color",
    "overlay", "soft_light", "hard_light", "vivid_light", "linear_light", "pin_light", "hard_mix",
    "difference", "exclusion", "subtract", "divide",
    "hue", "saturation", "color", "luminosity",
]
BLEND_MODES: tuple[str, ...] = BlendMode.__args__  # type: ignore[attr-defined]
Easing = Literal["linear", "ease", "hold", "bezier"]
# clip parameters that can be animated with keyframes
KEYABLE = ("x", "y", "scale", "rotation", "opacity", "volume", "anchor_x", "anchor_y")
MatteMode = Literal["alpha", "alpha_inverted", "luma", "luma_inverted"]
MaskMode = Literal["add", "subtract", "intersect", "difference"]
LayerSource = Literal["media", "solid", "adjustment", "text", "shape"]
FontFamily = Literal["sans", "display", "serif"]


class TextStyle(BaseModel):
    """A text layer: lines of text centred on the canvas (move it with the transform)."""

    text: str = Field("Text", min_length=1, max_length=2000)
    font: FontFamily = "sans"
    weight: int = Field(700, ge=100, le=900)
    size: float = Field(0.08, gt=0, le=1, description="Font size, fraction of the canvas short side")
    color: str = Field("#ffffff", pattern=r"^#[0-9a-fA-F]{6}$")
    align: Literal["left", "center", "right"] = "center"
    line_height: float = Field(1.2, ge=0.5, le=4)
    stroke_color: str = Field("#000000", pattern=r"^#[0-9a-fA-F]{6}$")
    stroke_width: float = Field(0.0, ge=0, le=0.3, description="Outline, fraction of the font size")


class ShapeStyle(BaseModel):
    """A shape layer: a rectangle (rounded corners) or ellipse centred on the canvas."""

    kind: Literal["rectangle", "ellipse"] = "rectangle"
    width: float = Field(0.4, gt=0, le=4, description="Fraction of the canvas width")
    height: float = Field(0.3, gt=0, le=4, description="Fraction of the canvas height")
    radius: float = Field(0.0, ge=0, le=1, description="Rectangle corner radius, fraction of the canvas short side")
    fill: str | None = Field("#ffffff", pattern=r"^#[0-9a-fA-F]{6}$", description="None = no fill")
    stroke_color: str = Field("#000000", pattern=r"^#[0-9a-fA-F]{6}$")
    stroke_width: float = Field(0.0, ge=0, le=0.5, description="Outline inside the edge, fraction of the canvas short side")


class Transform(BaseModel):
    """Placement of a layer on the canvas, applied after fit/fill and motion."""

    x: float = Field(0.0, ge=-4, le=4, description="Offset right, as a fraction of canvas width")
    y: float = Field(0.0, ge=-4, le=4, description="Offset down, as a fraction of canvas height")
    scale: float = Field(1.0, gt=0, le=20)
    rotation: float = Field(0.0, ge=-3600, le=3600, description="Degrees, clockwise")
    anchor_x: float = Field(0.0, ge=-4, le=4, description="Pivot for scale and rotation, right of the canvas centre (fraction of width)")
    anchor_y: float = Field(0.0, ge=-4, le=4, description="Pivot for scale and rotation, below the canvas centre (fraction of height)")

    @property
    def is_identity(self) -> bool:
        return self.x == 0 and self.y == 0 and self.scale == 1 and self.rotation == 0


class MaskPoint(BaseModel):
    """One vertex of a mask path, in layer space: 0..1 of the layer's source
    width/height (a solid or adjustment layer's source is the canvas). The
    tangents are offsets from the vertex, also in layer space; zero = a corner."""

    x: float = Field(ge=-2, le=3)
    y: float = Field(ge=-2, le=3)
    in_x: float = Field(0.0, ge=-3, le=3)
    in_y: float = Field(0.0, ge=-3, le=3)
    out_x: float = Field(0.0, ge=-3, le=3)
    out_y: float = Field(0.0, ge=-3, le=3)


class Mask(BaseModel):
    """A closed bezier path that limits where a layer shows (After Effects style)."""

    id: str = Field(default_factory=lambda: new_id("msk"))
    name: str = Field("", max_length=60)
    points: list[MaskPoint] = Field(min_length=3, max_length=2000)
    mode: MaskMode = "add"
    inverted: bool = False
    opacity: float = Field(1.0, ge=0, le=1)
    feather: float = Field(0.0, ge=0, le=0.5, description="Soft edge width, fraction of the canvas's short side")
    expansion: float = Field(0.0, ge=-0.5, le=0.5, description="Grow (+) or shrink (-) the shape, fraction of the short side")
    enabled: bool = True

    @staticmethod
    def rect(left: float, top: float, right: float, bottom: float, **kw) -> Mask:
        pts = [MaskPoint(x=left, y=top), MaskPoint(x=right, y=top), MaskPoint(x=right, y=bottom), MaskPoint(x=left, y=bottom)]
        return Mask(points=pts, **kw)

    @staticmethod
    def ellipse(cx: float, cy: float, rx: float, ry: float, **kw) -> Mask:
        k = 0.5522847498  # four cubic arcs approximate a circle
        pts = [
            MaskPoint(x=cx, y=cy - ry, in_x=-rx * k, out_x=rx * k),
            MaskPoint(x=cx + rx, y=cy, in_y=-ry * k, out_y=ry * k),
            MaskPoint(x=cx, y=cy + ry, in_x=rx * k, out_x=-rx * k),
            MaskPoint(x=cx - rx, y=cy, in_y=ry * k, out_y=-ry * k),
        ]
        return Mask(points=pts, **kw)


class Crop(BaseModel):
    """Fractions of the source trimmed from each edge (the rest stays in place)."""

    left: float = Field(0.0, ge=0, le=0.95)
    top: float = Field(0.0, ge=0, le=0.95)
    right: float = Field(0.0, ge=0, le=0.95)
    bottom: float = Field(0.0, ge=0, le=0.95)

    @property
    def is_empty(self) -> bool:
        return not (self.left or self.top or self.right or self.bottom)


class Keyframe(BaseModel):
    t: float = Field(ge=0, description="Seconds after the clip's start on the timeline")
    value: float
    ease: Easing = Field("ease", description="How the value moves from this keyframe to the next")
    # ease="bezier": the curve to the next key as CSS cubic-bezier(x1, y1, x2, y2), time and value 0..1
    bezier: tuple[float, float, float, float] | None = Field(None, description="x1 y1 x2 y2; x in 0..1")


def cubic_bezier(x1: float, y1: float, x2: float, y2: float, p: float) -> float:
    """CSS cubic-bezier: progress in time (0..1) -> progress in value."""
    if p <= 0 or p >= 1:
        return min(1.0, max(0.0, p))
    x1, x2 = min(1.0, max(0.0, x1)), min(1.0, max(0.0, x2))

    def bx(s: float) -> float:
        return 3 * x1 * s * (1 - s) ** 2 + 3 * x2 * s * s * (1 - s) + s**3

    lo, hi, s = 0.0, 1.0, p
    for _ in range(50):  # bisection: x(s) is monotonic for x1, x2 in 0..1
        x = bx(s) - p
        if abs(x) < 1e-7:
            break
        if x > 0:
            hi = s
        else:
            lo = s
        s = (lo + hi) / 2
    return 3 * y1 * s * (1 - s) ** 2 + 3 * y2 * s * s * (1 - s) + s**3


def keyframe_value(keys: list[Keyframe], t: float) -> float:
    """Value of an animated parameter t seconds into the clip (keys sorted by t)."""
    if t <= keys[0].t:
        return keys[0].value
    for a, b in zip(keys, keys[1:], strict=False):
        if t < b.t:
            if a.ease == "hold" or b.t - a.t < 1e-9:
                return a.value
            p = (t - a.t) / (b.t - a.t)
            if a.ease == "ease":
                p = p * p * (3 - 2 * p)
            elif a.ease == "bezier":
                p = cubic_bezier(*(a.bezier or (0.42, 0.0, 0.58, 1.0)), p)
            return a.value + (b.value - a.value) * p
    return keys[-1].value


class Clip(BaseModel):
    """One layer on a track: media (video/photo/audio), a solid colour, or an
    adjustment layer whose effects apply to everything below it."""

    id: str
    media_id: str = Field("", description="Empty for solid and adjustment layers")
    source: LayerSource = "media"
    color: str = Field("#000000", pattern=r"^#[0-9a-fA-F]{6}$", description="Solid layers: the colour")
    start: float = Field(ge=0, description="Timeline position in seconds")
    in_point: float = Field(ge=0, description="Source in point in seconds")
    out_point: float = Field(gt=0, description="Source out point in seconds")
    speed: float = Field(1.0, ge=0.1, le=10, description="Playback speed; 0.5 = slow motion")
    volume: float = Field(1.0, ge=0, le=4, description="Linear gain")
    muted: bool = False
    fade_in: float = Field(0.0, ge=0, le=30, description="Seconds (picture from black / audio from silence)")
    fade_out: float = Field(0.0, ge=0, le=30)
    fit: Literal["fill", "fit"] = "fill"
    motion: Motion = "none"
    transition: Transition | None = None
    ops: list[EditOp] = Field(default_factory=list, description="Per-clip color/effect stack")
    # layering (schema v2)
    transform: Transform = Field(default_factory=Transform)
    opacity: float = Field(1.0, ge=0, le=1)
    blend: BlendMode = "normal"
    crop: Crop = Field(default_factory=Crop)
    keyframes: dict[str, list[Keyframe]] = Field(default_factory=dict, description="Animated params: " + ", ".join(KEYABLE))
    link_id: str | None = Field(None, description="Clips sharing a link_id move and trim together (picture + its sound)")
    # schema v3
    masks: list[Mask] = Field(default_factory=list, description="Combined top to bottom; none = the whole layer shows")
    name: str = Field("", max_length=80)
    text: TextStyle | None = Field(None, description="Text layers")
    shape: ShapeStyle | None = Field(None, description="Shape layers")

    @property
    def vector(self) -> bool:
        """A text or shape layer (drawn by the engine, with its own transparency)."""
        return self.source in ("text", "shape")

    @property
    def generated(self) -> bool:
        """A solid, adjustment, text or shape layer (no media behind it)."""
        return self.source != "media"

    def value(self, name: str, t: float) -> float:
        """Animated value of a KEYABLE parameter at timeline time t."""
        keys = self.keyframes.get(name)
        if keys:
            return keyframe_value(keys, t - self.start)
        if name == "opacity":
            return self.opacity
        if name == "volume":
            return self.volume
        return float(getattr(self.transform, name))

    @property
    def is_layered(self) -> bool:
        """True when the clip may not cover the whole canvas opaquely."""
        return bool(
            not self.transform.is_identity or self.opacity < 1 or self.blend != "normal" or self.generated
            or not self.crop.is_empty or any(k in self.keyframes for k in ("x", "y", "scale", "rotation", "opacity"))
            or any(m.enabled for m in self.masks)
        )

    @property
    def source_duration(self) -> float:
        return self.out_point - self.in_point

    @property
    def duration(self) -> float:
        """Length on the timeline (speed applied)."""
        return (self.out_point - self.in_point) / self.speed

    @property
    def end(self) -> float:
        return self.start + self.duration

    def source_time(self, t: float) -> float:
        """Timeline time -> source time (clamped to the clip's source range)."""
        src = self.in_point + (t - self.start) * self.speed
        return min(max(src, self.in_point), self.out_point)


class Track(BaseModel):
    id: str
    kind: Literal["video", "audio", "overlay"] = "video"
    name: str = ""
    clips: list[Clip] = Field(default_factory=list)
    muted: bool = Field(False, description="Silence every clip on the track")
    hidden: bool = Field(False, description="Leave the track out of the picture")
    locked: bool = Field(False, description="Protect the track from edits")
    matte: MatteMode | None = Field(None, description="Use the track directly above as this track's matte (that track is then not drawn)")


class Marker(BaseModel):
    id: str
    t: float = Field(ge=0)
    name: str = Field("", max_length=80)
    color: str = Field("#f5c542", pattern=r"^#[0-9a-fA-F]{6}$")


CaptionStyle = Literal["clean", "bold_pop", "cinematic", "boxed", "title", "minimal"]
CaptionPosition = Literal["bottom", "center", "top", "lower_third", "center_low", "high"]


class Caption(BaseModel):
    """Timed text on top of the video (subtitles, titles)."""

    id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1, max_length=300)
    style: CaptionStyle = "clean"
    position: CaptionPosition | None = None  # None = the style's default
    kind: Literal["caption", "title"] = "caption"
    words: list[float] | None = Field(None, description="Start of each word, seconds after `start` (word-by-word highlight)")
    auto_position: bool = Field(False, description="position was chosen by the engine to avoid faces / on-screen text")


class ProjectSettings(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    letterbox: float | None = Field(None, description="Cinematic bars, e.g. 2.39 (aspect of the visible area)")
    duck_music: bool = True
    normalize_audio: bool = True
    fade_in: float = Field(0.0, ge=0, le=10, description="Whole-video fade from black")
    fade_out: float = Field(0.0, ge=0, le=10)
    progress_bar: Literal["off", "top", "bottom"] = "off"
    end_card: str | None = Field(None, max_length=120, description="Text on a closing card after the last clip")
    end_card_seconds: float = Field(1.5, ge=0.5, le=10)


class Project(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("prj"))
    name: str
    kind: ProjectKind = ProjectKind.VIDEO
    created_at: str = Field(default_factory=utc_now)
    settings: ProjectSettings = Field(default_factory=ProjectSettings)
    media: dict[str, MediaItem] = Field(default_factory=dict)
    tracks: list[Track] = Field(default_factory=list)
    edits: dict[str, list[EditOp]] = Field(default_factory=dict, description="media_id -> photo edit stack")
    grade: list[EditOp] = Field(default_factory=list, description="Video: look/color stack over the whole timeline")
    layers: dict[str, list[Track]] = Field(
        default_factory=dict, description="Photo: media_id -> layer tracks drawn over the edited photo (bottom first)"
    )
    captions: list[Caption] = Field(default_factory=list)
    markers: list[Marker] = Field(default_factory=list)

    @property
    def content_end(self) -> float:
        """End of the clips and captions (without the end card)."""
        ends = [c.end for t in self.tracks for c in t.clips]
        ends += [c.end for c in self.captions]
        return max(ends, default=0.0)

    @property
    def duration(self) -> float:
        end = self.content_end
        return end + self.settings.end_card_seconds if self.settings.end_card and end > 0 else end

    def all_tracks(self) -> list[Track]:
        """Timeline tracks, then every photo's layer tracks."""
        return [*self.tracks, *(t for stack in self.layers.values() for t in stack)]

    def stack_of(self, track: Track) -> list[Track]:
        """The list a track lives in: the timeline or one photo's layers."""
        if any(t is track for t in self.tracks):
            return self.tracks
        for stack in self.layers.values():
            if any(t is track for t in stack):
                return stack
        raise KeyError(f"track not found: {track.id}")

    def find_track(self, track_id: str) -> Track:
        for track in self.all_tracks():
            if track.id == track_id:
                return track
        raise KeyError(f"track not found: {track_id}")

    def find_clip(self, clip_id: str) -> tuple[Track, Clip]:
        for track in self.all_tracks():
            for clip in track.clips:
                if clip.id == clip_id:
                    return track, clip
        raise KeyError(f"clip not found: {clip_id}")
