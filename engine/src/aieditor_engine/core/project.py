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

SCHEMA_VERSION = 2  # v2: layered clips (transform, opacity, blend, crop, keyframes), track flags, markers


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


BlendMode = Literal["normal", "add", "screen", "multiply", "overlay"]
Easing = Literal["linear", "ease", "hold"]
# clip parameters that can be animated with keyframes
KEYABLE = ("x", "y", "scale", "rotation", "opacity", "volume")


class Transform(BaseModel):
    """Placement of a layer on the canvas, applied after fit/fill and motion."""

    x: float = Field(0.0, ge=-4, le=4, description="Offset right, as a fraction of canvas width")
    y: float = Field(0.0, ge=-4, le=4, description="Offset down, as a fraction of canvas height")
    scale: float = Field(1.0, gt=0, le=20)
    rotation: float = Field(0.0, ge=-3600, le=3600, description="Degrees, clockwise")

    @property
    def is_identity(self) -> bool:
        return self.x == 0 and self.y == 0 and self.scale == 1 and self.rotation == 0


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
            return a.value + (b.value - a.value) * p
    return keys[-1].value


class Clip(BaseModel):
    id: str
    media_id: str
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
            not self.transform.is_identity or self.opacity < 1 or self.blend != "normal"
            or not self.crop.is_empty or any(k in self.keyframes for k in ("x", "y", "scale", "rotation", "opacity"))
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


class Marker(BaseModel):
    id: str
    t: float = Field(ge=0)
    name: str = Field("", max_length=80)
    color: str = Field("#f5c542", pattern=r"^#[0-9a-fA-F]{6}$")


CaptionStyle = Literal["clean", "bold_pop", "cinematic", "boxed", "title", "minimal"]
CaptionPosition = Literal["bottom", "center", "top", "lower_third"]


class Caption(BaseModel):
    """Timed text on top of the video (subtitles, titles)."""

    id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1, max_length=300)
    style: CaptionStyle = "clean"
    position: CaptionPosition | None = None  # None = the style's default
    kind: Literal["caption", "title"] = "caption"


class ProjectSettings(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    letterbox: float | None = Field(None, description="Cinematic bars, e.g. 2.39 (aspect of the visible area)")
    duck_music: bool = True
    normalize_audio: bool = True
    fade_in: float = Field(0.0, ge=0, le=10, description="Whole-video fade from black")
    fade_out: float = Field(0.0, ge=0, le=10)


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
    captions: list[Caption] = Field(default_factory=list)
    markers: list[Marker] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        ends = [c.end for t in self.tracks for c in t.clips]
        ends += [c.end for c in self.captions]
        return max(ends, default=0.0)

    def find_track(self, track_id: str) -> Track:
        for track in self.tracks:
            if track.id == track_id:
                return track
        raise KeyError(f"track not found: {track_id}")

    def find_clip(self, clip_id: str) -> tuple[Track, Clip]:
        for track in self.tracks:
            for clip in track.clips:
                if clip.id == clip_id:
                    return track, clip
        raise KeyError(f"clip not found: {clip_id}")
