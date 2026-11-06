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

SCHEMA_VERSION = 1


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
