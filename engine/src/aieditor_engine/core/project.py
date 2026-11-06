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


class Clip(BaseModel):
    id: str
    media_id: str
    start: float = Field(ge=0, description="Timeline position in seconds")
    in_point: float = Field(ge=0, description="Source in point in seconds")
    out_point: float = Field(gt=0, description="Source out point in seconds")

    @property
    def duration(self) -> float:
        return self.out_point - self.in_point


class Track(BaseModel):
    id: str
    kind: Literal["video", "audio", "overlay"] = "video"
    name: str = ""
    clips: list[Clip] = Field(default_factory=list)


class ProjectSettings(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: float = 30.0


class Project(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: new_id("prj"))
    name: str
    kind: ProjectKind = ProjectKind.VIDEO
    created_at: str = Field(default_factory=utc_now)
    settings: ProjectSettings = Field(default_factory=ProjectSettings)
    media: dict[str, MediaItem] = Field(default_factory=dict)
    tracks: list[Track] = Field(default_factory=list)

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
