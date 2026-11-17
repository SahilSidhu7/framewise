"""Built-in project, media and timeline commands (M0 set).

Later milestones add photo/fx/text/audio tools the same way: one params model
plus one function registered with @command.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .commands import CommandError, command
from .project import (
    Clip,
    MediaItem,
    MediaKind,
    Project,
    ProjectSettings,
    ProxyStatus,
    Track,
    new_id,
    SubjectPick,
)

# ------------------------------------------------------------------ project


class RenameParams(BaseModel):
    name: str = Field(min_length=1, max_length=200)


@command("project.rename", RenameParams, "Rename the project.", "project")
def project_rename(p: Project, a: RenameParams) -> None:
    p.name = a.name


class SettingsParams(BaseModel):
    width: int | None = Field(None, ge=16, le=16384)
    height: int | None = Field(None, ge=16, le=16384)
    fps: float | None = Field(None, gt=0, le=240)
    letterbox: float | None = Field(None, ge=1.0, le=3.0, description="null removes the bars")
    duck_music: bool | None = None
    normalize_audio: bool | None = None
    fade_in: float | None = Field(None, ge=0, le=10)
    fade_out: float | None = Field(None, ge=0, le=10)


@command("project.set_settings", SettingsParams, "Set output size, frame rate, cinematic bars, fades and audio options.", "project")
def project_set_settings(p: Project, a: SettingsParams) -> None:
    changes = a.model_dump(exclude_unset=True)
    for key in ("width", "height", "fps", "duck_music", "normalize_audio", "fade_in", "fade_out"):
        if changes.get(key, 0) is None:
            changes.pop(key)  # only letterbox may be cleared with null
    p.settings = ProjectSettings(**{**p.settings.model_dump(), **changes})


# -------------------------------------------------------------------- media


class MediaAddParams(BaseModel):
    """Register already-probed media. The API's import endpoint probes files
    and then issues this command; the AI never invents media metadata."""

    name: str
    path: str
    kind: MediaKind
    width: int | None = None
    height: int | None = None
    duration: float | None = Field(None, ge=0)
    fps: float | None = None
    has_audio: bool = False
    thumb_path: str | None = None
    id: str | None = None


@command("media.add", MediaAddParams, "Add probed media to the project bin.", "media")
def media_add(p: Project, a: MediaAddParams) -> dict:
    media_id = a.id or new_id("med")
    if media_id in p.media:
        raise CommandError(f"media id already exists: {media_id}")
    data = a.model_dump(exclude={"id"})
    status = ProxyStatus.PENDING if a.kind == MediaKind.VIDEO else ProxyStatus.NONE
    p.media[media_id] = MediaItem(id=media_id, proxy_status=status, **data)
    return {"media_id": media_id}


class MediaRemoveParams(BaseModel):
    media_id: str


@command("media.remove", MediaRemoveParams, "Remove media and every clip that uses it.", "media")
def media_remove(p: Project, a: MediaRemoveParams) -> dict:
    if a.media_id not in p.media:
        raise CommandError(f"media not found: {a.media_id}")
    del p.media[a.media_id]
    p.edits.pop(a.media_id, None)
    removed = 0
    for track in p.tracks:
        before = len(track.clips)
        track.clips = [c for c in track.clips if c.media_id != a.media_id]
        removed += before - len(track.clips)
    return {"clips_removed": removed}


class MediaSubjectParams(BaseModel):
    media_id: str
    x: float | None = Field(None, ge=0, le=1, description="Subject position (0..1 of width); omit both x and y for automatic")
    y: float | None = Field(None, ge=0, le=1)
    t: float = Field(0.0, ge=0, description="Video: the time the click was made")


@command("media.set_subject", MediaSubjectParams, "Choose the subject of a video or photo by its position, or go back to automatic.", "media")
def media_set_subject(p: Project, a: MediaSubjectParams) -> dict:
    item = p.media.get(a.media_id)
    if item is None:
        raise CommandError(f"media not found: {a.media_id}")
    if (a.x is None) != (a.y is None):
        raise CommandError("give both x and y, or neither for automatic")
    item.subject = None if a.x is None else SubjectPick(x=a.x, y=a.y, t=a.t)  # type: ignore[arg-type]
    return {"subject": item.subject.model_dump() if item.subject else None}


# ----------------------------------------------------------------- timeline


class TrackAddParams(BaseModel):
    kind: Literal["video", "audio", "overlay"] = "video"
    name: str = ""


@command("track.add", TrackAddParams, "Add a timeline track.", "timeline")
def track_add(p: Project, a: TrackAddParams) -> dict:
    track = Track(id=new_id("trk"), kind=a.kind, name=a.name or f"{a.kind.title()} {len(p.tracks) + 1}")
    p.tracks.append(track)
    return {"track_id": track.id}


class ClipAddParams(BaseModel):
    track_id: str
    media_id: str
    start: float | None = Field(None, ge=0, description="Timeline position; default = end of track")
    in_point: float = Field(0.0, ge=0)
    out_point: float | None = Field(None, gt=0, description="Default = media duration")


DEFAULT_STILL_DURATION = 5.0


@command("clip.add", ClipAddParams, "Place media on a track.", "timeline")
def clip_add(p: Project, a: ClipAddParams) -> dict:
    track = p.find_track(a.track_id)
    media = p.media.get(a.media_id)
    if media is None:
        raise CommandError(f"media not found: {a.media_id}")
    source_len = media.duration if media.duration else DEFAULT_STILL_DURATION
    out_point = a.out_point if a.out_point is not None else source_len
    if media.kind != MediaKind.IMAGE and out_point > source_len + 1e-6:
        raise CommandError(f"out_point {out_point} beyond media duration {source_len}")
    if out_point <= a.in_point:
        raise CommandError("out_point must be after in_point")
    start = a.start if a.start is not None else _track_end(track)
    clip = Clip(id=new_id("clp"), media_id=media.id, start=start, in_point=a.in_point, out_point=out_point)
    track.clips.append(clip)
    track.clips.sort(key=lambda c: c.start)
    return {"clip_id": clip.id}


class ClipTrimParams(BaseModel):
    clip_id: str
    in_point: float | None = Field(None, ge=0)
    out_point: float | None = Field(None, gt=0)
    start: float | None = Field(None, ge=0, description="New timeline position (trimming the head keeps the tail in place)")


@command("clip.trim", ClipTrimParams, "Change a clip's source in/out points.", "timeline")
def clip_trim(p: Project, a: ClipTrimParams) -> None:
    _, clip = p.find_clip(a.clip_id)
    media = p.media[clip.media_id]
    new_in = a.in_point if a.in_point is not None else clip.in_point
    new_out = a.out_point if a.out_point is not None else clip.out_point
    if new_out <= new_in:
        raise CommandError("out_point must be after in_point")
    if media.kind != MediaKind.IMAGE and media.duration and new_out > media.duration + 1e-6:
        raise CommandError(f"out_point {new_out} beyond media duration {media.duration}")
    clip.in_point, clip.out_point = new_in, new_out
    if a.start is not None:
        clip.start = a.start
        track, _ = p.find_clip(a.clip_id)
        track.clips.sort(key=lambda c: c.start)


class ClipMoveParams(BaseModel):
    clip_id: str
    start: float = Field(ge=0)
    track_id: str | None = None


@command("clip.move", ClipMoveParams, "Move a clip in time and optionally to another track.", "timeline")
def clip_move(p: Project, a: ClipMoveParams) -> None:
    track, clip = p.find_clip(a.clip_id)
    clip.start = a.start
    if a.track_id and a.track_id != track.id:
        target = p.find_track(a.track_id)
        track.clips.remove(clip)
        target.clips.append(clip)
        track = target
    track.clips.sort(key=lambda c: c.start)


class ClipRemoveParams(BaseModel):
    clip_id: str


@command("clip.remove", ClipRemoveParams, "Delete a clip from the timeline.", "timeline")
def clip_remove(p: Project, a: ClipRemoveParams) -> None:
    track, clip = p.find_clip(a.clip_id)
    track.clips.remove(clip)


def _track_end(track: Track) -> float:
    return max((c.start + c.duration for c in track.clips), default=0.0)
