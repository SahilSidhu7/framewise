"""Video endpoints: scrub frames, preview renders, export, captions and the AI director."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..agent.options import OptionError, OutputOptions, catalog, save_style, style_from_reference
from ..agent.reels import make_reels
from ..agent.whisper import WhisperUnavailable, whisper_status
from ..core.commands import CommandError
from ..core.project import EditOp, MediaKind, Project
from ..dsl import DslError, compile_program
from ..video.captions import captions_from_transcripts
from ..video.compose import RenderSource, output_size, render_frame
from ..video.export import ExportSpec, export_video
from ..video.text import STYLE_INFO
from ..video.vdsl import VideoDslError, apply_video_program, timeline_to_dsl

if TYPE_CHECKING:
    from ..session import Session

PRESETS: dict[str, tuple[int, int, float | None]] = {
    "project": (0, 0, None),
    "youtube_1080p": (1920, 1080, None),
    "youtube_4k": (3840, 2160, None),
    "reel_1080": (1080, 1920, 30.0),
    "square_1080": (1080, 1080, 30.0),
    "portrait_4_5": (1080, 1350, 30.0),
    "hd_720p": (1280, 720, None),
}


class PreviewBody(BaseModel):
    max_side: int = Field(640, ge=160, le=1280)


class VideoExportBody(BaseModel):
    path: str
    preset: str = "project"
    quality: Literal["high", "standard", "small"] = "high"
    fps: float | None = Field(None, gt=0, le=120)


class ReelsBody(BaseModel):
    media_number: int | None = Field(None, ge=1, description="Which video in the bin (default: the longest one with speech)")
    count: int = Field(3, ge=1, le=10)
    length: float | None = Field(None, ge=5, le=180)
    style: str | None = None
    output: OutputOptions | None = None
    captions: bool | None = None
    tighten: bool = Field(True, description="Cut silences and filler words")
    progress_bar: bool = Field(True, description="Thin progress bar at the top")
    end_card: str | None = Field(None, max_length=120, description="Closing card text")


class StyleFromVideoBody(BaseModel):
    media_number: int = Field(ge=1, description="The reference video in the bin")
    name: str = Field("My style", min_length=1, max_length=40)


class AiApplyBody(BaseModel):
    program: str = Field(min_length=1, max_length=20000)
    mode: Literal["replace", "append"] = "replace"


class CaptionsBody(BaseModel):
    style: str = "clean"


class _FrameCache:
    def __init__(self, size: int = 240) -> None:
        self._items: OrderedDict[str, bytes] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            data = self._items.get(key)
            if data is not None:
                self._items.move_to_end(key)
            return data

    def put(self, key: str, data: bytes) -> None:
        with self._lock:
            self._items[key] = data
            while len(self._items) > self._size:
                self._items.popitem(last=False)


FRAMES = _FrameCache()


def _with_program(session: Session, program: str | None) -> tuple[Project, int]:
    with session.bus.lock:
        project = session.bus.project.model_copy(deep=True)
        rev = session.bus.revision
    if program:
        try:
            apply_video_program(project, program, transcripts=session.transcripts_for_timeline(project))
        except VideoDslError as exc:
            raise HTTPException(422, str(exc)) from exc
    return project, rev


def _jpeg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise HTTPException(500, "encode failed")
    return buf.tobytes()


def register(app: FastAPI, require, secured: list) -> None:  # noqa: ANN001 - FastAPI deps
    @app.get("/timeline/frame", dependencies=secured)
    def timeline_frame(
        t: float = Query(0.0, ge=0),
        max: int = Query(960, ge=64, le=3840),
        program: str | None = Query(None, max_length=20000),
        grade: str | None = Query(None, max_length=4000),
    ) -> Response:
        """One finished frame (from proxies). program=<Video DSL> previews an unapplied edit;
        grade=<Edit DSL> previews a different look for the whole video."""
        session: Session = require()
        project, rev = _with_program(session, program)
        if grade is not None:
            try:
                project.grade = [EditOp(id=f"g{i}", op=o["op"], params=o["params"]) for i, o in enumerate(compile_program(grade))]
            except DslError as exc:
                raise HTTPException(422, str(exc)) from exc
            program = (program or "") + "|grade|" + grade
        fps = project.settings.fps or 30
        t = round(t * fps) / fps
        w, h = output_size(project, max)
        key = hashlib.sha1(f"{rev}|{t:.4f}|{w}x{h}|{program or ''}".encode()).hexdigest()
        data = FRAMES.get(key)
        if data is None:
            try:
                frame = render_frame(project, t, w, h, RenderSource(session.paths.root, use_proxy=True, channel="scrub"))
            except (ValueError, OSError) as exc:
                raise HTTPException(422, str(exc)) from exc
            data = _jpeg(frame)
            FRAMES.put(key, data)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/timeline/dsl", dependencies=secured)
    def timeline_dsl() -> dict[str, Any]:
        session: Session = require()
        with session.bus.lock:
            project = session.bus.project.model_copy(deep=True)
        return {"program": timeline_to_dsl(project), "duration": project.duration}

    @app.post("/timeline/preview", dependencies=secured)
    def timeline_preview(body: PreviewBody) -> dict[str, Any]:
        """Render a small playable preview of the whole timeline in the background."""
        session: Session = require()
        project, rev = _with_program(session, None)
        if project.duration <= 0:
            raise HTTPException(422, "the timeline is empty")
        w, h = output_size(project, body.max_side)
        out = session.paths.root / "cache" / "preview" / f"timeline-{rev}-{w}x{h}.mp4"
        if out.is_file():
            return {"job": None, "revision": rev, "url": f"/timeline/preview/{out.name}"}
        spec = ExportSpec(out, w, h, min(project.settings.fps, 30.0), "small", use_proxy=True)

        def job(progress, cancelled) -> dict[str, Any]:
            res = export_video(project, session.paths.root, spec, progress, cancelled)
            _prune_previews(out.parent, keep=out.name)
            return {**res, "revision": rev, "url": f"/timeline/preview/{out.name}"}

        return {"job": session.start_job("preview", job, label="Preview"), "revision": rev, "url": None}

    @app.get("/timeline/preview/{name}", dependencies=secured)
    def timeline_preview_file(name: str) -> FileResponse:
        session: Session = require()
        try:
            path = session.paths.resolve(f"cache/preview/{name}")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not path.is_file() or path.suffix != ".mp4":
            raise HTTPException(404, "not available")
        return FileResponse(path, media_type="video/mp4")

    @app.post("/timeline/export", dependencies=secured)
    def timeline_export(body: VideoExportBody) -> dict[str, Any]:
        session: Session = require()
        project, _ = _with_program(session, None)
        if project.duration <= 0:
            raise HTTPException(422, "the timeline is empty")
        if body.preset not in PRESETS:
            raise HTTPException(422, f"unknown preset; use one of {', '.join(PRESETS)}")
        pw, ph, pfps = PRESETS[body.preset]
        w, h = (pw, ph) if pw else (project.settings.width, project.settings.height)
        fps = body.fps or pfps or project.settings.fps
        if pw:  # a preset with another shape reframes the project (fill)
            project.settings.width, project.settings.height = w, h
        out = Path(body.path).expanduser()
        if out.suffix.lower() != ".mp4":
            out = out.with_suffix(".mp4")
        spec = ExportSpec(out, w, h, fps, body.quality, use_proxy=False)

        def job(progress, cancelled) -> dict[str, Any]:
            return export_video(project, session.paths.root, spec, progress, cancelled)

        return {"job": session.start_job("export", job, label=out.name), "path": str(out)}

    @app.get("/jobs", dependencies=secured)
    def jobs() -> list[dict[str, Any]]:
        return list(require().jobs.values())

    @app.post("/jobs/{job_id}/cancel", dependencies=secured)
    def cancel_job(job_id: str) -> dict[str, bool]:
        return {"ok": require().cancel_job(job_id)}

    # ------------------------------------------------------------ captions
    @app.get("/captions/styles", dependencies=secured)
    def caption_styles() -> list[dict[str, str]]:
        return [{"name": k, "description": v} for k, v in STYLE_INFO.items()]

    @app.get("/ai/speech/status", dependencies=secured)
    def speech_status() -> dict[str, Any]:
        return whisper_status()

    @app.post("/timeline/auto_captions", dependencies=secured)
    def auto_captions(body: CaptionsBody) -> dict[str, Any]:
        """Transcribe the speech on the timeline (background job) and replace the captions."""
        session: Session = require()
        if body.style not in STYLE_INFO:
            raise HTTPException(422, "unknown caption style")
        if not whisper_status()["available"]:
            raise HTTPException(409, "speech recognition is not installed")
        with session.bus.lock:
            ids = {c.media_id for t in session.bus.project.tracks for c in t.clips
                   if c.media_id in session.bus.project.media and session.bus.project.media[c.media_id].kind == MediaKind.VIDEO}
        if not ids:
            raise HTTPException(422, "no videos on the timeline")

        def job(progress, cancelled) -> dict[str, Any]:
            for i, mid in enumerate(sorted(ids)):
                if cancelled():
                    from ..video.export import Cancelled

                    raise Cancelled()
                try:
                    session.transcript(mid)
                except WhisperUnavailable as exc:
                    raise RuntimeError(str(exc)) from exc
                progress((i + 1) / len(ids) * 0.95)
            with session.bus.lock:
                project = session.bus.project.model_copy(deep=True)
            caps = captions_from_transcripts(project, session.transcripts_for_timeline(project), body.style)
            payload = [{"start": c.start, "end": c.end, "text": c.text, "style": c.style} for c in caps]
            session.bus.execute("caption.replace_all", {"captions": payload}, source="user")
            return {"captions": len(payload)}

        return {"job": session.start_job("captions", job, label="Auto captions")}

    # ------------------------------------------------------------------ AI
    @app.get("/ai/options", dependencies=secured)
    def ai_options() -> dict[str, Any]:
        """Editing styles and desired outputs for the chips next to the prompt box."""
        session = require()
        return catalog(session.styles_dir)

    @app.post("/ai/reels", dependencies=secured)
    def ai_reels(body: ReelsBody) -> dict[str, Any]:
        """Find the best moments of one long video and return a ready reel program for each (nothing applied)."""
        session: Session = require()
        ctx = session.video_context(transcribe_speech=True)
        try:
            return make_reels(ctx, media_number=body.media_number, count=body.count, length=body.length, style=body.style,
                              output=body.output, captions=body.captions, tighten=body.tighten,
                              subject_of=session.subject_track, styles_dir=session.styles_dir,
                              progress_bar=body.progress_bar, end_card=body.end_card or None)
        except (OptionError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/ai/styles/from_video", dependencies=secured)
    def ai_style_from_video(body: StyleFromVideoBody) -> dict[str, Any]:
        """"Edit like this": measure a reference video and save it as a reusable style."""
        session: Session = require()
        ids = list(session.bus.project.media)
        if body.media_number > len(ids):
            raise HTTPException(404, "no such media")
        mid = ids[body.media_number - 1]
        if session.bus.project.media[mid].kind != MediaKind.VIDEO:
            raise HTTPException(422, "pick a video as the reference")
        style = style_from_reference(session.video_analysis(mid), body.name)
        key = save_style(session.styles_dir, body.name, style)
        return {"id": key, **style}

    @app.post("/ai/video/apply", dependencies=secured)
    def ai_video_apply(body: AiApplyBody) -> dict[str, Any]:
        session: Session = require()
        params = {"program": body.program, "mode": body.mode, "transcripts": session.transcripts_for_timeline() or None}
        try:
            result = session.bus.execute("timeline.apply_program", params, source="ai")
        except CommandError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"result": result, "revision": session.bus.revision}


def _prune_previews(folder: Path, keep: str, max_files: int = 4) -> None:
    files = sorted((p for p in folder.glob("timeline-*.mp4") if p.name != keep), key=lambda p: p.stat().st_mtime)
    for p in files[: max(0, len(files) - (max_files - 1))]:
        try:
            p.unlink()
        except OSError:
            pass  # still being played; try next time
