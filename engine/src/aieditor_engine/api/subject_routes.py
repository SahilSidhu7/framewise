"""Subject selection: pick by clicking, preview the outline, follow it in video clips.

    GET    /media/{id}/subject?point=x,y&t=   outline (photo) or tracked boxes (video)
    POST   /media/{id}/subject {x, y, t}      choose the subject (undoable command)
    DELETE /media/{id}/subject                back to automatic
    POST   /clips/{id}/follow_subject         keep the subject centred in this clip (x keyframes)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

from ..core.commands import CommandError
from ..core.project import MediaKind
from ..imaging.color import linear_to_srgb
from ..imaging.render import SOURCES
from ..imaging.subject import mask_box, parse_point, subject_mask

if TYPE_CHECKING:
    from ..session import Session


class PickBody(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    t: float = Field(0.0, ge=0)


def _outline(mask: np.ndarray, max_points: int = 160) -> list[list[list[float]]]:
    """Subject outline as normalised polygons (largest first) for the UI to draw."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
        if cv2.contourArea(c) < h * w * 0.002:
            continue
        eps = 0.002 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        while len(approx) > max_points:
            eps *= 1.5
            approx = cv2.approxPolyDP(c, eps, True)
        out.append([[round(float(p[0][0]) / w, 4), round(float(p[0][1]) / h, 4)] for p in approx])
    return out


def _video_info(session: Session, media_id: str, t: float | None) -> dict[str, Any]:
    item = session.bus.project.media[media_id]
    track = session.subject_track(media_id) or {}
    out: dict[str, Any] = {"kind": "video", "pick": item.subject.model_dump() if item.subject else None,
                           "method": track.get("method"), "box": None, "track": []}
    boxes, times = track.get("box") or [], track.get("t") or []
    if boxes:
        step = max(1, len(times) // 240)
        out["track"] = [[times[i], boxes[i]] for i in range(0, len(times), step)]
        if t is not None:
            i = int(np.argmin([abs(tt - t) for tt in times]))
            out["box"] = boxes[i]
    return out


def register(app: FastAPI, require, secured: list) -> None:  # noqa: ANN001 - FastAPI deps
    @app.get("/media/{media_id}/subject", dependencies=secured)
    def get_subject(media_id: str, point: str = Query("auto", max_length=40), t: float | None = Query(None, ge=0)) -> dict[str, Any]:
        session: Session = require()
        item = session.bus.project.media.get(media_id)
        if item is None:
            raise HTTPException(404, "media not found")
        if item.kind == MediaKind.VIDEO:
            return _video_info(session, media_id, t)
        if item.kind != MediaKind.IMAGE:
            raise HTTPException(422, "subjects can be chosen on photos and videos")
        try:
            pt = parse_point(point)
        except ValueError as exc:
            raise HTTPException(422, 'point must be "auto" or "x,y"') from exc
        img = SOURCES.get(Path(item.path), 900)
        display = (np.clip(linear_to_srgb(np.clip(img, 0, 1)), 0, 1) * 255).astype(np.uint8)
        mask = subject_mask(display, pt)
        return {"kind": "photo", "point": point, "box": mask_box(mask), "outline": _outline(mask), "coverage": round(float((mask > 0.5).mean()), 3)}

    @app.post("/media/{media_id}/subject", dependencies=secured)
    def pick_subject(media_id: str, body: PickBody) -> dict[str, Any]:
        """Choose the subject at a position. For videos this follows it through the whole clip."""
        session: Session = require()
        try:
            session.bus.execute("media.set_subject", {"media_id": media_id, "x": body.x, "y": body.y, "t": body.t}, source="user")
        except CommandError as exc:
            raise HTTPException(422, str(exc)) from exc
        item = session.bus.project.media[media_id]
        if item.kind == MediaKind.VIDEO:
            return _video_info(session, media_id, body.t)
        return {"kind": "photo", "pick": item.subject.model_dump() if item.subject else None}

    @app.delete("/media/{media_id}/subject", dependencies=secured)
    def clear_subject(media_id: str) -> dict[str, Any]:
        session: Session = require()
        try:
            session.bus.execute("media.set_subject", {"media_id": media_id}, source="user")
        except CommandError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"pick": None}

    @app.get("/media/{media_id}/frame", dependencies=secured)
    def media_frame(media_id: str, t: float = Query(0.0, ge=0), size: int = Query(960, alias="max", ge=64, le=1920)) -> Response:
        """The original (unedited, uncropped) video frame at source time t, for picking a subject."""
        import av

        session: Session = require()
        item = session.bus.project.media.get(media_id)
        if item is None or item.kind != MediaKind.VIDEO:
            raise HTTPException(404, "video not found")
        path = Path(item.path)
        if item.proxy_path and (session.paths.root / item.proxy_path).is_file():
            path = session.paths.root / item.proxy_path
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            c.seek(int(max(0.0, t - 1.0) / s.time_base), stream=s, backward=True)
            frame = None
            for f in c.decode(s):
                frame = f
                if f.time is not None and f.time >= t - 0.02:
                    break
        if frame is None:
            raise HTTPException(422, "could not read that frame")
        w0, h0 = frame.width, frame.height
        sc = min(1.0, size / max(w0, h0))
        rgb = frame.to_ndarray(format="rgb24", width=max(2, int(w0 * sc) // 2 * 2), height=max(2, int(h0 * sc) // 2 * 2))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/clips/{clip_id}/follow_subject", dependencies=secured)
    def follow_subject(clip_id: str) -> dict[str, Any]:
        """Animate the clip's horizontal position so the (picked or automatic) subject stays centred."""
        from ..video.reframe import reframe_keys

        session: Session = require()
        project = session.bus.project
        try:
            _, clip = project.find_clip(clip_id)
        except KeyError as exc:
            raise HTTPException(404, "clip not found") from exc
        track = session.subject_track(clip.media_id)
        if not track:
            raise HTTPException(422, "only video clips can follow a subject")
        s = project.settings
        keys = reframe_keys(track, clip.in_point, clip.out_point, s.width / s.height, clip.speed)
        if not keys:
            raise HTTPException(422, "this clip already fits the frame; following a subject works when the video is wider than "
                                     "the frame (for example 16:9 footage in a 9:16 project)")
        session.bus.execute("clip.set_keyframes", {"clip_id": clip_id, "param": "x",
                                                   "keys": [{"t": t, "value": v} for t, v in keys]}, source="user")
        return {"keys": len(keys), "method": track.get("method")}
