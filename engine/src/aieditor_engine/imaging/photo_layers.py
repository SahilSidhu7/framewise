"""Photo documents: layers (images, solids, adjustment layers, masks, blend
modes, mattes) drawn over the edited photo.

A photo's layers are ordinary tracks and clips (core.project.Project.layers),
composited at t=0 by the video compositor on a canvas the size of the edited
photo. Positions are fractions of that canvas, so a small preview and the
full-size export place every layer the same way.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..core.project import Project

Overlay = Callable[[np.ndarray], np.ndarray]


def has_layers(project: Project, media_id: str) -> bool:
    return any(t.clips and not t.hidden for t in project.layers.get(media_id, []))


def overlay(project: Project, media_id: str, seed: int = 0) -> tuple[Overlay | None, str]:
    """(function that draws the photo's layers over its edited picture, cache key);
    (None, "") when the photo has no visible layers. Takes a copy: safe off-lock."""
    if not has_layers(project, media_id):
        return None, ""
    tracks = [t.model_copy(deep=True) for t in project.layers[media_id]]
    used = {c.media_id for t in tracks for c in t.clips if c.media_id}
    media = {k: project.media[k].model_copy() for k in used if k in project.media}
    edits = {k: [o.model_copy() for o in project.edits.get(k, [])] for k in media}
    doc = Project(name="layers", media=media, edits=edits, tracks=[])
    stamps = []
    for k in sorted(media):
        path = Path(media[k].path)
        stamps.append([k, str(path), path.stat().st_mtime_ns if path.is_file() else 0])
    key_src = [[t.model_dump(mode="json") for t in tracks], stamps, {k: [o.model_dump(mode="json") for o in v] for k, v in edits.items()}]
    key = hashlib.sha1(json.dumps(key_src, sort_keys=True).encode()).hexdigest()

    def draw(base: np.ndarray) -> np.ndarray:
        from ..video.compose import RenderSource, composite_tracks

        h, w = base.shape[:2]
        doc.settings.width, doc.settings.height = w, h
        return composite_tracks(doc, 0.0, w, h, RenderSource(root=Path(".")), seed, tracks=tracks, base=base)

    return draw, key
