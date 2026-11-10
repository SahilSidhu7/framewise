"""Timeline compositor: project + time -> finished frame.

Pipeline per frame (linear-light float32, like the photo engine):
    for each video/overlay track, bottom to top:
        active clip(s) -> decode -> fit/fill + motion -> clip ops -> clip fades
        transitions blend the outgoing and incoming clip around the cut
    -> project grade (look/adjust/grain...)  -> letterbox -> global fades
    -> sRGB + dither (8-bit) -> captions/titles

The same function renders scrub frames (from proxies, small) and the final
export (from originals, full size), so preview matches export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.project import Clip, MediaKind, Project, Track
from ..imaging.ops import OPS
from ..imaging.render import SOURCES, apply_stack, make_context
from .decode import READERS, u8_to_linear
from .fastgrade import PLANS, encode_u8
from .text import draw_text


@dataclass
class RenderSource:
    """Where to read media from."""

    root: Path  # project dir (for proxies)
    use_proxy: bool = False
    channel: str = ""  # decoder pool channel (export gets its own)


# ---------------------------------------------------------------- geometry


def _cover_scale(sw: int, sh: int, w: int, h: int, fit: str) -> float:
    return max(w / sw, h / sh) if fit == "fill" else min(w / sw, h / sh)


def _motion(clip: Clip, t: float) -> tuple[float, float, float]:
    """(zoom, dx, dy) for Ken Burns style moves; dx/dy as fraction of the overflow."""
    if clip.motion == "none":
        return 1.0, 0.0, 0.0
    p = min(1.0, max(0.0, (t - clip.start) / max(clip.duration, 1e-6)))
    p = p * p * (3 - 2 * p)  # ease in-out
    if clip.motion == "zoom_in":
        return 1.0 + 0.12 * p, 0.0, 0.0
    if clip.motion == "zoom_out":
        return 1.12 - 0.12 * p, 0.0, 0.0
    if clip.motion == "pan_left":
        return 1.12, 0.5 - p, 0.0
    return 1.12, p - 0.5, 0.0  # pan_right


def _place(src: np.ndarray, w: int, h: int, fit: str, zoom: float, dx: float, dy: float) -> np.ndarray:
    """Scale/position a source into a w x h canvas with one affine warp."""
    sh, sw = src.shape[:2]
    s = _cover_scale(sw, sh, w, h, fit) * zoom
    out_w, out_h = sw * s, sh * s
    tx = (w - out_w) / 2 + dx * max(0.0, out_w - w)
    ty = (h - out_h) / 2 + dy * max(0.0, out_h - h)
    if abs(s - 1) < 1e-6 and abs(tx) < 1e-6 and abs(ty) < 1e-6 and (sw, sh) == (w, h):
        return src
    m = np.array([[s, 0, tx], [0, s, ty]], dtype=np.float32)
    interp = cv2.INTER_AREA if s < 0.75 else cv2.INTER_LINEAR
    if interp == cv2.INTER_AREA:
        # warpAffine has no area filter: pre-shrink, then warp the remainder
        pre = cv2.resize(src, (max(1, int(sw * s * 1.25)), max(1, int(sh * s * 1.25))), interpolation=cv2.INTER_AREA)
        k = pre.shape[1] / sw
        m[:, :2] /= k
        src, interp = pre, cv2.INTER_LINEAR
    return cv2.warpAffine(src, m, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


# ------------------------------------------------------------------ frames


def _decode_size(item_w: int, item_h: int, w: int, h: int, fit: str, zoom_max: float) -> tuple[int, int] | None:
    """Ask FFmpeg for a frame just big enough (saves most of the per-frame cost)."""
    s = _cover_scale(item_w, item_h, w, h, fit) * zoom_max
    if s >= 1:
        return None
    return max(2, int(item_w * s + 1) // 2 * 2), max(2, int(item_h * s + 1) // 2 * 2)


def clip_frame(project: Project, clip: Clip, t: float, w: int, h: int, src: RenderSource, seed: int) -> np.ndarray:
    item = project.media[clip.media_id]
    zoom, dx, dy = _motion(clip, t)
    if item.kind == MediaKind.IMAGE:
        img = SOURCES.get(Path(item.path), int(max(w, h) * 1.15))
        # photos in a video can carry their photo edit stack too
        stack = [o.model_dump(mode="json") for o in project.edits.get(item.id, [])]
        if stack:
            img = apply_stack(img, stack, make_context(seed))
    else:
        path = Path(item.path)
        if src.use_proxy and item.proxy_path:
            proxy = src.root / item.proxy_path
            if proxy.is_file():
                path = proxy
        reader_probe = READERS.get(path, None, src.channel)
        size = _decode_size(reader_probe.width, reader_probe.height, w, h, clip.fit, 1.12 if clip.motion != "none" else 1.0)
        reader = READERS.get(path, size, src.channel) if size else reader_probe
        img = u8_to_linear(reader.frame_at(clip.source_time(t)))
    img = _place(img, w, h, clip.fit, zoom, dx, dy)
    ops = [o.model_dump(mode="json") for o in clip.ops if not OPS[o.op].geometry]
    if ops:
        img = apply_stack(img, ops, make_context(seed))
    fade = 1.0
    if clip.fade_in > 0:
        fade = min(fade, (t - clip.start) / clip.fade_in)
    if clip.fade_out > 0:
        fade = min(fade, (clip.end - t) / clip.fade_out)
    if fade < 1:
        img = img * max(0.0, fade)
    return img


def _prev_clip(track: Track, clip: Clip) -> Clip | None:
    best = None
    for c in track.clips:
        if c is clip or c.start >= clip.start:
            continue
        if abs(c.end - clip.start) < 0.05 and (best is None or c.end > best.end):
            best = c
    return best


def _ease(p: float) -> float:
    return p * p * (3 - 2 * p)


def _blend(a: np.ndarray, b: np.ndarray, kind: str, p: float) -> np.ndarray:
    h, w = a.shape[:2]
    e = _ease(p)
    if kind == "crossfade":
        return a * (1 - e) + b * e
    if kind in ("fade_black", "fade_white"):
        mid = np.ones_like(a) if kind == "fade_white" else np.zeros_like(a)
        return a * (1 - e * 2) + mid * (e * 2) if p < 0.5 else mid * (2 - e * 2) + b * (e * 2 - 1)
    if kind in ("wipe_left", "wipe_right"):
        edge = int(w * e)
        out = a.copy()
        if kind == "wipe_left":
            out[:, w - edge :] = b[:, w - edge :]
        else:
            out[:, :edge] = b[:, :edge]
        return out
    if kind == "slide_left":
        off = int(w * e)
        out = np.empty_like(a)
        out[:, : w - off] = a[:, off:]
        out[:, w - off :] = b[:, : off] if off else out[:, w - off :]
        return out
    if kind == "slide_up":
        off = int(h * e)
        out = np.empty_like(a)
        out[: h - off] = a[off:]
        out[h - off :] = b[:off] if off else out[h - off :]
        return out
    if kind == "zoom":
        z = 1 + 0.25 * e
        za = cv2.warpAffine(a, np.float32([[z, 0, (1 - z) * w / 2], [0, z, (1 - z) * h / 2]]), (w, h), flags=cv2.INTER_LINEAR)
        return za * (1 - e) + b * e
    if kind == "blur":
        sigma = 1 + 18 * (1 - abs(2 * p - 1))
        mix = a * (1 - e) + b * e
        return cv2.GaussianBlur(mix, (0, 0), sigma * min(h, w) / 1000)
    return b if p >= 0.5 else a


def track_frame(project: Project, track: Track, t: float, w: int, h: int, src: RenderSource, seed: int) -> np.ndarray | None:
    for clip in track.clips:
        tr = clip.transition
        if tr is None:
            continue
        half = tr.duration / 2
        if clip.start - half <= t < clip.start + half:
            prev = _prev_clip(track, clip)
            p = (t - (clip.start - half)) / tr.duration
            b = clip_frame(project, clip, max(t, clip.start), w, h, src, seed)
            a = (
                clip_frame(project, prev, min(t, prev.end - 1e-4), w, h, src, seed)
                if prev is not None
                else np.zeros_like(b)
            )
            if prev is None and t < clip.start:
                return _blend(a, b, "crossfade", p) if tr.kind != "fade_white" else _blend(a, b, tr.kind, p)
            return _blend(a, b, tr.kind, p)
    for clip in track.clips:
        if clip.start <= t < clip.end:
            return clip_frame(project, clip, t, w, h, src, seed)
    return None


def render_frame(
    project: Project, t: float, w: int, h: int, src: RenderSource, *, captions: bool = True
) -> np.ndarray:
    """Finished uint8 RGB frame at time t."""
    fps = project.settings.fps or 30
    seed = int(t * fps)  # grain/dust change every frame
    canvas: np.ndarray | None = None
    for track in project.tracks:
        if track.kind == "audio":
            continue
        layer = track_frame(project, track, t, w, h, src, seed)
        if layer is not None:
            canvas = layer  # opaque layers: the upper track covers the lower
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.float32)
    grade = [o.model_dump(mode="json") for o in project.grade if not OPS[o.op].geometry]
    frame = PLANS.get(grade).finish(canvas, seed)
    s = project.settings
    total = project.duration
    fade = 1.0
    if s.fade_in > 0:
        fade = min(fade, t / s.fade_in)
    if s.fade_out > 0 and total > 0:
        fade = min(fade, (total - t) / s.fade_out)
    if fade < 1:
        frame = encode_u8(u8_to_linear(frame) * max(0.0, fade), seed)
    bar = 0
    if s.letterbox:
        visible_h = w / s.letterbox
        if visible_h < h:
            bar = int(round((h - visible_h) / 2))
            if not frame.flags.writeable or frame.base is not None:
                frame = frame.copy()
            frame[:bar] = 0
            frame[h - bar :] = 0
    if captions:
        for cap in project.captions:
            if cap.start <= t < cap.end:
                draw_text(frame, cap.text, cap.style, cap.position, t - cap.start, cap.end - t,
                          bar if cap.kind == "caption" else 0)
    return frame


def output_size(project: Project, max_side: int | None = None) -> tuple[int, int]:
    w, h = project.settings.width, project.settings.height
    if max_side and max(w, h) > max_side:
        s = max_side / max(w, h)
        w, h = int(w * s / 2 + 0.5) * 2, int(h * s / 2 + 0.5) * 2
    return max(2, w), max(2, h)


def describe_frame(project: Project, t: float) -> dict[str, Any]:
    """What is on screen at t (for UI tooltips and the AI)."""
    out: dict[str, Any] = {"t": t, "clips": [], "captions": []}
    for track in project.tracks:
        for clip in track.clips:
            if clip.start <= t < clip.end:
                out["clips"].append({"track": track.id, "clip": clip.id, "source_t": clip.source_time(t)})
    out["captions"] = [c.text for c in project.captions if c.start <= t < c.end]
    return out
