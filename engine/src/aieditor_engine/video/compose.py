"""Timeline compositor: project + time -> finished frame.

Pipeline per frame (linear-light float32, like the photo engine):
    for each video/overlay track, bottom to top:
        active clip(s) -> decode (or a solid colour) -> fit/fill + motion + transform/crop -> clip ops
        -> masks, opacity and fades (premultiplied alpha); keyframes animate these
        transitions blend the outgoing and incoming clip around the cut
        a track with a matte is cut by the track above it (which is then not drawn)
        an adjustment layer runs its effects on everything below it
        the layer is composited on the tracks below with its blend mode (video/layers.py)
    -> project grade (look/adjust/grain...)  -> letterbox -> global fades
    -> sRGB + dither (8-bit) -> captions/titles

The same function renders scrub frames (from proxies, small) and the final
export (from originals, full size), so preview matches export.
"""

from __future__ import annotations

import math
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
from .layers import composite, hex_to_linear, masks_coverage, matte_value, multiply_alpha
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


def _matrix(clip: Clip, t: float, sw: int, sh: int, w: int, h: int) -> tuple[np.ndarray, bool]:
    """Source pixel -> canvas pixel as a 2x3 affine: fit/fill and motion first,
    then the clip's own transform (scale and rotation about the canvas centre,
    then the offset). Also says whether the result covers the canvas completely."""
    zoom, dx, dy = _motion(clip, t)
    s = _cover_scale(sw, sh, w, h, clip.fit) * zoom
    out_w, out_h = sw * s, sh * s
    tx = (w - out_w) / 2 + dx * max(0.0, out_w - w)
    ty = (h - out_h) / 2 + dy * max(0.0, out_h - h)
    m = np.array([[s, 0, tx], [0, s, ty]], dtype=np.float64)
    k, rot = clip.value("scale", t), clip.value("rotation", t) % 360
    ox, oy = clip.value("x", t) * w, clip.value("y", t) * h
    if k != 1 or rot or ox or oy:
        th = math.radians(rot)
        c, sn = math.cos(th) * k, math.sin(th) * k
        cx, cy = w / 2 + clip.value("anchor_x", t) * w, h / 2 + clip.value("anchor_y", t) * h  # pivot
        u = np.array([[c, -sn, cx - c * cx + sn * cy + ox], [sn, c, cy - sn * cx - c * cy + oy]])
        m = np.hstack([u[:, :2] @ m[:, :2], (u[:, :2] @ m[:, 2] + u[:, 2])[:, None]])
    covers = bool(
        rot == 0 and clip.crop.is_empty
        and m[0, 2] <= 0.5 and m[1, 2] <= 0.5 and m[0, 2] + sw * m[0, 0] >= w - 0.5 and m[1, 2] + sh * m[1, 1] >= h - 0.5
    )
    return m, covers


def _place(src: np.ndarray, clip: Clip, t: float, w: int, h: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Put a source on the w x h canvas with one affine warp.

    Returns (picture, coverage). coverage is None when the picture fills the
    canvas; otherwise it is a (h, w) mask in 0..1 (antialiased edges, crop
    applied) and the picture is edge-extended, to be multiplied by the mask
    after the clip's effects ran."""
    sh, sw = src.shape[:2]
    m, covers = _matrix(clip, t, sw, sh, w, h)
    if covers and (sw, sh) == (w, h) and np.allclose(m, [[1, 0, 0], [0, 1, 0]], atol=1e-6):
        return src, None
    scale = math.sqrt(abs(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]))
    if scale < 0.75:
        # warpAffine has no area filter: pre-shrink, then warp the remainder
        pre = cv2.resize(src, (max(1, int(sw * scale * 1.25)), max(1, int(sh * scale * 1.25))), interpolation=cv2.INTER_AREA)
        m = m.copy()
        m[:, 0] /= pre.shape[1] / sw
        m[:, 1] /= pre.shape[0] / sh
        src = pre
    m32 = m.astype(np.float32)
    if covers:
        return cv2.warpAffine(src, m32, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)), None
    ph, pw = src.shape[:2]
    mask = np.ones((ph, pw), dtype=np.float32)
    cr = clip.crop
    if not cr.is_empty:
        mask[: int(round(ph * cr.top))] = 0
        mask[ph - int(round(ph * cr.bottom)) :] = 0
        mask[:, : int(round(pw * cr.left))] = 0
        mask[:, pw - int(round(pw * cr.right)) :] = 0
    img = cv2.warpAffine(src, m32, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    cover = cv2.warpAffine(mask, m32, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, cover


# ------------------------------------------------------------------ frames


def _decode_size(item_w: int, item_h: int, w: int, h: int, fit: str, zoom_max: float) -> tuple[int, int] | None:
    """Ask FFmpeg for a frame just big enough (saves most of the per-frame cost)."""
    s = _cover_scale(item_w, item_h, w, h, fit) * zoom_max
    if s >= 1:
        return None
    return max(2, int(item_w * s + 1) // 2 * 2), max(2, int(item_h * s + 1) // 2 * 2)


# A layer is (picture, alpha, blend). The picture is linear light and already
# multiplied by alpha; alpha is a (h, w) array or a plain float (1.0 = opaque).
Alpha = np.ndarray | float


def _max_scale(clip: Clip) -> float:
    keys = clip.keyframes.get("scale")
    return max([k.value for k in keys]) if keys else clip.transform.scale


def _layer_alpha(clip: Clip, t: float) -> float:
    a = min(1.0, max(0.0, clip.value("opacity", t)))
    if clip.fade_in > 0:
        a = min(a, max(0.0, (t - clip.start) / clip.fade_in))
    if clip.fade_out > 0:
        a = min(a, max(0.0, (clip.end - t) / clip.fade_out))
    return a


def _shape(clip: Clip, t: float, sw: int, sh: int, w: int, h: int, cover: np.ndarray | None) -> np.ndarray | None:
    """Coverage of the placed layer (edges, crop) times its masks; None = everywhere."""
    if not any(m.enabled for m in clip.masks):
        return cover
    masks = masks_coverage(clip.masks, sw, sh, _matrix(clip, t, sw, sh, w, h)[0], w, h)
    if masks is None:
        return cover
    return masks if cover is None else cover * masks


def clip_frame(project: Project, clip: Clip, t: float, w: int, h: int, src: RenderSource, seed: int) -> tuple[np.ndarray, Alpha]:
    if clip.source == "solid":
        img = np.empty((h, w, 3), np.float32)
        img[:] = hex_to_linear(clip.color)
        sh, sw = h, w
        img, cover = _place(img, clip, t, w, h)
        a = _layer_alpha(clip, t)
        cover = _shape(clip, t, sw, sh, w, h, cover)
        if cover is not None:
            alpha = cover * a
            return img * alpha[..., None], alpha
        return (img * a, a) if a < 1 else (img, 1.0)
    item = project.media[clip.media_id]
    if item.kind == MediaKind.IMAGE:
        img = SOURCES.get(Path(item.path), int(max(w, h) * 1.15 * min(4.0, max(1.0, _max_scale(clip)))))
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
        zoom_max = (1.12 if clip.motion != "none" else 1.0) * _max_scale(clip)
        size = _decode_size(reader_probe.width, reader_probe.height, w, h, clip.fit, zoom_max)
        reader = READERS.get(path, size, src.channel) if size else reader_probe
        img = u8_to_linear(reader.frame_at(clip.source_time(t)))
    sh, sw = img.shape[:2]
    img, cover = _place(img, clip, t, w, h)
    ops = [o.model_dump(mode="json") for o in clip.ops if not OPS[o.op].geometry and OPS[o.op].category != "subject"]
    if ops:
        img = apply_stack(img, ops, make_context(seed))
    a = _layer_alpha(clip, t)
    cover = _shape(clip, t, sw, sh, w, h, cover)
    if cover is not None:
        alpha = cover * a if a < 1 else cover
        return img * alpha[..., None], alpha
    if a < 1:
        return img * a, a
    return img, 1.0


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


_ALPHA_KINDS = ("crossfade", "wipe_left", "wipe_right", "slide_left", "slide_up", "zoom")


def _full(alpha: Alpha, h: int, w: int) -> np.ndarray:
    return alpha if isinstance(alpha, np.ndarray) else np.full((h, w), alpha, dtype=np.float32)


def track_frame(
    project: Project, track: Track, t: float, w: int, h: int, src: RenderSource, seed: int
) -> tuple[np.ndarray, Alpha, str] | None:
    """The track's layer at time t, or None when nothing is on it."""
    for clip in track.clips:
        tr = clip.transition
        if tr is None:
            continue
        half = tr.duration / 2
        if clip.start - half <= t < clip.start + half:
            prev = _prev_clip(track, clip)
            p = (t - (clip.start - half)) / tr.duration
            b, b_alpha = clip_frame(project, clip, max(t, clip.start), w, h, src, seed)
            kind = tr.kind
            if prev is not None:
                a, a_alpha = clip_frame(project, prev, min(t, prev.end - 1e-4), w, h, src, seed)
            else:
                # nothing before the clip: it appears out of the layers below (out of white for fade_white)
                a, a_alpha = np.zeros_like(b), (1.0 if kind == "fade_white" else 0.0)
                if t < clip.start and kind != "fade_white":
                    kind = "crossfade"
            img = _blend(a, b, kind, p)
            if isinstance(a_alpha, float) and isinstance(b_alpha, float) and a_alpha == b_alpha == 1.0:
                return img, 1.0, clip.blend
            alpha = _blend(_full(a_alpha, h, w), _full(b_alpha, h, w), kind if kind in _ALPHA_KINDS else "crossfade", p)
            return img, alpha, clip.blend
    for clip in track.clips:
        if clip.start <= t < clip.end:
            img, alpha = clip_frame(project, clip, t, w, h, src, seed)
            return img, alpha, clip.blend
    return None


def _active(track: Track, t: float) -> Clip | None:
    for clip in track.clips:
        if clip.start <= t < clip.end:
            return clip
    return None


def _adjust(clip: Clip, canvas: np.ndarray, t: float, w: int, h: int, seed: int) -> np.ndarray:
    """Adjustment layer: its effects run on everything below, inside its shape and opacity."""
    ops = [o.model_dump(mode="json") for o in clip.ops if not OPS[o.op].geometry and OPS[o.op].category != "subject"]
    a = _layer_alpha(clip, t)
    if not ops or a <= 0:
        return canvas
    _, cover = _place(np.ones((h, w, 3), np.float32), clip, t, w, h)
    cover = _shape(clip, t, w, h, w, h, cover)
    fx = apply_stack(canvas, ops, make_context(seed))
    if cover is None and a >= 1 and clip.blend == "normal":
        return fx
    alpha: Alpha = cover * a if cover is not None else a
    return composite(canvas, fx * (alpha[..., None] if isinstance(alpha, np.ndarray) else alpha), alpha, clip.blend)


def composite_tracks(project: Project, t: float, w: int, h: int, src: RenderSource, seed: int) -> np.ndarray:
    """All picture tracks, bottom to top, as one linear-light canvas."""
    canvas: np.ndarray | None = None
    tracks = [tr for tr in project.tracks if tr.kind != "audio"]
    skip: set[int] = set()
    for i, track in enumerate(tracks):  # bottom to top
        if i in skip or track.hidden:
            continue
        matte_track = tracks[i + 1] if track.matte and i + 1 < len(tracks) else None
        if matte_track is not None:
            skip.add(i + 1)  # a matte is not drawn itself
        active = _active(track, t)
        if active is not None and active.source == "adjustment":
            canvas = _adjust(active, canvas if canvas is not None else np.zeros((h, w, 3), np.float32), t, w, h, seed)
            continue
        layer = track_frame(project, track, t, w, h, src, seed)
        if layer is None:
            continue
        img, alpha, blend = layer
        if matte_track is not None and track.matte:
            m = track_frame(project, matte_track, t, w, h, src, seed)
            v = matte_value(m[0], m[1], track.matte, h, w) if m is not None else (1.0 if track.matte.endswith("inverted") else 0.0)
            img, alpha = multiply_alpha(img, alpha, v, h, w)
        canvas = composite(canvas, img, alpha, blend)
    return canvas if canvas is not None else np.zeros((h, w, 3), dtype=np.float32)


def _gpu_frame(project: Project, t: float, w: int, h: int, src: RenderSource, seed: int, grade: list[dict[str, Any]]) -> np.ndarray | None:
    from ..gpu.compositor import get

    gpu = get()
    if gpu is None:
        return None
    return gpu.render(project, t, w, h, src, seed, grade)


def render_frame(
    project: Project, t: float, w: int, h: int, src: RenderSource, *, captions: bool = True, use_gpu: bool = True
) -> np.ndarray:
    """Finished uint8 RGB frame at time t."""
    fps = project.settings.fps or 30
    seed = int(t * fps)  # grain/dust change every frame
    content_end = project.content_end
    if project.settings.end_card and t >= content_end > 0:
        return _end_card(project, t, w, h, src, content_end)
    grade = [o.model_dump(mode="json") for o in project.grade if not OPS[o.op].geometry and OPS[o.op].category != "subject"]
    frame = _gpu_frame(project, t, w, h, src, seed, grade) if use_gpu else None
    if frame is None:
        frame = PLANS.get(grade).finish(composite_tracks(project, t, w, h, src, seed), seed)
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
                if not frame.flags.writeable:
                    frame = frame.copy()
                draw_text(frame, cap.text, cap.style, cap.position, t - cap.start, cap.end - t,
                          bar if cap.kind == "caption" else 0, cap.words)
    if s.progress_bar != "off" and content_end > 0:
        if not frame.flags.writeable:
            frame = frame.copy()
        _progress_bar(frame, min(1.0, t / content_end), s.progress_bar, bar)
    return frame


def _progress_bar(frame: np.ndarray, p: float, where: str, bar: int) -> None:
    h, w = frame.shape[:2]
    th = max(3, int(round(min(w, h) * 0.008)))
    y0 = bar if where == "top" else h - bar - th
    region = frame[y0 : y0 + th].astype(np.float32)
    region *= 0.55  # dim track
    fill = int(round(w * p))
    if fill > 0:
        region[:, :fill] = region[:, :fill] * 0.1 + 255 * 0.9
    frame[y0 : y0 + th] = np.clip(region + 0.5, 0, 255).astype(np.uint8)


_CARD_CACHE: dict[tuple[int, int, int, float], np.ndarray] = {}


def _end_card(project: Project, t: float, w: int, h: int, src: RenderSource, content_end: float) -> np.ndarray:
    """Closing card: the last picture blurred and darkened, with the end card text."""
    s = project.settings
    fps = s.fps or 30
    key = (id(project), w, h, round(content_end, 3))
    bg = _CARD_CACHE.get(key)
    if bg is None:
        last = render_frame(project.model_copy(update={"settings": s.model_copy(update={
            "end_card": None, "progress_bar": "off", "fade_out": 0.0})}), max(0.0, content_end - 1.0 / fps), w, h, src,
            captions=False)
        small = cv2.resize(last, (max(2, w // 8), max(2, h // 8)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.5)
        bg = (cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) * 0.45).astype(np.uint8)
        if len(_CARD_CACHE) > 8:
            _CARD_CACHE.clear()
        _CARD_CACHE[key] = bg
    frame = bg.copy()
    age, remaining = t - content_end, project.duration - t
    draw_text(frame, s.end_card or "", "title", "center", age, remaining)
    if s.fade_out > 0 and remaining < s.fade_out:
        frame = (frame.astype(np.float32) * max(0.0, remaining / s.fade_out)).astype(np.uint8)
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
