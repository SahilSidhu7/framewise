"""Inspect media files and make thumbnails.

Uses PyAV (bundled FFmpeg libraries, no system ffmpeg required) and Pillow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import av
from PIL import Image, ImageOps, UnidentifiedImageError

from ..core.project import MediaKind

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".avif"}
THUMB_SIZE = (320, 320)


class ProbeError(Exception):
    pass


def probe(path: Path) -> dict[str, Any]:
    """Return metadata matching the `media.add` command params."""
    if not path.is_file():
        raise ProbeError(f"file not found: {path}")
    if path.suffix.lower() in IMAGE_EXTS:
        try:
            return _probe_image(path)
        except (UnidentifiedImageError, OSError):
            pass  # fall through: some "images" are really video containers
    return _probe_av(path)


def _probe_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)  # report the size the user actually sees
        width, height = img.size
    return {"name": path.name, "path": str(path.resolve()), "kind": MediaKind.IMAGE, "width": width, "height": height}


def _probe_av(path: Path) -> dict[str, Any]:
    try:
        container = av.open(str(path))
    except av.FFmpegError as exc:
        raise ProbeError(f"unsupported or corrupt media: {path.name}") from exc
    with container:
        video = next((s for s in container.streams.video), None)
        audio = next((s for s in container.streams.audio), None)
        if video is None and audio is None:
            raise ProbeError(f"no audio or video streams: {path.name}")
        duration = _duration(container, video or audio)
        info: dict[str, Any] = {
            "name": path.name,
            "path": str(path.resolve()),
            "duration": duration,
            "has_audio": audio is not None,
        }
        if video is not None:
            rate = video.average_rate or video.guessed_rate
            info.update(
                kind=MediaKind.VIDEO,
                width=video.codec_context.width,
                height=video.codec_context.height,
                fps=float(rate) if rate else None,
            )
        else:
            info["kind"] = MediaKind.AUDIO
        return info


def _duration(container: av.container.InputContainer, stream: av.stream.Stream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return None


def make_thumbnail(path: Path, kind: MediaKind, out_path: Path) -> bool:
    """Write a JPEG thumbnail. Returns False when the media has no picture."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if kind == MediaKind.IMAGE:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail(THUMB_SIZE)
            img.save(out_path, "JPEG", quality=85)
        return True
    if kind == MediaKind.VIDEO:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            # Seek ~10% in to skip black/fade-in first frames.
            if stream.duration and stream.duration > 0:
                container.seek(int(stream.duration * 0.1), stream=stream, any_frame=False)
            for frame in container.decode(stream):
                img = frame.to_image()
                img.thumbnail(THUMB_SIZE)
                img.save(out_path, "JPEG", quality=85)
                return True
    return False
