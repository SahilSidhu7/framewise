"""Render the timeline to an MP4 (H.264 + AAC).

- Frames render in parallel: the timeline is cut into short segments and each
  worker thread renders whole segments with its own decoders (sequential
  decode inside a segment, no thrashing). The main thread encodes in order.
- Encoders: hardware first (NVENC/QSV/AMF/MediaFoundation), then software.
  Each candidate is probed once with a tiny in-memory encode.
- Output is tagged BT.709 limited range, which is what players assume for HD.
"""

from __future__ import annotations

import io
import logging
import math
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import av
import numpy as np

from ..core.project import Project
from .audio import SR, mix_timeline
from .compose import RenderSource, render_frame
from .decode import READERS

log = logging.getLogger(__name__)

Quality = Literal["high", "standard", "small"]
ENCODERS = ("h264_nvenc", "h264_qsv", "h264_amf", "h264_mf", "libopenh264", "libx264", "mpeg4")
SEGMENT_SECONDS = 1.0


class Cancelled(Exception):
    pass


def _bitrate(w: int, h: int, fps: float, quality: Quality) -> int:
    base = 12_000_000 * (w * h * fps) / (1920 * 1080 * 30)
    return int(max(800_000, base * {"high": 1.6, "standard": 1.0, "small": 0.45}[quality]))


def _options(encoder: str, quality: Quality, bitrate: int) -> dict[str, str]:
    q = {"high": 0, "standard": 1, "small": 2}[quality]
    if encoder == "h264_nvenc":
        return {"preset": "p6", "tune": "hq", "rc": "vbr", "cq": ("19", "23", "28")[q],
                "maxrate": str(bitrate * 2), "bufsize": str(bitrate * 4), "profile": "high", "spatial-aq": "1"}
    if encoder == "h264_qsv":
        return {"preset": "slower", "global_quality": ("20", "24", "29")[q]}
    if encoder == "h264_amf":
        return {"quality": "quality", "rc": "cqp", "qp_i": ("18", "22", "27")[q], "qp_p": ("20", "24", "29")[q]}
    if encoder == "libx264":
        return {"preset": "slow", "crf": ("17", "20", "25")[q], "profile": "high"}
    return {}


@lru_cache(maxsize=32)
def encoder_works(name: str, w: int, h: int) -> bool:
    """Probe an encoder with a 3-frame in-memory encode (drivers may be missing)."""
    try:
        av.codec.Codec(name, "w")
    except Exception:
        return False
    try:
        buf = io.BytesIO()
        with av.open(buf, "w", format="mp4") as out:
            st = out.add_stream(name, rate=30)
            st.width, st.height, st.pix_fmt = w, h, "yuv420p"
            st.bit_rate = 2_000_000
            for i in range(3):
                f = av.VideoFrame.from_ndarray(np.zeros((h, w, 3), np.uint8), format="rgb24").reformat(format="yuv420p")
                f.pts = i
                out.mux(st.encode(f))
            out.mux(st.encode(None))
        return buf.tell() > 0
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable"
        log.info("encoder %s unusable: %s", name, exc)
        return False


def pick_encoder(w: int, h: int, prefer: str | None = None) -> str:
    order = ([prefer] if prefer else []) + [e for e in ENCODERS if e != prefer]
    for name in order:
        if encoder_works(name, w, h):
            return name
    raise RuntimeError("no working H.264/MPEG-4 encoder in this FFmpeg build")


@dataclass
class ExportSpec:
    path: Path
    width: int
    height: int
    fps: float
    quality: Quality = "high"
    use_proxy: bool = False  # previews read proxies; final exports read originals
    audio: bool = True
    encoder: str | None = None
    workers: int | None = None
    place_captions: bool = True  # move auto-positioned captions off faces and burned-in text


ProgressFn = Callable[[float], None]


def _rate(fps: float) -> Fraction:
    for std in (Fraction(24000, 1001), Fraction(30000, 1001), Fraction(60000, 1001)):
        if abs(float(std) - fps) < 0.01:
            return std
    return Fraction(fps).limit_denominator(1001)


def export_video(
    project: Project,
    root: Path,
    spec: ExportSpec,
    progress: ProgressFn | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    duration = project.duration
    if duration <= 0:
        raise ValueError("the timeline is empty")
    if spec.place_captions and any(c.kind == "caption" for c in project.captions):
        from .placement import place_captions

        project = project.model_copy(deep=True)  # never touch the caller's (undoable) project
        channel = f"place-{id(spec)}"
        try:
            placed = place_captions(project, RenderSource(root=root, use_proxy=True, channel=channel))
        finally:
            READERS.drop_channel(channel)
        log.info("caption placement: %s", placed)
    w, h = spec.width - spec.width % 2, spec.height - spec.height % 2
    rate = _rate(spec.fps)
    fps = float(rate)
    total = max(1, int(math.floor(duration * fps + 1e-6)))
    encoder = pick_encoder(w, h, spec.encoder)
    bitrate = _bitrate(w, h, fps, spec.quality)
    workers = spec.workers or max(1, min(8, (os.cpu_count() or 4) - 2))
    seg_len = max(1, int(round(SEGMENT_SECONDS * fps)))
    segments = [(s, min(total, s + seg_len)) for s in range(0, total, seg_len)]
    job = f"export-{id(spec)}"
    channels: set[str] = set()

    def render_segment(a: int, b: int) -> list[np.ndarray]:
        channel = f"{job}:{threading.get_ident()}"
        channels.add(channel)
        src = RenderSource(root=root, use_proxy=spec.use_proxy, channel=channel)
        frames = []
        for i in range(a, b):
            if cancelled():
                raise Cancelled()
            frames.append(render_frame(project, i / fps, w, h, src))
        return frames

    audio = mix_timeline(project, SR) if spec.audio else None
    has_audio = audio is not None and audio.size > 0 and bool(np.abs(audio).max() > 1e-5)

    spec.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = spec.path.with_name(spec.path.stem + ".part" + spec.path.suffix)
    try:
        with av.open(str(tmp), "w") as out:
            v = out.add_stream(encoder, rate=rate, options=_options(encoder, spec.quality, bitrate))
            v.width, v.height, v.pix_fmt = w, h, "yuv420p"
            v.bit_rate = bitrate
            v.codec_context.time_base = 1 / rate
            cc = v.codec_context
            cc.color_primaries, cc.color_trc, cc.colorspace, cc.color_range = 1, 1, 1, 1  # BT.709, limited
            a_stream = None
            if has_audio:
                a_stream = out.add_stream("aac", rate=SR, layout="stereo")
                a_stream.bit_rate = 192_000 if spec.quality != "small" else 128_000
            audio_pos = 0
            a_frame = 1024

            def mux_audio_until(t: float) -> None:
                nonlocal audio_pos
                if a_stream is None or audio is None:
                    return
                limit = audio.shape[1] if t == math.inf else min(audio.shape[1], int(t * SR))
                while audio_pos + a_frame <= limit or (t == math.inf and audio_pos < audio.shape[1]):
                    chunk = audio[:, audio_pos : audio_pos + a_frame]
                    if chunk.shape[1] < a_frame:
                        chunk = np.pad(chunk, ((0, 0), (0, a_frame - chunk.shape[1])))
                    af = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk), format="fltp", layout="stereo")
                    af.sample_rate, af.pts = SR, audio_pos
                    out.mux(a_stream.encode(af))
                    audio_pos += a_frame

            done = 0
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="render") as pool:
                pending: list[Future] = []
                it = iter(segments)
                for _ in range(workers * 2):
                    seg = next(it, None)
                    if seg:
                        pending.append(pool.submit(render_segment, *seg))
                try:
                    while pending:
                        frames = pending.pop(0).result()
                        seg = next(it, None)
                        if seg:
                            pending.append(pool.submit(render_segment, *seg))
                        for frame in frames:
                            vf = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(
                                format="yuv420p", dst_colorspace="ITU709", dst_color_range="MPEG"
                            )
                            vf.pts = done
                            out.mux(v.encode(vf))
                            done += 1
                            mux_audio_until(done / fps)
                        if progress:
                            progress(done / total)
                        if cancelled():
                            raise Cancelled()
                finally:
                    for f in pending:
                        f.cancel()
            out.mux(v.encode(None))
            mux_audio_until(math.inf)
            if a_stream is not None:
                out.mux(a_stream.encode(None))
        os.replace(tmp, spec.path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        for ch in channels:
            READERS.drop_channel(ch)
    return {
        "path": str(spec.path),
        "width": w,
        "height": h,
        "fps": round(fps, 3),
        "frames": done,
        "duration": round(done / fps, 3),
        "encoder": encoder,
        "audio": has_audio,
    }
