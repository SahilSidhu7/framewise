"""Low-resolution preview proxies for video.

Preview always plays proxies; final export always reads the originals.
Encoder choice prefers hardware encoders, then LGPL-safe software encoders.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import av

PROXY_HEIGHT = 720
# Hardware first; libopenh264 (BSD) before libx264 (GPL, only if the local build has it);
# mpeg4 is the always-available last resort.
VIDEO_ENCODERS = ("h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox", "h264_mf", "libopenh264", "libx264", "mpeg4")
TIME_BASE = Fraction(1, 1000)  # ms; small denominator keeps every encoder happy

ProgressFn = Callable[[float], None]


class ProxyError(Exception):
    pass


def proxy_size(width: int, height: int, target_h: int = PROXY_HEIGHT) -> tuple[int, int]:
    """Scale down (never up) to target height, keeping aspect and even dimensions."""
    if height <= target_h:
        return width - width % 2, height - height % 2
    scale = target_h / height
    w = max(2, int(width * scale / 2 + 0.5) * 2)  # round half up (round() is banker's)
    return w, target_h - target_h % 2


def make_proxy(src: Path, dst: Path, progress: ProgressFn | None = None) -> str:
    """Encode a proxy. Returns the encoder used. Tries encoders until one opens."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for encoder in VIDEO_ENCODERS:
        try:
            _encode(src, dst, encoder, progress)
            return encoder
        except (av.FFmpegError, ValueError, ProxyError) as exc:
            errors.append(f"{encoder}: {exc}")
            dst.unlink(missing_ok=True)
    raise ProxyError("no usable encoder: " + " | ".join(errors))


def _encode(src: Path, dst: Path, encoder: str, progress: ProgressFn | None) -> None:
    try:
        av.codec.Codec(encoder, "w")
    except (av.codec.codec.UnknownCodecError, ValueError) as exc:
        raise ProxyError("not in this FFmpeg build") from exc

    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        v_in = inp.streams.video[0]
        v_in.thread_type = "AUTO"
        a_in = next(iter(inp.streams.audio), None)

        width, height = proxy_size(v_in.codec_context.width, v_in.codec_context.height)
        rate = v_in.average_rate or v_in.guessed_rate or Fraction(30)
        v_out = out.add_stream(encoder, rate=rate)
        v_out.width, v_out.height = width, height
        v_out.pix_fmt = "yuv420p"
        v_out.codec_context.time_base = TIME_BASE
        v_out.bit_rate = 4_000_000

        a_out = resampler = None
        if a_in is not None:
            a_out = out.add_stream("aac", rate=48000, layout="stereo")
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=48000)

        total = float(v_in.duration * v_in.time_base) if v_in.duration else None
        last_pts = -1
        streams = [s for s in (v_in, a_in) if s is not None]
        for packet in inp.demux(*streams):
            for frame in packet.decode():
                if isinstance(frame, av.VideoFrame):
                    if frame.time is None:
                        continue
                    pts = max(round(frame.time / TIME_BASE), last_pts + 1)
                    last_pts = pts
                    scaled = frame.reformat(width=width, height=height, format="yuv420p")
                    scaled.pts, scaled.time_base = pts, TIME_BASE
                    out.mux(v_out.encode(scaled))
                    if progress and total:
                        progress(min(frame.time / total, 1.0))
                elif a_out is not None and resampler is not None:
                    frame.pts = None
                    for rf in resampler.resample(frame):
                        out.mux(a_out.encode(rf))

        out.mux(v_out.encode(None))
        if a_out is not None and resampler is not None:
            for rf in resampler.resample(None):
                out.mux(a_out.encode(rf))
            out.mux(a_out.encode(None))
        if last_pts < 0:
            raise ProxyError("no video frames decoded")
    if progress:
        progress(1.0)
