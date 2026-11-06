"""Timeline audio: decode, place, fade, crossfade, duck music under speech,
normalize loudness (ITU-R BS.1770 style, -14 LUFS for streaming) and limit peaks.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np

from ..core.project import Clip, MediaKind, Project, Track

SR = 48000
DECLICK = 0.005  # seconds of micro-fade at every edit


class _AudioCache:
    def __init__(self, size: int = 16) -> None:
        self._items: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get(self, path: Path, rate: int = SR) -> np.ndarray:
        key = (str(path), path.stat().st_mtime_ns, rate)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        data = decode_audio(path, rate)
        with self._lock:
            self._items[key] = data
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return data


def decode_audio(path: Path, rate: int = SR, mono: bool = False) -> np.ndarray:
    """Whole file as float32 (channels, samples). Empty if there is no audio."""
    layout = "mono" if mono else "stereo"
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = next(iter(container.streams.audio), None)
        if stream is None:
            return np.zeros((1 if mono else 2, 0), np.float32)
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
        for frame in container.decode(stream):
            frame.pts = None
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray())
        for rf in resampler.resample(None):
            chunks.append(rf.to_ndarray())
    if not chunks:
        return np.zeros((1 if mono else 2, 0), np.float32)
    return np.concatenate(chunks, axis=1).astype(np.float32)


AUDIO = _AudioCache()


def _ramp(n: int, a: int, b: int) -> np.ndarray:
    """Gain envelope: rises over the first a samples, falls over the last b."""
    env = np.ones(n, np.float32)
    if a > 0:
        a = min(a, n)
        env[:a] = np.linspace(0, 1, a, dtype=np.float32) ** 1.5
    if b > 0:
        b = min(b, n)
        env[n - b :] = np.minimum(env[n - b :], np.linspace(1, 0, b, dtype=np.float32) ** 1.5)
    return env


def _next_transition(track: Track, clip: Clip) -> float:
    for c in track.clips:
        if c is not clip and c.transition and abs(c.start - clip.end) < 0.05:
            return c.transition.duration
    return 0.0


def _clip_audio(project: Project, track: Track, clip: Clip, rate: int) -> tuple[int, np.ndarray] | None:
    item = project.media.get(clip.media_id)
    if item is None or item.kind == MediaKind.IMAGE or clip.muted or clip.volume <= 0:
        return None
    if item.kind == MediaKind.VIDEO and not item.has_audio:
        return None
    src = AUDIO.get(Path(item.path), rate)
    if src.shape[1] == 0:
        return None
    # Transitions crossfade the audio too: extend into the neighbours' handles.
    pre = clip.transition.duration / 2 if clip.transition else 0.0
    post = _next_transition(track, clip) / 2
    pre = min(pre, clip.in_point / clip.speed)
    src_end = (item.duration or src.shape[1] / rate)
    post = min(post, max(0.0, (src_end - clip.out_point) / clip.speed))
    s0 = int(round((clip.in_point - pre * clip.speed) * rate))
    s1 = int(round((clip.out_point + post * clip.speed) * rate))
    seg = src[:, max(0, s0) : max(0, s1)]
    if seg.shape[1] == 0:
        return None
    if abs(clip.speed - 1) > 1e-3:
        n_out = max(1, int(seg.shape[1] / clip.speed))
        x = np.linspace(0, seg.shape[1] - 1, n_out)
        seg = np.stack([np.interp(x, np.arange(seg.shape[1]), ch) for ch in seg]).astype(np.float32)
    n = seg.shape[1]
    fade_in = int((clip.fade_in + 2 * pre + DECLICK) * rate)
    fade_out = int((clip.fade_out + 2 * post + DECLICK) * rate)
    seg = seg * _ramp(n, fade_in, fade_out) * clip.volume
    start = int(round((clip.start - pre) * rate))
    return start, seg


def _place(bus: np.ndarray, start: int, seg: np.ndarray) -> None:
    n = bus.shape[1]
    if start < 0:
        seg, start = seg[:, -start:], 0
    end = min(n, start + seg.shape[1])
    if end > start:
        bus[:, start:end] += seg[:, : end - start]


def _envelope(x: np.ndarray, rate: int, win: float) -> np.ndarray:
    """RMS envelope sampled every 10 ms."""
    hop = int(rate * 0.01)
    mono = np.abs(x).mean(axis=0) if x.ndim == 2 else np.abs(x)
    m = len(mono) // hop
    if m == 0:
        return np.zeros(0, np.float32)
    blocks = (mono[: m * hop] ** 2).reshape(m, hop).mean(axis=1)
    k = max(1, int(win / 0.01))
    return np.sqrt(np.convolve(blocks, np.ones(k) / k, mode="same")).astype(np.float32)


def duck(music: np.ndarray, voice: np.ndarray, rate: int, depth_db: float = -12.0) -> np.ndarray:
    env = _envelope(voice, rate, 0.08)
    if env.size == 0:
        return music
    active = (env > 10 ** (-38 / 20)).astype(np.float32)
    # fast attack (50 ms), slow release (400 ms): dilate, then smooth
    k_rel = 40
    dil = np.convolve(active, np.ones(k_rel), mode="full")[: active.size] > 0
    smooth = np.convolve(dil.astype(np.float32), np.ones(5) / 5, mode="same")
    gain_blocks = 1 - (1 - 10 ** (depth_db / 20)) * smooth
    hop = int(rate * 0.01)
    x = np.arange(music.shape[1]) / hop
    gain = np.interp(x, np.arange(gain_blocks.size), gain_blocks).astype(np.float32)
    return music * gain


# -------------------------------------------------------------- loudness


def _k_weight_power(x: np.ndarray, rate: int) -> np.ndarray:
    """Per-sample K-weighted power (sum over channels), via FFT magnitude response
    of the BS.1770 pre-filter + RLB high-pass. Phase does not matter for power."""
    chunk = 1 << 18
    out = np.zeros(x.shape[1], np.float32)
    freqs = np.fft.rfftfreq(chunk, 1 / rate)
    # high shelf ~ +4 dB above ~1.7 kHz, high-pass ~ 38 Hz (2nd order)
    shelf = 1 + (10 ** (4 / 20) - 1) / (1 + (1681.97 / np.maximum(freqs, 1e-3)) ** 2)
    hp = (freqs / 38.13) ** 2 / np.sqrt(1 + (freqs / 38.13) ** 4)
    resp = (shelf * hp).astype(np.float32)
    for s in range(0, x.shape[1], chunk):
        seg = x[:, s : s + chunk]
        n = seg.shape[1]
        spec = np.fft.rfft(seg, n=chunk, axis=1) * resp
        y = np.fft.irfft(spec, n=chunk, axis=1)[:, :n]
        out[s : s + n] = (y**2).sum(axis=0)
    return out


def loudness_lufs(x: np.ndarray, rate: int = SR) -> float:
    if x.shape[1] < rate * 0.4:
        return -70.0
    p = _k_weight_power(x, rate)
    block, hop = int(0.4 * rate), int(0.1 * rate)
    m = 1 + (len(p) - block) // hop
    cs = np.concatenate([[0], np.cumsum(p, dtype=np.float64)])
    starts = np.arange(m) * hop
    z = (cs[starts + block] - cs[starts]) / block
    lk = -0.691 + 10 * np.log10(np.maximum(z, 1e-12))
    z = z[lk > -70]
    if z.size == 0:
        return -70.0
    rel = -0.691 + 10 * np.log10(z.mean()) - 10
    z2 = z[-0.691 + 10 * np.log10(z) > rel]
    return float(-0.691 + 10 * np.log10((z2 if z2.size else z).mean()))


def soft_limit(x: np.ndarray, ceiling: float = 0.891) -> np.ndarray:
    """Soft knee above `ceiling` (-1 dBFS); never exceeds 1.0."""
    knee = ceiling * 0.9
    a = np.abs(x)
    over = a > knee
    if not over.any():
        return x
    y = x.copy()
    span = 1.0 - knee
    y[over] = np.sign(x[over]) * (knee + span * np.tanh((a[over] - knee) / span))
    return y * min(1.0, ceiling / max(float(np.abs(y).max()), 1e-9))


# ------------------------------------------------------------------- mix


def mix_timeline(project: Project, rate: int = SR, *, target_lufs: float = -14.0) -> np.ndarray:
    """Final stereo mix (2, samples) for the whole timeline."""
    n = int(np.ceil(project.duration * rate))
    dialog = np.zeros((2, n), np.float32)
    music = np.zeros((2, n), np.float32)
    for track in project.tracks:
        bus = music if track.kind == "audio" else dialog
        for clip in track.clips:
            placed = _clip_audio(project, track, clip, rate)
            if placed is not None:
                _place(bus, *placed)
    s = project.settings
    if s.duck_music and music.any() and dialog.any():
        music = duck(music, dialog, rate)
    master = dialog + music
    if n and (s.fade_in > 0 or s.fade_out > 0):
        master *= _ramp(n, int(s.fade_in * rate), int(s.fade_out * rate))
    if s.normalize_audio and master.any():
        lufs = loudness_lufs(master, rate)
        if lufs > -60:
            gain_db = min(12.0, target_lufs - lufs)
            master *= 10 ** (gain_db / 20)
    return soft_limit(master) if master.any() else master
