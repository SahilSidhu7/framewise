"""L1/L2 measurements of video and music for the AI's media brief.

Video: per-second brightness, sharpness, motion, colorfulness, shot cuts and
an "interest" score used to pick the best moments. Music: tempo and beat grid
(onset envelope + autocorrelation; no heavy dependencies).
Everything is cheap: frames are sampled ~4x a second at 160 px wide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np

from .audio import decode_audio

SAMPLE_FPS = 4.0
SMALL_W = 160


def analyze_video(path: Path) -> dict[str, Any]:
    times: list[float] = []
    bright, sharp, motion, color = [], [], [], []
    hists: list[np.ndarray] = []
    prev = None
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        w0, h0 = stream.codec_context.width, stream.codec_context.height
        known = float(stream.duration * stream.time_base) if stream.duration and stream.time_base else (c.duration / av.time_base if c.duration else None)
        sw = SMALL_W
        sh = max(2, int(h0 * sw / max(w0, 1)) // 2 * 2)
        next_t = 0.0
        for frame in c.decode(stream):
            t = frame.time
            if t is None or t + 1e-6 < next_t:
                continue
            next_t = t + 1 / SAMPLE_FPS
            rgb = frame.to_ndarray(format="rgb24", width=sw, height=sh)
            grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            times.append(float(t))
            bright.append(float(grey.mean() / 255))
            sharp.append(float(cv2.Laplacian(grey, cv2.CV_32F).var()))
            rg = rgb[..., 0].astype(np.float32) - rgb[..., 1]
            yb = 0.5 * (rgb[..., 0].astype(np.float32) + rgb[..., 1]) - rgb[..., 2]
            color.append(float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 100)
            motion.append(0.0 if prev is None else float(np.abs(grey.astype(np.int16) - prev).mean() / 255))
            hist = cv2.calcHist([rgb], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).ravel()
            hists.append(hist / max(hist.sum(), 1))
            prev = grey.astype(np.int16)
    if not times:
        return {"duration": 0, "notes": ["no frames"], "seconds": [], "cuts": [], "best": []}
    duration = known if known else times[-1] + 1 / SAMPLE_FPS
    cuts = [times[i] for i in range(1, len(hists)) if 0.5 * np.abs(hists[i] - hists[i - 1]).sum() > 0.45]

    b, s, m, col = map(np.asarray, (bright, sharp, motion, color))
    # interest: exposed well, sharp, some (not chaotic) motion, colorful
    exposure_ok = 1 - np.clip(np.abs(b - 0.48) / 0.4, 0, 1)
    sharp_n = np.clip(s / (np.percentile(s, 90) + 1e-6), 0, 1)
    motion_n = np.clip(m / 0.06, 0, 1)
    motion_ok = motion_n * (1 - np.clip((m - 0.12) / 0.1, 0, 1))  # very fast = shaky / whip
    score = 0.35 * exposure_ok + 0.3 * sharp_n + 0.2 * motion_ok + 0.15 * np.clip(col, 0, 1)
    # cut frames are bad in-points
    for ct in cuts:
        score[np.abs(np.asarray(times) - ct) < 0.3] *= 0.6

    seconds = []
    n_sec = int(np.ceil(duration))
    tarr = np.asarray(times)
    for sec in range(n_sec):
        mask = (tarr >= sec) & (tarr < sec + 1)
        if mask.any():
            seconds.append(round(float(score[mask].mean()), 3))
        else:
            seconds.append(seconds[-1] if seconds else 0.0)

    notes = []
    if b.mean() < 0.28:
        notes.append("dark")
    elif b.mean() > 0.75:
        notes.append("bright / overexposed")
    if m.mean() > 0.09:
        notes.append("lots of motion")
    elif m.mean() < 0.01:
        notes.append("static shot")
    if np.std(m) > 0.05:
        notes.append("shaky in places")
    if s.mean() < 40:
        notes.append("soft / out of focus")
    if col.mean() < 0.12:
        notes.append("muted colors")
    if len(cuts) > 0:
        notes.append(f"{len(cuts)} scene cut(s)")
    return {
        "duration": round(duration, 2),
        "size": [w0, h0],
        "orientation": "landscape" if w0 > h0 * 1.05 else "portrait" if h0 > w0 * 1.05 else "square",
        "brightness": round(float(b.mean()), 3),
        "motion": round(float(m.mean()), 4),
        "colorfulness": round(float(col.mean()), 3),
        "cuts": [round(c, 2) for c in cuts],
        "seconds": seconds,
        "notes": notes,
    }


def best_window(seconds: list[float], length: float, avoid: list[tuple[float, float]] | None = None, duration: float | None = None) -> tuple[float, float]:
    """Start/end of the highest-scoring window of `length` seconds, avoiding ranges already used."""
    total = duration if duration is not None else float(len(seconds))
    if length >= total or not seconds:
        return 0.0, total
    best, best_t = -1.0, 0.0
    step = 0.25
    t = 0.0
    arr = np.asarray(seconds, dtype=np.float32)
    while t + length <= total + 1e-6:
        if avoid and any(t < b and t + length > a for a, b in avoid):
            t += step
            continue
        a, b = int(t), max(int(t) + 1, int(np.ceil(t + length)))
        val = float(arr[a:b].mean()) if b > a else 0.0
        # mild preference for not starting at frame 0 (camera shake / fumbling)
        val -= 0.03 if t < 0.3 else 0.0
        if val > best:
            best, best_t = val, t
        t += step
    if best < 0:  # everything overlapped: fall back to the start
        return 0.0, min(length, total)
    return round(best_t, 2), round(min(total, best_t + length), 2)


# ------------------------------------------------------------------- music


def analyze_music(path: Path) -> dict[str, Any]:
    sr = 22050
    y = decode_audio(path, sr, mono=True)[0] if path else np.zeros(0, np.float32)
    if y.size < sr * 2:
        return {"duration": round(y.size / sr, 2), "bpm": None, "beats": [], "loudness": None}
    hop, n_fft = 512, 1024
    frames = 1 + (y.size - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    win = np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(y[idx] * win, axis=1)).astype(np.float32)
    logm = np.log1p(10 * mag)
    diff = np.maximum(0, np.diff(logm, axis=0))
    low = int(160 / (sr / n_fft)) + 1  # kick drum band: cut on the strong beat, not the hi-hat
    flux_low = diff[:, :low].sum(axis=1)
    flux_all = diff.sum(axis=1)
    flux = flux_low / (flux_low.max() + 1e-9) * 2 + flux_all / (flux_all.max() + 1e-9)
    flux = np.concatenate([[0], flux])
    flux = flux - np.convolve(flux, np.ones(16) / 16, mode="same")
    onset = np.maximum(flux, 0)
    onset /= onset.max() + 1e-9
    fps = sr / hop
    # tempo by autocorrelation, 70-180 BPM, mild prior around 115
    ac = np.correlate(onset, onset, mode="full")[onset.size - 1 :]
    lags = np.arange(ac.size)
    bpm_l = 60 * fps / np.maximum(lags, 1)
    valid = (bpm_l >= 70) & (bpm_l <= 180)
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm_l, 1) / 115) / 0.9) ** 2)
    weighted = np.where(valid, ac * prior, 0)
    lag = int(np.argmax(weighted))
    bpm = None
    beats: list[float] = []
    if lag > 1:
        # sub-frame period + phase: comb search around the autocorrelation peak
        best, period, phase = -1.0, float(lag), 0.0
        for cand in np.arange(lag - 1.0, lag + 1.0001, 0.02):
            k = np.arange(int((onset.size - 1) / cand))
            for ph in np.arange(0, cand, 0.5):
                pos = np.round(ph + k * cand).astype(int)
                val = float(onset[pos[pos < onset.size]].sum())
                if val > best:
                    best, period, phase = val, float(cand), float(ph)
        bpm = 60 * fps / period
        n_beats = int((onset.size - 1 - phase) / period) + 1
        beats = [round(float(max(0.0, (phase + i * period) / fps)), 3) for i in range(n_beats)]
    rms = float(np.sqrt(np.mean(y**2)))
    return {
        "duration": round(y.size / sr, 2),
        "bpm": round(bpm, 1) if bpm else None,
        "beats": beats,
        "loudness": round(float(20 * np.log10(max(rms, 1e-9))), 1),
        "energy": [round(float(v), 3) for v in _energy_per_second(y, sr)],
    }


def _energy_per_second(y: np.ndarray, sr: int) -> np.ndarray:
    n = y.size // sr
    if n == 0:
        return np.zeros(0)
    e = np.sqrt((y[: n * sr].reshape(n, sr) ** 2).mean(axis=1))
    return e / (e.max() + 1e-9)


def analyze_audio_track(path: Path) -> dict[str, Any]:
    """Speech/loudness summary of a video's own sound (for the brief)."""
    y = decode_audio(path, 16000, mono=True)
    if y.size == 0 or y.shape[1] == 0:
        return {"has_audio": False}
    y = y[0]
    rms = float(np.sqrt(np.mean(y**2)))
    # crude speech hint: energy fluctuating at syllable rate in the voice band
    hop = 160
    n = y.size // hop
    env = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) if n else np.zeros(1)
    active = float((env > max(rms * 0.5, 1e-4)).mean())
    mod = float(np.std(env) / (np.mean(env) + 1e-9))
    syllabic = _syllabic_ratio(y, 16000) if rms >= 0.003 else 0.0
    return {
        "has_audio": True,
        "loudness_db": round(float(20 * np.log10(max(rms, 1e-9))), 1),
        "silent": rms < 0.003,
        "energy": [round(float(v), 3) for v in _energy_per_second(y, 16000)],  # loudness per second, 0..1
        "speech_likely": bool(rms >= 0.003 and (syllabic > 0.43 or (mod > 0.9 and 0.2 < active < 0.95))),
    }


def _syllabic_ratio(y: np.ndarray, sr: int) -> float:
    """Share of the voice-band envelope's modulation energy at syllable rate (2-8 Hz).

    Speech sits around 0.45-0.5; music, waves, wind and traffic stay below ~0.4,
    even when background noise hides the pauses between words.
    """
    spec = np.fft.rfft(y)
    f = np.fft.rfftfreq(y.size, 1 / sr)
    spec[(f < 300) | (f > 3400)] = 0
    v = np.fft.irfft(spec, y.size)
    hop = sr // 100
    n = v.size // hop
    if n < 200:  # under 2 s: not enough to judge
        return 0.0
    env = np.sqrt((v[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    m = np.abs(np.fft.rfft(env - env.mean())) ** 2
    mf = np.fft.rfftfreq(n, 1 / 100)
    return float(m[(mf >= 2) & (mf <= 8)].sum() / (m[(mf >= 0.2) & (mf <= 20)].sum() + 1e-12))
