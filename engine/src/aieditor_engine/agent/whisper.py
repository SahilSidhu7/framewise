"""Local speech-to-text with whisper.cpp (MIT) for auto-captions.

The engine looks for `whisper-cli(.exe)` under <app data>/runtime/whisper-*/
(or AIEDITOR_WHISPER) and a ggml model under <app data>/models/ggml-*.bin
(or AIEDITOR_WHISPER_MODEL). Audio is decoded with PyAV to 16 kHz mono WAV.
Output: {"language", "words": [{start, end, text}], "segments": [...]}
in source seconds. Results are cached per media in the project folder.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from ..video.audio import decode_audio
from .llm import app_data_root

from ..paths import env
log = logging.getLogger(__name__)
MODEL_PREFERENCE = ("ggml-small", "ggml-base", "ggml-tiny")


class WhisperUnavailable(RuntimeError):
    pass


def find_whisper() -> Path | None:
    value = env("WHISPER")
    if value and Path(value).is_file():
        return Path(value)
    exe = "whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"
    runtime = app_data_root() / "runtime"
    if runtime.is_dir():
        for found in sorted(runtime.glob(f"whisper*/**/{exe}"), reverse=True):
            return found
    return None


def find_whisper_model() -> Path | None:
    value = env("WHISPER_MODEL")
    if value and Path(value).is_file():
        return Path(value)
    models = app_data_root() / "models"
    if not models.is_dir():
        return None
    found = sorted(models.glob("ggml-*.bin"))
    for prefix in MODEL_PREFERENCE:
        for m in found:
            if m.name.startswith(prefix):
                return m
    return found[0] if found else None


def whisper_status() -> dict[str, Any]:
    exe, model = find_whisper(), find_whisper_model()
    return {"available": bool(exe and model), "model": model.name if model else None}


def _write_wav(path: Path, mono16k: np.ndarray) -> None:
    pcm = np.clip(mono16k * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


_TOKEN_JUNK = re.compile(r"^\[.*\]$|^<\|.*\|>$")


def transcribe(media_path: Path, language: str = "auto", timeout: float = 900) -> dict[str, Any]:
    exe, model = find_whisper(), find_whisper_model()
    if exe is None or model is None:
        raise WhisperUnavailable("speech recognition is not installed")
    audio = decode_audio(media_path, 16000, mono=True)
    if audio.size == 0:
        return {"language": None, "words": [], "segments": []}
    with tempfile.TemporaryDirectory(prefix="framewise-asr-") as tmp:
        wav = Path(tmp) / "audio.wav"
        _write_wav(wav, audio[0])
        out_base = Path(tmp) / "out"
        threads = max(1, min(8, (os.cpu_count() or 4) - 1))
        args = [
            str(exe), "-m", str(model), "-f", str(wav), "-l", language,
            "-ml", "1", "-sow", "-oj", "-of", str(out_base), "-t", str(threads), "-np",
        ]
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        from ..procs import bind_to_engine

        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                creationflags=flags, cwd=str(exe.parent))
        bind_to_engine(proc)  # a killed engine must not leave whisper running
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise WhisperUnavailable(f"whisper took longer than {timeout:.0f} s") from None
        result = out_base.with_suffix(".json")
        if proc.returncode != 0 or not result.is_file():
            tail = err.decode("utf-8", "replace")[-400:]
            raise WhisperUnavailable(f"whisper failed: {tail}")
        data = json.loads(result.read_text(encoding="utf-8", errors="replace"))
    words = []
    for item in data.get("transcription", []):
        text = str(item.get("text", "")).strip()
        if not text or _TOKEN_JUNK.match(text):
            continue
        off = item.get("offsets", {})
        s, e = off.get("from", 0) / 1000.0, off.get("to", 0) / 1000.0
        raw = str(item.get("text", ""))
        if words and not raw.startswith(" "):
            words[-1]["text"] += text  # sub-word token or punctuation: glue to the previous word
            words[-1]["end"] = e
            continue
        words.append({"start": s, "end": max(e, s + 0.05), "text": text})
    segments = _segments(words)
    lang = data.get("result", {}).get("language")
    return {"language": lang, "words": words, "segments": segments}


def _segments(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segs: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > 0.7 or cur[-1]["text"][-1:] in ".?!"):
            segs.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": " ".join(x["text"] for x in cur)})
            cur = []
        cur.append(w)
    if cur:
        segs.append({"start": cur[0]["start"], "end": cur[-1]["end"], "text": " ".join(x["text"] for x in cur)})
    return segs
