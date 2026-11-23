"""What the clipping tools know about the media in a project: one MediaInfo
per item (analysis, audio, transcript, music) and the VideoContext around them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# looks that suit most footage; reels pick from these when no style names one
SAFE_VIDEO_LOOKS = ("vivid_pop", "film_portra", "golden_hour", "bright_airy", "faded_film")



@dataclass
class MediaInfo:
    number: int
    media_id: str
    name: str
    kind: str  # video | image | audio
    duration: float
    orientation: str = "landscape"
    analysis: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    transcript: dict[str, Any] | None = None
    music: dict[str, Any] | None = None

    @property
    def speech(self) -> bool:
        if self.transcript is not None:
            return len(self.transcript.get("words", [])) >= 3
        return bool(self.audio.get("speech_likely"))


@dataclass
class VideoContext:
    media: list[MediaInfo]
    current_program: str = ""
    # media_id -> subject track (video/reframe.py); looked up only when a wide clip goes on a taller canvas
    subject_of: Callable[[str], dict[str, Any] | None] | None = None

    @property
    def visual(self) -> list[MediaInfo]:
        return [m for m in self.media if m.kind in ("video", "image")]

    @property
    def songs(self) -> list[MediaInfo]:
        return [m for m in self.media if m.kind == "audio"]


def video_brief(ctx: VideoContext) -> str:
    lines = ["media:"]
    for m in ctx.media:
        if m.kind == "audio":
            mu = m.music or {}
            bpm = f", {mu['bpm']:g} bpm" if mu.get("bpm") else ""
            lines.append(f"#{m.number} {m.name}: music {m.duration:.0f}s{bpm}")
            continue
        if m.kind == "image":
            lines.append(f"#{m.number} {m.name}: photo {m.orientation}")
            continue
        notes = ", ".join(m.analysis.get("notes", [])) or "ok"
        bits = [f"#{m.number} {m.name}: video {m.duration:.1f}s {m.orientation}", notes]
        if m.speech:
            text = " ".join(s["text"] for s in (m.transcript or {}).get("segments", []))[:120]
            bits.append(f'speech: "{text}"' if text else "has speech")
        lines.append("; ".join(bits))
    lines.append("current edit: " + (ctx.current_program.replace("\n", " | ") if ctx.current_program.strip() else "empty timeline"))
    return "\n".join(lines)
