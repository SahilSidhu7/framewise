"""EditLM for video: request + media brief -> Video Edit DSL programs.

Division of labour (PLAN.md section 1):
- The model (or the rule layer) decides WHAT: aspect, pace, transition style,
  look, captions, title, music. It writes a short *intent* program.
- The director (deterministic code) decides HOW: which seconds of each clip
  (best-moment windows from the analysis), shot lengths snapped to the music's
  beat grid, ordering, fades, ducking. It expands intent into a concrete
  program that the engine applies as one undo step.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..imaging.recipes import load_recipes
from ..video.analysis import best_window
from ..video.reframe import reframe_keys, x_param
from ..video.vdsl import ASPECTS, CAPTION_STYLES, TRANSITIONS
from .llm import LLM, LLMUnavailable
from .planner import INTENSITY, _match_looks, wants

log = logging.getLogger(__name__)

SAFE_VIDEO_LOOKS = ("vivid_pop", "film_portra", "golden_hour", "bright_airy", "faded_film")
PACE_SHOT = {"fast": 1.1, "medium": 2.4, "slow": 4.0}
FAST_WORDS = r"\b(reels?|tiktok|shorts|energetic|fast|hype|upbeat|montage|quick|dynamic|action|party|sports?|beat|beats|punchy)\b"
SLOW_WORDS = r"\b(cinematic|calm|relax\w*|slow|emotional|wedding|romantic|documentary|peaceful|film|movie|dreamy|nostalgic|elegant)\b"
CAPTION_WORDS = r"\b(captions?|subtitles?|subs|transcri\w*|text on screen|words on screen)\b"
MUSIC_WORDS = r"\b(music|song|soundtrack|beat|beats|track)\b"
SLOWMO_WORDS = r"\b(slow ?mo\w*|slow motion)\b"
HIGHLIGHT_WORDS = r"\b(highlights?|best (parts|moments|bits|shots)|recap|montage|teaser|trailer)\b"
ASPECT_WORDS = [
    (r"\b(reels?|tiktok|shorts|stor(y|ies)|vertical|9:16|portrait video)\b", "9:16"),
    (r"\b(square|1:1)\b", "1:1"),
    (r"\b(instagram post|feed post|4:5)\b", "4:5"),
    (r"\b(youtube|widescreen|16:9|landscape|cinematic|film|movie|trailer)\b", "16:9"),
]


# ------------------------------------------------------------------ context


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


# -------------------------------------------------------------------- intent


@dataclass
class Intent:
    aspect: str
    pace: str  # fast | medium | slow
    transition: str  # none | crossfade | ...
    look: str | None = None
    strength: float = 1.0
    finishing: list[str] = field(default_factory=list)  # extra grade statements, e.g. "grain 0.15"
    letterbox: float | None = None
    captions: str | None = None  # style
    title: str | None = None
    title_style: str = "title"
    music: int | None = None  # media number
    length: float | None = None  # target seconds
    highlights: bool = False
    slowmo: bool = False
    rationale: str = ""

    def to_program(self) -> str:
        lines = [f"# {self.rationale}" if self.rationale else "# edit plan", f"aspect {self.aspect}", f"pace {self.pace}", f"transition {self.transition}"]
        if self.look:
            lines.append(f"look {self.look} s={self.strength:g}")
        lines.extend(self.finishing)
        if self.letterbox:
            lines.append(f"letterbox {self.letterbox:g}")
        if self.captions:
            lines.append(f"captions {self.captions}")
        if self.title:
            lines.append(f'title "{self.title}"')
        if self.music:
            lines.append(f"music {self.music}")
        return "\n".join(lines)


@dataclass
class VideoHints:
    aspect: str | None  # requested explicitly
    pace: str
    transition: str
    looks: tuple[str, ...]
    captions: str | None
    title: str | None
    music: int | None
    letterbox: float | None
    length: float | None
    highlights: bool
    slowmo: bool
    strength: float
    text: str
    style_captions: str | None = None  # the chosen style's caption look, used when captions get switched on
    fixed_pace: str | None = None  # set when a style was chosen: the model may not change the pace


def _default_aspect(ctx: VideoContext) -> str:
    vis = ctx.visual
    portrait = sum(1 for m in vis if m.orientation == "portrait")
    return "9:16" if vis and portrait > len(vis) / 2 else "16:9"


def _length(text: str) -> float | None:
    m = re.search(r"\b(\d{1,3})\s*(s|sec|secs|seconds?)\b", text)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d{1,2}|a|one)\s*(min|minute)s?\b", text)
    if m:
        n = m.group(1)
        return 60.0 * (1 if n in ("a", "one") else int(n))
    return None


def video_hints(prompt: str, ctx: VideoContext) -> VideoHints:
    text = " " + prompt.lower().strip() + " "
    aspect = next((a for pattern, a in ASPECT_WORDS if re.search(pattern, text)), None)
    fast, slow = bool(re.search(FAST_WORDS, text)), bool(re.search(SLOW_WORDS, text))
    pace = "fast" if fast and not slow else "slow" if slow and not fast else "medium"
    if "slow" in text and re.search(SLOWMO_WORDS, text) and not re.search(SLOW_WORDS.replace("slow|", ""), text):
        pace = "medium"  # "slow motion" alone is not a slow-paced edit
    transition = "none" if pace == "fast" else "crossfade"
    if re.search(r"fade to black|dip to black", text):
        transition = "fade_black"
    elif re.search(r"\bwipes?\b", text):
        transition = "wipe_left"
    elif re.search(r"\bzoom transitions?\b", text):
        transition = "zoom"
    elif re.search(r"\b(smooth|soft) (cuts|transitions)\b", text):
        transition = "crossfade"
    elif re.search(r"\b(hard|jump|quick) cuts\b|\bno transitions?\b", text):
        transition = "none"
    looks = tuple(_match_looks(text))
    speech = any(m.speech for m in ctx.visual)
    captions = None
    if wants(CAPTION_WORDS, text):
        captions = "bold_pop" if pace == "fast" else "cinematic" if re.search(r"\b(cinematic|film|movie)\b", text) else "clean"
        for st in CAPTION_STYLES:
            if st != "title" and re.search(rf"\b({st}|{st.replace('_', ' ')})\b", text):
                captions = st
    quoted = re.search(r"[\"“]([^\"”]{1,60})[\"”]", prompt)
    title = quoted.group(1).strip() if quoted else None
    songs = ctx.songs
    music = None
    if songs and not re.search(r"\bno music\b|\bwithout music\b", text):
        if wants(MUSIC_WORDS, text) or not speech:
            music = songs[0].number
    cinematic = bool(re.search(r"\b(cinematic|film|movie|trailer)\b", text))
    letterbox = 2.39 if cinematic and (aspect or _default_aspect(ctx)) == "16:9" else None
    strength = next((s for pattern, s in INTENSITY if re.search(pattern, text)), 1.0)
    return VideoHints(
        aspect=aspect, pace=pace, transition=transition, looks=looks, captions=captions if (captions and speech) else None,
        title=title, music=music, letterbox=letterbox, length=_length(text),
        highlights=bool(re.search(HIGHLIGHT_WORDS, text)) or pace == "fast", slowmo=bool(re.search(SLOWMO_WORDS, text)),
        strength=strength, text=text,
    )


def intent_from_hints(h: VideoHints, ctx: VideoContext, look_index: int = 0, pace: str | None = None) -> Intent:
    look = h.looks[look_index] if len(h.looks) > look_index else None
    finishing = []
    if look in ("teal_orange_cinematic", "moody_dark", "noir") or (h.letterbox and not look):
        finishing.append("grain 0.12")
    pace = pace or h.pace
    title_style = "bold_pop" if pace == "fast" else "title"
    why = [f"{pace} pace"]
    if look:
        why.append(f"{load_recipes()[look].title} look")
    if h.music:
        why.append("cuts on the music beat")
    if h.captions:
        why.append(f"{h.captions} captions")
    return Intent(
        aspect=h.aspect or _default_aspect(ctx), pace=pace, transition="none" if pace == "fast" and h.transition == "crossfade" else h.transition,
        look=look, strength=h.strength, finishing=finishing, letterbox=h.letterbox, captions=h.captions,
        title=h.title, title_style=title_style, music=h.music, length=h.length, highlights=h.highlights, slowmo=h.slowmo,
        rationale=", ".join(why),
    )


# ------------------------------------------------------------------ director


@dataclass
class Shot:
    media: MediaInfo
    a: float
    b: float
    speed: float = 1.0
    motion: str = "none"
    speech: bool = False
    cut: str = ""  # incoming transition chosen by the director ("none" = hard cut); "" = the intent's default
    volume: float | None = None

    @property
    def duration(self) -> float:
        return (self.b - self.a) / self.speed


def _speech_range(m: MediaInfo) -> tuple[float, float]:
    words = (m.transcript or {}).get("words", [])
    if words:
        return max(0.0, words[0]["start"] - 0.25), min(m.duration, words[-1]["end"] + 0.4)
    return 0.0, m.duration


def _speech_cut(m: MediaInfo, budget: float) -> tuple[float, float]:
    """Whole sentences from the start, as many as fit the budget (at least one)."""
    a, b = _speech_range(m)
    segs = (m.transcript or {}).get("segments", [])
    if not segs or b - a <= budget:
        return a, b
    end = segs[0]["end"]
    for seg in segs[1:]:
        if seg["end"] - a > budget:
            break
        end = seg["end"]
    return a, min(m.duration, end + 0.4)


def _interleave(groups: list[list[Shot]]) -> list[Shot]:
    """Round-robin across clips so two shots of the same clip are never adjacent if avoidable."""
    out: list[Shot] = []
    queues = [list(g) for g in groups if g]
    while queues:
        for q in list(queues):
            out.append(q.pop(0))
            if not q:
                queues.remove(q)
    return out


def _beats(ctx: VideoContext, number: int | None) -> list[float]:
    for s in ctx.songs:
        if s.number == number and s.music:
            return list(s.music.get("beats", []))
    return []


def _song(ctx: VideoContext, number: int | None) -> MediaInfo | None:
    return next((s for s in ctx.songs if s.number == number), None)


def _norm(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = float(np.percentile(values, 10)), float(np.percentile(values, 90))
    return [float(np.clip((v - lo) / (hi - lo), 0, 1)) if hi - lo > 1e-6 else 0.5 for v in values]


def _energy_curve(energy: list[float], end: float) -> list[float]:
    """Per-second loudness smoothed over a bar or two (beat-level pulsing is not the song's shape), 0..1."""
    raw = np.asarray(energy[: int(np.ceil(end)) + 4], dtype=np.float32)
    if raw.size == 0:
        return [0.5]
    k = min(5, raw.size)
    smooth = np.convolve(np.pad(raw, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    return _norm([float(v) for v in smooth])


# shot-length rhythm on top of the music's energy: steady lengths read as a slideshow
RHYTHM = (1.0, 0.75, 1.25, 0.85, 1.1, 0.7, 1.2, 0.9)


def _mean(values: list[float], a: float, b: float, default: float) -> float:
    i, j = int(a), max(int(a) + 1, int(np.ceil(b)))
    part = values[i:j]
    return float(np.mean(part)) if part else default


def _cut_times(beats: list[float], energy: list[float], target: float, base: float, song_len: float) -> tuple[list[float], set[float]]:
    """Cut points that follow the music: on beats (downbeats preferred), long shots where the song is
    quiet and shorter ones where it builds, ending on a phrase boundary. Returns (cuts incl. 0 and the
    end, phrase starts) - phrase starts are where a dissolve reads as a new chapter."""
    beat = float(np.median(np.diff(beats)))
    downbeats = beats[::4]
    phrases = beats[::16] if len(beats) > 16 else downbeats[::2]
    usable = [p for p in phrases if p >= 2 * base and p <= song_len - 0.5]
    end = min(usable, key=lambda p: abs(p - target)) if usable else min(target, song_len)
    e = _energy_curve(energy, end)
    cuts, t = [0.0], 0.0
    while True:
        level = _mean(e, t, t + base, 0.5)
        want = base * (1.4 - 0.75 * level)  # calm: ~1.4x the pace's shot, peak: ~0.65x
        want *= 1.35 if len(cuts) == 1 else RHYTHM[len(cuts) % len(RHYTHM)]  # the opening shot carries the title
        lo, hi = t + max(1.2, 2 * beat, want * 0.7), t + want * 1.3
        if end <= hi or end - (t + want) < base * 0.6:
            cuts.append(end)
            break
        near = [p for p in phrases if lo <= p <= hi] or [d for d in downbeats if lo <= d <= hi] or [b for b in beats if lo <= b <= hi]
        nxt = min(near, key=lambda x: abs(x - (t + want))) if near else t + want
        cuts.append(nxt)
        t = nxt
    return cuts, {round(p, 3) for p in phrases}


def _music_cut(intent: Intent, broll: list[MediaInfo], song: MediaInfo, target: float, base: float) -> list[Shot]:
    """Music-led montage (no speech): rhythm from the song, shots matched to its energy.

    - cut lengths vary with the music (_cut_times), hard cuts on the beat, a dissolve only where a phrase starts
    - calm, well-exposed shots open and close the film; the moving shots go where the music peaks
    - two shots of the same clip are never back to back, and every clip gets used before any repeats
    """
    music = song.music or {}
    beats = list(music.get("beats", []))
    beat = float(np.median(np.diff(beats)))
    cuts, phrase_starts = _cut_times(beats, list(music.get("energy", [])), target, base, song.duration)
    energy = _energy_curve(list(music.get("energy", [])), cuts[-1])
    all_motion = [v for m in broll for v in m.analysis.get("motion_s", [])]
    lo, hi = (float(np.percentile(all_motion, 10)), float(np.percentile(all_motion, 90))) if all_motion else (0.0, 1.0)
    used: dict[str, list[tuple[float, float]]] = {m.media_id: [] for m in broll}
    uses = dict.fromkeys(used, 0)
    seq: list[Shot] = []
    n = len(cuts) - 1
    for i in range(n):
        t0, t1 = cuts[i], cuts[i + 1]
        dur = t1 - t0
        calm = i == 0 or i == n - 1
        want_motion = 0.15 if calm else 0.15 + 0.75 * _mean(energy, t0, t1, 0.5)
        best: tuple[float, Shot] | None = None
        for m in broll:
            if seq and seq[-1].media is m and len(broll) > 1:
                continue
            speed = 0.5 if intent.slowmo else 1.0
            if m.kind == "image":
                shot, q, mot = Shot(m, 0.0, dur, motion=("zoom_in", "pan_right", "zoom_out", "pan_left")[uses[m.media_id] % 4]), 0.55, 0.3
            else:
                src = dur * speed
                if src > m.duration:
                    continue
                gap = max(1.5, src * 0.5)
                blocked = [(a - gap, b + gap) for a, b in used[m.media_id]]
                a, b = best_window(m.analysis.get("seconds", []), src, blocked, m.duration)
                if b - a < src - 0.05 or any(a < ub and b > ua for ua, ub in blocked):
                    continue
                q = _mean(m.analysis.get("seconds", []), a, b, 0.5)
                ms = m.analysis.get("motion_s")
                raw = _mean(ms, a, b, 0.5) if ms else min(1.0, m.analysis.get("motion", 0.04) / 0.08)
                mot = float(np.clip((raw - lo) / (hi - lo), 0, 1)) if hi - lo > 1e-6 else 0.5
                shot = Shot(m, a, b, speed=speed, motion="zoom_in" if raw < 0.08 else "none")
            # a second moment of one drone flight often looks like the first: use every clip before repeating
            # (long clips travel further, so their second moment is likelier to look new)
            repeat = 0.45 * min(1.5, (150 / max(m.duration, 30)) ** 0.6)
            score = q - 0.35 * abs(mot - want_motion) - repeat * uses[m.media_id]
            if len(seq) > 1 and seq[-2].media is m:
                score -= 0.15
            if best is None or score > best[0]:
                best = (score, shot)
        if best is None:
            break
        shot = best[1]
        if i > 0:
            at_phrase = any(abs(t0 - p) < 0.05 for p in phrase_starts)
            if at_phrase and intent.transition != "none":
                kind = intent.transition if intent.transition != "crossfade" else "crossfade"
                shot.cut = f"{kind}:{min(1.0, beat if intent.pace == 'slow' else beat / 2):.2f}"
            else:
                shot.cut = "none"
        shot.volume = 0.0  # drone wind and handling noise fight the score
        used[shot.media.media_id].append((shot.a, shot.b))
        uses[shot.media.media_id] += 1
        seq.append(shot)
    return seq


def direct(intent: Intent, ctx: VideoContext) -> str:
    """Expand an intent into a concrete Video Edit DSL program."""
    vis = ctx.visual
    if not vis:
        raise ValueError("import some videos or photos first")
    shot_len = PACE_SHOT.get(intent.pace, 2.4)
    beats = _beats(ctx, intent.music)
    beat = (beats[1] - beats[0]) if len(beats) > 1 else None
    if beat:
        n = max(1, round(shot_len / beat))
        if intent.pace == "fast":
            n = max(2, n) if beat < 0.45 else max(1, n)
        shot_len = n * beat

    talk = [m for m in vis if m.kind == "video" and m.speech]
    broll = [m for m in vis if m not in talk]
    material = sum(m.duration if m.kind == "video" else 3.0 for m in broll)
    # social-length defaults: viewers drop off fast, and long edits repeat the same clips
    default_len = 20.0 if intent.pace == "fast" else 40.0 if intent.pace == "medium" else 45.0
    # playbook: shot variety - more than a few moments from one clip reads as repetitive
    max_shots = 8 if len(broll) == 1 else {"fast": 4}.get(intent.pace, 3)
    # fast edits keep only the first sentences; music-led edits shouldn't stall on a long talk
    speech_budget = (intent.length or default_len) * (0.45 if intent.pace == "fast" else 0.8)
    talk_cuts = {m.media_id: _speech_cut(m, speech_budget / max(1, len(talk))) for m in talk}
    talk_len = sum(b - a for a, b in talk_cuts.values())
    target = intent.length or (talk_len + min(material * 0.7, max(default_len - talk_len, shot_len * 4)))
    broll_budget = max(0.0, target - talk_len) if talk else target

    song = _song(ctx, intent.music)
    if song and len(beats) > 16 and not talk and intent.pace != "fast" and len(broll) > 1:
        seq = _music_cut(intent, broll, song, target, PACE_SHOT.get(intent.pace, 2.4))
        if seq:
            return _emit(intent, ctx, seq, talk, music_led=True)

    # allocate b-roll shots: more shots for longer / better clips
    shots_by_media: dict[str, list[Shot]] = {}
    if broll:
        weights = []
        for m in broll:
            q = sum(m.analysis.get("seconds", [0.5])) / max(1, len(m.analysis.get("seconds", [1])))
            # sqrt: a long clip earns more shots, but not all of them (keeps variety)
            weights.append((m.duration if m.kind == "video" else 3.0) ** 0.5 * (0.5 + q))
        total_w = sum(weights) or 1.0
        motions = ["zoom_in", "pan_right", "zoom_out", "pan_left"]
        mi = 0
        for m, wgt in zip(broll, weights, strict=True):
            alloc = broll_budget * wgt / total_w
            if m.kind == "image":
                d = max(2.0, shot_len * 1.3)
                if beat:
                    d = max(beat, round(d / beat) * beat)
                shots_by_media[m.media_id] = [Shot(m, 0.0, d, motion=motions[mi % 4])]
                mi += 1
                continue
            speed = 0.5 if intent.slowmo and not m.speech else 1.0
            src_len = shot_len * speed
            # moments from one clip are spread out (a gap between them) so cuts never look like jump cuts
            gap = max(1.5, src_len)
            k = max(1, min(int((m.duration + gap) // (src_len + gap)), round(alloc / shot_len), max_shots))
            used: list[tuple[float, float]] = []
            shots = []
            for _ in range(k):
                blocked = [(ua - gap, ub + gap) for ua, ub in used]
                a, b = best_window(m.analysis.get("seconds", []), src_len, blocked, m.duration)
                if b - a < 0.3 or any(a < ub and b > ua for ua, ub in blocked):
                    break
                used.append((a, b))
                shots.append(Shot(m, a, b, speed=speed))
            shots.sort(key=lambda s: s.a)  # keep time order inside a clip
            shots_by_media[m.media_id] = shots

    # order: talking parts carry the story; b-roll as intro, then interleaved
    seq: list[Shot] = []
    broll_order = list(broll)
    if intent.highlights and not talk:
        broll_order.sort(key=lambda m: -sum(m.analysis.get("seconds", [0])) / max(1, len(m.analysis.get("seconds", [1]))))
    groups = [shots_by_media.get(m.media_id, []) for m in broll_order]
    if intent.pace == "fast" or intent.highlights:
        pool = _interleave(groups)
    else:
        pool = [s for g in groups for s in g]
    if talk:
        intro = pool[:2]
        rest = pool[2:]
        seq.extend(intro)
        for i, m in enumerate(talk):
            a, b = talk_cuts[m.media_id]
            seq.append(Shot(m, a, b, speech=True))
            share = len(rest) // max(1, len(talk) - i)
            seq.extend(rest[:share])
            rest = rest[share:]
        seq.extend(rest)
    else:
        seq = pool

    # snap cut points to beats (b-roll only; never cut a sentence)
    if beat and beats:
        t = 0.0
        for s in seq:
            end = t + s.duration
            if not s.speech:
                cand = min(beats, key=lambda x: abs(x - end))
                if abs(cand - end) < beat * 0.6 and cand - t > 0.4:
                    want = (cand - t) * s.speed
                    if s.media.kind == "image":
                        s.b = s.a + want
                    elif s.a + want <= s.media.duration:
                        s.b = s.a + want
                    elif want <= s.media.duration:
                        s.a, s.b = s.media.duration - want, s.media.duration
            t += s.duration

    # trim to target (drop trailing b-roll first)
    while len(seq) > 1 and sum(s.duration for s in seq) > target * 1.15 and not seq[-1].speech:
        seq.pop()
    return _emit(intent, ctx, seq, talk)


def _emit(intent: Intent, ctx: VideoContext, seq: list[Shot], talk: list[MediaInfo], music_led: bool = False) -> str:
    """Write the shot list (plus music, grade, titles) as a Video Edit DSL program."""
    lines = [f"# {intent.rationale}" if intent.rationale else "# edit", f"aspect {intent.aspect}"]
    aw, ah = ASPECTS[intent.aspect]
    tracks: dict[str, dict[str, Any] | None] = {}

    def follow(s: Shot) -> str:
        """Keep the subject in frame when a wide clip is cropped to a taller canvas."""
        if s.media.kind != "video" or s.media.orientation != "landscape" or aw / ah >= 1.5 or ctx.subject_of is None:
            return ""
        if s.media.media_id not in tracks:
            tracks[s.media.media_id] = ctx.subject_of(s.media.media_id)
        keys = reframe_keys(tracks[s.media.media_id], s.a, s.b, aw / ah, s.speed)
        if s.duration < 3.0 and len(keys) > 1:
            keys = [(0.0, round(sum(v for _, v in keys) / len(keys), 4))]
        return x_param(keys).strip()

    trans = intent.transition
    tdur = {"fast": 0.25, "medium": 0.5, "slow": 0.8}.get(intent.pace, 0.5)
    for i, s in enumerate(seq):
        parts = [f"clip {s.media.number}"]
        if s.media.kind == "image":
            parts.append(f"dur={s.b - s.a:.2f}")
        else:
            parts.append(f"{s.a:.2f}-{s.b:.2f}")
        if abs(s.speed - 1) > 1e-6:
            parts.append(f"speed={s.speed:g}")
        if s.motion != "none":
            parts.append(f"motion={s.motion}")
        if s.cut:
            if s.cut != "none":
                kind, _, d = s.cut.partition(":")
                parts.append(f"transition={kind}:{min(float(d), s.duration / 2):.2f}")
        elif i > 0 and trans != "none" and not (s.speech and seq[i - 1].speech):
            parts.append(f"transition={trans}:{min(tdur, s.duration / 2):.2f}")
        if s.volume is not None and s.media.kind == "video":
            parts.append(f"volume={s.volume:g}")
        elif intent.music and not s.speech and s.media.kind == "video":
            parts.append("volume=0.25")
        framing = follow(s)
        if framing:
            parts.append(framing)
        lines.append(" ".join(parts))
    if intent.music:
        # music-led: the film ends on a phrase, so let the song breathe out over the last shot
        fade = min(4.0, seq[-1].duration * 0.8) if music_led else 1.5 if intent.pace != "fast" else 0.8
        lines.append(f"music {intent.music} volume=0.85 fade={fade:.1f}" if music_led else f"music {intent.music} volume=0.8 fade={fade}")
    if intent.look:
        lines.append(f"look {intent.look} s={intent.strength:g}")
    lines.extend(intent.finishing)
    if intent.letterbox and intent.aspect == "16:9":
        lines.append(f"letterbox {intent.letterbox:g}")
    if intent.pace != "fast":
        lines.append(f"fade_in {0.6 if intent.pace == 'medium' else 1.0}")
    lines.append(f"fade_out {min(2.5, seq[-1].duration * 0.5):.1f}" if music_led else f"fade_out {0.6 if intent.pace == 'fast' else 1.2}")
    if intent.title:
        title = intent.title.replace('"', "'")
        style = intent.title_style
        if music_led:  # the title rides the opening shot, in after the fade-in, out before the first cut
            at = 1.0
            dur = max(2.0, min(4.5, seq[0].duration - at - 0.4))
            lines.append(f'title "{title}" at={at:.1f} dur={dur:.1f} style={style}')
        else:
            lines.append(f'title "{title}" at=0.3 dur={2.2 if intent.pace == "fast" else 3.2} style={style}' + (" pos=center" if style == "bold_pop" else ""))
    if intent.captions and talk:
        lines.append(f"captions {intent.captions}")
    return "\n".join(lines)


# ----------------------------------------------------------------- the model

VIDEO_PLAYBOOK = """Video editing rules:
- Pace: reels/tiktok/energetic -> fast (hard cuts on the beat). Cinematic/wedding/calm -> slow (crossfades). Otherwise medium.
- Transitions: use one style for the whole edit. Fast edits use none (hard cuts).
- Pick ONE look that matches the mood; s=0.6-1.0 for video. Cinematic -> teal_orange_cinematic and letterbox 2.39.
- Captions only when the user asks and someone speaks; bold_pop for reels, clean or cinematic otherwise.
- Use music when there is a music file and the user wants music or nobody speaks.
- Aspect: reels/tiktok/stories -> 9:16, youtube/cinematic -> 16:9, instagram post -> 4:5."""

VIDEO_EXAMPLES = [
    (
        "media:\n#1 beach.mp4: video 12.0s landscape; ok\n#2 city.mp4: video 9.5s landscape; lots of motion\n#3 song.mp3: music 95s, 124 bpm\ncurrent edit: empty timeline\nhints: allowed looks: vivid_pop, golden_hour; aspect: 9:16 requested; captions: no speech\nrequest: make an energetic reel with music",
        "# upbeat reel: hard cuts on the beat, punchy color\naspect 9:16\npace fast\ntransition none\nlook vivid_pop s=0.8\nmusic 3",
    ),
    (
        "media:\n#1 talk.mp4: video 40.0s landscape; static shot; speech: \"so today we are going to the old town\"\n#2 street.mp4: video 15.0s landscape; ok\ncurrent edit: empty timeline\nhints: allowed looks: teal_orange_cinematic, film_portra; aspect: any; captions: allowed\nrequest: cinematic vlog with subtitles",
        "# cinematic vlog: slow crossfades, movie grade, bars, readable subtitles\naspect 16:9\npace slow\ntransition crossfade\nlook teal_orange_cinematic s=0.8\ngrain 0.12\nletterbox 2.39\ncaptions cinematic",
    ),
]


def _intent_grammar(h: VideoHints, ctx: VideoContext) -> str:
    def lit(s: str) -> str:
        return '"' + s.replace('"', '\\"') + '"'

    looks = h.looks or SAFE_VIDEO_LOOKS  # vague request: crowd-pleasers only
    aspects = (h.aspect,) if h.aspect else tuple(ASPECTS)[:4]
    paces = (h.fixed_pace,) if h.fixed_pace else tuple(PACE_SHOT)
    songs = [h.music] if h.music else []  # the rules decide whether music fits
    speech = any(m.speech for m in ctx.visual)
    lines = [
        'root ::= rationale nl aspect nl pace nl trans (nl look)? (nl fin){0,2}' + (" (nl lbox)?" if h.letterbox else "") + (" (nl caps)?" if speech and h.captions else "")
        + (" (nl title)?" if h.title else "") + (" (nl music)?" if songs else "") + " nl?",
        'rationale ::= "# " [^\\n]{3,120}',
        'nl ::= "\\n"',
        "aspect ::= \"aspect \" (" + " | ".join(lit(a) for a in aspects) + ")",
        "pace ::= \"pace \" (" + " | ".join(lit(x) for x in paces) + ")",
        "trans ::= \"transition \" (" + " | ".join(lit(t) for t in ("none",) + TRANSITIONS[:4]) + ")",
        "look ::= \"look \" (" + " | ".join(lit(n) for n in looks) + ') " s=" ("0." [4-9] | "1" ("." [0-2])?)',
        'fin ::= ("grain 0." [0-3] [0-9] | "vignette 0." [1-4] | "bloom 0." [1-3])',
        'lbox ::= "letterbox 2.39"',
        'caps ::= "captions " (' + " | ".join(lit(s) for s in CAPTION_STYLES if s != "title") + ")",
    ]
    if h.title:
        lines.append("title ::= " + lit(f'title "{h.title}"'))
    if songs:
        lines.append('music ::= "music " (' + " | ".join(lit(str(n)) for n in songs) + ")")
    return "\n".join(lines) + "\n"


def parse_intent(text: str, h: VideoHints, ctx: VideoContext) -> Intent:
    base = intent_from_hints(h, ctx)
    it = replace(base, look=None, finishing=[], letterbox=None, captions=None, music=None)
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("#"):
            it.rationale = line.lstrip("# ").strip()
            continue
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == "aspect" and len(parts) > 1 and parts[1] in ASPECTS:
            it.aspect = parts[1]
        elif key == "pace" and len(parts) > 1 and parts[1] in PACE_SHOT:
            it.pace = parts[1]
        elif key == "transition" and len(parts) > 1:
            it.transition = parts[1] if parts[1] in TRANSITIONS + ("none",) else it.transition
        elif key == "look" and len(parts) > 1 and parts[1] in load_recipes():
            it.look = parts[1]
            m = re.search(r"s=([\d.]+)", line)
            it.strength = float(m.group(1)) if m else 1.0
        elif key in ("grain", "vignette", "bloom"):
            it.finishing.append(line)
        elif key == "letterbox":
            it.letterbox = 2.39 if h.letterbox else None
        elif key == "captions" and len(parts) > 1:
            it.captions = parts[1]
        elif key == "music" and len(parts) > 1 and parts[1].isdigit():
            it.music = int(parts[1]) if h.music else None
    it.title_style = "bold_pop" if it.pace == "fast" else "title"
    if any(it.rationale and it.rationale in ans for _, ans in VIDEO_EXAMPLES):
        it.rationale = base.rationale  # the small model parroted an example; describe the plan instead
    # rules the model must not override
    if h.music and it.music is None and wants(MUSIC_WORDS, h.text):
        it.music = h.music
    if h.captions and it.captions is None:
        it.captions = h.captions
    if it.pace == "fast" and it.transition == "crossfade" and h.transition == "none":
        it.transition = "none"
    return it


def _llm_intents(h: VideoHints, ctx: VideoContext, prompt: str, n: int) -> list[Intent]:
    hint_line = "hints: " + "; ".join([
        "allowed looks: " + ", ".join(h.looks) if h.looks else "looks: any that fits, or none",
        f"aspect: {h.aspect} requested" if h.aspect else "aspect: any",
        ("captions: allowed" if h.captions else "captions: not requested") if any(m.speech for m in ctx.visual) else "captions: no speech",
    ])
    sys_prompt = (
        "You are EditLM, the editing brain of a video editor. Read the media list and the request, then write a short edit "
        "plan: first '# ' + reason, then aspect, pace, transition, optional look/grain/letterbox/captions/title/music lines.\n\n"
        + VIDEO_PLAYBOOK
    )
    msgs = [{"role": "system", "content": sys_prompt}]
    for user, answer in VIDEO_EXAMPLES:
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    msgs.append({"role": "user", "content": f"{video_brief(ctx)}\n{hint_line}\nrequest: {prompt}"})
    grammar = _intent_grammar(h, ctx)
    out = []
    for i, temp in enumerate([0.2, 0.8, 1.1][:n]):
        text = LLM.chat(msgs, grammar=grammar, temperature=temp, seed=4321 + i, max_tokens=220)
        out.append(parse_intent(text, h, ctx))
    return out


def plan_video(
    prompt: str, ctx: VideoContext, variations: int = 3, use_llm: bool = True,
    style: str | None = None, output: Any = None, styles_dir: Any = None,
) -> dict[str, Any]:
    """style / output are the optional prompt options (agent/options.py): explicit
    choices are hard constraints, the prompt fills whatever is left on Auto."""
    from .options import apply_video_options

    h = apply_video_options(video_hints(prompt, ctx), ctx, style, output, styles_dir)
    if output is not None and getattr(output, "versions", None):
        variations = output.versions
    intents: list[tuple[Intent, str]] = []
    llm_error = None
    if use_llm and LLM.available():
        try:
            intents = [(it, "llm") for it in _llm_intents(h, ctx, prompt, variations)]
        except LLMUnavailable as exc:
            llm_error = str(exc)
    # rule variations: matched look / second look / other pace
    alt_pace = h.fixed_pace or {"fast": "medium", "medium": "fast" if h.music else "slow", "slow": "medium"}[h.pace]
    rules = [intent_from_hints(h, ctx), intent_from_hints(h, ctx, 1), intent_from_hints(h, ctx, 0, alt_pace), intent_from_hints(h, ctx, 2)]
    if not h.looks:
        safe = ("vivid_pop", "film_portra", "golden_hour") if h.pace != "slow" else ("film_portra", "teal_orange_cinematic", "faded_film")
        rules = [replace(rules[0])] + [replace(rules[0], look=s, strength=0.7, rationale=rules[0].rationale + f", {load_recipes()[s].title} look") for s in safe]
    intents += [(r, "rules") for r in rules]

    results, seen = [], set()
    for it, source in intents:
        try:
            program = direct(it, ctx)
        except ValueError as exc:
            return {"variations": [], "brief": video_brief(ctx), "llm_error": llm_error, "error": str(exc)}
        body = "\n".join(ln for ln in program.splitlines() if not ln.startswith("#"))
        if body in seen:
            continue
        seen.add(body)
        results.append({"program": program, "rationale": it.rationale, "source": source, "intent": it.to_program(), "warnings": []})
        if len(results) >= variations:
            break
    return {"variations": results, "brief": video_brief(ctx), "llm_error": llm_error}
