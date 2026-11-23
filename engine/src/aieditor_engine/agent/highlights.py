"""Find the best moments in a long video.

Signals, fused per candidate window:
- what is said: every sentence gets a hook score from cheap text features
  (question, "you", numbers, strong words, contrast, punchy length) and,
  when the local model is running, a 1-5 rating from it;
- how it sounds: loudness per second (laughter, cheering, raised voices);
- how it looks: the per-second visual interest from the video analysis.

A moment always starts on a sentence start and ends on a sentence end, so a
clip never begins or ends mid-thought. Videos without speech fall back to
sound + picture only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .llm import LLM, LLMUnavailable
from .media_context import MediaInfo

log = logging.getLogger(__name__)

STRONG = (
    r"\b(never|always|best|worst|secret|mistake|truth|nobody|everyone|amazing|incredible|crazy|insane|love|hate|"
    r"biggest|important|surprising|shocking|huge|wrong|problem|imagine|remember|finally|actually|literally|"
    r"changed|discovered|realized|learned|genius|impossible|dangerous|free|money|first|only)\b"
)
OPENERS = r"^(here'?s|this is|the (reason|problem|truth|secret|thing)|what if|did you know|let me|i (think|believe|learned|realized)|one of|most people|imagine|stop)\b"
WEAK_START = r"^(and|but|so|or|because|which|that|then|also|um+|uh+|er+|well|yeah|okay|ok|like)\b"
FILLER_ONLY = r"^(um+|uh+|er+|ah+|hmm+|yeah|okay|ok|right|so|well)[.,!?]*$"


@dataclass
class Moment:
    media_number: int
    media_id: str
    start: float
    end: float
    score: float  # 0..1, comparable within one video
    title: str
    reason: str
    text: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["start"], d["end"], d["score"] = round(self.start, 2), round(self.end, 2), round(self.score, 3)
        return d


def sentences(transcript: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Sentences with timings: whisper segments, split further at . ? ! and long pauses."""
    if not transcript:
        return []
    words = [w for w in transcript.get("words", []) if str(w.get("text", "")).strip()]
    if not words:
        return [dict(s) for s in transcript.get("segments", []) if str(s.get("text", "")).strip()]
    out: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    for w in words:
        if cur and (float(w["start"]) - float(cur[-1]["end"]) > 0.8 or str(cur[-1]["text"]).rstrip()[-1:] in ".?!"):
            out.append({"start": float(cur[0]["start"]), "end": float(cur[-1]["end"]), "text": " ".join(str(x["text"]).strip() for x in cur)})
            cur = []
        cur.append(w)
    if cur:
        out.append({"start": float(cur[0]["start"]), "end": float(cur[-1]["end"]), "text": " ".join(str(x["text"]).strip() for x in cur)})
    return out


def text_score(text: str) -> tuple[float, list[str]]:
    """0..1 hook score of one sentence and the reasons behind it."""
    t = text.strip()
    low = t.lower()
    n = len(low.split())
    if n == 0 or re.match(FILLER_ONLY, low):
        return 0.0, []
    score, why = 0.3, []
    if t.endswith("?") or re.match(r"^(why|how|what|who|when|where|do you|have you|did you|can you|would you)\b", low):
        score += 0.22
        why.append("asks a question")
    if re.search(r"\b(you|your|you're)\b", low):
        score += 0.1
        why.append("speaks to the viewer")
    if re.search(r"\b\d+([.,]\d+)?\s*(%|percent|x|times|years?|days?|dollars?|million|billion|k)?\b", low) and re.search(r"\d", low):
        score += 0.1
        why.append("has a number")
    strong = len(re.findall(STRONG, low))
    if strong:
        score += min(0.2, 0.1 * strong)
        why.append("strong wording")
    if re.match(OPENERS, low):
        score += 0.15
        why.append("opens like a hook")
    if re.search(r"\b(but|however|instead|turns out|except)\b", low):
        score += 0.06
        why.append("has a twist")
    if t.endswith("!"):
        score += 0.06
    if 5 <= n <= 18:
        score += 0.08
    elif n < 3:
        score -= 0.15
    elif n > 40:
        score -= 0.1
    if re.match(WEAK_START, low):
        score -= 0.15
    return float(min(1.0, max(0.0, score))), why


def _llm_ratings(sents: list[dict[str, Any]]) -> list[float] | None:
    """One 1-5 rating per sentence from the local model (None if it is not running)."""
    if not sents or not LLM.available():
        return None
    sents = sents[:60]
    listing = "\n".join(f"{i + 1}. {s['text'][:160]}" for i, s in enumerate(sents))
    msgs = [
        {"role": "system", "content": (
            "You pick the lines of a talk that would make people stop scrolling. Rate each numbered line from 1 (boring, "
            "filler, needs context) to 5 (surprising, emotional, funny, or a strong claim that works on its own). "
            "Answer with one digit per line, in order, nothing else."
        )},
        {"role": "user", "content": listing},
    ]
    grammar = "root ::= " + " ".join(['[1-5] "\\n"'] * len(sents)) + "\n"
    try:
        text = LLM.chat(msgs, grammar=grammar, temperature=0.1, seed=7, max_tokens=len(sents) * 2 + 8)
    except LLMUnavailable as exc:
        log.warning("highlight rating skipped: %s", exc)
        return None
    digits = [int(c) for c in text if c in "12345"]
    if len(digits) < len(sents):
        return None
    return [(d - 1) / 4 for d in digits[: len(sents)]]


def _norm(values: list[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=np.float32)
    if a.size == 0:
        return a
    lo, hi = float(np.percentile(a, 10)), float(np.percentile(a, 95))
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def _mean(arr: np.ndarray, a: float, b: float, default: float = 0.5) -> float:
    if arr.size == 0:
        return default
    i, j = int(max(0, a)), int(min(arr.size, max(a + 1, np.ceil(b))))
    return float(arr[i:j].mean()) if j > i else default


def _title(text: str) -> str:
    words = re.sub(r"[^\w\s'-]", "", text).split()
    title = " ".join(words[:7])
    return (title[:1].upper() + title[1:] + ("…" if len(words) > 7 else "")) if title else "Highlight"


def find_moments(m: MediaInfo, length: float = 30.0, count: int = 3, use_llm: bool = True) -> list[Moment]:
    """Up to `count` non-overlapping moments of about `length` seconds, best first."""
    length = float(min(max(3.0, length), max(3.0, m.duration)))
    visual = _norm(m.analysis.get("seconds", []))
    energy = _norm(m.audio.get("energy", []))
    sents = sentences(m.transcript) if m.speech else []
    cands = _speech_windows(m, sents, length, visual, energy, use_llm) if len(sents) >= 2 else _quiet_windows(m, length, visual, energy)
    cands.sort(key=lambda c: -c.score)
    picked: list[Moment] = []
    for c in cands:
        if all(min(c.end, p.end) - max(c.start, p.start) <= 0.2 * (c.end - c.start) for p in picked):
            picked.append(c)
        if len(picked) >= count:
            break
    return picked


def _speech_windows(
    m: MediaInfo, sents: list[dict[str, Any]], length: float, visual: np.ndarray, energy: np.ndarray, use_llm: bool
) -> list[Moment]:
    scored = [text_score(s["text"]) for s in sents]
    base = np.array([s for s, _ in scored], dtype=np.float32)
    ratings = _llm_ratings(sents) if use_llm else None
    if ratings is not None:
        r = np.array(ratings + [0.5] * (len(sents) - len(ratings)), dtype=np.float32)
        base = 0.5 * base + 0.5 * r
    out: list[Moment] = []
    lo, hi = length * 0.6, length * 1.15
    for i, first in enumerate(sents):
        if base[i] < 0.2:
            continue  # never open on filler
        best_j, best_fit = None, 1e9
        for j in range(i, len(sents)):
            d = sents[j]["end"] - first["start"]
            if d > hi:
                break
            if d >= lo or j == len(sents) - 1:
                ends_clean = str(sents[j]["text"]).rstrip()[-1:] in ".?!"
                fit = abs(d - length) - (0.15 * length if ends_clean else 0)
                if fit < best_fit:
                    best_j, best_fit = j, fit
        if best_j is None:
            continue
        a = max(0.0, first["start"] - 0.2)
        b = min(m.duration, sents[best_j]["end"] + 0.35)
        if b - a < min(lo, 3.0):
            continue
        body = float(base[i : best_j + 1].mean())
        e, v = _mean(energy, a, b), _mean(visual, a, b)
        score = 0.42 * float(base[i]) + 0.33 * body + 0.15 * e + 0.10 * v
        why = list(scored[i][1])
        if ratings is not None and i < len(ratings) and ratings[i] >= 0.75:
            why.insert(0, "the AI rated the opening line highly")
        if e > 0.7:
            why.append("loud, lively delivery")
        if not why:
            why.append("a complete thought that stands on its own")
        text = " ".join(str(s["text"]) for s in sents[i : best_j + 1])
        out.append(Moment(m.number, m.media_id, a, b, float(min(1.0, score)), _title(first["text"]), "; ".join(why[:3]), text[:400]))
    return out


def _quiet_windows(m: MediaInfo, length: float, visual: np.ndarray, energy: np.ndarray) -> list[Moment]:
    """No speech: slide a window over picture interest + loudness."""
    n = max(visual.size, energy.size, int(m.duration))
    if n == 0:
        return [Moment(m.number, m.media_id, 0.0, min(length, m.duration), 0.3, "Highlight", "the opening", "")]

    def pad(a: np.ndarray) -> np.ndarray:
        return np.pad(a, (0, n - a.size), constant_values=float(a.mean()) if a.size else 0.5) if a.size < n else a[:n]

    both = 0.6 * pad(visual) + 0.4 * pad(energy)
    peak = pad(energy)
    out: list[Moment] = []
    step = max(0.5, length / 6)
    t = 0.0
    while t + length <= m.duration + 1e-6:
        a, b = t, min(m.duration, t + length)
        loud = _mean(peak, a, b) > 0.7
        out.append(Moment(m.number, m.media_id, round(a, 2), round(b, 2), _mean(both, a, b), "Highlight",
                          "loudest part" if loud else "liveliest picture", ""))
        t += step
    if not out:
        out.append(Moment(m.number, m.media_id, 0.0, m.duration, _mean(both, 0, m.duration), "Highlight", "the whole clip", ""))
    return out
