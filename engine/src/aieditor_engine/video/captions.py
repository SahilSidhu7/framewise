"""Transcripts -> timed captions on the timeline.

Transcripts are in source time, per media: {"words": [{"start","end","text"}],
"segments": [...]}. Words are mapped through every clip that uses the media
(trim + speed), then grouped into readable captions following the playbook:
at most ~42 characters a line and 2 lines, break on punctuation and pauses,
short punchy chunks for the bold_pop style.
"""

from __future__ import annotations

from typing import Any

from ..core.project import Caption, MediaKind, Project, new_id

LIMITS = {  # style -> (max chars per caption, max seconds)
    "bold_pop": (16, 1.6),
    "minimal": (60, 4.0),
}
DEFAULT_LIMIT = (80, 4.5)
PAUSE = 0.45


def _words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words = [w for w in transcript.get("words", []) if str(w.get("text", "")).strip()]
    if words:
        return words
    out = []
    for seg in transcript.get("segments", []):
        toks = str(seg.get("text", "")).split()
        if not toks:
            continue
        span = max(0.05, float(seg["end"]) - float(seg["start"]))
        total = sum(len(t) + 1 for t in toks)
        t = float(seg["start"])
        for tok in toks:
            d = span * (len(tok) + 1) / total
            out.append({"start": t, "end": t + d, "text": tok})
            t += d
    return out


def timeline_words(p: Project, transcripts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for track in p.tracks:
        if track.kind == "audio":
            continue
        for clip in track.clips:
            item = p.media.get(clip.media_id)
            if item is None or item.kind != MediaKind.VIDEO or clip.muted or clip.media_id not in transcripts:
                continue
            for w in _words(transcripts[clip.media_id]):
                s, e = float(w["start"]), float(w["end"])
                mid = (s + e) / 2
                if not (clip.in_point <= mid < clip.out_point):
                    continue
                ts = clip.start + (max(s, clip.in_point) - clip.in_point) / clip.speed
                te = clip.start + (min(e, clip.out_point) - clip.in_point) / clip.speed
                out.append({"start": ts, "end": max(te, ts + 0.05), "text": str(w["text"]).strip()})
    out.sort(key=lambda w: w["start"])
    return out


def group_words(words: list[dict[str, Any]], style: str) -> list[tuple[float, float, str]]:
    max_chars, max_dur = LIMITS.get(style, DEFAULT_LIMIT)
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for w in words:
        if cur:
            text_len = sum(len(x["text"]) + 1 for x in cur) + len(w["text"])
            gap = w["start"] - cur[-1]["end"]
            if text_len > max_chars or w["end"] - cur[0]["start"] > max_dur or gap > PAUSE or cur[-1]["text"][-1:] in ".?!":
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    out = []
    for i, g in enumerate(groups):
        start, end = g[0]["start"], g[-1]["end"]
        nxt = groups[i + 1][0]["start"] if i + 1 < len(groups) else end + 1.0
        end = min(max(end, start + 0.7), nxt)  # minimum reading time, never overlap the next
        text = " ".join(x["text"] for x in g).strip()
        if text:
            out.append((start, max(end, start + 0.2), text))
    return out


def captions_from_transcripts(p: Project, transcripts: dict[str, dict[str, Any]], style: str) -> list[Caption]:
    words = timeline_words(p, transcripts)
    return [
        Caption(id=new_id("cap"), start=round(s, 3), end=round(e, 3), text=text[:300], style=style, kind="caption")  # type: ignore[arg-type]
        for s, e, text in group_words(words, style)
    ]
