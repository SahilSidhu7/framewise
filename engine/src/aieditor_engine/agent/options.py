"""Prompt options: editing style ("how it should feel") and desired output
("what comes out"), chosen next to the prompt box or sent by an automation.

Both are optional. An explicit option is a hard constraint for the planner;
whatever is left on Auto is decided from the prompt and the media, as before.
Presets are YAML in ../styles, so adding one needs no code.
"""

from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, Field

from ..imaging.recipes import load_recipes

if TYPE_CHECKING:
    from .planner import Hints
    from .video_planner import VideoContext, VideoHints

STYLE_DIR = Path(__file__).resolve().parent.parent / "styles"
PACES = ("fast", "medium", "slow")
MUSIC_MODES = ("auto", "on", "off")


class OutputOptions(BaseModel):
    """Desired output, as sent by the UI chips or an automation."""

    platform: str | None = Field(None, description="Key from /ai/options, e.g. tiktok, youtube, instagram_post")
    length: float | None = Field(None, gt=0, le=3600, description="Target length in seconds (video)")
    versions: int | None = Field(None, ge=1, le=5, description="How many alternatives to return")
    captions: bool | None = Field(None, description="Force captions on or off (video)")
    music: bool | None = Field(None, description="Force music on or off (video)")
    quality: Literal["draft", "standard", "high"] | None = None


class OptionError(ValueError):
    pass


def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((STYLE_DIR / name).read_text(encoding="utf-8")) or {}


def _clean_video_style(key: str, raw: dict[str, Any]) -> dict[str, Any]:
    looks = [n for n in raw.get("looks", []) if n in load_recipes()]
    pace = raw.get("pace", "medium")
    music = str(raw.get("music", "auto")).lower()
    return {
        "id": key,
        "title": raw.get("title", key.replace("_", " ").title()),
        "description": raw.get("description", ""),
        "pace": pace if pace in PACES else "medium",
        "transition": raw.get("transition", "none" if pace == "fast" else "crossfade"),
        "looks": looks,
        "captions": raw.get("captions"),
        "letterbox": raw.get("letterbox"),
        "highlights": bool(raw.get("highlights", False)),
        "music": music if music in MUSIC_MODES else "auto",
        "aspect": raw.get("aspect"),
        "length": raw.get("length"),
        "custom": bool(raw.get("custom", False)),
    }


@lru_cache(maxsize=1)
def _builtin_video_styles() -> dict[str, dict[str, Any]]:
    return {k: _clean_video_style(k, v) for k, v in _load("video.yaml").items()}


def video_styles(extra_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Built-in styles plus the project's saved ones (<project>/styles/*.yaml)."""
    styles = dict(_builtin_video_styles())
    if extra_dir and extra_dir.is_dir():
        for path in sorted(extra_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(raw, dict):
                styles[path.stem] = _clean_video_style(path.stem, {**raw, "custom": True})
    return styles


@lru_cache(maxsize=1)
def photo_styles() -> dict[str, dict[str, Any]]:
    out = {}
    for key, raw in _load("photo.yaml").items():
        out[key] = {
            "id": key, "title": raw.get("title", key), "description": raw.get("description", ""),
            "looks": [n for n in raw.get("looks", []) if n in load_recipes()], "words": raw.get("words", ""),
        }
    return out


@lru_cache(maxsize=1)
def outputs() -> dict[str, dict[str, dict[str, Any]]]:
    raw = _load("outputs.yaml")
    return {kind: {k: {"id": k, **v} for k, v in raw.get(kind, {}).items()} for kind in ("video", "photo")}


def catalog(extra_dir: Path | None = None) -> dict[str, Any]:
    """Everything the UI needs to draw the option chips."""
    return {
        "video": {"styles": list(video_styles(extra_dir).values()), "outputs": list(outputs()["video"].values())},
        "photo": {"styles": list(photo_styles().values()), "outputs": list(outputs()["photo"].values())},
        "lengths": [15, 30, 60],
    }


# --------------------------------------------------------------------- video


def apply_video_options(
    h: VideoHints, ctx: VideoContext, style: str | None, output: OutputOptions | None, extra_dir: Path | None = None
) -> VideoHints:
    """Overlay the chosen options on the hints read from the prompt."""
    if style and style != "auto":
        styles = video_styles(extra_dir)
        if style not in styles:
            raise OptionError(f"unknown style `{style}`; choose one of {', '.join(styles)}")
        st = styles[style]
        speech = any(m.speech for m in ctx.visual)
        songs = ctx.songs
        music = h.music
        if st["music"] == "off":
            music = None
        elif st["music"] == "on" and songs:
            music = songs[0].number
        aspect = h.aspect or st["aspect"]
        h = replace(
            h,
            pace=st["pace"],
            fixed_pace=st["pace"],
            transition=st["transition"],
            # words in the prompt may still pick among the style's looks; otherwise the style's order wins
            looks=tuple([n for n in h.looks if n in st["looks"]] + [n for n in st["looks"] if n not in h.looks]) or h.looks,
            captions=(st["captions"] or h.captions) if (h.captions and speech) else h.captions,
            letterbox=st["letterbox"],
            highlights=h.highlights or st["highlights"],
            music=music,
            aspect=aspect,
            length=h.length or st["length"],
            style_captions=st["captions"],
        )
    if output:
        if output.platform:
            plats = outputs()["video"]
            if output.platform not in plats:
                raise OptionError(f"unknown output `{output.platform}`; choose one of {', '.join(plats)}")
            plat = plats[output.platform]
            h = replace(h, aspect=plat.get("aspect") or h.aspect)
            cap = plat.get("max_length")
            if cap and (h.length or 0) > cap:
                h = replace(h, length=float(cap))
            if cap and h.length is None and output.length is None and h.highlights:
                h = replace(h, length=float(plat.get("default_length") or cap))
        if output.length:
            h = replace(h, length=float(output.length))
        if output.captions is not None:
            speech = any(m.speech for m in ctx.visual)
            default = h.style_captions or ("bold_pop" if h.pace == "fast" else "clean")
            h = replace(h, captions=(h.captions or default) if (output.captions and speech) else None)
        if output.music is not None:
            h = replace(h, music=(h.music or (ctx.songs[0].number if ctx.songs else None)) if output.music else None)
    if h.letterbox and (h.aspect or "16:9") != "16:9":
        h = replace(h, letterbox=None)  # bars only make sense on widescreen
    return h


def style_from_reference(analysis: dict[str, Any], title: str) -> dict[str, Any]:
    """"Edit like this": measure a reference video and describe it as a style.

    Uses what the analysis already has: cut rhythm, brightness, colourfulness, shape."""
    duration = float(analysis.get("duration") or 0)
    cuts = analysis.get("cuts", [])
    shot = duration / (len(cuts) + 1) if duration else 3.0
    pace = "fast" if shot < 1.8 else "medium" if shot < 3.8 else "slow"
    colour = float(analysis.get("colorfulness", 0.3))
    bright = float(analysis.get("brightness", 0.5))
    if colour < 0.06:
        looks = ["bw_classic", "noir"]
    elif bright < 0.3:
        looks = ["moody_dark", "teal_orange_cinematic"]
    elif colour > 0.45:
        looks = ["vivid_pop", "golden_hour"]
    elif colour < 0.2:
        looks = ["faded_film", "film_portra"]
    elif bright > 0.62:
        looks = ["bright_airy", "aesthetic_soft"]
    else:
        looks = ["film_portra", "vivid_pop"]
    size = analysis.get("size") or [16, 9]
    ratio = size[0] / max(1, size[1])
    aspect = "9:16" if ratio < 0.7 else "4:5" if ratio < 0.9 else "1:1" if ratio < 1.15 else "16:9"
    return {
        "title": title[:40] or "My style",
        "description": f"Like the reference: {pace} pace (about {shot:.1f} s per shot), {load_recipes()[looks[0]].title} colour, {aspect}.",
        "pace": pace,
        "transition": "none" if pace == "fast" else "crossfade",
        "looks": looks,
        "captions": "bold_pop" if pace == "fast" else "clean",
        "highlights": pace == "fast",
        "music": "auto",
        "aspect": aspect,
        "length": round(min(duration, 180.0)) if 5 <= duration <= 180 else None,
    }


def save_style(folder: Path, name: str, style: dict[str, Any]) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40] or "my_style"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{key}.yaml").write_text(yaml.safe_dump(style, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return key


# --------------------------------------------------------------------- photo


def apply_photo_options(h: Hints, style: str | None, output: OutputOptions | None) -> Hints:
    if style and style != "auto":
        styles = photo_styles()
        if style not in styles:
            raise OptionError(f"unknown style `{style}`; choose one of {', '.join(styles)}")
        st = styles[style]
        if not st["looks"]:
            h = replace(h, looks=(), fix_only=True)
        else:
            picked = [n for n in (h.looks or ()) if n in st["looks"]]
            h = replace(h, looks=tuple(picked + [n for n in st["looks"] if n not in picked]), fix_only=False)
        if st["words"]:
            h = replace(h, text=h.text + st["words"] + " ")  # direction words (e.g. dark) the sanitizer reads
    if output and output.platform:
        plats = outputs()["photo"]
        if output.platform not in plats:
            raise OptionError(f"unknown output `{output.platform}`; choose one of {', '.join(plats)}")
        aspect = plats[output.platform].get("aspect")
        if aspect:
            h = replace(h, aspect=aspect)
    return h


def photo_rule_words(prompt: str, style: str | None, output: OutputOptions | None) -> str:
    """The same options as plain words, for the rule-based planner (which reads text)."""
    words: list[str] = []
    if style and style != "auto":
        st = photo_styles().get(style)
        if st is None:
            raise OptionError(f"unknown style `{style}`; choose one of {', '.join(photo_styles())}")
        if not st["looks"]:
            words.append("fix")
        else:
            from .planner import _match_looks

            if not any(n in st["looks"] for n in _match_looks(" " + prompt.lower() + " ")):
                words.append(st["looks"][0].replace("_", " "))
        if st["words"]:
            words.append(st["words"])
    if output and output.platform:
        aspect = outputs()["photo"].get(output.platform, {}).get("aspect")
        if aspect:
            words.append(f"crop {aspect}")
    return (" " + " ".join(words)) if words else ""
