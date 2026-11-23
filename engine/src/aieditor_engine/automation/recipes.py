"""Recipes: what an automation job can do.

Each recipe takes the job's input files (numbered 1..n in the order given, the
same numbers the Edit DSL uses), the prompt options (style + output) and its
own small options model, and writes its results into the job's `out/` folder.

    reel            one long video -> N ranked vertical clips (highlights + reframe + captions)
    highlights      the best moments as JSON, optionally each cut as a clip
    captions        burned-in captions + .srt subtitles
    photo_edit      prompt (+ style/output) -> edited photos, N versions each
    render_program  run a Video or Photo Edit DSL program you wrote
    transcribe      speech -> .srt, .txt and word-timed .json
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..agent.options import OptionError, OutputOptions
from ..core.project import MediaKind, ProjectKind
from ..video.vdsl import ASPECTS, CAPTION_STYLES

if TYPE_CHECKING:
    from ..session import Session


class RecipeError(ValueError):
    """Bad request for this recipe (reported to the caller as a 422 / failed job)."""


class Cancelled(Exception):
    pass


# ------------------------------------------------------------------ options
class ReelOptions(BaseModel):
    count: int = Field(3, ge=1, le=10, description="How many reels")
    length: float | None = Field(None, ge=5, le=180, description="Target seconds per reel (default from style/platform)")
    captions: bool | None = None
    tighten: bool = Field(True, description="Cut long pauses and filler words")
    source: int | None = Field(None, ge=1, description="Which input is the long video (default: longest with speech)")
    progress_bar: bool = Field(True, description="Thin progress bar at the top")
    end_card: str | None = Field(None, max_length=120, description="Text on a 1.5 s closing card, e.g. 'Follow for part 2'")


class HighlightOptions(BaseModel):
    count: int = Field(5, ge=1, le=20)
    length: float = Field(30, ge=3, le=600)
    clips: bool = Field(True, description="Also export each moment as a clip")
    source: int | None = Field(None, ge=1)


class CaptionOptions(BaseModel):
    style: str = Field("bold_pop", description=f"One of {', '.join(CAPTION_STYLES)}")
    burn: bool = Field(True, description="Burn captions into the video (else only .srt)")
    source: int | None = Field(None, ge=1)


class PhotoEditOptions(BaseModel):
    format: Literal["jpeg", "png", "webp", "tiff"] = "jpeg"
    quality: int = Field(92, ge=50, le=100)
    long_edge: int | None = Field(None, ge=64, le=20000)


class RenderOptions(BaseModel):
    program: str = Field(min_length=1, max_length=50000, description="Video Edit DSL, or Photo Edit DSL for photo inputs")
    name: str = Field("render", pattern=r"^[\w\- ]{1,60}$")
    format: Literal["jpeg", "png", "webp", "tiff"] = "jpeg"


class TranscribeOptions(BaseModel):
    pass


@dataclass
class Recipe:
    name: str
    title: str
    description: str
    options: type[BaseModel]
    run: Callable[[JobContext], dict[str, Any]]
    needs_prompt: bool = False
    media: str = "video"  # video | photo | any


# ------------------------------------------------------------------ context
@dataclass
class JobContext:
    session: Session
    inputs: list[Path]
    out: Path
    options: Any
    prompt: str | None = None
    style: str | None = None
    output: OutputOptions | None = None
    report: Callable[[float, str], None] = lambda frac, stage: None
    cancelled: Callable[[], bool] = lambda: False
    outputs: list[dict[str, Any]] = field(default_factory=list)

    def check(self) -> None:
        if self.cancelled():
            raise Cancelled()

    def stage(self, label: str, lo: float, hi: float) -> Callable[[float], None]:
        self.check()
        self.report(lo, label)
        return lambda f: self.report(lo + (hi - lo) * max(0.0, min(1.0, f)), label)

    def media_ids(self) -> list[str]:
        return list(self.session.bus.project.media)

    def add_output(self, path: Path, kind: str, **meta: Any) -> None:
        self.outputs.append({"name": path.name, "kind": kind, "size": path.stat().st_size, **meta})

    def write_json(self, name: str, data: Any) -> Path:
        p = self.out / name
        p.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        self.add_output(p, "json")
        return p

    @property
    def quality(self) -> str:
        return (self.output.quality if self.output and self.output.quality else None) or "standard"


QUALITY = {"draft": (1280, "small"), "standard": (1920, "standard"), "high": (3840, "high")}


def _apply_video(ctx: JobContext, program: str) -> None:
    from ..core.commands import CommandError

    s = ctx.session
    try:
        s.bus.execute("timeline.apply_program", {"program": program, "transcripts": s.transcripts_for_timeline() or None}, source="ai")
    except CommandError as exc:
        raise RecipeError(f"program failed: {exc}") from exc


def _export(ctx: JobContext, name: str, progress: Callable[[float], None], **meta: Any) -> Path:
    from ..video.compose import output_size
    from ..video.export import Cancelled as ExportCancelled
    from ..video.export import ExportSpec, export_video

    s = ctx.session
    with s.bus.lock:
        project = s.bus.project.model_copy(deep=True)
    max_side, q = QUALITY[ctx.quality]
    w, h = output_size(project, max_side)
    dst = ctx.out / name
    try:
        res = export_video(project, s.paths.root, ExportSpec(dst, w, h, min(project.settings.fps or 30, 60), q), progress, ctx.cancelled)
    except ExportCancelled as exc:
        raise Cancelled() from exc
    ctx.add_output(dst, "video", width=res["width"], height=res["height"], duration=res["duration"], **meta)
    return dst


def _videos(ctx: JobContext) -> list[tuple[int, str]]:
    media = ctx.session.bus.project.media
    return [(n, mid) for n, mid in enumerate(media, start=1) if media[mid].kind == MediaKind.VIDEO]


def _nearest_aspect(w: int, h: int) -> str:
    r = w / max(h, 1)
    return min(ASPECTS, key=lambda k: abs(ASPECTS[k][0] / ASPECTS[k][1] - r))


def _source(ctx: JobContext, wanted: int | None, need_speech: bool = False) -> tuple[int, str]:
    vids = _videos(ctx)
    if not vids:
        raise RecipeError("this recipe needs a video input")
    if wanted is not None:
        for n, mid in vids:
            if n == wanted:
                return n, mid
        raise RecipeError(f"input {wanted} is not a video")
    media = ctx.session.bus.project.media
    return max(vids, key=lambda v: media[v[1]].duration or 0)


def _need_whisper() -> None:
    from ..agent.whisper import whisper_status

    if not whisper_status()["available"]:
        raise RecipeError("speech recognition (whisper) is not installed on this computer")


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments: list[dict[str, Any]]) -> str:
    out = []
    for i, seg in enumerate(segments, start=1):
        out.append(f"{i}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n{seg['text'].strip()}\n")
    return "\n".join(out)


# ------------------------------------------------------------------ recipes
def run_reel(ctx: JobContext) -> dict[str, Any]:
    from ..agent.reels import make_reels

    o: ReelOptions = ctx.options
    ctx.stage("Understanding the video", 0.0, 0.3)
    s = ctx.session
    vc = s.video_context(transcribe_speech=True)
    ctx.check()
    try:
        res = make_reels(vc, media_number=o.source, count=o.count, length=o.length, style=ctx.style, output=ctx.output,
                         captions=o.captions, tighten=o.tighten, subject_of=s.subject_track, styles_dir=s.styles_dir,
                         progress_bar=o.progress_bar, end_card=o.end_card or None)
    except (OptionError, ValueError) as exc:
        raise RecipeError(str(exc)) from exc
    reels = []
    n = max(1, len(res["variations"]))
    for i, v in enumerate(res["variations"], start=1):
        prog = ctx.stage(f"Rendering reel {i} of {n}", 0.3 + 0.7 * (i - 1) / n, 0.3 + 0.7 * i / n)
        _apply_video(ctx, v["program"])
        meta = {"title": v["title"], "score": v["score"], "rationale": v["rationale"], "moment": v["moment"]}
        _export(ctx, f"reel_{i}.mp4", prog, **meta)
        reels.append({**meta, "file": f"reel_{i}.mp4", "info": v["info"], "program": v["program"]})
    ctx.write_json("reels.json", reels)
    return {"reels": len(reels), "style": res["style"], "aspect": res["aspect"], "tracking": res["tracking"]}


def run_highlights(ctx: JobContext) -> dict[str, Any]:
    from ..agent.highlights import find_moments
    from ..agent.reels import pick_source

    o: HighlightOptions = ctx.options
    ctx.stage("Understanding the video", 0.0, 0.4)
    s = ctx.session
    n, mid = _source(ctx, o.source)
    vc = s.video_context(transcribe_speech=True)
    m = pick_source(vc, n)
    moments = [mo.to_json() for mo in find_moments(m, o.length, o.count)]
    item = s.bus.project.media[mid]
    aspect = _nearest_aspect(item.width or 1920, item.height or 1080)
    for i, mo in enumerate(moments, start=1):
        mo["rank"] = i
        if o.clips:
            prog = ctx.stage(f"Cutting moment {i} of {len(moments)}", 0.4 + 0.6 * (i - 1) / len(moments), 0.4 + 0.6 * i / len(moments))
            _apply_video(ctx, f"aspect {aspect}\nclip {n} {mo['start']:.2f}-{mo['end']:.2f}\nfade_in 0.2\nfade_out 0.3")
            _export(ctx, f"highlight_{i}.mp4", prog, title=mo["title"], score=mo["score"], start=mo["start"], end=mo["end"])
            mo["file"] = f"highlight_{i}.mp4"
    ctx.write_json("highlights.json", moments)
    return {"moments": len(moments), "speech": m.speech}


def run_captions(ctx: JobContext) -> dict[str, Any]:
    o: CaptionOptions = ctx.options
    if o.style not in CAPTION_STYLES:
        raise RecipeError(f"caption style must be one of {', '.join(CAPTION_STYLES)}")
    _need_whisper()
    n, mid = _source(ctx, o.source)
    s = ctx.session
    ctx.stage("Listening", 0.0, 0.35)
    tr = s.transcript(mid)
    item = s.bus.project.media[mid]
    stem = Path(item.name).stem
    srt = ctx.out / f"{stem}.srt"
    srt.write_text(to_srt(tr.get("segments", [])), encoding="utf-8")
    ctx.add_output(srt, "subtitles")
    if o.burn:
        if not tr.get("words"):
            raise RecipeError("no speech found to caption")
        prog = ctx.stage("Rendering captions", 0.35, 1.0)
        aspect = _nearest_aspect(item.width or 1920, item.height or 1080)
        _apply_video(ctx, f"aspect {aspect}\nclip {n} 0-{(item.duration or 0):.3f} volume=1\ncaptions {o.style}")
        _export(ctx, f"{stem}_captioned.mp4", prog)
    return {"segments": len(tr.get("segments", [])), "language": tr.get("language")}


def _photo_ids(ctx: JobContext) -> list[str]:
    media = ctx.session.bus.project.media
    ids = [mid for mid in media if media[mid].kind == MediaKind.IMAGE]
    if not ids:
        raise RecipeError("this recipe needs photo inputs")
    return ids


EXT = {"jpeg": ".jpg", "png": ".png", "webp": ".webp", "tiff": ".tif"}


def run_photo_edit(ctx: JobContext) -> dict[str, Any]:
    from ..agent.planner import plan
    from ..dsl import compile_program
    from ..imaging.render import export_image

    o: PhotoEditOptions = ctx.options
    s = ctx.session
    ids = _photo_ids(ctx)
    versions = ctx.output.versions if ctx.output and ctx.output.versions else 1
    prompt = ctx.prompt or "make it look professional"
    edits = []
    for k, mid in enumerate(ids):
        ctx.stage(f"Editing photo {k + 1} of {len(ids)}", k / len(ids), (k + 1) / len(ids))
        path, _ = s.photo_stack(mid)
        try:
            res = plan(prompt, s.analysis(mid), "", versions, ctx.style, ctx.output)
        except OptionError as exc:
            raise RecipeError(str(exc)) from exc
        for j, v in enumerate(res["variations"][:versions], start=1):
            ctx.check()
            name = f"{path.stem}_edit{j}{EXT[o.format]}" if versions > 1 else f"{path.stem}_edit{EXT[o.format]}"
            dst = ctx.out / name
            export_image(path, compile_program(v["program"]), dst, o.format, o.quality, 8, o.long_edge)
            ctx.add_output(dst, "photo", source=path.name, rationale=v.get("rationale", ""))
            edits.append({"file": name, "source": path.name, "program": v["program"], "rationale": v.get("rationale", "")})
    ctx.write_json("edits.json", edits)
    return {"photos": len(ids), "files": len(edits)}


def run_render_program(ctx: JobContext) -> dict[str, Any]:
    from ..dsl import DslError, compile_program
    from ..imaging.render import export_image

    o: RenderOptions = ctx.options
    media = ctx.session.bus.project.media
    photos_only = all(m.kind == MediaKind.IMAGE for m in media.values())
    first = o.program.strip().split(None, 1)[0] if o.program.strip() else ""
    if photos_only and first not in ("aspect", "clip", "overlay", "music", "title", "text"):
        try:
            ops = compile_program(o.program)
        except DslError as exc:
            raise RecipeError(str(exc)) from exc
        for k, mid in enumerate(media):
            ctx.stage(f"Rendering photo {k + 1}", k / len(media), (k + 1) / len(media))
            path, _ = ctx.session.photo_stack(mid)
            dst = ctx.out / f"{path.stem}_{o.name}{EXT[o.format]}"
            export_image(path, ops, dst, o.format, 92, 8, None)
            ctx.add_output(dst, "photo", source=path.name)
        return {"files": len(media)}
    _apply_video(ctx, o.program)
    _export(ctx, f"{o.name}.mp4", ctx.stage("Rendering", 0.0, 1.0))
    return {"duration": round(ctx.session.bus.project.duration, 2)}


def run_transcribe(ctx: JobContext) -> dict[str, Any]:
    _need_whisper()
    media = ctx.session.bus.project.media
    ids = [mid for mid in media if media[mid].kind in (MediaKind.VIDEO, MediaKind.AUDIO)]
    if not ids:
        raise RecipeError("transcribe needs video or audio inputs")
    for k, mid in enumerate(ids):
        ctx.stage(f"Listening to {media[mid].name}", k / len(ids), (k + 1) / len(ids))
        tr = ctx.session.transcript(mid) or {}
        stem = Path(media[mid].name).stem
        srt = ctx.out / f"{stem}.srt"
        srt.write_text(to_srt(tr.get("segments", [])), encoding="utf-8")
        ctx.add_output(srt, "subtitles")
        txt = ctx.out / f"{stem}.txt"
        txt.write_text("\n".join(seg["text"] for seg in tr.get("segments", [])), encoding="utf-8")
        ctx.add_output(txt, "text")
        ctx.write_json(f"{stem}.json", {k2: tr.get(k2) for k2 in ("language", "words", "segments")})
    return {"files": len(ids)}


RECIPES: dict[str, Recipe] = {r.name: r for r in (
    Recipe("reel", "Make reels", "One long video in, N ranked vertical clips out: best moments, pauses cut, speaker kept in frame, captions.",
           ReelOptions, run_reel),
    Recipe("highlights", "Find highlights", "The best moments of a video as JSON (start, end, title, reason), optionally cut as clips.",
           HighlightOptions, run_highlights),
    Recipe("captions", "Add captions", "Transcribe speech; burn styled captions into the video and write an .srt file.",
           CaptionOptions, run_captions),
    Recipe("photo_edit", "Edit photos from a prompt", "Fix and style each photo; one file per photo and version.",
           PhotoEditOptions, run_photo_edit, media="photo"),
    Recipe("render_program", "Render a DSL program", "Run your own Video Edit DSL (or Photo Edit DSL for photos). Inputs are numbered 1..n.",
           RenderOptions, run_render_program, media="any"),
    Recipe("transcribe", "Transcribe", "Speech to .srt subtitles, plain text and word-timed JSON.",
           TranscribeOptions, run_transcribe),
)}

ALIASES = {"reels": "reel", "highlight": "highlights", "caption": "captions", "photo": "photo_edit",
           "render": "render_program", "program": "render_program", "transcript": "transcribe"}


def resolve(name: str) -> Recipe:
    key = ALIASES.get(name, name)
    if key not in RECIPES:
        raise RecipeError(f"unknown recipe `{name}`; use one of {', '.join(RECIPES)}")
    return RECIPES[key]


def parse_options(recipe: Recipe, raw: dict[str, Any] | None) -> BaseModel:
    try:
        return recipe.options.model_validate(raw or {})
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ())) or "options"
        raise RecipeError(f"options.{where}: {first.get('msg')}") from exc


def guess_recipe(prompt: str | None, files: list[Path]) -> str:
    """No recipe given: photos -> photo_edit, a video -> reel."""
    from ..media.probe import IMAGE_EXTS

    if files and all(f.suffix.lower() in IMAGE_EXTS for f in files):
        return "photo_edit"
    if files:
        return "reel"
    raise RecipeError("give a recipe and input files")


def project_kind(recipe: Recipe, files: list[Path]) -> ProjectKind:
    from ..media.probe import IMAGE_EXTS

    if recipe.media == "photo" or (files and all(f.suffix.lower() in IMAGE_EXTS for f in files)):
        return ProjectKind.PHOTO
    return ProjectKind.VIDEO


def catalog() -> list[dict[str, Any]]:
    return [{"name": r.name, "title": r.title, "description": r.description, "needs_prompt": r.needs_prompt,
             "inputs": r.media, "options": r.options.model_json_schema()} for r in RECIPES.values()]


def copy_outputs(out: Path, dest: Path, outputs: list[dict[str, Any]]) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    done = []
    for o in outputs:
        shutil.copy2(out / o["name"], dest / o["name"])
        done.append(str(dest / o["name"]))
    return done
