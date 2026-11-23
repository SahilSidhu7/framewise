"""GPU timeline compositor (OpenGL 3.3 through moderngl).

Same result as the CPU compositor in video/compose.py, which stays the
reference and the fallback:

    prepare()  CPU, runs in the export worker threads in parallel:
               decode frames, rasterise masks, work out each layer's matrix;
               layers with effects (clip ops), blur transitions and the like
               are rendered on the CPU here and passed on as finished pictures
    execute()  GPU, one frame at a time under a lock: upload, place layers,
               transitions, mattes, blend modes, then the finishing pass
               (sRGB encode + dither + baked grade LUT), read back 8-bit

Adjustment layers and grades with spatial effects (grain, vignette, ...) read
the canvas back and finish on the CPU, so every project renders; the GPU just
does less of the work.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.project import BLEND_MODES, Clip, MediaKind, Project, Track
from ..imaging.ops import OPS, _vignette_gain, grain_noise
from ..imaging.render import SOURCES, apply_stack, make_context
from ..video import compose as C
from ..video.decode import _LIN_LUT, READERS
from ..video.fastgrade import _ENC, NOISE, PLANS, GradePlan
from ..video.layers import hex_to_linear, masks_coverage

log = logging.getLogger(__name__)

SHADER_DIR = Path(__file__).resolve().parent / "shaders"
MODE_INDEX = {m: i for i, m in enumerate(BLEND_MODES)}
TRANSITION_INDEX = {"crossfade": 0, "fade_black": 1, "fade_white": 2, "wipe_left": 3, "wipe_right": 4,
                    "slide_left": 5, "slide_up": 6, "zoom": 7}
GPU_TRANSITIONS = set(TRANSITION_INDEX) | {"none"}
VERTEX = """
in vec2 pos;
void main() { gl_Position = vec4(pos, 0.0, 1.0); }
"""
GL_SRGB8 = 0x8C41


def shader_source(name: str, header: str = "#version 330 core\n") -> str:
    """A fragment shader with its includes, for this engine (or, with another header, for WebGL2)."""
    body = (SHADER_DIR / name).read_text(encoding="utf-8")
    needs = []
    for line in body.splitlines():
        if line.startswith("// needs:"):
            needs = [n.strip() for n in line.split(":", 1)[1].split(",")]
            break
    return header + "".join((SHADER_DIR / n).read_text(encoding="utf-8") + "\n" for n in needs) + body


# ------------------------------------------------------------------ plan


@dataclass
class GpuLayer:
    """A layer the GPU places itself."""

    src: np.ndarray | None  # uint8 sRGB (video) or float32 linear (photo); None for a solid
    color: tuple[float, float, float]
    inv: np.ndarray  # 3x3 canvas px -> source px
    src_size: tuple[int, int]
    keep: tuple[float, float, float, float]
    pre: tuple[float, float]  # coverage is measured in these source-pixel units (the CPU pre-shrinks)
    covers: bool
    opacity: float
    mask: np.ndarray | None
    mip: bool
    key: Any = None  # cache key for still sources

    @property
    def opaque(self) -> bool:
        return self.covers and self.mask is None and self.opacity >= 1.0


@dataclass
class CpuLayer:
    """A picture the CPU already finished (premultiplied) with its alpha."""

    img: np.ndarray
    alpha: np.ndarray | float

    @property
    def opaque(self) -> bool:
        return isinstance(self.alpha, float) and self.alpha >= 1.0


@dataclass
class ClearLayer:
    alpha: float

    @property
    def opaque(self) -> bool:
        return self.alpha >= 1.0


Layer = GpuLayer | CpuLayer | ClearLayer


@dataclass
class TransitionLayer:
    a: Layer
    b: Layer
    kind: str
    alpha_kind: str
    p: float


@dataclass
class Step:
    layer: Layer | TransitionLayer | None = None
    blend: str = "normal"
    matte: str | None = None
    matte_layer: Layer | TransitionLayer | None = None
    adjust: Clip | None = None


@dataclass
class FramePlan:
    w: int
    h: int
    t: float
    seed: int
    steps: list[Step] = field(default_factory=list)
    grade: GradePlan | None = None
    post: list[tuple[int, float, np.ndarray, Any]] | None = None  # GPU grade effects: (kind, amount, field, cache key)


class Unsupported(Exception):
    pass


def _gpu_clip(project: Project, clip: Clip, t: float, w: int, h: int, src: C.RenderSource, seed: int) -> Layer:
    ops = [o for o in clip.ops if not OPS[o.op].geometry and OPS[o.op].category != "subject"]
    if ops or clip.vector:  # effects run on the placed picture, text and shapes are drawn: CPU
        img, alpha = C.clip_frame(project, clip, t, w, h, src, seed)
        return CpuLayer(img, alpha)
    key = None
    if clip.source == "solid":
        pic, sw, sh, mip = None, w, h, False
        color = tuple(float(v) for v in hex_to_linear(clip.color))
    else:
        color = (0.0, 0.0, 0.0)
        item = project.media[clip.media_id]
        if item.kind == MediaKind.IMAGE:
            side = int(max(w, h) * 1.15 * min(4.0, max(1.0, C._max_scale(clip))))
            pic = SOURCES.get(Path(item.path), side)
            stack = [o.model_dump(mode="json") for o in project.edits.get(item.id, [])]
            if stack:
                pic = apply_stack(pic, stack, make_context(seed))
            else:
                key = (item.path, side)
        else:
            path = Path(item.path)
            if src.use_proxy and item.proxy_path:
                proxy = src.root / item.proxy_path
                if proxy.is_file():
                    path = proxy
            probe = READERS.get(path, None, src.channel)
            zoom_max = (1.12 if clip.motion != "none" else 1.0) * C._max_scale(clip)
            size = C._decode_size(probe.width, probe.height, w, h, clip.fit, zoom_max)
            reader = READERS.get(path, size, src.channel) if size else probe
            pic = reader.frame_at(clip.source_time(t))
        sh, sw = pic.shape[:2]
    m, covers = C._matrix(clip, t, sw, sh, w, h)
    full = np.vstack([m, [0.0, 0.0, 1.0]])
    inv = np.linalg.inv(full)
    scale = math.sqrt(abs(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]))
    cr = clip.crop
    pw, ph = sw, sh
    if scale < 0.75:  # compose._place shrinks the source first; its edges are measured in those pixels
        pw, ph = max(1, int(sw * scale * 1.25)), max(1, int(sh * scale * 1.25))
    keep = (float(round(pw * cr.left)), float(round(ph * cr.top)),
            float(pw - round(pw * cr.right)), float(ph - round(ph * cr.bottom)))
    mask = _masks(clip, sw, sh, m, w, h) if any(k.enabled for k in clip.masks) else None
    return GpuLayer(pic, color, inv, (sw, sh), keep, (pw / sw, ph / sh), covers, C._layer_alpha(clip, t), mask, scale < 0.75, key)


POST_OPS = {"vignette": 1, "grain": 2}


def _post_ops(grade: GradePlan, w: int, h: int, seed: int) -> list[tuple[int, float, np.ndarray, Any]] | None:
    """The grade's spatial effects as GPU fields, or None if one of them has no GPU version."""
    if len(grade.rest) > 4 or any(name not in POST_OPS for name, _ in grade.rest):
        return None
    out = []
    ctx = make_context(seed)
    for name, params in grade.rest:
        p = OPS[name].params_model.model_validate(params)
        if name == "vignette":
            gain = _vignette_gain(h, w, p.amount, p.midpoint, p.feather)
            out.append((1, 0.0, gain, ("vignette", h, w, p.amount, p.midpoint, p.feather)))
        else:
            field_ = grain_noise(h, w, p, ctx) if p.amount > 0 else None
            out.append((2, p.amount if p.amount > 0 else 0.0, field_, None))
    return out


_MASKS: OrderedDict[Any, np.ndarray] = OrderedDict()
_MASKS_LOCK = threading.Lock()


def _masks(clip: Clip, sw: int, sh: int, m: np.ndarray, w: int, h: int) -> np.ndarray:
    """masks_coverage, cached: a still layer's feathered mask is the same every frame."""
    key = (tuple(k.model_dump_json() for k in clip.masks), sw, sh, w, h, np.round(m, 6).tobytes())
    with _MASKS_LOCK:
        hit = _MASKS.get(key)
        if hit is not None:
            _MASKS.move_to_end(key)
            return hit
    cov = masks_coverage(clip.masks, sw, sh, m, w, h)
    assert cov is not None
    with _MASKS_LOCK:
        _MASKS[key] = cov
        while len(_MASKS) > 12:
            _MASKS.popitem(last=False)
    return cov


def _track_layer(project: Project, track: Track, t: float, w: int, h: int, src: C.RenderSource, seed: int) -> tuple[Layer | TransitionLayer, str] | None:
    """Mirrors compose.track_frame: the layer and its blend mode."""
    for clip in track.clips:
        tr = clip.transition
        if tr is None:
            continue
        half = tr.duration / 2
        if clip.start - half <= t < clip.start + half:
            kind = tr.kind
            if kind not in GPU_TRANSITIONS:
                res = C.track_frame(project, track, t, w, h, src, seed)
                return (CpuLayer(res[0], res[1]), res[2]) if res else None
            prev = C._prev_clip(track, clip)
            p = (t - (clip.start - half)) / tr.duration
            b = _gpu_clip(project, clip, max(t, clip.start), w, h, src, seed)
            if prev is not None:
                a: Layer = _gpu_clip(project, prev, min(t, prev.end - 1e-4), w, h, src, seed)
            else:
                a = ClearLayer(1.0 if kind == "fade_white" else 0.0)
                if t < clip.start and kind != "fade_white":
                    kind = "crossfade"
            if kind == "none":
                kind = "cut"
            alpha_kind = kind if kind in C._ALPHA_KINDS else "crossfade"
            return TransitionLayer(a, b, kind, alpha_kind, p), clip.blend
    clip = C._active(track, t)
    if clip is None:
        return None
    return _gpu_clip(project, clip, t, w, h, src, seed), clip.blend


def prepare(project: Project, t: float, w: int, h: int, src: C.RenderSource, seed: int, grade: list[dict[str, Any]]) -> FramePlan:
    plan = FramePlan(w, h, t, seed, grade=PLANS.get(grade) if grade else None)
    if plan.grade is not None and plan.grade.rest:
        plan.post = _post_ops(plan.grade, w, h, seed)
    tracks = [tr for tr in project.tracks if tr.kind != "audio"]
    skip: set[int] = set()
    for i, track in enumerate(tracks):  # mirrors compose.composite_tracks
        if i in skip or track.hidden:
            continue
        matte_track = tracks[i + 1] if track.matte and i + 1 < len(tracks) else None
        if matte_track is not None:
            skip.add(i + 1)
        active = C._active(track, t)
        if active is not None and active.source == "adjustment":
            plan.steps.append(Step(adjust=active))
            continue
        found = _track_layer(project, track, t, w, h, src, seed)
        if found is None:
            continue
        step = Step(layer=found[0], blend=found[1])
        if matte_track is not None and track.matte:
            step.matte = track.matte
            m = _track_layer(project, matte_track, t, w, h, src, seed)
            step.matte_layer = m[0] if m else None
        plan.steps.append(step)
    return plan


# -------------------------------------------------------------------- GL


class GpuCompositor:
    """Owns one OpenGL context on its own thread; every GL call runs there."""

    def __init__(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._gl = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self._gl.submit(self._init).result()

    def _init(self) -> None:
        import moderngl

        self.mgl = moderngl
        self.ctx = moderngl.create_standalone_context(require=330)
        self.renderer = self.ctx.info.get("GL_RENDERER", "?")
        quad = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")  # one triangle covering the screen
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.progs = {n: self.ctx.program(vertex_shader="#version 330 core\n" + VERTEX, fragment_shader=shader_source(n + ".frag"))
                      for n in ("layer", "composite", "transition", "matte", "finish")}
        self.vaos = {n: self.ctx.vertex_array(p, [(self.vbo, "2f", "pos")]) for n, p in self.progs.items()}
        enc = np.ascontiguousarray(_ENC.reshape(1, -1), dtype="f4")
        self.enc = self.ctx.texture((enc.shape[1], 1), 1, enc.tobytes(), dtype="f4")
        self.enc.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.lin8 = self.ctx.texture((256, 1), 1, np.ascontiguousarray(_LIN_LUT, dtype="f4").tobytes(), dtype="f4")
        self.lin8.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._targets: dict[tuple[str, int, int], Any] = {}
        self._uploads: dict[tuple[Any, ...], Any] = {}
        self._stills: OrderedDict[Any, Any] = OrderedDict()
        self._luts: OrderedDict[int, Any] = OrderedDict()
        self._noise: tuple[tuple[int, int], Any] | None = None
        self._dummy = self.ctx.texture((1, 1), 4, np.zeros(4, "f4").tobytes(), dtype="f4")
        self._dummy3 = self.ctx.texture3d((1, 1, 1), 3, bytes(3))

    # ------------------------------------------------------------ helpers

    def _target(self, name: str, w: int, h: int) -> Any:
        key = (name, w, h)
        fbo = self._targets.get(key)
        if fbo is None:
            dtype = "f1" if name == "out" else "f4"
            tex = self.ctx.texture((w, h), 4, dtype=dtype)
            tex.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
            fbo = self._targets[key] = self.ctx.framebuffer([tex])
        return fbo

    def _upload(self, layer: GpuLayer) -> Any:
        pic = layer.src
        assert pic is not None
        if layer.key is not None and layer.key in self._stills:
            self._stills.move_to_end(layer.key)
            return self._stills[layer.key]
        sh, sw = pic.shape[:2]
        if pic.dtype == np.uint8:
            key = ("u8", sw, sh, layer.mip)
            tex = self._uploads.get(key)
            if tex is None:
                tex = self._uploads[key] = self.ctx.texture((sw, sh), 3, dtype="f1", internal_format=GL_SRGB8)
            tex.write(np.ascontiguousarray(pic).tobytes())
        else:
            data = np.ascontiguousarray(pic, dtype="f4").tobytes()
            if layer.key is not None:
                tex = self.ctx.texture((sw, sh), 3, data, dtype="f4")
                self._stills[layer.key] = tex
                while len(self._stills) > 6:
                    self._stills.popitem(last=False)[1].release()
            else:
                key = ("f4", sw, sh, layer.mip)
                tex = self._uploads.get(key)
                if tex is None:
                    tex = self._uploads[key] = self.ctx.texture((sw, sh), 3, dtype="f4")
                tex.write(data)
        tex.repeat_x = tex.repeat_y = False  # clamp to edge = cv2 BORDER_REPLICATE
        if layer.mip:
            tex.build_mipmaps()
            tex.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
        else:
            tex.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
        return tex

    def _canvas_tex(self, arr: np.ndarray, w: int, h: int, name: str) -> Any:
        """A CPU array (top row first) as a canvas-sized float texture."""
        key = (name, w, h, arr.shape[-1] if arr.ndim == 3 else 1)
        tex = self._uploads.get(key)
        comps = key[3]
        if tex is None:
            tex = self._uploads[key] = self.ctx.texture((w, h), comps, dtype="f4")
            tex.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
        tex.write(np.ascontiguousarray(arr, dtype="f4").tobytes())
        return tex

    def _draw(self, name: str, fbo: Any, **uniforms: Any) -> None:
        prog = self.progs[name]
        unit = 0
        for key, value in uniforms.items():
            if key not in prog:
                continue
            if hasattr(value, "use"):  # texture
                value.use(location=unit)
                prog[key].value = unit
                unit += 1
            elif isinstance(value, np.ndarray):
                prog[key].write(np.ascontiguousarray(value, dtype="f4").tobytes())
            else:
                prog[key].value = value
        fbo.use()
        self.vaos[name].render(self.mgl.TRIANGLES)

    # ------------------------------------------------------------- layers

    def _layer(self, layer: Layer | TransitionLayer, w: int, h: int, name: str) -> tuple[Any, bool]:
        """Render a layer into the named target; returns (fbo, opaque)."""
        fbo = self._target(name, w, h)
        if isinstance(layer, ClearLayer):
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, layer.alpha)
            return fbo, layer.opaque
        if isinstance(layer, CpuLayer):
            alpha = layer.alpha if isinstance(layer.alpha, np.ndarray) else np.full((h, w), layer.alpha, np.float32)
            rgba = np.concatenate([layer.img, alpha[..., None]], axis=-1)[::-1]  # bottom-up like a render
            fbo.color_attachments[0].write(np.ascontiguousarray(rgba, dtype="f4").tobytes())
            return fbo, layer.opaque
        if isinstance(layer, TransitionLayer):
            fa, oa = self._layer(layer.a, w, h, name + "_a")
            fb, ob = self._layer(layer.b, w, h, name + "_b")
            self._draw("transition", fbo, ta=fa.color_attachments[0], tb=fb.color_attachments[0],
                       kind=TRANSITION_INDEX.get(layer.kind, 8), alpha_kind=TRANSITION_INDEX.get(layer.alpha_kind, 0),
                       p=float(layer.p), e=C._ease(layer.p), edge=(int(w * C._ease(layer.p)), int(h * C._ease(layer.p))),
                       size=(w, h))
            if oa and ob:  # both opaque: the CPU keeps alpha at exactly 1
                return fbo, True
            return fbo, False
        tex = self._upload(layer) if layer.src is not None else self._dummy
        mask = self._canvas_tex(layer.mask, w, h, "mask_" + name) if layer.mask is not None else self._dummy
        self._draw("layer", fbo, src=tex, solid=int(layer.src is None), color=layer.color,
                   inv=layer.inv.T.astype("f4"), src_size=tuple(float(v) for v in layer.src_size),
                   keep=layer.keep, pre=layer.pre, covers=int(layer.covers), opacity=float(layer.opacity),
                   has_mask=int(layer.mask is not None), mask=mask, height=h)
        return fbo, layer.opaque

    def _field(self, arr: np.ndarray, key: Any, w: int, h: int, slot: int) -> Any:
        """A per-pixel effect field (top row first); still ones (vignette) stay on the GPU."""
        if key is not None and key in self._stills:
            self._stills.move_to_end(key)
            return self._stills[key]
        data = np.ascontiguousarray(arr.reshape(h, w), dtype="f4").tobytes()
        if key is not None:
            tex = self.ctx.texture((w, h), 1, data, dtype="f4")
            tex.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
            self._stills[key] = tex
            while len(self._stills) > 6:
                self._stills.popitem(last=False)[1].release()
            return tex
        return self._canvas_tex(arr.reshape(h, w), w, h, f"field{slot}")

    def _read_canvas(self, fbo: Any, w: int, h: int) -> np.ndarray:
        raw = np.frombuffer(fbo.color_attachments[0].read(), dtype="f4").reshape(h, w, 4)
        return np.ascontiguousarray(raw[::-1, :, :3])

    def _lut(self, plan: GradePlan) -> Any:
        key = id(plan)
        if key in self._luts:
            self._luts.move_to_end(key)
            return self._luts[key]
        assert plan.lut is not None
        tex = self.ctx.texture3d((256, 256, 256), 3, np.ascontiguousarray(plan.lut).tobytes())
        tex.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
        self._luts[key] = tex
        while len(self._luts) > 2:
            self._luts.popitem(last=False)[1].release()
        return tex

    def _noise_tex(self, w: int, h: int) -> Any:
        if self._noise is None or self._noise[0] != (w, h):
            if self._noise is not None:
                self._noise[1].release()
            arr = np.ascontiguousarray(NOISE.full(h, w), dtype="f4")
            tex = self.ctx.texture((w + 32, h + 32), 3, arr.tobytes(), dtype="f4")
            tex.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
            self._noise = ((w, h), tex)
        return self._noise[1]

    # -------------------------------------------------------------- frame

    def execute(self, plan: FramePlan) -> np.ndarray:
        """Run a prepared frame on the GL thread (callers may be any thread).
        A grade with spatial effects finishes back on the caller's thread."""
        out = self._gl.submit(self._execute, plan).result()
        if out.dtype != np.uint8:
            assert plan.grade is not None
            return plan.grade.finish(out, plan.seed)
        return out

    def _drop_other_sizes(self, w: int, h: int) -> None:
        """Free render targets of other frame sizes and old uploads (only between frames)."""
        if len(self._uploads) > 16:
            for k in list(self._uploads)[:-8]:
                self._uploads.pop(k).release()
        old = [k for k in self._targets if k[1:] != (w, h)]
        if len(old) > 8:
            for k in old:
                f = self._targets.pop(k)
                f.color_attachments[0].release()
                f.release()

    def _execute(self, plan: FramePlan) -> np.ndarray:
        w, h = plan.w, plan.h
        self._drop_other_sizes(w, h)
        cur, other = self._target("canvas0", w, h), self._target("canvas1", w, h)
        cur.use()
        cur.clear(0.0, 0.0, 0.0, 1.0)
        for step in plan.steps:
            if step.adjust is not None:
                canvas = self._read_canvas(cur, w, h)
                canvas = C._adjust(step.adjust, canvas, plan.t, w, h, plan.seed)
                rgba = np.concatenate([canvas, np.ones((h, w, 1), np.float32)], axis=-1)[::-1]
                cur.color_attachments[0].write(np.ascontiguousarray(rgba, dtype="f4").tobytes())
                continue
            assert step.layer is not None
            fbo, _ = self._layer(step.layer, w, h, "layer")
            if step.matte:
                mfbo = self._layer(step.matte_layer, w, h, "matte")[0] if step.matte_layer is not None else None
                out = self._target("matted", w, h)
                self._draw("matte", out, layer=fbo.color_attachments[0],
                           matte=mfbo.color_attachments[0] if mfbo else self._dummy, use_matte=int(mfbo is not None),
                           fallback=1.0 if step.matte.endswith("inverted") else 0.0,
                           luma=int(step.matte.startswith("luma")), inverted=int(step.matte.endswith("inverted")))
                fbo = out
            self._draw("composite", other, base=cur.color_attachments[0], layer=fbo.color_attachments[0],
                       mode=MODE_INDEX[step.blend])
            cur, other = other, cur
        grade = plan.grade
        if grade is not None and grade.rest and plan.post is None:  # effects without a GPU version: CPU finishes
            return self._read_canvas(cur, w, h)
        out = self._target("out", w, h)
        oy, ox = (plan.seed * 7) % 32, (plan.seed * 13) % 32
        use_lut = grade is not None and grade.lut is not None
        post = plan.post or []
        fields = {}
        kinds, amounts = [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0]
        for i, (kind, amount, arr, key) in enumerate(post):
            kinds[i], amounts[i] = kind, amount
            fields[f"op{i}"] = self._field(arr, key, w, h, i) if arr is not None else self._dummy
        for i in range(len(post), 4):
            fields[f"op{i}"] = self._dummy
        self._draw("finish", out, canvas=cur.color_attachments[0], enc=self.enc, lin8=self.lin8, noise=self._noise_tex(w, h),
                   offset=(ox, oy), use_lut=int(use_lut), lut=self._lut(grade) if use_lut else self._dummy3, height=h,
                   n_ops=len(post), op_kind=tuple(kinds), op_amount=tuple(amounts), **fields)
        raw = np.frombuffer(out.read(components=3, alignment=1), dtype=np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(raw[::-1])

    def render(self, project: Project, t: float, w: int, h: int, src: C.RenderSource, seed: int,
               grade: list[dict[str, Any]]) -> np.ndarray:
        return self.execute(prepare(project, t, w, h, src, seed, grade))


# ----------------------------------------------------------------- access

_INSTANCE: GpuCompositor | None = None
_FAILED = False
_INIT = threading.Lock()


def get() -> GpuCompositor | None:
    """The shared compositor, or None when there is no usable GPU (or it is turned off)."""
    global _INSTANCE, _FAILED
    from ..paths import env

    if (env("GPU") or "").lower() in ("0", "off", "false", "no") or _FAILED:
        return None
    if _INSTANCE is not None:
        return _INSTANCE
    with _INIT:
        if _INSTANCE is None and not _FAILED:
            try:
                _INSTANCE = GpuCompositor()
                log.info("GPU compositor on %s", _INSTANCE.renderer)
            except Exception as exc:  # no OpenGL 3.3 driver, headless server, ...
                log.warning("GPU compositor unavailable, using the CPU: %s", exc)
                _FAILED = True
    return _INSTANCE
