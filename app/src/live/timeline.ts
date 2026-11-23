// Timeline maths for the live preview. A port of the engine's pure functions
// (core/project.py keyframes, video/compose.py placement), so the live picture
// lines up with the exact frames the engine renders.

import type { Clip, Keyframe, Project, Track } from "../engine";
import { clipDuration } from "../engine";

export const clipEnd = (c: Clip) => c.start + clipDuration(c);

export function cubicBezier(x1: number, y1: number, x2: number, y2: number, p: number): number {
  if (p <= 0 || p >= 1) return Math.min(1, Math.max(0, p));
  x1 = Math.min(1, Math.max(0, x1));
  x2 = Math.min(1, Math.max(0, x2));
  const bx = (s: number) => 3 * x1 * s * (1 - s) ** 2 + 3 * x2 * s * s * (1 - s) + s ** 3;
  let lo = 0;
  let hi = 1;
  let s = p;
  for (let i = 0; i < 50; i++) {
    const x = bx(s) - p;
    if (Math.abs(x) < 1e-7) break;
    if (x > 0) hi = s;
    else lo = s;
    s = (lo + hi) / 2;
  }
  return 3 * y1 * s * (1 - s) ** 2 + 3 * y2 * s * s * (1 - s) + s ** 3;
}

export function keyframeValue(keys: Keyframe[], t: number): number {
  if (t <= keys[0].t) return keys[0].value;
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i];
    const b = keys[i + 1];
    if (t < b.t) {
      if (a.ease === "hold" || b.t - a.t < 1e-9) return a.value;
      let p = (t - a.t) / (b.t - a.t);
      if (a.ease === "ease") p = p * p * (3 - 2 * p);
      else if (a.ease === "bezier") {
        const [x1, y1, x2, y2] = a.bezier ?? [0.42, 0, 0.58, 1];
        p = cubicBezier(x1, y1, x2, y2, p);
      }
      return a.value + (b.value - a.value) * p;
    }
  }
  return keys[keys.length - 1].value;
}

const TRANSFORM_DEFAULTS: Record<string, number> = { x: 0, y: 0, scale: 1, rotation: 0, anchor_x: 0, anchor_y: 0 };

/** Animated value of a keyable parameter at timeline time t. */
export function clipValue(c: Clip, name: string, t: number): number {
  const keys = c.keyframes?.[name];
  if (keys && keys.length) return keyframeValue(keys, t - c.start);
  if (name === "opacity") return c.opacity ?? 1;
  if (name === "volume") return c.volume;
  const tr = c.transform as Record<string, number | undefined> | undefined;
  return tr?.[name] ?? TRANSFORM_DEFAULTS[name] ?? 0;
}

export function sourceTime(c: Clip, t: number): number {
  const s = c.in_point + (t - c.start) * c.speed;
  return Math.min(Math.max(s, c.in_point), c.out_point);
}

function motion(c: Clip, t: number): [number, number, number] {
  if (c.motion === "none") return [1, 0, 0];
  let p = Math.min(1, Math.max(0, (t - c.start) / Math.max(clipDuration(c), 1e-6)));
  p = p * p * (3 - 2 * p);
  if (c.motion === "zoom_in") return [1 + 0.12 * p, 0, 0];
  if (c.motion === "zoom_out") return [1.12 - 0.12 * p, 0, 0];
  if (c.motion === "pan_left") return [1.12, 0.5 - p, 0];
  return [1.12, p - 0.5, 0];
}

/** Source pixel -> canvas pixel as [a, b, c, d, e, f] (x' = a x + b y + c, y' = d x + e y + f), and whether it covers the canvas. */
export function matrix(c: Clip, t: number, sw: number, sh: number, w: number, h: number): { m: number[]; covers: boolean } {
  const [zoom, dx, dy] = motion(c, t);
  const fit = c.fit === "fit";
  const s = (fit ? Math.min(w / sw, h / sh) : Math.max(w / sw, h / sh)) * zoom;
  const ow = sw * s;
  const oh = sh * s;
  const tx = (w - ow) / 2 + dx * Math.max(0, ow - w);
  const ty = (h - oh) / 2 + dy * Math.max(0, oh - h);
  let m = [s, 0, tx, 0, s, ty];
  const k = clipValue(c, "scale", t);
  const rot = ((clipValue(c, "rotation", t) % 360) + 360) % 360;
  const ox = clipValue(c, "x", t) * w;
  const oy = clipValue(c, "y", t) * h;
  if (k !== 1 || rot || ox || oy) {
    const th = (rot * Math.PI) / 180;
    const cs = Math.cos(th) * k;
    const sn = Math.sin(th) * k;
    const cx = w / 2 + clipValue(c, "anchor_x", t) * w;
    const cy = h / 2 + clipValue(c, "anchor_y", t) * h;
    const u = [cs, -sn, cx - cs * cx + sn * cy + ox, sn, cs, cy - sn * cx - cs * cy + oy];
    m = [
      u[0] * m[0] + u[1] * m[3], u[0] * m[1] + u[1] * m[4], u[0] * m[2] + u[1] * m[5] + u[2],
      u[3] * m[0] + u[4] * m[3], u[3] * m[1] + u[4] * m[4], u[3] * m[2] + u[4] * m[5] + u[5],
    ];
  }
  const crop = c.crop;
  const cropEmpty = !crop || !(crop.left || crop.top || crop.right || crop.bottom);
  const covers = rot === 0 && cropEmpty && m[2] <= 0.5 && m[5] <= 0.5 && m[2] + sw * m[0] >= w - 0.5 && m[5] + sh * m[4] >= h - 0.5;
  return { m, covers };
}

/** 3x3 inverse of the affine, column-major for a GLSL mat3 uniform. */
export function inverseColumnMajor(m: number[]): Float32Array {
  const [a, b, c, d, e, f] = m;
  const det = a * e - b * d || 1e-12;
  const ia = e / det;
  const ib = -b / det;
  const id = -d / det;
  const ie = a / det;
  const ic = -(ia * c + ib * f);
  const if_ = -(id * c + ie * f);
  // rows: [ia ib ic], [id ie if], [0 0 1] -> columns
  return new Float32Array([ia, id, 0, ib, ie, 0, ic, if_, 1]);
}

export const maxScale = (c: Clip) => {
  const keys = c.keyframes?.scale;
  return keys && keys.length ? Math.max(...keys.map((k) => k.value)) : c.transform?.scale ?? 1;
};

export function layerAlpha(c: Clip, t: number): number {
  let a = Math.min(1, Math.max(0, clipValue(c, "opacity", t)));
  if (c.fade_in > 0) a = Math.min(a, Math.max(0, (t - c.start) / c.fade_in));
  if (c.fade_out > 0) a = Math.min(a, Math.max(0, (clipEnd(c) - t) / c.fade_out));
  return a;
}

export function prevClip(track: Track, clip: Clip): Clip | null {
  let best: Clip | null = null;
  for (const c of track.clips) {
    if (c === clip || c.start >= clip.start) continue;
    if (Math.abs(clipEnd(c) - clip.start) < 0.05 && (!best || clipEnd(c) > clipEnd(best))) best = c;
  }
  return best;
}

export function activeClip(track: Track, t: number): Clip | null {
  for (const c of track.clips) if (c.start <= t && t < clipEnd(c)) return c;
  return null;
}

export const ease = (p: number) => p * p * (3 - 2 * p);
export const ALPHA_KINDS = new Set(["crossfade", "wipe_left", "wipe_right", "slide_left", "slide_up", "zoom"]);

export type LayerRef = { clip: Clip; t: number } | { clear: number };
export interface TrackLayer {
  kind: "clip" | "transition";
  b: LayerRef;
  a?: LayerRef;
  transition?: string;
  alphaKind?: string;
  p?: number;
  blend: string;
}

/** Mirrors compose.track_frame: what a track shows at time t. */
export function trackLayer(track: Track, t: number): TrackLayer | null {
  for (const clip of track.clips) {
    const tr = clip.transition;
    if (!tr) continue;
    const half = tr.duration / 2;
    if (clip.start - half <= t && t < clip.start + half) {
      let kind: string = tr.kind;
      const prev = prevClip(track, clip);
      const p = (t - (clip.start - half)) / tr.duration;
      const b: LayerRef = { clip, t: Math.max(t, clip.start) };
      let a: LayerRef;
      if (prev) a = { clip: prev, t: Math.min(t, clipEnd(prev) - 1e-4) };
      else {
        a = { clear: kind === "fade_white" ? 1 : 0 };
        if (t < clip.start && kind !== "fade_white") kind = "crossfade";
      }
      if (kind === "blur") kind = "crossfade"; // no live blur: crossfade stands in
      return { kind: "transition", a, b, transition: kind, alphaKind: ALPHA_KINDS.has(kind) ? kind : "crossfade", p, blend: clip.blend ?? "normal" };
    }
  }
  const c = activeClip(track, t);
  return c ? { kind: "clip", b: { clip: c, t }, blend: c.blend ?? "normal" } : null;
}

export const pictureTracks = (p: Project) => p.tracks.filter((t) => t.kind !== "audio");
