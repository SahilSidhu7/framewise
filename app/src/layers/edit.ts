// Layer editing helpers shared by the layers panel and the viewer handles.
// A value change goes to the keyframes when the parameter is animated (a key at
// the playhead), else to the clip's static value — like After Effects' stopwatch.

import { clipDuration, type Clip, type Keyframe, type Mask, type MaskPoint, type Project, type Track } from "../engine";
import { clipValue } from "../live/timeline";

export type RunFn = (name: string, params: Record<string, unknown>) => Promise<Record<string, unknown> | undefined>;

export const KEYABLE = ["x", "y", "scale", "rotation", "opacity", "anchor_x", "anchor_y"] as const;
export type Param = (typeof KEYABLE)[number];
export type Values = Partial<Record<Param, number>>;

const KEY_EPS = 1 / 120;

/** Clip-local time for the playhead, clamped into the clip. */
export const localTime = (clip: Clip, t: number) => Math.min(Math.max(t - clip.start, 0), clipDuration(clip));

export const animated = (clip: Clip, p: Param) => (clip.keyframes?.[p]?.length ?? 0) > 0;

export const keyAt = (clip: Clip, p: Param, t: number) => clip.keyframes?.[p]?.find((k) => Math.abs(k.t - localTime(clip, t)) < KEY_EPS);

function upsert(keys: Keyframe[], t: number, value: number): Keyframe[] {
  const i = keys.findIndex((k) => Math.abs(k.t - t) < KEY_EPS);
  const out = keys.slice();
  if (i >= 0) out[i] = { ...out[i], value };
  else out.push({ t: +t.toFixed(4), value, ease: "linear" });
  return out.sort((a, b) => a.t - b.t);
}

const round = (v: number) => +v.toFixed(5);

/** The clip with new values applied at time t (for drawing while dragging). */
export function applyValues(clip: Clip, v: Values, t: number): Clip {
  const out: Clip = { ...clip, transform: { x: 0, y: 0, scale: 1, rotation: 0, ...clip.transform }, keyframes: { ...clip.keyframes } };
  for (const [p, value] of Object.entries(v) as [Param, number][]) {
    if (animated(clip, p)) out.keyframes![p] = upsert(clip.keyframes![p], localTime(clip, t), value);
    else if (p === "opacity") out.opacity = value;
    else (out.transform as Record<string, number>)[p] = value;
  }
  return out;
}

/** Save new values: static ones in one clip.set_layer, animated ones as keys at the playhead. */
export async function commitValues(run: RunFn, clip: Clip, v: Values, t: number): Promise<void> {
  const flat: Record<string, number> = {};
  for (const [p, value] of Object.entries(v) as [Param, number][]) {
    if (animated(clip, p)) await run("clip.set_keyframes", { clip_id: clip.id, param: p, keys: upsert(clip.keyframes![p], localTime(clip, t), round(value)) });
    else flat[p] = round(value);
  }
  if (Object.keys(flat).length) await run("clip.set_layer", { clip_id: clip.id, ...flat });
}

/** Stopwatch: start animating from the current value, or stop and keep the current value. */
export async function toggleStopwatch(run: RunFn, clip: Clip, p: Param, t: number): Promise<void> {
  const value = round(clipValue(clip, p, t));
  if (animated(clip, p)) {
    await run("clip.set_keyframes", { clip_id: clip.id, param: p, keys: [] });
    await run("clip.set_layer", { clip_id: clip.id, [p]: value });
  } else {
    await run("clip.set_keyframes", { clip_id: clip.id, param: p, keys: [{ t: +localTime(clip, t).toFixed(4), value, ease: "linear" }] });
  }
}

/** Add a key at the playhead with the current value, or remove the one there. */
export async function toggleKeyHere(run: RunFn, clip: Clip, p: Param, t: number): Promise<void> {
  const keys = clip.keyframes?.[p] ?? [];
  const here = keyAt(clip, p, t);
  const next = here ? keys.filter((k) => k !== here) : upsert(keys, localTime(clip, t), round(clipValue(clip, p, t)));
  if (here && next.length === 0) {
    await run("clip.set_keyframes", { clip_id: clip.id, param: p, keys: [] });
    await run("clip.set_layer", { clip_id: clip.id, [p]: round(here.value) });
  } else {
    await run("clip.set_keyframes", { clip_id: clip.id, param: p, keys: next });
  }
}

/** Every key time of a clip (any parameter), timeline seconds. */
export function keyTimes(clip: Clip, only?: Param): number[] {
  const set = new Set<number>();
  for (const [p, keys] of Object.entries(clip.keyframes ?? {})) {
    if (only && p !== only) continue;
    for (const k of keys) set.add(+(clip.start + k.t).toFixed(4));
  }
  return [...set].sort((a, b) => a - b);
}

export function replaceClip(project: Project, clip: Clip): Project {
  return { ...project, tracks: project.tracks.map((tr) => (tr.clips.some((c) => c.id === clip.id) ? { ...tr, clips: tr.clips.map((c) => (c.id === clip.id ? clip : c)) } : tr)) };
}

/** Source picture size used for layer space (solids and adjustment layers fill the canvas). */
export function sourceSize(project: Project, clip: Clip): [number, number] {
  const { width: W, height: H } = project.settings;
  if ((clip.source ?? "media") !== "media") return [W, H];
  const m = project.media[clip.media_id];
  return m?.width && m?.height ? [m.width, m.height] : [W, H];
}

export const activeOn = (track: Track, t: number) => track.clips.find((c) => t >= c.start && t < c.start + clipDuration(c)) ?? null;

// ---- masks

let maskSeq = 0;
export const newMaskId = () => `msk_${Date.now().toString(36)}${(maskSeq++).toString(36)}`;

const pt = (x: number, y: number, extra: Partial<MaskPoint> = {}): MaskPoint => ({ x, y, in_x: 0, in_y: 0, out_x: 0, out_y: 0, ...extra });

export function maskOf(points: MaskPoint[], name: string): Mask {
  return { id: newMaskId(), name, points, mode: "add", inverted: false, opacity: 1, feather: 0, expansion: 0, enabled: true };
}

export function rectMask(l = 0.2, t = 0.2, r = 0.8, b = 0.8): Mask {
  return maskOf([pt(l, t), pt(r, t), pt(r, b), pt(l, b)], "Rectangle");
}

export function ellipseMask(cx = 0.5, cy = 0.5, rx = 0.3, ry = 0.3): Mask {
  const k = 0.5522847498;
  return maskOf(
    [
      pt(cx, cy - ry, { in_x: -rx * k, out_x: rx * k }),
      pt(cx + rx, cy, { in_y: -ry * k, out_y: ry * k }),
      pt(cx, cy + ry, { in_x: rx * k, out_x: -rx * k }),
      pt(cx - rx, cy, { in_y: ry * k, out_y: -ry * k }),
    ],
    "Ellipse",
  );
}

export const roundPoint = (p: MaskPoint): MaskPoint => ({
  x: round(p.x), y: round(p.y), in_x: round(p.in_x), in_y: round(p.in_y), out_x: round(p.out_x), out_y: round(p.out_y),
});
