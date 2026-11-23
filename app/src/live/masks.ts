// Layer masks for the live preview: the engine's rules (video/layers.py
// masks_coverage) drawn with a 2D canvas. Coverage is kept in the red channel.

import type { Mask } from "../engine";

const cache = new Map<string, HTMLCanvasElement>();

function canvas(w: number, h: number, fill: string): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d")!;
  g.fillStyle = fill;
  g.fillRect(0, 0, w, h);
  return c;
}

/** The closed bezier path in canvas pixels (layer space -> source px -> canvas px). */
function path(mask: Mask, sw: number, sh: number, m: number[]): Path2D {
  const map = (x: number, y: number): [number, number] => {
    const px = x * sw;
    const py = y * sh;
    // +0.5: canvas paths use pixel edges, the engine uses pixel centres
    return [m[0] * px + m[1] * py + m[2] + 0.5, m[3] * px + m[4] * py + m[5] + 0.5];
  };
  const p = new Path2D();
  const pts = mask.points;
  p.moveTo(...map(pts[0].x, pts[0].y));
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % pts.length];
    p.bezierCurveTo(...map(a.x + a.out_x, a.y + a.out_y), ...map(b.x + b.in_x, b.y + b.in_y), ...map(b.x, b.y));
  }
  p.closePath();
  return p;
}

function shape(mask: Mask, sw: number, sh: number, m: number[], w: number, h: number): HTMLCanvasElement {
  const short = Math.min(w, h);
  const c = canvas(w, h, "#000");
  const g = c.getContext("2d")!;
  const p = path(mask, sw, sh, m);
  const grow = mask.expansion * short;
  const sigma = mask.feather > 0 ? Math.max(0.5, (mask.feather * short) / 2) : 0;
  const layer = sigma ? canvas(w, h, "#000") : c;
  const lg = layer.getContext("2d")!;
  lg.fillStyle = "#fff";
  lg.fill(p);
  if (Math.abs(grow) >= 0.5) {
    lg.lineJoin = "round";
    lg.lineWidth = 2 * Math.abs(grow);
    lg.strokeStyle = grow > 0 ? "#fff" : "#000";
    lg.stroke(p);
  }
  if (sigma) {
    g.filter = `blur(${sigma}px)`;
    g.drawImage(layer, 0, 0);
    g.filter = "none";
  }
  if (mask.inverted) {
    g.globalCompositeOperation = "difference";
    g.fillStyle = "#fff";
    g.fillRect(0, 0, w, h);
    g.globalCompositeOperation = "source-over";
  }
  if (mask.opacity < 1) {
    const o = canvas(w, h, "#000");
    const og = o.getContext("2d")!;
    og.globalAlpha = mask.opacity;
    og.drawImage(c, 0, 0);
    return o;
  }
  return c;
}

/** All of a layer's enabled masks combined, or null if it has none. Cached per shape and placement. */
export function masksCanvas(masks: Mask[] | undefined, sw: number, sh: number, m: number[], w: number, h: number): { canvas: HTMLCanvasElement; key: string } | null {
  const active = (masks ?? []).filter((k) => k.enabled);
  if (!active.length) return null;
  const key = JSON.stringify([active, sw, sh, w, h, m.map((v) => Math.round(v * 1000) / 1000)]);
  const hit = cache.get(key);
  if (hit) return { canvas: hit, key };
  const first = active[0].mode;
  const acc = canvas(w, h, first === "subtract" || first === "intersect" ? "#fff" : "#000");
  const g = acc.getContext("2d")!;
  for (const k of active) {
    const s = shape(k, sw, sh, m, w, h);
    if (k.mode === "add") g.globalCompositeOperation = "screen"; // a + m - a*m
    else if (k.mode === "intersect") g.globalCompositeOperation = "multiply"; // a * m
    else if (k.mode === "difference") g.globalCompositeOperation = "difference"; // |a - m|
    else {
      // subtract: a * (1 - m)
      const inv = s.getContext("2d")!;
      inv.globalCompositeOperation = "difference";
      inv.fillStyle = "#fff";
      inv.fillRect(0, 0, w, h);
      g.globalCompositeOperation = "multiply";
    }
    g.drawImage(s, 0, 0);
    g.globalCompositeOperation = "source-over";
  }
  cache.set(key, acc);
  if (cache.size > 24) cache.delete(cache.keys().next().value as string);
  return { canvas: acc, key };
}
