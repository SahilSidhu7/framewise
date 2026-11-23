// Handles over the viewer for the selected layer: drag to move, corners to
// scale, the top knob to rotate, the crosshair to move the anchor point; mask
// points to drag (Alt-drag pulls curve handles, double-click makes a corner);
// and the pen tool, which draws a new mask point by point.

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import type { Clip, Mask, MaskPoint, Project } from "../engine";
import { clipValue, matrix } from "../live/timeline";
import { maskOf, roundPoint, sourceSize, type Values } from "./edit";
import type { Tool } from "./LayersPanel";

type M = number[];
const apply = (m: M, x: number, y: number): [number, number] => [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]];
function invert(m: M): M {
  const [a, b, c, d, e, f] = m;
  const det = a * e - b * d || 1e-12;
  const ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
  return [ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)];
}

interface Props {
  project: Project;
  clip: Clip;
  t: number;
  tool: Tool;
  locked: boolean;
  onDraft: (v: { values?: Values; masks?: Mask[] } | null) => void;
  onCommit: (values: Values) => void;
  onMasks: (masks: Mask[]) => void;
  onPenDone: () => void;
}

type Drag =
  | { kind: "move" | "scale" | "rotate" | "anchor"; x0: number; y0: number; base: Record<string, number>; pivot: [number, number] }
  | { kind: "point"; mask: number; point: number; curve: boolean; x0: number; y0: number; masks: Mask[] }
  | { kind: "pen"; x0: number; y0: number };

export function ViewerOverlay({ project, clip, t, tool, locked, onDraft, onCommit, onMasks, onPenDone }: Props) {
  const { width: W, height: H } = project.settings;
  const svg = useRef<SVGSVGElement>(null);
  const [k, setK] = useState(1); // canvas px per screen px
  const drag = useRef<Drag | null>(null);
  const last = useRef<Values | null>(null);
  const lastMasks = useRef<Mask[] | null>(null);
  const [pen, setPen] = useState<MaskPoint[]>([]);
  const [hover, setHover] = useState<[number, number] | null>(null);

  useLayoutEffect(() => {
    const el = svg.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      if (r.width && r.height) setK(Math.max(W / r.width, H / r.height));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [W, H]);

  useEffect(() => setPen([]), [clip.id, tool]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (tool !== "pen") return;
      if (e.key === "Escape") {
        if (pen.length) setPen([]);
        else onPenDone();
      } else if (e.key === "Enter" && pen.length >= 3) finishPen(pen);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const [sw, sh] = sourceSize(project, clip);
  const { m } = matrix(clip, t, sw, sh, W, H);
  const inv = invert(m);
  const crop = clip.crop ?? { left: 0, top: 0, right: 0, bottom: 0 };
  const corners = [
    apply(m, crop.left * sw, crop.top * sh),
    apply(m, (1 - crop.right) * sw, crop.top * sh),
    apply(m, (1 - crop.right) * sw, (1 - crop.bottom) * sh),
    apply(m, crop.left * sw, (1 - crop.bottom) * sh),
  ];
  const v = (p: string) => clipValue(clip, p, t);
  const pivot: [number, number] = [W / 2 + v("anchor_x") * W + v("x") * W, H / 2 + v("anchor_y") * H + v("y") * H];
  const topMid: [number, number] = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2];
  const edge = Math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]) || 1;
  const up: [number, number] = [(corners[0][1] - corners[1][1]) / edge, (corners[1][0] - corners[0][0]) / edge]; // outward normal of the top edge
  const knob: [number, number] = [topMid[0] + up[0] * 28 * k, topMid[1] + up[1] * 28 * k];

  // layer space (0..1 of the source) <-> canvas pixels
  const toCanvas = (x: number, y: number) => apply(m, x * sw, y * sh);
  const toLayer = (X: number, Y: number): [number, number] => {
    const [x, y] = apply(inv, X, Y);
    return [x / sw, y / sh];
  };
  const point = (e: React.PointerEvent | React.MouseEvent): [number, number] => {
    const ctm = svg.current!.getScreenCTM();
    if (!ctm) return [0, 0];
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    return [p.x, p.y];
  };

  const masks = clip.masks ?? [];
  const pathD = (pts: MaskPoint[], closed = true) => {
    if (!pts.length) return "";
    const P = (x: number, y: number) => toCanvas(x, y).map((n) => n.toFixed(2)).join(" ");
    let d = `M ${P(pts[0].x, pts[0].y)}`;
    const n = closed ? pts.length : pts.length - 1;
    for (let i = 0; i < n; i++) {
      const a = pts[i];
      const b = pts[(i + 1) % pts.length];
      d += ` C ${P(a.x + a.out_x, a.y + a.out_y)} ${P(b.x + b.in_x, b.y + b.in_y)} ${P(b.x, b.y)}`;
    }
    return closed ? d + " Z" : d;
  };

  // ---- transform drags
  const startTransform = (kind: "move" | "scale" | "rotate" | "anchor") => (e: React.PointerEvent) => {
    if (locked || tool !== "move") return;
    e.stopPropagation();
    e.preventDefault();
    (e.target as Element).setPointerCapture(e.pointerId);
    const [x0, y0] = point(e);
    drag.current = { kind, x0, y0, pivot, base: { x: v("x"), y: v("y"), scale: v("scale"), rotation: v("rotation"), anchor_x: v("anchor_x"), anchor_y: v("anchor_y") } };
  };

  const onMove = (e: React.PointerEvent) => {
    const d = drag.current;
    const [X, Y] = point(e);
    if (tool === "pen") setHover([X, Y]);
    if (!d) return;
    if (d.kind === "point") {
      const [lx, ly] = toLayer(X, Y);
      const ms = d.masks.map((mk, i) => {
        if (i !== d.mask) return mk;
        const pts = mk.points.map((p, j) => {
          if (j !== d.point) return p;
          if (d.curve) {
            const ox = lx - p.x, oy = ly - p.y;
            return { ...p, out_x: ox, out_y: oy, in_x: -ox, in_y: -oy };
          }
          return { ...p, x: lx, y: ly };
        });
        return { ...mk, points: pts };
      });
      lastMasks.current = ms;
      onDraft({ masks: ms });
      return;
    }
    if (d.kind === "pen") {
      if (Math.hypot(X - d.x0, Y - d.y0) < 4 * k) return; // a click, not a curve
      setPen((ps) => {
        if (!ps.length) return ps;
        const [lx, ly] = toLayer(X, Y);
        const p = ps[ps.length - 1];
        const ox = lx - p.x, oy = ly - p.y;
        return [...ps.slice(0, -1), { ...p, out_x: ox, out_y: oy, in_x: -ox, in_y: -oy }];
      });
      return;
    }
    const b = d.base;
    let out: Values = {};
    if (d.kind === "move") {
      out = { x: b.x + (X - d.x0) / W, y: b.y + (Y - d.y0) / H };
      if (e.shiftKey) Math.abs(X - d.x0) > Math.abs(Y - d.y0) ? (out.y = b.y) : (out.x = b.x);
    } else if (d.kind === "scale") {
      const r0 = Math.hypot(d.x0 - d.pivot[0], d.y0 - d.pivot[1]) || 1;
      const r1 = Math.hypot(X - d.pivot[0], Y - d.pivot[1]);
      out = { scale: Math.min(20, Math.max(0.01, b.scale * (r1 / r0))) };
    } else if (d.kind === "rotate") {
      const a0 = Math.atan2(d.y0 - d.pivot[1], d.x0 - d.pivot[0]);
      const a1 = Math.atan2(Y - d.pivot[1], X - d.pivot[0]);
      let r = b.rotation + ((a1 - a0) * 180) / Math.PI;
      if (e.shiftKey) r = Math.round(r / 15) * 15;
      out = { rotation: r };
    } else {
      // anchor: move the pivot and offset the position so the picture stays put
      const ax = b.anchor_x + (X - d.x0) / W;
      const ay = b.anchor_y + (Y - d.y0) / H;
      const th = (b.rotation * Math.PI) / 180;
      const cs = Math.cos(th) * b.scale, sn = Math.sin(th) * b.scale;
      const dx = (ax - b.anchor_x) * W, dy = (ay - b.anchor_y) * H;
      out = { anchor_x: ax, anchor_y: ay, x: b.x + (cs * dx - sn * dy - dx) / W, y: b.y + (sn * dx + cs * dy - dy) / H };
    }
    last.current = out;
    onDraft({ values: out });
  };

  const onUp = () => {
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    if (d.kind === "point") {
      if (lastMasks.current) onMasks(lastMasks.current.map((mk) => ({ ...mk, points: mk.points.map(roundPoint) })));
      lastMasks.current = null;
      return;
    }
    if (d.kind === "pen") return;
    if (last.current) onCommit(last.current);
    else onDraft(null);
    last.current = null;
  };

  // ---- mask points
  const startPoint = (mask: number, pointIdx: number) => (e: React.PointerEvent) => {
    if (locked || tool !== "move") return;
    e.stopPropagation();
    e.preventDefault();
    (e.target as Element).setPointerCapture(e.pointerId);
    const [x0, y0] = point(e);
    drag.current = { kind: "point", mask, point: pointIdx, curve: e.altKey, x0, y0, masks };
  };
  const cornerPoint = (mask: number, pointIdx: number) => (e: React.MouseEvent) => {
    e.stopPropagation();
    if (locked) return;
    onMasks(masks.map((mk, i) => (i !== mask ? mk : { ...mk, points: mk.points.map((p, j) => (j === pointIdx ? { ...p, in_x: 0, in_y: 0, out_x: 0, out_y: 0 } : p)) })));
  };

  // ---- pen
  function finishPen(pts: MaskPoint[]) {
    setPen([]);
    onMasks([...masks, maskOf(pts.map(roundPoint), `Mask ${masks.length + 1}`)]);
    onPenDone();
  }
  const penDown = (e: React.PointerEvent) => {
    if (tool !== "pen" || locked) return;
    e.stopPropagation();
    e.preventDefault();
    (e.target as Element).setPointerCapture(e.pointerId);
    const [X, Y] = point(e);
    if (pen.length >= 3) {
      const [fx, fy] = toCanvas(pen[0].x, pen[0].y);
      if (Math.hypot(X - fx, Y - fy) < 10 * k) return finishPen(pen);
    }
    const [lx, ly] = toLayer(X, Y);
    setPen((ps) => [...ps, { x: lx, y: ly, in_x: 0, in_y: 0, out_x: 0, out_y: 0 }]);
    drag.current = { kind: "pen", x0: X, y0: Y };
  };

  const r = 5 * k;
  const stroke = { vectorEffect: "non-scaling-stroke" as const };
  const poly = corners.map((c) => c.join(",")).join(" ");
  const lastPen = pen.length ? toCanvas(pen[pen.length - 1].x, pen[pen.length - 1].y) : null;

  return (
    <svg
      ref={svg}
      className={`v-overlay ${tool}`}
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      onPointerDown={penDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerLeave={() => setHover(null)}
      onPointerCancel={onUp}
    >
      {/* mask outlines */}
      {masks.map((mk) => (
        <path key={mk.id} d={pathD(mk.points)} className={`ov-mask ${mk.enabled ? "" : "off"}`} style={stroke} />
      ))}
      {tool === "move" && (
        <>
          <polygon points={poly} className="ov-box" style={stroke} onPointerDown={startTransform("move")} />
          <line x1={topMid[0]} y1={topMid[1]} x2={knob[0]} y2={knob[1]} className="ov-line" style={stroke} />
          <circle cx={knob[0]} cy={knob[1]} r={r} className="ov-rot" style={stroke} onPointerDown={startTransform("rotate")}>
            <title>Rotate (Shift: 15° steps)</title>
          </circle>
          {corners.map((c, i) => (
            <rect key={i} x={c[0] - r} y={c[1] - r} width={2 * r} height={2 * r} className="ov-handle" style={stroke} onPointerDown={startTransform("scale")}>
              <title>Scale</title>
            </rect>
          ))}
          <g className="ov-anchor" onPointerDown={(e) => startTransform(e.altKey ? "anchor" : "move")(e)}>
            <circle cx={pivot[0]} cy={pivot[1]} r={r * 1.4} style={stroke} />
            <line x1={pivot[0] - r * 2.4} y1={pivot[1]} x2={pivot[0] + r * 2.4} y2={pivot[1]} style={stroke} />
            <line x1={pivot[0]} y1={pivot[1] - r * 2.4} x2={pivot[0]} y2={pivot[1] + r * 2.4} style={stroke} />
            <title>Anchor point (rotation and scale pivot): Alt-drag to move it</title>
          </g>
          {masks.map((mk, i) =>
            mk.points.map((p, j) => {
              const [cx, cy] = toCanvas(p.x, p.y);
              return (
                <rect key={`${mk.id}-${j}`} x={cx - r * 0.8} y={cy - r * 0.8} width={1.6 * r} height={1.6 * r} className="ov-point" style={stroke}
                  onPointerDown={startPoint(i, j)} onDoubleClick={cornerPoint(i, j)}>
                  <title>Drag to move · Alt-drag for a curve · Double-click for a corner</title>
                </rect>
              );
            }),
          )}
        </>
      )}
      {tool === "pen" && (
        <>
          <rect x={0} y={0} width={W} height={H} className="ov-catch" />
          {pen.length > 0 && <path d={pathD(pen, false)} className="ov-pen" style={stroke} />}
          {lastPen && hover && <line x1={lastPen[0]} y1={lastPen[1]} x2={hover[0]} y2={hover[1]} className="ov-pen ghost" style={stroke} />}
          {pen.map((p, j) => {
            const [cx, cy] = toCanvas(p.x, p.y);
            return <rect key={j} x={cx - r * 0.8} y={cy - r * 0.8} width={1.6 * r} height={1.6 * r} className={`ov-point ${j === 0 && pen.length >= 3 ? "close" : ""}`} style={stroke} />;
          })}
        </>
      )}
    </svg>
  );
}
