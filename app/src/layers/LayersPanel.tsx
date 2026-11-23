// Layers tab: the layer stack (tracks, top first), the selected layer's
// properties with keyframe stopwatches, and its masks.

import { useEffect, useState } from "react";

import { BLEND_MODES, type Clip, type LookInfo, type Mask, type Project, type ShapeStyle, type TextStyle, type Track } from "../engine";
import { clipValue } from "../live/timeline";
import { activeOn, animated, ellipseMask, keyAt, keyTimes, rectMask, toggleKeyHere, toggleStopwatch, commitValues, type Param, type RunFn } from "./edit";

const label = (s: string) => s.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const MATTES = ["none", "alpha", "alpha_inverted", "luma", "luma_inverted"] as const;
const MATTE_LABEL: Record<string, string> = { none: "No matte", alpha: "Alpha matte", alpha_inverted: "Alpha inverted", luma: "Luma matte", luma_inverted: "Luma inverted" };

export type Tool = "move" | "pen";

interface Props {
  project: Project;
  playhead: number;
  selClip: string | null;
  onSelectClip: (id: string | null) => void;
  onSeek: (t: number) => void;
  run: RunFn;
  tool: Tool;
  setTool: (t: Tool) => void;
  /** Photo document: the photo's media id. Layers are added to its stack, and there is no time. */
  doc?: string;
  /** Looks offered on adjustment layers */
  looks?: LookInfo[];
}

const layerName = (project: Project, c: Clip) =>
  c.name || project.media[c.media_id]?.name || (c.source === "text" ? `“${(c.text?.text ?? "").split("\n")[0].slice(0, 30)}”` : c.source === "shape" ? (c.shape?.kind === "ellipse" ? "Ellipse" : "Rectangle") : c.source === "solid" ? "Solid" : c.source === "adjustment" ? "Adjustment layer" : "Layer");

export function LayersPanel({ project, playhead, selClip, onSelectClip, onSeek, run, tool, setTool, doc, looks }: Props) {
  const still = doc !== undefined;
  const images = still ? Object.values(project.media).filter((m) => m.kind === "image" && m.id !== doc) : [];
  const visual = project.tracks.filter((t) => t.kind !== "audio");
  const top = [...visual].reverse();
  const entry = selClip ? project.tracks.flatMap((track) => track.clips.map((clip) => ({ track, clip }))).find((e) => e.clip.id === selClip) : undefined;
  const sel = entry && entry.track.kind !== "audio" ? entry : undefined;
  const { width: W, height: H } = project.settings;
  const end = Math.max(0, ...project.tracks.flatMap((t) => t.clips.map((c) => c.start + (c.out_point - c.in_point) / c.speed)));

  const move = (track: Track, dir: 1 | -1) => {
    const i = visual.indexOf(track);
    const other = visual[i + dir];
    if (!other) return;
    run("track.set", { track_id: track.id, index: project.tracks.indexOf(other) });
  };
  const addLayer = async (source: "solid" | "adjustment" | "media" | "text" | "shape", mediaId = "") => {
    const color = source === "solid" ? "#3a6df0" : "#000000";
    if (still) {
      const r = await run("layer.add", { source, doc, media_id: mediaId, color });
      if (r?.clip_id) onSelectClip(r.clip_id as string);
      return;
    }
    const start = Math.min(playhead, Math.max(0, end - 0.5));
    const r = await run("layer.add", { source, start, duration: Math.max(1, Math.min(5, (end || 5) - start)), color });
    if (r?.clip_id) onSelectClip(r.clip_id as string);
  };

  return (
    <>
      <div className="group">
        <div className="row between">
          <h4>Layers</h4>
          <div className="row tools">
            <button className={tool === "move" ? "active" : ""} onClick={() => setTool("move")} title="Move, scale and rotate in the viewer (V)">✥ Move</button>
            <button className={tool === "pen" ? "active" : ""} onClick={() => setTool("pen")} disabled={!sel} title="Draw a mask on the selected layer (P)">✎ Pen</button>
          </div>
        </div>
        <ul className="layer-list">
          {top.map((track, i) => {
            const active = activeOn(track, playhead);
            const shown = sel?.track.id === track.id ? sel.clip : active ?? track.clips[0];
            const isSel = sel?.track.id === track.id;
            const below = top[i + 1];
            return (
              <li key={track.id} className={`layer ${isSel ? "selected" : ""} ${track.hidden ? "off" : ""}`} onClick={() => shown && onSelectClip(shown.id)}>
                <button className="icon" title={track.hidden ? "Show" : "Hide"} onClick={(e) => { e.stopPropagation(); run("track.set", { track_id: track.id, hidden: !track.hidden }); }}>
                  {track.hidden ? "◌" : "👁"}
                </button>
                <button className="icon" title={track.locked ? "Unlock" : "Lock"} onClick={(e) => { e.stopPropagation(); run("track.set", { track_id: track.id, locked: !track.locked }); }}>
                  {track.locked ? "🔒" : "🔓"}
                </button>
                {shown?.source === "solid" && <span className="swatch" style={{ background: shown.color }} />}
                <div className="layer-text">
                  <TrackName track={track} run={run} />
                  <span className="muted small">
                    {shown ? layerName(project, shown) : "Empty"}
                    {shown && shown.blend && shown.blend !== "normal" ? ` · ${label(shown.blend)}` : ""}
                    {shown && (shown.masks?.length ?? 0) > 0 ? ` · ${shown.masks!.length} mask${shown.masks!.length > 1 ? "s" : ""}` : ""}
                    {!active && shown && !still ? " · not at playhead" : ""}
                  </span>
                </div>
                {below && (
                  <select
                    className="matte"
                    value={below.matte ?? "none"}
                    title={`Use this layer to cut “${below.name}” below it`}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => run("track.set", { track_id: below.id, matte: e.target.value })}
                  >
                    {MATTES.map((m) => <option key={m} value={m}>{m === "none" ? "—" : MATTE_LABEL[m]}</option>)}
                  </select>
                )}
                <div className="updown">
                  <button className="icon" disabled={i === 0} title="Move up" onClick={(e) => { e.stopPropagation(); move(track, 1); }}>▲</button>
                  <button className="icon" disabled={i === top.length - 1} title="Move down" onClick={(e) => { e.stopPropagation(); move(track, -1); }}>▼</button>
                </div>
              </li>
            );
          })}
          {still && (
            <li className={`layer base ${sel ? "" : "selected"}`} onClick={() => onSelectClip(null)} title="The photo with its edits (Adjust, Looks, Crop tabs)">
              <div className="layer-text">
                <span className="layer-name">Background</span>
                <span className="muted small">{project.media[doc!]?.name ?? "Photo"} · edits</span>
              </div>
            </li>
          )}
        </ul>
        <p className="muted small">A matte on a layer cuts the layer below it, and is not drawn itself.</p>
        <div className="row" style={{ flexWrap: "wrap" }}>
          {still && (
            <select value="" title="Place another photo of this project as a layer" disabled={images.length === 0}
              onChange={(e) => e.target.value && addLayer("media", e.target.value)}>
              <option value="">{images.length ? "+ Image…" : "+ Image (import more photos)"}</option>
              {images.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          )}
          <button onClick={() => addLayer("text")}>+ Text</button>
          <button onClick={() => addLayer("shape")}>+ Shape</button>
          <button onClick={() => addLayer("solid")}>+ Solid</button>
          <button onClick={() => addLayer("adjustment")}>+ Adjustment</button>
          {!still && <button onClick={() => run("track.add", { kind: "overlay" })}>+ Track</button>}
        </div>
      </div>

      {sel ? (
        <>
          <LayerProps key={sel.clip.id} project={project} clip={sel.clip} track={sel.track} playhead={playhead} run={run} onSeek={onSeek} W={W} H={H} still={still} looks={looks} />
          <MasksSection clip={sel.clip} run={run} onPen={() => setTool("pen")} />
        </>
      ) : (
        <p className="muted small">Select a layer to change its blend mode, transform{still ? "" : ", keyframes"} and masks.</p>
      )}
    </>
  );
}

function TrackName({ track, run }: { track: Track; run: RunFn }) {
  const [edit, setEdit] = useState<string | null>(null);
  if (edit === null)
    return (
      <span className="layer-name" title="Double-click to rename" onDoubleClick={(e) => { e.stopPropagation(); setEdit(track.name); }}>
        {track.name}
      </span>
    );
  const done = () => {
    if (edit.trim() && edit !== track.name) run("track.set", { track_id: track.id, name: edit.trim().slice(0, 40) });
    setEdit(null);
  };
  return (
    <input
      autoFocus
      className="layer-name"
      value={edit}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => setEdit(e.target.value)}
      onBlur={done}
      onKeyDown={(e) => {
        if (e.key === "Enter") done();
        if (e.key === "Escape") setEdit(null);
      }}
    />
  );
}

// ---------------------------------------------------------------- properties

interface PropDef {
  p: Param;
  label: string;
  min: number;
  max: number;
  step: number;
  fmt: (v: number) => string;
}

function LayerProps({ project, clip, track, playhead, run, onSeek, W, H, still, looks }: { project: Project; clip: Clip; track: Track; playhead: number; run: RunFn; onSeek: (t: number) => void; W: number; H: number; still: boolean; looks?: LookInfo[] }) {
  const look = clip.ops?.find((o) => o.op === "look");
  const [name, setName] = useState(clip.name ?? "");
  useEffect(() => setName(clip.name ?? ""), [clip.name]);
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const px = (n: number) => (v: number) => `${Math.round(v * n)} px`;
  const defs: PropDef[] = [
    { p: "opacity", label: "Opacity", min: 0, max: 1, step: 0.01, fmt: pct },
    { p: "x", label: "Position X", min: -1.5, max: 1.5, step: 0.001, fmt: px(W) },
    { p: "y", label: "Position Y", min: -1.5, max: 1.5, step: 0.001, fmt: px(H) },
    { p: "scale", label: "Scale", min: 0.05, max: 4, step: 0.01, fmt: pct },
    { p: "rotation", label: "Rotation", min: -180, max: 180, step: 0.5, fmt: (v) => `${v.toFixed(1)}°` },
    { p: "anchor_x", label: "Anchor X", min: -1, max: 1, step: 0.001, fmt: px(W) },
    { p: "anchor_y", label: "Anchor Y", min: -1, max: 1, step: 0.001, fmt: px(H) },
  ];
  const keys = keyTimes(clip);
  const prev = [...keys].reverse().find((k) => k < playhead - 1e-3);
  const next = keys.find((k) => k > playhead + 1e-3);
  const crop = clip.crop ?? { left: 0, top: 0, right: 0, bottom: 0 };
  const locked = !!track.locked;

  return (
    <>
      <div className="group">
        <h4>Layer</h4>
        <input
          placeholder={project.media[clip.media_id]?.name ?? "Layer name"}
          value={name}
          disabled={locked}
          onChange={(e) => setName(e.target.value)}
          onBlur={() => name !== (clip.name ?? "") && run("clip.set_layer", { clip_id: clip.id, name: name.slice(0, 80) })}
        />
        {clip.source === "solid" && (
          <label className="row between small">
            <span>Colour</span>
            <input type="color" style={{ width: 48 }} value={clip.color ?? "#000000"} disabled={locked} onChange={(e) => run("clip.set_layer", { clip_id: clip.id, color: e.target.value })} />
          </label>
        )}
        {clip.source === "adjustment" && (looks?.length ? (
          <>
            <label className="small">Look (applies to every layer below)</label>
            <select value={look ? String(look.params.name) : ""} disabled={locked}
              onChange={(e) => run("clip.apply_look", { clip_id: clip.id, name: e.target.value || null })}>
              <option value="">None</option>
              {looks.map((l) => <option key={l.name} value={l.name}>{l.title}</option>)}
            </select>
          </>
        ) : <p className="muted small">An adjustment layer applies its effects (Clip tab → look) to every layer below it.</p>)}
        <label className="small">Blend mode</label>
        <select value={clip.blend ?? "normal"} disabled={locked} onChange={(e) => run("clip.set_layer", { clip_id: clip.id, blend: e.target.value })}>
          {BLEND_MODES.map((b) => <option key={b} value={b}>{label(b)}</option>)}
        </select>
        {locked && <p className="muted small">This layer's track is locked.</p>}
      </div>
      {clip.text && <TextProps clip={clip} text={clip.text} run={run} disabled={locked} />}
      {clip.shape && <ShapeProps clip={clip} shape={clip.shape} run={run} disabled={locked} />}
      <div className="group">
        <div className="row between">
          <h4>Transform</h4>
          {!still && <div className="row tools">
            <button className="icon" disabled={prev === undefined} title="Previous keyframe" onClick={() => prev !== undefined && onSeek(prev)}>◀◆</button>
            <button className="icon" disabled={next === undefined} title="Next keyframe" onClick={() => next !== undefined && onSeek(next)}>◆▶</button>
          </div>}
        </div>
        {defs.map((d) => (
          <PropRow key={d.p} def={d} clip={clip} playhead={playhead} run={run} disabled={locked} still={still} />
        ))}
        <button className="small" disabled={locked} onClick={() => run("clip.set_layer", { clip_id: clip.id, x: 0, y: 0, scale: 1, rotation: 0, anchor_x: 0, anchor_y: 0 })}>
          Reset transform
        </button>
      </div>
      {(clip.source ?? "media") === "media" && (
        <div className="group">
          <h4>Crop</h4>
          {(["left", "top", "right", "bottom"] as const).map((side) => (
            <Slider key={side} label={label(side)} value={crop[side]} min={0} max={0.9} step={0.005} fmt={(v) => `${(v * 100).toFixed(1)}%`} disabled={locked}
              onCommit={(v) => run("clip.set_layer", { clip_id: clip.id, [`crop_${side}`]: v })} />
          ))}
        </div>
      )}
    </>
  );
}

function PropRow({ def, clip, playhead, run, disabled, still }: { def: PropDef; clip: Clip; playhead: number; run: RunFn; disabled: boolean; still: boolean }) {
  const anim = animated(clip, def.p);
  const here = !!keyAt(clip, def.p, playhead);
  const value = clipValue(clip, def.p, playhead);
  return (
    <div className={`prop ${anim ? "anim" : ""}`}>
      {!still && <button className={`icon stopwatch ${anim ? "on" : ""}`} disabled={disabled} title={anim ? "Stop animating (keeps the current value)" : "Animate: add a keyframe here"}
        onClick={() => toggleStopwatch(run, clip, def.p, playhead)}>⏱</button>}
      <Slider label={def.label} value={value} min={def.min} max={def.max} step={def.step} fmt={def.fmt} disabled={disabled}
        onCommit={(v) => commitValues(run, clip, { [def.p]: v }, playhead)} />
      {!still && <button className={`icon key ${here ? "on" : ""}`} disabled={disabled || !anim} title={here ? "Remove this keyframe" : "Add a keyframe here"}
        onClick={() => toggleKeyHere(run, clip, def.p, playhead)}>◆</button>}
    </div>
  );
}

function Slider({ label: text, value, min, max, step, fmt, onCommit, disabled }: { label: string; value: number; min: number; max: number; step: number; fmt: (v: number) => string; onCommit: (v: number) => void; disabled?: boolean }) {
  const [v, setV] = useState(value);
  const [typing, setTyping] = useState<string | null>(null);
  useEffect(() => setV(value), [value]);
  const commit = (x: number) => {
    if (Number.isFinite(x) && Math.abs(x - value) > 1e-9) onCommit(x);
  };
  return (
    <div className="slider">
      <div className="row between small">
        <span>{text}</span>
        {typing === null ? (
          <span className="val" title="Click to type a value" onClick={() => !disabled && setTyping(String(+v.toFixed(4)))}>{fmt(v)}</span>
        ) : (
          <input
            autoFocus
            className="val-input"
            value={typing}
            onChange={(e) => setTyping(e.target.value)}
            onBlur={() => { commit(Number(typing)); setTyping(null); }}
            onKeyDown={(e) => {
              if (e.key === "Enter") (e.target as HTMLInputElement).blur();
              if (e.key === "Escape") setTyping(null);
            }}
          />
        )}
      </div>
      <input type="range" min={min} max={max} step={step} value={Math.min(max, Math.max(min, v))} disabled={disabled}
        onChange={(e) => setV(Number(e.target.value))}
        onPointerUp={() => commit(v)}
        onKeyUp={() => commit(v)} />
    </div>
  );
}

// ---------------------------------------------------------------- text and shapes

const WEIGHTS: [number, string][] = [[300, "Light"], [400, "Regular"], [500, "Medium"], [600, "Semibold"], [700, "Bold"], [800, "Extra bold"], [900, "Black"]];

function TextProps({ clip, text, run, disabled }: { clip: Clip; text: TextStyle; run: RunFn; disabled: boolean }) {
  const [words, setWords] = useState(text.text);
  useEffect(() => setWords(text.text), [text.text]);
  const set = (p: Partial<TextStyle>) => run("clip.set_text", { clip_id: clip.id, ...p });
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  return (
    <div className="group">
      <h4>Text</h4>
      <textarea rows={3} value={words} disabled={disabled} onChange={(e) => setWords(e.target.value)}
        onBlur={() => words.trim() && words !== text.text && set({ text: words.slice(0, 2000) })} />
      <div className="row">
        <select value={text.font} disabled={disabled} onChange={(e) => set({ font: e.target.value as TextStyle["font"] })} title="Font">
          <option value="sans">Inter (sans)</option>
          <option value="display">Montserrat (display)</option>
          <option value="serif">Playfair (serif)</option>
        </select>
        <select value={text.weight} disabled={disabled} onChange={(e) => set({ weight: Number(e.target.value) })} title="Weight">
          {WEIGHTS.map(([w, n]) => <option key={w} value={w}>{n}</option>)}
        </select>
      </div>
      <div className="row">
        <label className="row small">
          <span>Colour</span>
          <input type="color" style={{ width: 40 }} value={text.color} disabled={disabled} onChange={(e) => set({ color: e.target.value })} />
        </label>
        <div className="row tools">
          {(["left", "center", "right"] as const).map((a) => (
            <button key={a} className={text.align === a ? "active" : ""} disabled={disabled} onClick={() => set({ align: a })} title={`Align ${a}`}>
              {a === "left" ? "⯇" : a === "center" ? "≡" : "⯈"}
            </button>
          ))}
        </div>
      </div>
      <Slider label="Size" value={text.size} min={0.01} max={0.5} step={0.002} fmt={pct} disabled={disabled} onCommit={(v) => set({ size: v })} />
      <Slider label="Line height" value={text.line_height} min={0.6} max={3} step={0.05} fmt={(v) => v.toFixed(2)} disabled={disabled} onCommit={(v) => set({ line_height: v })} />
      <div className="prop">
        <Slider label="Outline" value={text.stroke_width} min={0} max={0.3} step={0.005} fmt={pct} disabled={disabled} onCommit={(v) => set({ stroke_width: v })} />
        <input type="color" style={{ width: 40 }} value={text.stroke_color} disabled={disabled} title="Outline colour" onChange={(e) => set({ stroke_color: e.target.value })} />
      </div>
    </div>
  );
}

function ShapeProps({ clip, shape, run, disabled }: { clip: Clip; shape: ShapeStyle; run: RunFn; disabled: boolean }) {
  const set = (p: Record<string, unknown>) => run("clip.set_shape", { clip_id: clip.id, ...p });
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  return (
    <div className="group">
      <h4>Shape</h4>
      <select value={shape.kind} disabled={disabled} onChange={(e) => set({ kind: e.target.value })}>
        <option value="rectangle">Rectangle</option>
        <option value="ellipse">Ellipse</option>
      </select>
      <Slider label="Width" value={shape.width} min={0.01} max={2} step={0.005} fmt={pct} disabled={disabled} onCommit={(v) => set({ width: v })} />
      <Slider label="Height" value={shape.height} min={0.01} max={2} step={0.005} fmt={pct} disabled={disabled} onCommit={(v) => set({ height: v })} />
      {shape.kind === "rectangle" && (
        <Slider label="Corner radius" value={shape.radius} min={0} max={0.5} step={0.002} fmt={pct} disabled={disabled} onCommit={(v) => set({ radius: v })} />
      )}
      <div className="row">
        <label className="row small">
          <input type="checkbox" style={{ width: "auto" }} checked={!!shape.fill} disabled={disabled || (!!shape.fill && shape.stroke_width === 0)}
            title={shape.fill && shape.stroke_width === 0 ? "Add an outline first" : undefined}
            onChange={(e) => set({ fill: e.target.checked ? "#ffffff" : "none" })} />
          <span>Fill</span>
        </label>
        {shape.fill && <input type="color" style={{ width: 40 }} value={shape.fill} disabled={disabled} onChange={(e) => set({ fill: e.target.value })} />}
      </div>
      <div className="prop">
        <Slider label="Outline" value={shape.stroke_width} min={0} max={0.2} step={0.001} fmt={pct} disabled={disabled}
          onCommit={(v) => (v > 0 || shape.fill) && set({ stroke_width: v })} />
        <input type="color" style={{ width: 40 }} value={shape.stroke_color} disabled={disabled} title="Outline colour" onChange={(e) => set({ stroke_color: e.target.value })} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- masks

function MasksSection({ clip, run, onPen }: { clip: Clip; run: RunFn; onPen: () => void }) {
  const masks = clip.masks ?? [];
  const save = (next: Mask[]) => run("clip.set_masks", { clip_id: clip.id, masks: next });
  const patch = (i: number, p: Partial<Mask>) => save(masks.map((m, j) => (j === i ? { ...m, ...p } : m)));
  return (
    <div className="group">
      <h4>Masks</h4>
      <div className="row">
        <button onClick={() => save([...masks, rectMask()])}>+ Rectangle</button>
        <button onClick={() => save([...masks, ellipseMask()])}>+ Ellipse</button>
        <button onClick={onPen} title="Click points in the viewer; click the first point to close. Drag while clicking for curves.">✎ Pen</button>
      </div>
      {masks.length === 0 && <p className="muted small">No masks: the whole layer shows. Drag mask points in the viewer; Alt-drag a point for curves, double-click it for a sharp corner.</p>}
      {masks.map((m, i) => (
        <div key={m.id} className={`mask ${m.enabled ? "" : "off"}`}>
          <div className="row between">
            <label className="row small">
              <input type="checkbox" style={{ width: "auto" }} checked={m.enabled} onChange={(e) => patch(i, { enabled: e.target.checked })} />
              <span>{m.name || `Mask ${i + 1}`}</span>
            </label>
            <button className="icon" title="Delete mask" onClick={() => save(masks.filter((_, j) => j !== i))}>✕</button>
          </div>
          <div className="row">
            <select value={m.mode} onChange={(e) => patch(i, { mode: e.target.value as Mask["mode"] })}>
              {(["add", "subtract", "intersect", "difference"] as const).map((k) => <option key={k} value={k}>{label(k)}</option>)}
            </select>
            <label className="row small">
              <input type="checkbox" style={{ width: "auto" }} checked={m.inverted} onChange={(e) => patch(i, { inverted: e.target.checked })} />
              <span>Invert</span>
            </label>
          </div>
          <Slider label="Feather" value={m.feather} min={0} max={0.2} step={0.002} fmt={(v) => `${(v * 100).toFixed(1)}%`} onCommit={(v) => patch(i, { feather: v })} />
          <Slider label="Expansion" value={m.expansion} min={-0.2} max={0.2} step={0.002} fmt={(v) => `${(v * 100).toFixed(1)}%`} onCommit={(v) => patch(i, { expansion: v })} />
          <Slider label="Opacity" value={m.opacity} min={0} max={1} step={0.01} fmt={(v) => `${Math.round(v * 100)}%`} onCommit={(v) => patch(i, { opacity: v })} />
        </div>
      ))}
    </div>
  );
}
