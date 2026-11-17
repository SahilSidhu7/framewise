import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  EngineClient,
  isTauri,
  type AiStatus,
  type AiVariation,
  type EditOp,
  type ExportOptions,
  type LookInfo,
  type MediaItem,
  type Snapshot,
} from "./engine";

type Tab = "ai" | "looks" | "adjust" | "crop" | "edits";

interface SliderSpec {
  op: string;
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  neutral?: number;
}

const ADJUST_GROUPS: { title: string; sliders: SliderSpec[] }[] = [
  {
    title: "Light",
    sliders: [
      { op: "adjust", key: "exposure", label: "Exposure", min: -3, max: 3, step: 0.05 },
      { op: "adjust", key: "contrast", label: "Contrast", min: -100, max: 100, step: 1 },
      { op: "adjust", key: "highlights", label: "Highlights", min: -100, max: 100, step: 1 },
      { op: "adjust", key: "shadows", label: "Shadows", min: -100, max: 100, step: 1 },
      { op: "adjust", key: "whites", label: "Whites", min: -100, max: 100, step: 1 },
      { op: "adjust", key: "blacks", label: "Blacks", min: -100, max: 100, step: 1 },
    ],
  },
  {
    title: "Color",
    sliders: [
      { op: "adjust", key: "temperature", label: "Temperature", min: -100, max: 100, step: 1 },
      { op: "adjust", key: "tint", label: "Tint", min: -100, max: 100, step: 1 },
      { op: "adjust", key: "vibrance", label: "Vibrance", min: -100, max: 100, step: 1 },
      { op: "adjust", key: "saturation", label: "Saturation", min: -100, max: 100, step: 1 },
    ],
  },
  {
    title: "Detail",
    sliders: [
      { op: "adjust", key: "clarity", label: "Clarity", min: -100, max: 100, step: 1 },
      { op: "sharpen", key: "amount", label: "Sharpen", min: 0, max: 150, step: 1 },
      { op: "denoise", key: "strength", label: "Noise reduction", min: 0, max: 100, step: 1 },
      { op: "dehaze", key: "amount", label: "Dehaze", min: -100, max: 100, step: 1 },
    ],
  },
  {
    title: "Effects",
    sliders: [
      { op: "vignette", key: "amount", label: "Vignette", min: -1, max: 1, step: 0.01 },
      { op: "grain", key: "amount", label: "Grain", min: 0, max: 1, step: 0.01 },
      { op: "fade", key: "amount", label: "Fade", min: 0, max: 1, step: 0.01 },
      { op: "bloom", key: "amount", label: "Glow", min: 0, max: 1, step: 0.01 },
    ],
  },
];

const AI_CHIPS = [
  "Make it aesthetic",
  "Give it an ancient look",
  "Cinematic movie look",
  "Fix the lighting",
  "Vintage film",
  "Black and white, dramatic",
  "Dreamy and soft",
  "Crop it for an Instagram post",
];

const ASPECTS: [string, string | null][] = [
  ["Original", null],
  ["1:1", "1:1"],
  ["4:5", "4:5"],
  ["9:16", "9:16"],
  ["16:9", "16:9"],
  ["3:2", "3:2"],
];

const OP_LABELS: Record<string, string> = {
  adjust: "Adjust",
  look: "Look",
  crop: "Crop",
  rotate: "Rotate",
  straighten: "Straighten",
  flip: "Flip",
  black_white: "Black & white",
  color_grade: "Color grade",
  hsl: "Color mix",
  chromatic_aberration: "Lens fringing",
  light_leak: "Light leak",
  dust_scratches: "Dust & scratches",
};

const DRAG_PREVIEW_MAX = 900;

interface Props {
  client: EngineClient;
  snapshot: Snapshot;
  item: MediaItem;
  onNotice: (msg: string) => void;
}

function opParams(stack: EditOp[], op: string): Record<string, unknown> | undefined {
  return stack.find((o) => o.op === op)?.params;
}

export function PhotoEditor({ client, snapshot, item, onNotice }: Props) {
  const stack = useMemo(() => snapshot.project.edits[item.id] ?? [], [snapshot, item.id]);
  const [tab, setTab] = useState<Tab>("ai");
  const [draft, setDraft] = useState<EditOp[] | null>(null);
  const [draftUrl, setDraftUrl] = useState<string | null>(null);
  const [compare, setCompare] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const previewSeq = useRef(0);
  const timer = useRef<number | undefined>(undefined);

  // committed preview; `rev` changes after every edit so the browser refetches
  const committedUrl = client.renderUrl(item.id, { max: 1600, rev: snapshot.revision });
  const shownUrl = compare ? client.renderUrl(item.id, { max: 1600, bypass: true }) : draftUrl ?? committedUrl;

  useEffect(() => setLoaded(false), [committedUrl]);
  useEffect(() => {
    // new committed state arrived: drop the drag draft
    setDraft(null);
    setDraftUrl((u) => {
      if (u) URL.revokeObjectURL(u);
      return null;
    });
  }, [snapshot.revision, item.id]);

  const run = useCallback(
    async (name: string, params: Record<string, unknown>, asAi = false) => {
      try {
        await (asAi ? client.runAi(name, params) : client.run(name, params));
      } catch (e) {
        onNotice(e instanceof Error ? e.message : String(e));
      }
    },
    [client, onNotice],
  );

  const schedulePreview = useCallback(
    (ops: EditOp[]) => {
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(async () => {
        const seq = ++previewSeq.current;
        try {
          const url = await client.preview(item.id, ops, DRAG_PREVIEW_MAX);
          if (seq !== previewSeq.current) return URL.revokeObjectURL(url);
          setDraftUrl((old) => {
            if (old) URL.revokeObjectURL(old);
            return url;
          });
        } catch {
          /* transient while dragging */
        }
      }, 70);
    },
    [client, item.id],
  );

  // ----------------------------------------------------- slider plumbing
  const current = draft ?? stack;
  const valueOf = (s: SliderSpec): number => {
    const v = opParams(current, s.op)?.[s.key];
    return typeof v === "number" ? v : s.neutral ?? 0;
  };

  const onSlide = (s: SliderSpec, value: number) => {
    const base = draft ?? stack;
    const idx = base.findIndex((o) => o.op === s.op);
    const next = idx >= 0
      ? base.map((o, i) => (i === idx ? { ...o, params: { ...o.params, [s.key]: value } } : o))
      : [...base, { id: "draft", op: s.op, params: { [s.key]: value }, enabled: true }];
    setDraft(next);
    schedulePreview(next);
  };

  const onCommit = async (s: SliderSpec, value: number) => {
    const existing = stack.find((o) => o.op === s.op);
    const neutral = s.neutral ?? 0;
    if (existing) {
      const others = Object.entries(existing.params).filter(([k, v]) => k !== s.key && v !== 0);
      if (value === neutral && others.length === 0) {
        await run("photo.remove_op", { media_id: item.id, op_id: existing.id });
      } else {
        await run("photo.set_op", { media_id: item.id, op_id: existing.id, params: { [s.key]: value } });
      }
    } else if (value !== neutral) {
      // corrections go before the look, finishing effects after
      const lookIdx = stack.findIndex((o) => o.op === "look");
      const before = ["adjust", "denoise", "dehaze"].includes(s.op) && lookIdx >= 0;
      await run("photo.add_op", { media_id: item.id, op: s.op, params: { [s.key]: value }, index: before ? lookIdx : null });
    }
  };

  const resetSlider = (s: SliderSpec) => {
    onSlide(s, s.neutral ?? 0);
    void onCommit(s, s.neutral ?? 0);
  };

  return (
    <div className="photo">
      <div className="photo-stage">
        <div className="photo-tools">
          <button
            className={compare ? "active" : ""}
            onPointerDown={() => setCompare(true)}
            onPointerUp={() => setCompare(false)}
            onPointerLeave={() => setCompare(false)}
            title="Hold to see the original"
          >
            Hold for before
          </button>
          <div className="spacer" />
          <button className="primary" onClick={() => setShowExport(true)}>
            Export
          </button>
        </div>
        <div className="photo-canvas">
          <img src={shownUrl} alt={item.name} onLoad={() => setLoaded(true)} className={loaded || draftUrl ? "" : "loading"} />
          {!loaded && !draftUrl && <div className="spinner" />}
        </div>
      </div>

      <aside className="photo-panel">
        <nav className="tabs">
          {(["ai", "looks", "adjust", "crop", "edits"] as Tab[]).map((t) => (
            <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
              {t === "ai" ? "Auto edit" : t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        <div className="panel-body">
          {tab === "ai" && <AiPanel client={client} item={item} revision={snapshot.revision} run={run} onNotice={onNotice} />}
          {tab === "looks" && <LooksPanel client={client} item={item} stack={stack} revision={snapshot.revision} run={run} />}
          {tab === "adjust" &&
            ADJUST_GROUPS.map((g) => (
              <section key={g.title} className="group">
                <h4>{g.title}</h4>
                {g.sliders.map((s) => (
                  <Slider key={s.op + s.key} spec={s} value={valueOf(s)} onSlide={onSlide} onCommit={onCommit} onReset={resetSlider} />
                ))}
              </section>
            ))}
          {tab === "crop" && <CropPanel item={item} stack={stack} run={run} />}
          {tab === "edits" && <EditsPanel item={item} stack={stack} run={run} />}
        </div>
      </aside>

      {showExport && <ExportDialog client={client} item={item} onClose={() => setShowExport(false)} onNotice={onNotice} />}
    </div>
  );
}

// ------------------------------------------------------------------ slider

function Slider({
  spec,
  value,
  onSlide,
  onCommit,
  onReset,
}: {
  spec: SliderSpec;
  value: number;
  onSlide: (s: SliderSpec, v: number) => void;
  onCommit: (s: SliderSpec, v: number) => void;
  onReset: (s: SliderSpec) => void;
}) {
  const fmt = spec.step < 1 ? value.toFixed(2) : String(Math.round(value));
  return (
    <label className="slider" onDoubleClick={() => onReset(spec)} title="Double-click to reset">
      <span className="slider-head">
        <span>{spec.label}</span>
        <span className="muted">{fmt}</span>
      </span>
      <input
        type="range"
        min={spec.min}
        max={spec.max}
        step={spec.step}
        value={value}
        onChange={(e) => onSlide(spec, Number(e.target.value))}
        onPointerUp={(e) => onCommit(spec, Number((e.target as HTMLInputElement).value))}
        onKeyUp={(e) => onCommit(spec, Number((e.target as HTMLInputElement).value))}
      />
    </label>
  );
}

// ---------------------------------------------------------------------- AI

type RunFn = (name: string, params: Record<string, unknown>, asAi?: boolean) => Promise<void>;

function AiPanel({ client, item, revision, run, onNotice }: { client: EngineClient; item: MediaItem; revision: number; run: RunFn; onNotice: (m: string) => void }) {
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<AiStatus | null>(null);
  const [results, setResults] = useState<{ prompt: string; variations: AiVariation[] } | null>(null);
  const [applied, setApplied] = useState<number | null>(null);

  useEffect(() => {
    client.aiWarmup().then(setStatus, () => undefined);
    const id = window.setInterval(() => client.aiStatus().then(setStatus, () => undefined), 3000);
    return () => window.clearInterval(id);
  }, [client]);

  const ask = async (text: string) => {
    if (!text.trim() || busy) return;
    setBusy(true);
    setApplied(null);
    try {
      const res = await client.aiEdit(item.id, text.trim());
      setResults({ prompt: text.trim(), variations: res.variations });
      if (res.variations[0]) {
        await run("photo.apply_program", { media_id: item.id, program: res.variations[0].program, mode: "replace" }, true);
        setApplied(0);
      }
    } catch (e) {
      onNotice(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const apply = async (i: number) => {
    if (!results) return;
    await run("photo.apply_program", { media_id: item.id, program: results.variations[i].program, mode: "replace" }, true);
    setApplied(i);
  };

  const modelLabel =
    status?.state === "ready" ? "On-device model ready" :
    status?.state === "idle" ? "Loading model…" :
    status?.state === "not_installed" ? "Rules only · model not installed" :
    status?.state === "error" ? "Rules only · model failed to start" : "…";

  return (
    <div className="ai">
      <p className="muted small">Describe the result you want. You get three versions; the first one is applied and Undo reverts it.</p>
      <textarea
        value={prompt}
        placeholder="e.g. make it look like an old film photo"
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            void ask(prompt);
          }
        }}
        rows={3}
      />
      <div className="row between">
        <span className={`small ${status?.state === "ready" ? "ok" : "muted"}`}>{modelLabel}</span>
        <button className="primary" disabled={busy || !prompt.trim()} onClick={() => ask(prompt)}>
          {busy ? "Working…" : "Apply"}
        </button>
      </div>
      <div className="examples">
        <span className="label">Examples</span>
        {AI_CHIPS.slice(0, 4).map((c) => (
          <button key={c} type="button" className="example" disabled={busy} onClick={() => { setPrompt(c); void ask(c); }}>
            {c}
          </button>
        ))}
      </div>

      {results && (
        <div className="variations">
          <h4>“{results.prompt}”</h4>
          {results.variations.map((v, i) => (
            <div key={v.program} className={`variation ${applied === i ? "applied" : ""}`} onClick={() => apply(i)}>
              <img src={client.renderUrl(item.id, { max: 360, program: v.program })} alt="" />
              <div className="variation-body">
                <div className="row between">
                  <strong>Version {i + 1}</strong>
                  <span className="muted small">{applied === i ? "applied" : v.source === "llm" ? "model" : "preset"}</span>
                </div>
                {v.rationale && <p className="small">{v.rationale}</p>}
                <details onClick={(e) => e.stopPropagation()}>
                  <summary className="small muted">What it does</summary>
                  <pre>{v.program}</pre>
                </details>
              </div>
            </div>
          ))}
        </div>
      )}
      <span hidden>{revision}</span>
    </div>
  );
}

// ------------------------------------------------------------------- looks

function LooksPanel({ client, item, stack, revision, run }: { client: EngineClient; item: MediaItem; stack: EditOp[]; revision: number; run: RunFn }) {
  const [looks, setLooks] = useState<LookInfo[]>([]);
  useEffect(() => {
    client.looks().then(setLooks, () => undefined);
  }, [client]);
  const current = stack.find((o) => o.op === "look");
  const currentName = current?.params.name as string | undefined;
  const [strength, setStrength] = useState<number>((current?.params.strength as number) ?? 1);
  useEffect(() => setStrength((current?.params.strength as number) ?? 1), [current]);
  // thumbnails: the photo's other edits + each look, small; stable across slider changes
  const baseRev = useMemo(() => stack.filter((o) => o.op !== "look").map((o) => o.id + JSON.stringify(o.params)).join("|"), [stack]);

  return (
    <div>
      {current && (
        <section className="group">
          <h4>{looks.find((l) => l.name === currentName)?.title ?? currentName}</h4>
          <label className="slider">
            <span className="slider-head">
              <span>Strength</span>
              <span className="muted">{Math.round(strength * 100)}%</span>
            </span>
            <input
              type="range" min={0} max={1.5} step={0.05} value={strength}
              onChange={(e) => setStrength(Number(e.target.value))}
              onPointerUp={() => run("photo.set_op", { media_id: item.id, op_id: current.id, params: { strength } })}
            />
          </label>
          <button onClick={() => run("photo.remove_op", { media_id: item.id, op_id: current.id })}>Remove look</button>
        </section>
      )}
      <div className="look-grid">
        {looks.map((l) => (
          <button
            key={l.name}
            className={`look ${l.name === currentName ? "active" : ""}`}
            title={l.description}
            onClick={() => run("photo.apply_look", { media_id: item.id, name: l.name, strength: 1 })}
          >
            <img loading="lazy" src={client.renderUrl(item.id, { max: 220, look: l.name, rev: hash(baseRev) })} alt="" />
            <span>{l.title}</span>
          </button>
        ))}
      </div>
      <span hidden>{revision}</span>
    </div>
  );
}

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return h;
}

// -------------------------------------------------------------------- crop

function CropPanel({ item, stack, run }: { item: MediaItem; stack: EditOp[]; run: RunFn }) {
  const crop = stack.find((o) => o.op === "crop");
  const rotate = stack.find((o) => o.op === "rotate");
  const straighten = stack.find((o) => o.op === "straighten");
  const flip = stack.find((o) => o.op === "flip");
  const aspect = (crop?.params.aspect as string | undefined) ?? null;

  const setAspect = (a: string | null) => {
    if (!a) return crop && run("photo.remove_op", { media_id: item.id, op_id: crop.id });
    if (crop) return run("photo.set_op", { media_id: item.id, op_id: crop.id, params: { aspect: a } });
    return run("photo.add_op", { media_id: item.id, op: "crop", params: { aspect: a } });
  };
  const turn = (delta: number) => {
    const angle = ((((rotate?.params.angle as number) ?? 0) + delta + 540) % 360) - 180;
    if (rotate && angle === 0) return run("photo.remove_op", { media_id: item.id, op_id: rotate.id });
    if (rotate) return run("photo.set_op", { media_id: item.id, op_id: rotate.id, params: { angle } });
    return run("photo.add_op", { media_id: item.id, op: "rotate", params: { angle }, index: 0 });
  };

  return (
    <div>
      <section className="group">
        <h4>Aspect ratio</h4>
        <div className="button-grid">
          {ASPECTS.map(([label, a]) => (
            <button key={label} className={aspect === a ? "active" : ""} onClick={() => setAspect(a)}>
              {label}
            </button>
          ))}
        </div>
      </section>
      <section className="group">
        <h4>Orientation</h4>
        <div className="button-grid">
          <button onClick={() => turn(90)}>⟲ Rotate left</button>
          <button onClick={() => turn(-90)}>⟳ Rotate right</button>
          <button
            className={flip ? "active" : ""}
            onClick={() => (flip ? run("photo.remove_op", { media_id: item.id, op_id: flip.id }) : run("photo.add_op", { media_id: item.id, op: "flip", params: { horizontal: true } }))}
          >
            ⇋ Flip
          </button>
          <button
            className={straighten ? "active" : ""}
            onClick={() =>
              straighten
                ? run("photo.remove_op", { media_id: item.id, op_id: straighten.id })
                : run("photo.add_op", { media_id: item.id, op: "straighten", params: {}, index: 0 })
            }
          >
            ⊾ Auto-straighten
          </button>
        </div>
      </section>
    </div>
  );
}

// ------------------------------------------------------------------- edits

function EditsPanel({ item, stack, run }: { item: MediaItem; stack: EditOp[]; run: RunFn }) {
  if (stack.length === 0) return <p className="muted small">No edits yet. Start with Auto edit or a look.</p>;
  return (
    <div>
      <ul className="edit-list">
        {stack.map((o) => (
          <li key={o.id} className={o.enabled ? "" : "off"}>
            <input
              type="checkbox"
              checked={o.enabled}
              onChange={() => run("photo.set_op", { media_id: item.id, op_id: o.id, enabled: !o.enabled })}
              title="Toggle"
            />
            <div className="grow">
              <strong>{OP_LABELS[o.op] ?? o.op[0].toUpperCase() + o.op.slice(1)}</strong>
              <div className="muted small">{describe(o)}</div>
            </div>
            <button className="icon" title="Remove" onClick={() => run("photo.remove_op", { media_id: item.id, op_id: o.id })}>
              ✕
            </button>
          </li>
        ))}
      </ul>
      <button onClick={() => run("photo.reset", { media_id: item.id })}>Reset all edits</button>
    </div>
  );
}

function describe(o: EditOp): string {
  if (o.op === "look") return `${String(o.params.name).replace(/_/g, " ")} · ${Math.round(Number(o.params.strength ?? 1) * 100)}%`;
  return Object.entries(o.params)
    .map(([k, v]) => `${k} ${typeof v === "number" ? +v.toFixed(2) : String(v)}`)
    .join(", ");
}

// ------------------------------------------------------------------ export

function ExportDialog({ client, item, onClose, onNotice }: { client: EngineClient; item: MediaItem; onClose: () => void; onNotice: (m: string) => void }) {
  const [format, setFormat] = useState<ExportOptions["format"]>("jpeg");
  const [quality, setQuality] = useState(95);
  const [bitDepth, setBitDepth] = useState<8 | 16>(8);
  const [size, setSize] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const ext = { jpeg: "jpg", png: "png", tiff: "tif", webp: "webp" }[format];
  const baseName = item.name.replace(/\.[^.]+$/, "") + "_edit";

  const go = async () => {
    let path: string | null = null;
    if (isTauri()) {
      const { save } = await import("@tauri-apps/plugin-dialog");
      path = await save({ defaultPath: `${baseName}.${ext}`, filters: [{ name: format.toUpperCase(), extensions: [ext] }] });
    } else {
      const dir = item.path.replace(/[\\/][^\\/]*$/, "");
      path = window.prompt("Save to:", `${dir}\\${baseName}.${ext}`);
    }
    if (!path) return;
    setBusy(true);
    try {
      const res = await client.exportImage(item.id, {
        path,
        format,
        quality,
        bit_depth: format === "png" || format === "tiff" ? bitDepth : 8,
        long_edge: size,
      });
      onNotice(`Exported ${res.width}×${res.height} → ${res.path}`);
      onClose();
    } catch (e) {
      onNotice(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="card modal" onClick={(e) => e.stopPropagation()}>
        <h3>Export photo</h3>
        <label>
          Format
          <select value={format} onChange={(e) => setFormat(e.target.value as ExportOptions["format"])}>
            <option value="jpeg">JPEG (share anywhere)</option>
            <option value="png">PNG (lossless)</option>
            <option value="tiff">TIFF (print / archive)</option>
            <option value="webp">WebP (small, web)</option>
          </select>
        </label>
        {(format === "jpeg" || format === "webp") && (
          <label className="slider">
            <span className="slider-head">
              <span>Quality</span>
              <span className="muted">{quality}</span>
            </span>
            <input type="range" min={60} max={100} value={quality} onChange={(e) => setQuality(Number(e.target.value))} />
          </label>
        )}
        {(format === "png" || format === "tiff") && (
          <label>
            Bit depth
            <select value={bitDepth} onChange={(e) => setBitDepth(Number(e.target.value) as 8 | 16)}>
              <option value={8}>8-bit</option>
              <option value={16}>16-bit (best for further editing)</option>
            </select>
          </label>
        )}
        <label>
          Size
          <select value={size ?? ""} onChange={(e) => setSize(e.target.value ? Number(e.target.value) : null)}>
            <option value="">Original resolution</option>
            <option value={4096}>4096 px long edge</option>
            <option value={2048}>2048 px (great for sharing)</option>
            <option value={1080}>1080 px (Instagram)</option>
          </select>
        </label>
        <div className="row end">
          <button onClick={onClose}>Cancel</button>
          <button className="primary" disabled={busy} onClick={go}>
            {busy ? "Exporting…" : "Export"}
          </button>
        </div>
      </div>
    </div>
  );
}
