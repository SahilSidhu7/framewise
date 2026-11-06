import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  clipDuration,
  EngineClient,
  isTauri,
  projectDuration,
  type Caption,
  type Clip,
  type EngineEvent,
  type LookInfo,
  type Snapshot,
  type Track,
  type VideoVariation,
} from "./engine";

type Tab = "ai" | "clip" | "look" | "text" | "export";
type RunFn = (name: string, params: Record<string, unknown>) => Promise<Record<string, unknown> | undefined>;

const AI_CHIPS = [
  "Make an energetic reel with music",
  "Cinematic video with captions",
  "Combine the clips into a short story",
  "Aesthetic slow vlog",
  "Vintage film look, calm pace",
  "Highlights in 30 seconds",
];
const TRANSITIONS = ["none", "crossfade", "fade_black", "fade_white", "wipe_left", "wipe_right", "slide_left", "slide_up", "zoom", "blur"];
const MOTIONS = ["none", "zoom_in", "zoom_out", "pan_left", "pan_right"];
const PRESETS: [string, string][] = [
  ["project", "Project size"],
  ["youtube_1080p", "YouTube 1080p (16:9)"],
  ["youtube_4k", "YouTube 4K (16:9)"],
  ["reel_1080", "Reel / TikTok / Shorts (9:16)"],
  ["portrait_4_5", "Instagram post (4:5)"],
  ["square_1080", "Square (1:1)"],
  ["hd_720p", "720p (small file)"],
];
const SNAP_PX = 8;

const fmt = (t: number) => {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s < 10 ? "0" : ""}${s.toFixed(1)}`;
};
const label = (s: string) => s.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

interface Props {
  client: EngineClient;
  snapshot: Snapshot;
  onNotice: (m: string | null) => void;
}

export function VideoEditor({ client, snapshot, onNotice }: Props) {
  const { project, revision } = snapshot;
  const duration = projectDuration(project);
  const fps = project.settings.fps || 30;
  const [tab, setTab] = useState<Tab>("ai");
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selClip, setSelClip] = useState<string | null>(null);
  const [selCaption, setSelCaption] = useState<string | null>(null);
  const [zoom, setZoom] = useState(60);
  const [preview, setPreview] = useState<{ url: string; revision: number } | null>(null);
  const [previewProgress, setPreviewProgress] = useState<number | null>(null);
  const [jobProgress, setJobProgress] = useState<Record<string, number>>({});
  const [exportJob, setExportJob] = useState<string | null>(null);
  const [captionJob, setCaptionJob] = useState<string | null>(null);
  const [frameLoading, setFrameLoading] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const wantPlay = useRef(false);

  const run: RunFn = useCallback(
    async (name, params) => {
      try {
        return (await client.run(name, params)).result;
      } catch (e) {
        onNotice(e instanceof Error ? e.message : String(e));
        return undefined;
      }
    },
    [client, onNotice],
  );

  const clipById = useMemo(() => {
    const m = new Map<string, { track: Track; clip: Clip }>();
    for (const track of project.tracks) for (const clip of track.clips) m.set(clip.id, { track, clip });
    return m;
  }, [project]);
  const selected = selClip ? clipById.get(selClip) : undefined;
  const caption = project.captions.find((c) => c.id === selCaption);

  // ---- engine events: render jobs
  useEffect(
    () =>
      client.events((e: EngineEvent) => {
        const kind = e.kind as string | undefined;
        if (e.type === "job_progress") {
          if (kind === "preview") setPreviewProgress(e.progress as number);
          else setJobProgress((p) => ({ ...p, [e.job as string]: e.progress as number }));
        } else if (e.type === "job_done") {
          const result = (e.result ?? {}) as Record<string, unknown>;
          if (kind === "preview") {
            setPreviewProgress(null);
            setPreview({ url: client.fileUrl(result.url as string), revision: result.revision as number });
          } else if (kind === "export") {
            setExportJob(null);
            onNotice(`Exported ${result.path} (${result.width}×${result.height}, ${result.duration}s, ${result.encoder})`);
          } else if (kind === "captions") {
            setCaptionJob(null);
            onNotice(`Added ${result.captions} captions`);
          }
        } else if (e.type === "job_failed" || e.type === "job_cancelled") {
          if (kind === "preview") setPreviewProgress(null);
          if (kind === "export") setExportJob(null);
          if (kind === "captions") setCaptionJob(null);
          if (e.type === "job_failed") onNotice(`${kind} failed: ${e.error}`);
        }
      }),
    [client, onNotice],
  );

  // ---- playable preview: re-render shortly after every change
  useEffect(() => {
    if (duration <= 0) return;
    const timer = window.setTimeout(() => {
      client.previewRender().then(
        (r) => {
          if (r.url) setPreview({ url: client.fileUrl(r.url), revision: r.revision });
          else setPreviewProgress(0);
        },
        () => undefined,
      );
    }, 900);
    return () => window.clearTimeout(timer);
  }, [client, revision, duration]);

  const fresh = preview !== null && preview.revision === revision;

  useEffect(() => {
    if (!fresh && playing) {
      videoRef.current?.pause();
      setPlaying(false);
      wantPlay.current = true; // resume once the new preview is ready
    }
  }, [fresh, playing]);

  const play = useCallback(() => {
    const v = videoRef.current;
    if (!fresh || !v) {
      wantPlay.current = true;
      return;
    }
    if (playhead >= duration - 0.05) setPlayhead(0);
    v.currentTime = playhead >= duration - 0.05 ? 0 : playhead;
    v.play().then(() => setPlaying(true), () => undefined);
  }, [fresh, playhead, duration]);

  const pause = useCallback(() => {
    videoRef.current?.pause();
    wantPlay.current = false;
    setPlaying(false);
  }, []);

  useEffect(() => {
    if (fresh && wantPlay.current) {
      wantPlay.current = false;
      const t = window.setTimeout(play, 50);
      return () => window.clearTimeout(t);
    }
  }, [fresh, play]);

  // follow the video while playing
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const v = videoRef.current;
      if (v) setPlayhead(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const seek = useCallback(
    (t: number) => {
      const c = Math.max(0, Math.min(duration, Math.round(t * fps) / fps));
      setPlayhead(c);
      if (playing && videoRef.current) videoRef.current.currentTime = c;
    },
    [duration, fps, playing],
  );

  // ---- editing actions
  const split = useCallback(async () => {
    const target =
      (selected && playhead > selected.clip.start && playhead < selected.clip.start + clipDuration(selected.clip) ? selected : undefined) ??
      [...clipById.values()].find(({ track, clip }) => track.kind !== "audio" && playhead > clip.start && playhead < clip.start + clipDuration(clip));
    if (!target) return onNotice("Put the playhead over a clip to split it.");
    const res = await run("clip.split", { clip_id: target.clip.id, at: playhead });
    if (res?.clip_id) setSelClip(res.clip_id as string);
  }, [selected, playhead, clipById, run, onNotice]);

  const removeSelected = useCallback(async () => {
    if (selected) {
      await run(selected.track.kind === "audio" ? "clip.remove" : "clip.ripple_delete", { clip_id: selected.clip.id });
      setSelClip(null);
    } else if (caption) {
      await run("caption.remove", { caption_id: caption.id });
      setSelCaption(null);
    }
  }, [selected, caption, run]);

  // ---- keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.closest("input, textarea, select")) return;
      if (e.key === " ") {
        e.preventDefault();
        playing ? pause() : play();
      } else if (e.key === "s" || e.key === "S") split();
      else if (e.key === "Delete" || e.key === "Backspace") removeSelected();
      else if (e.key === "ArrowRight") seek(playhead + (e.shiftKey ? 1 : 1 / fps));
      else if (e.key === "ArrowLeft") seek(playhead - (e.shiftKey ? 1 : 1 / fps));
      else if (e.key === "Home") seek(0);
      else if (e.key === "End") seek(duration);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [playing, play, pause, split, removeSelected, seek, playhead, fps, duration]);

  const frameSrc = duration > 0 ? client.frameUrl(Math.min(playhead, Math.max(0, duration - 1 / fps)), { max: 1280, rev: revision }) : null;
  useEffect(() => setFrameLoading(true), [frameSrc]);

  const empty = project.tracks.every((t) => t.clips.length === 0);

  return (
    <div className="video">
      <div className="video-top">
        <section className="v-viewer">
          <div className="v-stage">
            {empty ? (
              <div className="v-empty muted">
                <p>Your timeline is empty.</p>
                <p className="small">Import clips, then ask the AI (right) to edit them, or double-click media to add it yourself.</p>
              </div>
            ) : (
              <>
                {frameSrc && (
                  <img
                    className={`${playing ? "hidden" : ""} ${frameLoading ? "loading" : ""}`}
                    src={frameSrc}
                    alt=""
                    onLoad={() => setFrameLoading(false)}
                    onError={() => setFrameLoading(false)}
                  />
                )}
                {preview && (
                  <video
                    ref={videoRef}
                    className={playing ? "" : "hidden"}
                    src={preview.url}
                    preload="auto"
                    onEnded={() => {
                      setPlaying(false);
                      setPlayhead(duration);
                    }}
                    onPause={() => setPlaying(false)}
                  />
                )}
              </>
            )}
          </div>
          <div className="transport">
            <button onClick={() => seek(0)} title="Start (Home)">⏮</button>
            <button className="primary play" onClick={playing ? pause : play} disabled={empty} title="Play / pause (Space)">
              {playing ? "❚❚" : "▶"}
            </button>
            <span className="tc">
              {fmt(playhead)} / {fmt(duration)}
            </span>
            <div className="spacer" />
            {previewProgress !== null && !fresh && (
              <span className="muted small">Preparing playback {Math.round(previewProgress * 100)}%</span>
            )}
            {fresh && <span className="ok small">Ready to play</span>}
          </div>
        </section>

        <aside className="photo-panel v-panel">
          <nav className="tabs">
            {(["ai", "clip", "look", "text", "export"] as Tab[]).map((t) => (
              <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
                {{ ai: "AI", clip: "Clip", look: "Look", text: "Text", export: "Export" }[t]}
              </button>
            ))}
          </nav>
          <div className="panel-body">
            {tab === "ai" && <VideoAi client={client} snapshot={snapshot} onNotice={onNotice} />}
            {tab === "clip" && (selected ? <ClipPanel client={client} snapshot={snapshot} entry={selected} run={run} onSplit={split} onDelete={removeSelected} /> : <p className="muted small">Select a clip on the timeline.</p>)}
            {tab === "look" && <LookPanel client={client} snapshot={snapshot} playhead={playhead} run={run} />}
            {tab === "text" && (
              <TextPanel
                client={client}
                snapshot={snapshot}
                playhead={playhead}
                run={run}
                selected={selCaption}
                onSelect={(id) => setSelCaption(id)}
                captionJob={captionJob}
                setCaptionJob={setCaptionJob}
                onNotice={onNotice}
              />
            )}
            {tab === "export" && (
              <ExportPanel client={client} snapshot={snapshot} job={exportJob} setJob={setExportJob} progress={exportJob ? jobProgress[exportJob] ?? 0 : 0} onNotice={onNotice} />
            )}
          </div>
        </aside>
      </div>

      <Timeline
        client={client}
        snapshot={snapshot}
        duration={duration}
        playhead={playhead}
        zoom={zoom}
        setZoom={setZoom}
        selClip={selClip}
        selCaption={selCaption}
        onSeek={seek}
        onSelectClip={(id) => {
          setSelClip(id);
          setSelCaption(null);
          if (id) setTab("clip");
        }}
        onSelectCaption={(id) => {
          setSelCaption(id);
          setSelClip(null);
          setTab("text");
        }}
        run={run}
        onSplit={split}
        onDelete={removeSelected}
      />
    </div>
  );
}

// ------------------------------------------------------------------ timeline

interface Drag {
  clipId: string;
  mode: "move" | "head" | "tail";
  x0: number;
  dx: number;
}

function Timeline(props: {
  client: EngineClient;
  snapshot: Snapshot;
  duration: number;
  playhead: number;
  zoom: number;
  setZoom: (z: number) => void;
  selClip: string | null;
  selCaption: string | null;
  onSeek: (t: number) => void;
  onSelectClip: (id: string | null) => void;
  onSelectCaption: (id: string) => void;
  run: RunFn;
  onSplit: () => void;
  onDelete: () => void;
}) {
  const { client, snapshot, duration, playhead, zoom, setZoom, selClip, selCaption, onSeek, onSelectClip, onSelectCaption, run } = props;
  const { project } = snapshot;
  const [drag, setDrag] = useState<Drag | null>(null);
  const laneRef = useRef<HTMLDivElement>(null);
  const width = Math.max(800, (duration + 10) * zoom);
  const tracks = [...project.tracks.filter((t) => t.kind !== "audio"), ...project.tracks.filter((t) => t.kind === "audio")];

  const snap = (t: number, exclude: string) => {
    let best = t;
    let bestPx = SNAP_PX;
    for (const t2 of project.tracks)
      for (const c of t2.clips) {
        if (c.id === exclude) continue;
        for (const e of [c.start, c.start + clipDuration(c)]) {
          const px = Math.abs(e - t) * zoom;
          if (px < bestPx) [best, bestPx] = [e, px];
        }
      }
    for (const e of [0, playhead]) {
      const px = Math.abs(e - t) * zoom;
      if (px < bestPx) [best, bestPx] = [e, px];
    }
    return best;
  };

  const geometry = (c: Clip) => {
    let start = c.start;
    let dur = clipDuration(c);
    if (drag && drag.clipId === c.id) {
      const dt = drag.dx / zoom;
      if (drag.mode === "move") start = Math.max(0, c.start + dt);
      else if (drag.mode === "head") {
        const d = Math.min(Math.max(dt, -c.in_point / c.speed), dur - 0.1);
        start = c.start + d;
        dur -= d;
      } else {
        const media = project.media[c.media_id];
        const maxDur = media?.kind === "image" ? 600 : ((media?.duration ?? c.out_point) - c.in_point) / c.speed;
        dur = Math.min(Math.max(0.1, dur + dt), maxDur);
      }
    }
    return { start, dur };
  };

  const onPointerDown = (e: React.PointerEvent, c: Clip, mode: Drag["mode"]) => {
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setDrag({ clipId: c.id, mode, x0: e.clientX, dx: 0 });
    onSelectClip(c.id);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (drag) setDrag({ ...drag, dx: e.clientX - drag.x0 });
  };
  const onPointerUp = async (c: Clip) => {
    if (!drag || drag.clipId !== c.id) return;
    const d = drag;
    setDrag(null);
    if (Math.abs(d.dx) < 3) return;
    const g = geometry(c);
    if (d.mode === "move") {
      await run("clip.move", { clip_id: c.id, start: Math.max(0, snap(g.start, c.id)) });
    } else if (d.mode === "head") {
      const start = snap(g.start, c.id);
      const inPoint = Math.max(0, c.in_point + (start - c.start) * c.speed);
      await run("clip.trim", { clip_id: c.id, in_point: inPoint, start: Math.max(0, start) });
    } else {
      const end = snap(g.start + g.dur, c.id);
      await run("clip.trim", { clip_id: c.id, out_point: c.in_point + (end - c.start) * c.speed });
    }
  };

  const seekFromEvent = (e: React.PointerEvent | React.MouseEvent) => {
    const rect = laneRef.current?.getBoundingClientRect();
    if (!rect) return;
    onSeek((e.clientX - rect.left) / zoom);
  };

  const ticks = [];
  const step = zoom > 80 ? 1 : zoom > 30 ? 2 : zoom > 12 ? 5 : 10;
  for (let t = 0; t <= duration + 10; t += step) ticks.push(t);

  return (
    <section className="tl">
      <div className="tl-tools">
        <button onClick={props.onSplit} title="Split at playhead (S)">✂ Split</button>
        <button onClick={props.onDelete} disabled={!selClip && !selCaption} title="Delete (Del)">
          Delete
        </button>
        <span className="muted small">Drag clips to move, drag edges to trim. Space plays.</span>
        <div className="spacer" />
        <span className="muted small">Zoom</span>
        <input type="range" min={8} max={200} value={zoom} onChange={(e) => setZoom(Number(e.target.value))} style={{ width: 140 }} />
      </div>
      <div className="tl-scroll">
        <div className="tl-inner" style={{ width: width + 90 }}>
          <div className="tl-row tl-ruler-row">
            <div className="tl-label" />
            <div
              className="tl-lane tl-ruler"
              ref={laneRef}
              style={{ width }}
              onPointerDown={(e) => {
                (e.target as HTMLElement).setPointerCapture(e.pointerId);
                seekFromEvent(e);
              }}
              onPointerMove={(e) => e.buttons === 1 && seekFromEvent(e)}
            >
              {ticks.map((t) => (
                <span key={t} className="tick" style={{ left: t * zoom }}>
                  {fmt(t).replace(/\.0$/, "")}
                </span>
              ))}
            </div>
          </div>
          {tracks.map((track) => (
            <div key={track.id} className={`tl-row ${track.kind}`}>
              <div className="tl-label small">{track.name}</div>
              <div className="tl-lane" style={{ width }} onPointerDown={(e) => { seekFromEvent(e); onSelectClip(null); }}>
                {track.clips.map((c) => {
                  const g = geometry(c);
                  const media = project.media[c.media_id];
                  return (
                    <div
                      key={c.id}
                      className={`tl-clip ${track.kind} ${selClip === c.id ? "selected" : ""}`}
                      style={{
                        left: g.start * zoom,
                        width: Math.max(6, g.dur * zoom),
                        backgroundImage: media?.thumb_path && track.kind !== "audio" ? `url(${client.fileUrl(`/media/${media.id}/thumb`)})` : undefined,
                      }}
                      onPointerDown={(e) => onPointerDown(e, c, "move")}
                      onPointerMove={onPointerMove}
                      onPointerUp={() => onPointerUp(c)}
                      title={`${media?.name ?? "?"} · ${clipDuration(c).toFixed(2)}s${c.speed !== 1 ? ` · ${c.speed}×` : ""}`}
                    >
                      {c.transition && <span className="tl-trans" title={`${label(c.transition.kind)} ${c.transition.duration}s`} style={{ left: -6 }} />}
                      <span className="tl-name">
                        {media?.name}
                        {c.speed !== 1 ? ` · ${c.speed}×` : ""}
                        {c.muted ? " · muted" : ""}
                      </span>
                      <span className="tl-handle head" onPointerDown={(e) => onPointerDown(e, c, "head")} onPointerMove={onPointerMove} onPointerUp={() => onPointerUp(c)} />
                      <span className="tl-handle tail" onPointerDown={(e) => onPointerDown(e, c, "tail")} onPointerMove={onPointerMove} onPointerUp={() => onPointerUp(c)} />
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
          <div className="tl-row captions">
            <div className="tl-label small">Text</div>
            <div className="tl-lane" style={{ width }} onPointerDown={seekFromEvent}>
              {project.captions.map((c) => (
                <div
                  key={c.id}
                  className={`tl-cap ${c.kind} ${selCaption === c.id ? "selected" : ""}`}
                  style={{ left: c.start * zoom, width: Math.max(6, (c.end - c.start) * zoom) }}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    onSelectCaption(c.id);
                    onSeek(c.start + Math.min(0.4, (c.end - c.start) / 2));
                  }}
                  title={c.text}
                >
                  {c.text}
                </div>
              ))}
            </div>
          </div>
          <div className="tl-playhead" style={{ left: 90 + playhead * zoom }} />
        </div>
      </div>
    </section>
  );
}

// ------------------------------------------------------------------- AI tab

function VideoAi({ client, snapshot, onNotice }: { client: EngineClient; snapshot: Snapshot; onNotice: (m: string | null) => void }) {
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [vars, setVars] = useState<VideoVariation[]>([]);
  const [applied, setApplied] = useState<number | null>(null);
  const [heading, setHeading] = useState("");
  const [status, setStatus] = useState<string>("");
  const hasMedia = Object.keys(snapshot.project.media).length > 0;

  useEffect(() => {
    client.aiWarmup().then((s) => setStatus(s.state === "not_installed" ? "Basic AI (model not installed)" : "Local AI model"), () => undefined);
  }, [client]);

  const apply = async (i: number, list = vars) => {
    try {
      await client.aiVideoApply(list[i].program);
      setApplied(i);
    } catch (e) {
      onNotice(e instanceof Error ? e.message : String(e));
    }
  };

  const go = async (text: string) => {
    if (!text.trim()) return;
    setBusy(true);
    setVars([]);
    setApplied(null);
    setHeading(text);
    try {
      const res = await client.aiVideo(text);
      if (res.error) throw new Error(res.error);
      setVars(res.variations);
      if (res.variations.length) await apply(0, res.variations);
    } catch (e) {
      onNotice(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="ai">
      <div className="row between">
        <h4 style={{ margin: 0 }}>Ask the AI editor</h4>
        <span className="muted small">{status}</span>
      </div>
      <textarea
        rows={3}
        placeholder={hasMedia ? 'e.g. "fast travel reel on the beat with bold captions"' : "Import your clips first"}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            go(prompt);
          }
        }}
        disabled={!hasMedia}
      />
      <div className="row end">
        <button className="primary" disabled={busy || !hasMedia || !prompt.trim()} onClick={() => go(prompt)}>
          {busy ? "Editing…" : "Edit my video"}
        </button>
      </div>
      <div className="chips">
        {AI_CHIPS.map((c) => (
          <button key={c} className="chip" disabled={busy || !hasMedia} onClick={() => { setPrompt(c); go(c); }}>
            {c}
          </button>
        ))}
      </div>
      {busy && <p className="muted small">Watching your clips and planning the edit…</p>}
      {vars.length > 0 && (
        <div className="variations">
          <h4>{heading}</h4>
          {vars.map((v, i) => (
            <div key={i} className={`variation ${applied === i ? "applied" : ""}`} onClick={() => apply(i)}>
              <img src={client.frameUrl(2, { max: 320, program: v.program })} alt="" loading="lazy" />
              <div className="variation-body">
                <p className="small">
                  <strong>Option {i + 1}</strong> {applied === i && <span className="ok">· applied</span>}
                </p>
                <p className="small muted">{v.rationale}</p>
                <details onClick={(e) => e.stopPropagation()}>
                  <summary className="small muted">Edit program</summary>
                  <pre>{v.program}</pre>
                </details>
              </div>
            </div>
          ))}
          <p className="muted small">Every AI edit is one Undo step.</p>
        </div>
      )}
    </div>
  );
}

// ----------------------------------------------------------------- Clip tab

function NumberSlider({ label: text, value, min, max, step, onCommit, fmt: f }: { label: string; value: number; min: number; max: number; step: number; onCommit: (v: number) => void; fmt?: (v: number) => string }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  const commit = () => v !== value && onCommit(v);
  return (
    <label className="slider">
      <span className="slider-head">
        <span>{text}</span>
        <span className="muted">{f ? f(v) : v}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={v} onChange={(e) => setV(Number(e.target.value))} onPointerUp={commit} onKeyUp={commit} onBlur={commit} />
    </label>
  );
}

function ClipPanel({ client, snapshot, entry, run, onSplit, onDelete }: { client: EngineClient; snapshot: Snapshot; entry: { track: Track; clip: Clip }; run: RunFn; onSplit: () => void; onDelete: () => void }) {
  const { clip, track } = entry;
  const media = snapshot.project.media[clip.media_id];
  const [looks, setLooks] = useState<LookInfo[]>([]);
  useEffect(() => {
    client.looks().then(setLooks, () => undefined);
  }, [client]);
  const set = (params: Record<string, unknown>) => run("clip.set", { clip_id: clip.id, ...params });
  const clipLook = clip.ops.find((o) => o.op === "look");
  const isVisual = track.kind !== "audio";
  return (
    <>
      <div className="group">
        <h4>{media?.name}</h4>
        <p className="muted small">
          {clip.in_point.toFixed(2)}s – {clip.out_point.toFixed(2)}s of the source · {clipDuration(clip).toFixed(2)}s on the timeline
        </p>
        <div className="row">
          <button onClick={onSplit}>✂ Split at playhead</button>
          <button onClick={onDelete}>Delete</button>
        </div>
      </div>
      <div className="group">
        <h4>Speed & sound</h4>
        <NumberSlider label="Speed" value={clip.speed} min={0.25} max={4} step={0.05} fmt={(v) => `${v.toFixed(2)}×`} onCommit={(v) => set({ speed: v })} />
        <NumberSlider label="Volume" value={clip.volume} min={0} max={2} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`} onCommit={(v) => set({ volume: v })} />
        <label className="row between small">
          <span>Mute</span>
          <input type="checkbox" style={{ width: "auto" }} checked={clip.muted} onChange={(e) => set({ muted: e.target.checked })} />
        </label>
        <NumberSlider label="Fade in" value={clip.fade_in} min={0} max={3} step={0.1} fmt={(v) => `${v.toFixed(1)}s`} onCommit={(v) => set({ fade_in: v })} />
        <NumberSlider label="Fade out" value={clip.fade_out} min={0} max={3} step={0.1} fmt={(v) => `${v.toFixed(1)}s`} onCommit={(v) => set({ fade_out: v })} />
      </div>
      {isVisual && (
        <>
          <div className="group">
            <h4>Transition into this clip</h4>
            <select value={clip.transition?.kind ?? "none"} onChange={(e) => run("clip.set_transition", { clip_id: clip.id, kind: e.target.value, duration: clip.transition?.duration ?? 0.5 })}>
              {TRANSITIONS.map((t) => (
                <option key={t} value={t}>{label(t)}</option>
              ))}
            </select>
            {clip.transition && (
              <NumberSlider label="Duration" value={clip.transition.duration} min={0.1} max={2} step={0.05} fmt={(v) => `${v.toFixed(2)}s`}
                onCommit={(v) => run("clip.set_transition", { clip_id: clip.id, kind: clip.transition!.kind, duration: v })} />
            )}
            <button className="small" onClick={() => run("timeline.set_transitions", { track_id: track.id, kind: clip.transition?.kind ?? "none", duration: clip.transition?.duration ?? 0.5 })}>
              Use on every cut
            </button>
          </div>
          <div className="group">
            <h4>Framing</h4>
            <label className="small">Motion (great for photos)</label>
            <select value={clip.motion} onChange={(e) => set({ motion: e.target.value })}>
              {MOTIONS.map((m) => (
                <option key={m} value={m}>{label(m)}</option>
              ))}
            </select>
            <label className="small">Fit</label>
            <select value={clip.fit} onChange={(e) => set({ fit: e.target.value })}>
              <option value="fill">Fill the frame (crop)</option>
              <option value="fit">Show everything (bars)</option>
            </select>
          </div>
          <div className="group">
            <h4>Look for this clip only</h4>
            <select value={clipLook ? String(clipLook.params.name) : ""} onChange={(e) => run("clip.apply_look", { clip_id: clip.id, name: e.target.value || null })}>
              <option value="">None (use the video's look)</option>
              {looks.map((l) => (
                <option key={l.name} value={l.name}>{l.title}</option>
              ))}
            </select>
          </div>
        </>
      )}
    </>
  );
}

// ----------------------------------------------------------------- Look tab

const GRADE_SLIDERS: { op: string; key: string; label: string; min: number; max: number; step: number }[] = [
  { op: "adjust", key: "exposure", label: "Brightness", min: -2, max: 2, step: 0.05 },
  { op: "adjust", key: "contrast", label: "Contrast", min: -100, max: 100, step: 1 },
  { op: "adjust", key: "temperature", label: "Warmth", min: -100, max: 100, step: 1 },
  { op: "adjust", key: "saturation", label: "Saturation", min: -100, max: 100, step: 1 },
  { op: "grain", key: "amount", label: "Film grain", min: 0, max: 1, step: 0.01 },
  { op: "vignette", key: "amount", label: "Vignette", min: -1, max: 1, step: 0.01 },
];

function LookPanel({ client, snapshot, playhead, run }: { client: EngineClient; snapshot: Snapshot; playhead: number; run: RunFn }) {
  const { project } = snapshot;
  const [looks, setLooks] = useState<LookInfo[]>([]);
  useEffect(() => {
    client.looks().then(setLooks, () => undefined);
  }, [client]);
  const look = project.grade.find((o) => o.op === "look");
  const current = look ? String(look.params.name) : null;
  const strength = look ? Number(look.params.strength ?? 1) : 1;
  const t = Math.round(playhead * 2) / 2;
  const setGrade = async (op: string, key: string, value: number) => {
    const existing = project.grade.find((o) => o.op === op);
    if (existing) await run("timeline.set_grade_op", { op_id: existing.id, params: { [key]: value } });
    else await run("timeline.add_grade_op", { op, params: { [key]: value } });
  };
  const s = project.settings;
  return (
    <>
      <div className="group">
        <h4>Look of the whole video</h4>
        <div className="look-grid">
          <button className={`look ${!current ? "active" : ""}`} onClick={() => run("timeline.apply_look", { name: null })}>
            <img src={client.frameUrl(t, { max: 320, grade: "" })} alt="" loading="lazy" />
            <span>Original</span>
          </button>
          {looks.map((l) => (
            <button key={l.name} className={`look ${current === l.name ? "active" : ""}`} title={l.description} onClick={() => run("timeline.apply_look", { name: l.name, strength })}>
              <img src={client.frameUrl(t, { max: 320, grade: `look ${l.name} s=1` })} alt="" loading="lazy" />
              <span>{l.title}</span>
            </button>
          ))}
        </div>
        {current && (
          <NumberSlider label="Strength" value={strength} min={0} max={1.5} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`} onCommit={(v) => run("timeline.apply_look", { name: current, strength: v })} />
        )}
      </div>
      <div className="group">
        <h4>Fine-tune</h4>
        {GRADE_SLIDERS.map((g) => {
          const op = project.grade.find((o) => o.op === g.op);
          const v = op ? Number(op.params[g.key] ?? 0) : 0;
          return <NumberSlider key={g.op + g.key} label={g.label} value={v} min={g.min} max={g.max} step={g.step} onCommit={(nv) => setGrade(g.op, g.key, nv)} />;
        })}
      </div>
      <div className="group">
        <h4>Finishing</h4>
        <label className="row between small">
          <span>Cinematic bars (2.39:1)</span>
          <input type="checkbox" style={{ width: "auto" }} checked={!!s.letterbox} onChange={(e) => run("project.set_settings", { letterbox: e.target.checked ? 2.39 : null })} />
        </label>
        <NumberSlider label="Fade in from black" value={s.fade_in} min={0} max={3} step={0.1} fmt={(v) => `${v.toFixed(1)}s`} onCommit={(v) => run("project.set_settings", { fade_in: v })} />
        <NumberSlider label="Fade out to black" value={s.fade_out} min={0} max={3} step={0.1} fmt={(v) => `${v.toFixed(1)}s`} onCommit={(v) => run("project.set_settings", { fade_out: v })} />
        <label className="row between small">
          <span>Lower music when people talk</span>
          <input type="checkbox" style={{ width: "auto" }} checked={s.duck_music} onChange={(e) => run("project.set_settings", { duck_music: e.target.checked })} />
        </label>
        <label className="row between small">
          <span>Even out loudness (−14 LUFS)</span>
          <input type="checkbox" style={{ width: "auto" }} checked={s.normalize_audio} onChange={(e) => run("project.set_settings", { normalize_audio: e.target.checked })} />
        </label>
      </div>
    </>
  );
}

// ----------------------------------------------------------------- Text tab

function TextPanel(props: {
  client: EngineClient;
  snapshot: Snapshot;
  playhead: number;
  run: RunFn;
  selected: string | null;
  onSelect: (id: string) => void;
  captionJob: string | null;
  setCaptionJob: (id: string | null) => void;
  onNotice: (m: string | null) => void;
}) {
  const { client, snapshot, playhead, run, selected, onSelect, captionJob, setCaptionJob, onNotice } = props;
  const [styles, setStyles] = useState<{ name: string; description: string }[]>([]);
  const [style, setStyle] = useState("clean");
  const [speech, setSpeech] = useState<boolean | null>(null);
  useEffect(() => {
    client.captionStyles().then(setStyles, () => undefined);
    client.speechStatus().then((s) => setSpeech(s.available), () => setSpeech(false));
  }, [client]);
  const caps = snapshot.project.captions;
  const auto = async () => {
    try {
      const r = await client.autoCaptions(style);
      setCaptionJob(r.job);
    } catch (e) {
      onNotice(e instanceof Error ? e.message : String(e));
    }
  };
  const add = async (kind: "title" | "caption") => {
    const res = await run("caption.add", {
      start: playhead,
      end: playhead + 3,
      text: kind === "title" ? "Your title" : "Your text",
      style: kind === "title" ? "title" : style,
      kind,
    });
    if (res?.caption_id) onSelect(res.caption_id as string);
  };
  return (
    <>
      <div className="group">
        <h4>Auto captions</h4>
        <select value={style} onChange={(e) => setStyle(e.target.value)}>
          {styles.filter((s) => s.name !== "title").map((s) => (
            <option key={s.name} value={s.name}>{label(s.name)} — {s.description}</option>
          ))}
        </select>
        <div className="row">
          <button className="primary" disabled={!speech || !!captionJob} onClick={auto}>
            {captionJob ? "Listening…" : "Generate captions from speech"}
          </button>
          {caps.some((c) => c.kind === "caption") && (
            <button onClick={() => run("caption.style_all", { style, kind: "caption" })}>Restyle all</button>
          )}
        </div>
        {speech === false && <p className="muted small">Speech recognition isn't installed (whisper.cpp + a ggml model in the AI folder).</p>}
      </div>
      <div className="group">
        <h4>Titles & text</h4>
        <div className="row">
          <button onClick={() => add("title")}>+ Title at playhead</button>
          <button onClick={() => add("caption")}>+ Text at playhead</button>
        </div>
      </div>
      <ul className="edit-list caption-list">
        {caps.map((c) => (
          <CaptionRow key={c.id} c={c} styles={styles.map((s) => s.name)} active={selected === c.id} run={run} onSelect={() => onSelect(c.id)} />
        ))}
      </ul>
      {caps.length === 0 && <p className="muted small">No text yet.</p>}
    </>
  );
}

function CaptionRow({ c, styles, active, run, onSelect }: { c: Caption; styles: string[]; active: boolean; run: RunFn; onSelect: () => void }) {
  const [text, setText] = useState(c.text);
  useEffect(() => setText(c.text), [c.text]);
  const ref = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: "nearest" });
  }, [active]);
  return (
    <li ref={ref} className={active ? "active" : ""} onClick={onSelect} style={{ flexDirection: "column", alignItems: "stretch" }}>
      <div className="row between small muted">
        <span>
          {c.kind === "title" ? "Title" : "Caption"} · {fmt(c.start)}–{fmt(c.end)}
        </span>
        <button className="icon" title="Delete" onClick={(e) => { e.stopPropagation(); run("caption.remove", { caption_id: c.id }); }}>✕</button>
      </div>
      <input value={text} onChange={(e) => setText(e.target.value)} onBlur={() => text.trim() && text !== c.text && run("caption.set", { caption_id: c.id, text })}
        onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()} />
      {active && (
        <div className="row">
          <select value={c.style} onChange={(e) => run("caption.set", { caption_id: c.id, style: e.target.value })}>
            {styles.map((s) => (
              <option key={s} value={s}>{label(s)}</option>
            ))}
          </select>
          <select value={c.position ?? ""} onChange={(e) => run("caption.set", { caption_id: c.id, position: e.target.value || null })}>
            <option value="">Default place</option>
            <option value="top">Top</option>
            <option value="center">Center</option>
            <option value="lower_third">Lower third</option>
            <option value="bottom">Bottom</option>
          </select>
        </div>
      )}
      {active && (
        <div className="row">
          <NumberSlider label="Length" value={Number((c.end - c.start).toFixed(2))} min={0.3} max={10} step={0.1} fmt={(v) => `${v.toFixed(1)}s`} onCommit={(v) => run("caption.set", { caption_id: c.id, end: c.start + v })} />
        </div>
      )}
    </li>
  );
}

// --------------------------------------------------------------- Export tab

async function pickSavePath(defaultName: string): Promise<string | null> {
  if (isTauri()) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    return (await save({ defaultPath: defaultName, filters: [{ name: "MP4 video", extensions: ["mp4"] }] })) ?? null;
  }
  return window.prompt("Save to (full path of the .mp4 file):", defaultName);
}

function ExportPanel({ client, snapshot, job, setJob, progress, onNotice }: { client: EngineClient; snapshot: Snapshot; job: string | null; setJob: (j: string | null) => void; progress: number; onNotice: (m: string | null) => void }) {
  const { project } = snapshot;
  const portrait = project.settings.height > project.settings.width;
  const [preset, setPreset] = useState(portrait ? "reel_1080" : "project");
  const [quality, setQuality] = useState<"high" | "standard" | "small">("high");
  const start = async () => {
    const path = await pickSavePath(`${project.name.replace(/[\\/:*?"<>|]/g, "_")}.mp4`);
    if (!path) return;
    try {
      const r = await client.exportVideo(path, preset, quality);
      setJob(r.job);
      onNotice(null);
    } catch (e) {
      onNotice(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <div className="group">
      <h4>Export video</h4>
      <label className="small">Made for</label>
      <select value={preset} onChange={(e) => setPreset(e.target.value)}>
        {PRESETS.map(([k, v]) => (
          <option key={k} value={k}>
            {v}
            {k === "project" ? ` (${project.settings.width}×${project.settings.height})` : ""}
          </option>
        ))}
      </select>
      <label className="small">Quality</label>
      <select value={quality} onChange={(e) => setQuality(e.target.value as typeof quality)}>
        <option value="high">Best (bigger file)</option>
        <option value="standard">Standard</option>
        <option value="small">Small file</option>
      </select>
      <p className="muted small">Renders from your original files at full quality: H.264 + AAC, loudness evened out for streaming.</p>
      {job ? (
        <>
          <div className="progress">
            <div style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
          <div className="row between">
            <span className="small muted">Exporting… {Math.round(progress * 100)}%</span>
            <button onClick={() => client.cancelJob(job)}>Cancel</button>
          </div>
        </>
      ) : (
        <button className="primary" onClick={start} disabled={projectDuration(project) <= 0}>
          Export
        </button>
      )}
    </div>
  );
}
