// Client for the local Python engine (see engine/src/aieditor_engine/api/server.py).

export type MediaKind = "image" | "video" | "audio";
export type ProxyStatus = "none" | "pending" | "ready" | "failed";

export interface MediaItem {
  id: string;
  name: string;
  path: string;
  kind: MediaKind;
  width: number | null;
  height: number | null;
  duration: number | null;
  fps: number | null;
  has_audio: boolean;
  thumb_path: string | null;
  proxy_path: string | null;
  proxy_status: ProxyStatus;
  subject?: { x: number; y: number; t: number } | null;
}

export interface ModelStatus {
  name: string;
  title: string;
  purpose: string;
  size_mb: number;
  licence: string;
  installed: boolean;
  memory_gb: number | null;
  fits: boolean;
  download: { state: "running" | "done" | "failed"; progress: number; error: string | null } | null;
}

export interface SubjectInfo {
  kind: "photo" | "video";
  point?: string;
  outline?: number[][][];
  box?: number[] | null;
  coverage?: number;
  pick?: { x: number; y: number; t: number } | null;
  method?: string | null;
  track?: [number, number[]][];
}

export type TransitionKind = "crossfade" | "fade_black" | "fade_white" | "wipe_left" | "wipe_right" | "slide_left" | "slide_up" | "zoom" | "blur";
export type Motion = "none" | "zoom_in" | "zoom_out" | "pan_left" | "pan_right";

export interface Clip {
  id: string;
  media_id: string;
  start: number;
  in_point: number;
  out_point: number;
  speed: number;
  volume: number;
  muted: boolean;
  fade_in: number;
  fade_out: number;
  fit: "fill" | "fit";
  motion: Motion;
  transition: { kind: TransitionKind; duration: number } | null;
  ops: EditOp[];
}

export const clipDuration = (c: Clip) => (c.out_point - c.in_point) / (c.speed || 1);

export interface Caption {
  id: string;
  start: number;
  end: number;
  text: string;
  style: string;
  position: string | null;
  kind: "caption" | "title";
}

export interface Job {
  id: string;
  kind: string;
  label: string;
  state: "queued" | "running" | "done" | "failed" | "cancelled";
  progress: number;
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface StyleOption {
  id: string;
  title: string;
  description: string;
  custom?: boolean;
}
export interface OutputChoice {
  id: string;
  title: string;
  aspect?: string;
}
export interface AiOptions {
  video: { styles: StyleOption[]; outputs: OutputChoice[] };
  photo: { styles: StyleOption[]; outputs: OutputChoice[] };
  lengths: number[];
}
export interface OutputOptions {
  platform?: string | null;
  length?: number | null;
  versions?: number | null;
  captions?: boolean | null;
  music?: boolean | null;
}
export interface ReelVariation {
  program: string;
  title: string;
  rationale: string;
  score: number;
  moment: { start: number; end: number; text: string };
  info: { pieces: number; duration: number; silence_removed: number; reframed: boolean };
}

export interface VideoVariation {
  program: string;
  rationale: string;
  source: "llm" | "rules" | "highlights";
  intent: string;
  warnings: string[];
}

export interface Track {
  id: string;
  kind: "video" | "audio" | "overlay";
  name: string;
  clips: Clip[];
}

export interface EditOp {
  id: string;
  op: string;
  params: Record<string, unknown>;
  enabled: boolean;
}

export interface Project {
  id: string;
  name: string;
  kind: "photo" | "video" | "collage";
  settings: {
    width: number;
    height: number;
    fps: number;
    letterbox: number | null;
    duck_music: boolean;
    normalize_audio: boolean;
    fade_in: number;
    fade_out: number;
  };
  media: Record<string, MediaItem>;
  tracks: Track[];
  edits: Record<string, EditOp[]>;
  grade: EditOp[];
  captions: Caption[];
}

export const projectDuration = (p: Project) =>
  Math.max(0, ...p.tracks.flatMap((t) => t.clips.map((c) => c.start + clipDuration(c))), ...p.captions.map((c) => c.end));

export interface LookInfo {
  name: string;
  title: string;
  description: string;
  tags: string[];
}

export interface AiVariation {
  program: string;
  rationale: string;
  source: "llm" | "rules" | "highlights";
  warnings: string[];
}

export interface AiStatus {
  state: "ready" | "idle" | "not_installed" | "error";
  model: string | null;
  error: string | null;
}

export interface ExportOptions {
  path: string;
  format: "jpeg" | "png" | "tiff" | "webp";
  quality: number;
  bit_depth: 8 | 16;
  long_edge: number | null;
}

export interface Snapshot {
  revision: number;
  can_undo: boolean;
  can_redo: boolean;
  undo_label: string | null;
  redo_label: string | null;
  project: Project;
}

export interface EngineEvent {
  type: string;
  revision: number | null;
  [key: string]: unknown;
}

export interface SystemInfo {
  os: string;
  cpu: string;
  cores: number;
  memory_gb: number | null;
  disk_free_gb: number;
  hardware_encoder: "nvenc" | null;
  ai: {
    editing_model: { ready: boolean; name: string | null };
    speech: { ready: boolean; name: string | null };
    face_model: { ready: boolean };
  };
  data_folder: string;
}

export interface Benchmark {
  ms_per_frame: number;
  workers: number;
  reel_seconds: number;
  tier: "fast" | "good" | "basic";
  summary: string;
}

export type Focus = "photos" | "videos" | "reels" | "automation";
export interface AppSettings {
  onboarded: boolean;
  focus: Focus | null;
}

interface EngineInfo {
  port: number;
  token: string;
}

export const isTauri = () => "__TAURI_INTERNALS__" in window;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function resolveInfo(): Promise<EngineInfo> {
  if (isTauri()) {
    const { invoke } = await import("@tauri-apps/api/core");
    // The Rust shell starts the engine; poll until it reports ready.
    for (let i = 0; i < 150; i++) {
      try {
        return await invoke<EngineInfo>("engine_info");
      } catch (err) {
        if (String(err) !== "engine starting") throw new Error(String(err));
        await sleep(200);
      }
    }
    throw new Error("engine did not start within 30 s");
  }
  // Browser dev mode: run the engine yourself and pass ?port=&token= (or VITE_ENGINE_PORT/TOKEN).
  const q = new URLSearchParams(location.search);
  const port = Number(q.get("port") ?? import.meta.env.VITE_ENGINE_PORT);
  const token = q.get("token") ?? import.meta.env.VITE_ENGINE_TOKEN;
  if (!port || !token) throw new Error("No engine connection. Open with ?port=<port>&token=<token> from the engine's ready line.");
  return { port, token };
}

export class EngineError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export class EngineClient {
  private constructor(private base: string, private token: string) {}

  static async connect(): Promise<EngineClient> {
    const info = await resolveInfo();
    return new EngineClient(`http://127.0.0.1:${info.port}`, info.token);
  }

  /** URL usable in <img>/<video> tags (token as query param). */
  fileUrl(path: string, revision?: number): string {
    const rev = revision === undefined ? "" : `&rev=${revision}`;
    return `${this.base}${path}?token=${encodeURIComponent(this.token)}${rev}`;
  }

  async request<T>(method: "GET" | "POST" | "DELETE", path: string, body?: unknown): Promise<T> {
    const res = await fetch(this.base + path, {
      method,
      headers: { "X-Engine-Token": this.token, ...(body ? { "Content-Type": "application/json" } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        detail = (await res.json()).detail ?? detail;
      } catch {
        /* non-JSON error body */
      }
      throw new EngineError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return res.json() as Promise<T>;
  }

  project = () => this.request<Snapshot>("GET", "/project");
  createProject = (path: string, name: string, kind: Project["kind"]) =>
    this.request<Snapshot>("POST", "/projects", { path, name, kind });
  openProject = (path: string) => this.request<Snapshot>("POST", "/projects/open", { path });
  importMedia = (paths: string[]) =>
    this.request<{ imported: string[]; failed: { path: string; error: string }[] }>("POST", "/media/import", { paths });
  run = (name: string, params: Record<string, unknown> = {}) =>
    this.request<{ result: Record<string, unknown>; revision: number }>("POST", `/commands/${name}`, { params, source: "user" });
  undo = () => this.request<{ ok: boolean }>("POST", "/undo");
  redo = () => this.request<{ ok: boolean }>("POST", "/redo");

  // ---- photo
  looks = () => this.request<LookInfo[]>("GET", "/looks");
  runAi = (name: string, params: Record<string, unknown>) =>
    this.request<{ result: Record<string, unknown>; revision: number }>("POST", `/commands/${name}`, { params, source: "ai" });

  /** Preview URL of the committed stack (or variants). `rev` busts the browser cache. */
  renderUrl(mediaId: string, opts: { max?: number; bypass?: boolean; look?: string; s?: number; program?: string; rev?: number }): string {
    const q = new URLSearchParams({ token: this.token, max: String(opts.max ?? 1600) });
    if (opts.bypass) q.set("bypass", "1");
    if (opts.look) q.set("look", opts.look);
    if (opts.s !== undefined) q.set("s", String(opts.s));
    if (opts.program !== undefined) q.set("program", opts.program);
    if (opts.rev !== undefined) q.set("rev", String(opts.rev));
    return `${this.base}/media/${mediaId}/render?${q}`;
  }

  /** Render an uncommitted stack (slider drags). Returns an object URL; caller revokes it. */
  async preview(mediaId: string, ops: Pick<EditOp, "op" | "params" | "enabled">[], max: number): Promise<string> {
    const res = await fetch(`${this.base}/media/${mediaId}/preview`, {
      method: "POST",
      headers: { "X-Engine-Token": this.token, "Content-Type": "application/json" },
      body: JSON.stringify({ ops, max }),
    });
    if (!res.ok) throw new EngineError(res.status, await res.text());
    return URL.createObjectURL(await res.blob());
  }

  // ---- onboarding
  appSettings = () => this.request<AppSettings>("GET", "/app/settings");
  saveAppSettings = (changes: Partial<AppSettings>) => this.request<AppSettings>("POST", "/app/settings", changes);
  system = () => this.request<SystemInfo>("GET", "/system");
  benchmark = () => this.request<Benchmark>("POST", "/system/benchmark");

  // ---- subject selection
  subject = (mediaId: string, point = "auto", t?: number) =>
    this.request<SubjectInfo>("GET", `/media/${mediaId}/subject?point=${encodeURIComponent(point)}${t !== undefined ? `&t=${t.toFixed(3)}` : ""}`);
  pickSubject = (mediaId: string, x: number, y: number, t = 0) => this.request<SubjectInfo>("POST", `/media/${mediaId}/subject`, { x, y, t });
  clearSubject = (mediaId: string) => this.request<{ pick: null }>("DELETE", `/media/${mediaId}/subject`);
  followSubject = (clipId: string) => this.request<{ keys: number; method: string }>("POST", `/clips/${clipId}/follow_subject`);
  models = () => this.request<ModelStatus[]>("GET", "/models");
  installModel = (name: string) => this.request<{ state: string; progress: number; error: string | null }>("POST", `/models/${name}/install`);
  mediaFrameUrl(mediaId: string, t: number, max = 960): string {
    return `${this.base}/media/${mediaId}/frame?token=${encodeURIComponent(this.token)}&t=${t.toFixed(3)}&max=${max}`;
  }

  aiStatus = () => this.request<AiStatus>("GET", "/ai/status");
  aiWarmup = () => this.request<AiStatus>("POST", "/ai/warmup");
  aiEdit = (mediaId: string, prompt: string, style?: string | null, output?: OutputOptions | null) =>
    this.request<{ variations: AiVariation[]; brief: string; llm_error: string | null }>("POST", "/ai/edit", {
      media_id: mediaId,
      prompt,
      variations: 3,
      style: style || null,
      output: output || null,
    });
  aiOptions = () => this.request<AiOptions>("GET", "/ai/options");
  exportImage = (mediaId: string, opts: ExportOptions) =>
    this.request<{ path: string; width: number; height: number }>("POST", `/media/${mediaId}/export`, opts);

  // ---- video
  frameUrl(t: number, opts: { max?: number; rev?: number; program?: string; grade?: string } = {}): string {
    const q = new URLSearchParams({ token: this.token, t: t.toFixed(3), max: String(opts.max ?? 960) });
    if (opts.rev !== undefined) q.set("rev", String(opts.rev));
    if (opts.program) q.set("program", opts.program);
    if (opts.grade !== undefined) q.set("grade", opts.grade);
    return `${this.base}/timeline/frame?${q}`;
  }
  previewRender = (maxSide = 640) =>
    this.request<{ job: string | null; revision: number; url: string | null }>("POST", "/timeline/preview", { max_side: maxSide });
  exportVideo = (path: string, preset: string, quality: "high" | "standard" | "small") =>
    this.request<{ job: string; path: string }>("POST", "/timeline/export", { path, preset, quality });
  jobs = () => this.request<Job[]>("GET", "/jobs");
  cancelJob = (id: string) => this.request<{ ok: boolean }>("POST", `/jobs/${id}/cancel`);
  captionStyles = () => this.request<{ name: string; description: string }[]>("GET", "/captions/styles");
  speechStatus = () => this.request<{ available: boolean; model: string | null }>("GET", "/ai/speech/status");
  autoCaptions = (style: string) => this.request<{ job: string }>("POST", "/timeline/auto_captions", { style });
  timelineDsl = () => this.request<{ program: string; duration: number }>("GET", "/timeline/dsl");
  aiReels = (body: { count?: number; style?: string | null; output?: OutputOptions | null; captions?: boolean | null }) =>
    this.request<{ variations: ReelVariation[]; aspect: string; style: string; export: string | null; tracking: string | null }>(
      "POST",
      "/ai/reels",
      body,
    );
  styleFromVideo = (mediaNumber: number, name: string) =>
    this.request<StyleOption & { pace: string }>("POST", "/ai/styles/from_video", { media_number: mediaNumber, name });
  aiVideoApply = (program: string, mode: "replace" | "append" = "replace") =>
    this.request<{ result: Record<string, unknown>; revision: number }>("POST", "/ai/video/apply", { program, mode });

  /** Subscribe to engine events; reconnects automatically. Returns a stop function. */
  events(onEvent: (e: EngineEvent) => void): () => void {
    let ws: WebSocket | null = null;
    let stopped = false;
    const open = () => {
      ws = new WebSocket(`${this.base.replace("http", "ws")}/ws?token=${encodeURIComponent(this.token)}`);
      ws.onmessage = (m) => onEvent(JSON.parse(m.data));
      ws.onclose = () => {
        if (!stopped) setTimeout(open, 1000);
      };
    };
    open();
    return () => {
      stopped = true;
      ws?.close();
    };
  }
}
