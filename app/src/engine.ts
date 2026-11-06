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
}

export interface Clip {
  id: string;
  media_id: string;
  start: number;
  in_point: number;
  out_point: number;
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
  settings: { width: number; height: number; fps: number };
  media: Record<string, MediaItem>;
  tracks: Track[];
  edits: Record<string, EditOp[]>;
}

export interface LookInfo {
  name: string;
  title: string;
  description: string;
  tags: string[];
}

export interface AiVariation {
  program: string;
  rationale: string;
  source: "llm" | "rules";
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

  async request<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
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

  aiStatus = () => this.request<AiStatus>("GET", "/ai/status");
  aiWarmup = () => this.request<AiStatus>("POST", "/ai/warmup");
  aiEdit = (mediaId: string, prompt: string) =>
    this.request<{ variations: AiVariation[]; brief: string; llm_error: string | null }>("POST", "/ai/edit", {
      media_id: mediaId,
      prompt,
      variations: 3,
    });
  exportImage = (mediaId: string, opts: ExportOptions) =>
    this.request<{ path: string; width: number; height: number }>("POST", `/media/${mediaId}/export`, opts);

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
