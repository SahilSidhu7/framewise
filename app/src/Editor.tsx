import { useState } from "react";
import { EngineClient, isTauri, type MediaItem, type Snapshot } from "./engine";
import { PhotoEditor } from "./PhotoEditor";
import { VideoEditor } from "./VideoEditor";

const PX_PER_SECOND = 40;

async function pickFiles(): Promise<string[]> {
  if (isTauri()) {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open({
      multiple: true,
      filters: [{ name: "Media", extensions: ["jpg", "jpeg", "png", "webp", "heic", "tif", "tiff", "mp4", "mov", "mkv", "webm", "avi", "mp3", "wav", "m4a"] }],
    });
    return Array.isArray(picked) ? picked : picked ? [picked] : [];
  }
  const raw = window.prompt("Paste one or more file paths (one per line):") ?? "";
  return raw.split(/\r?\n/).map((s) => s.trim().replace(/^"|"$/g, "")).filter(Boolean);
}

interface Props {
  client: EngineClient;
  snapshot: Snapshot;
  proxyProgress: Record<string, number>;
}

export function Editor({ client, snapshot, proxyProgress }: Props) {
  const { project } = snapshot;
  const [selected, setSelected] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const media = Object.values(project.media);
  // default selection: first photo in photo/collage projects
  const fallback = project.kind !== "video" ? media.find((m) => m.kind === "image")?.id ?? null : null;
  const current = (selected && project.media[selected]) || (fallback ? project.media[fallback] : undefined);
  const videoMode = project.kind === "video";
  const photoMode = !videoMode && current?.kind === "image";

  const guard = async (fn: () => Promise<unknown>) => {
    setNotice(null);
    try {
      await fn();
    } catch (e) {
      setNotice(e instanceof Error ? e.message : String(e));
    }
  };

  const importFiles = () =>
    guard(async () => {
      const paths = await pickFiles();
      if (!paths.length) return;
      const res = await client.importMedia(paths);
      if (res.imported[0]) setSelected(res.imported[0]);
      if (res.failed.length) setNotice(`Skipped ${res.failed.length}: ${res.failed.map((f) => f.error).join("; ")}`);
    });

  const addToTimeline = (item: MediaItem) =>
    guard(async () => {
      const kind = item.kind === "audio" ? "audio" : "video";
      let trackId = project.tracks.find((t) => t.kind === kind)?.id;
      if (!trackId) {
        const res = await client.run("track.add", { kind });
        trackId = res.result.track_id as string;
      }
      await client.run("clip.add", { track_id: trackId, media_id: item.id });
    });

  return (
    <div className={`editor ${photoMode || videoMode ? "photo-mode" : ""}`}>
      <header className="toolbar">
        <strong>{project.name}</strong>
        <span className="muted small">
          {project.kind} · {project.settings.width}×{project.settings.height} @ {project.settings.fps} fps
        </span>
        <div className="spacer" />
        <button disabled={!snapshot.can_undo} title={snapshot.undo_label ?? ""} onClick={() => guard(client.undo)}>
          Undo
        </button>
        <button disabled={!snapshot.can_redo} title={snapshot.redo_label ?? ""} onClick={() => guard(client.redo)}>
          Redo
        </button>
      </header>

      {notice && (
        <div className="notice" onClick={() => setNotice(null)}>
          {notice}
        </div>
      )}

      <aside className="bin">
        <div className="bin-head">
          <span>Media</span>
          <button className="primary" onClick={importFiles}>
            Import
          </button>
        </div>
        {media.length === 0 && <p className="muted small pad">No media yet. Import photos, videos or audio.</p>}
        <ul>
          {media.map((m) => (
            <li key={m.id} className={m.id === current?.id ? "active" : ""} onClick={() => setSelected(m.id)} onDoubleClick={() => (videoMode || m.kind !== "image") && addToTimeline(m)}
              title={videoMode ? "Double-click to add to the timeline" : undefined}>
              {m.thumb_path ? (
                <img src={client.fileUrl(`/media/${m.id}/thumb`)} alt="" />
              ) : (
                <div className="thumb-placeholder">{m.kind}</div>
              )}
              <div className="meta">
                <span className="name">{m.name}</span>
                <span className="muted small">
                  {m.width && m.height ? `${m.width}×${m.height}` : m.kind}
                  {m.duration ? ` · ${m.duration.toFixed(1)}s` : ""}
                </span>
                <ProxyBadge item={m} progress={proxyProgress[m.id]} />
              </div>
            </li>
          ))}
        </ul>
      </aside>

      {videoMode ? (
        <main className="photo-host">
          <VideoEditor client={client} snapshot={snapshot} onNotice={setNotice} />
        </main>
      ) : photoMode && current ? (
        <main className="photo-host">
          <PhotoEditor client={client} snapshot={snapshot} item={current} onNotice={setNotice} />
        </main>
      ) : (
      <>
      <main className="viewer">
        {current ? <Preview client={client} item={current} /> : <p className="muted">Select media to preview. Double-click to add it to the timeline.</p>}
        {current && (
          <button className="add" onClick={() => addToTimeline(current)}>
            Add to timeline
          </button>
        )}
      </main>

      <section className="timeline">
        {project.tracks.length === 0 && <p className="muted small pad">Timeline is empty.</p>}
        {project.tracks.map((t) => (
          <div key={t.id} className="track">
            <div className="track-label small">{t.name}</div>
            <div className="lane">
              {t.clips.map((c) => (
                <div
                  key={c.id}
                  className={`clip ${t.kind}`}
                  style={{ left: c.start * PX_PER_SECOND, width: Math.max(8, (c.out_point - c.in_point) * PX_PER_SECOND) }}
                  title={`${project.media[c.media_id]?.name ?? "?"} (${(c.out_point - c.in_point).toFixed(2)}s)`}
                  onDoubleClick={() => guard(() => client.run("clip.remove", { clip_id: c.id }))}
                >
                  {project.media[c.media_id]?.name}
                </div>
              ))}
            </div>
          </div>
        ))}
      </section>
      </>
      )}
    </div>
  );
}

function ProxyBadge({ item, progress }: { item: MediaItem; progress?: number }) {
  if (item.kind !== "video") return null;
  switch (item.proxy_status) {
    case "ready":
      return <span className="badge ok">proxy ready</span>;
    case "failed":
      return <span className="badge err">proxy failed</span>;
    case "pending":
      return <span className="badge">proxy {progress !== undefined ? `${Math.round(progress * 100)}%` : "queued"}</span>;
    default:
      return null;
  }
}

function Preview({ client, item }: { client: EngineClient; item: MediaItem }) {
  if (item.kind === "video" && item.proxy_status === "ready") {
    return <video key={item.id} src={client.fileUrl(`/media/${item.id}/proxy`)} controls />;
  }
  if (item.thumb_path) {
    // M0 shows the thumbnail for photos; the full-res render path arrives with the M1 photo engine.
    return <img src={client.fileUrl(`/media/${item.id}/thumb`)} alt={item.name} />;
  }
  return <p className="muted">{item.kind === "audio" ? "Audio file" : "Preparing preview…"}</p>;
}
