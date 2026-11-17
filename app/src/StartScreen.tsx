import { useState } from "react";
import { EngineClient, isTauri, type Focus, type LicenceStatus, type Project, type Snapshot } from "./engine";
import { ProBadge } from "./ProPanel";
import { Shell } from "./Shell";

async function pickFolder(): Promise<string | null> {
  if (!isTauri()) return null;
  const { open } = await import("@tauri-apps/plugin-dialog");
  const picked = await open({ directory: true, multiple: false });
  return typeof picked === "string" ? picked : null;
}

interface Props {
  client: EngineClient;
  onOpened: (s: Snapshot) => void;
  focus?: Focus | null;
  licence?: LicenceStatus | null;
  onOpenPro?: () => void;
  onSetup?: () => void;
}

const HINT: Record<Focus, string> = {
  videos: "Add your clips, then describe the video you want.",
  reels: "Import a long video, then use Make reels.",
  photos: "Add pictures, then describe the look you want.",
  automation: "Automation runs from the aieditor command and aieditor serve. You can still edit here.",
};

const KINDS: { id: Project["kind"]; title: string; text: string }[] = [
  { id: "video", title: "Video", text: "Clips, music, captions, reels" },
  { id: "photo", title: "Photo", text: "Single photos and batches" },
  { id: "collage", title: "Collage", text: "Several photos on one canvas" },
];

export function StartScreen({ client, onOpened, focus, licence, onOpenPro, onSetup }: Props) {
  const [path, setPath] = useState("");
  const [name, setName] = useState(focus === "reels" ? "My reels" : focus === "photos" ? "My photos" : "My project");
  const [kind, setKind] = useState<Project["kind"]>(focus === "photos" ? "photo" : "video");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const act = async (fn: () => Promise<Snapshot>) => {
    setBusy(true);
    setError(null);
    try {
      onOpened(await fn());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const browse = async () => {
    const p = await pickFolder();
    if (p) setPath(p);
  };

  return (
    <Shell
      side={
        <div className="side-stack">
          {onOpenPro && (
            <div className="side-block">
              <span className="label">Plan</span>
              <ProBadge status={licence ?? null} onClick={onOpenPro} />
            </div>
          )}
          {onSetup && (
            <button type="button" className="link" onClick={onSetup}>
              Run setup again
            </button>
          )}
        </div>
      }
      footer={
        <>
          <button disabled={busy || !path} onClick={() => act(() => client.openProject(path))}>
            Open existing project
          </button>
          <button className="primary" disabled={busy || !path || !name} onClick={() => act(() => client.createProject(path, name, kind))}>
            Create project
          </button>
        </>
      }
    >
      <div className="ob-body">
        <h1>New project</h1>
        <p className="lead">{focus ? HINT[focus] : "Create a project, or open one you made before."}</p>

        <div className="kind-grid" role="radiogroup" aria-label="Project type">
          {KINDS.map((k) => (
            <button key={k.id} type="button" role="radio" aria-checked={kind === k.id} className={`focus ${kind === k.id ? "sel" : ""}`} onClick={() => setKind(k.id)}>
              <strong>{k.title}</strong>
              <span>{k.text}</span>
            </button>
          ))}
        </div>

        <div className="form-grid">
          <label htmlFor="project-name">Name</label>
          <input id="project-name" value={name} onChange={(e) => setName(e.target.value)} />
          <label htmlFor="project-folder">Folder</label>
          <div className="row">
            <input id="project-folder" value={path} onChange={(e) => setPath(e.target.value)} placeholder="C:\Videos\Holiday" />
            {isTauri() && (
              <button type="button" onClick={browse}>
                Browse…
              </button>
            )}
          </div>
        </div>
        <p className="muted small">The project folder holds the project file, previews and exports. Your original files stay where they are.</p>
        {error && <p className="error">{error}</p>}
      </div>
    </Shell>
  );
}
