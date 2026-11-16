import { useState } from "react";
import { EngineClient, isTauri, type Focus, type LicenceStatus, type Project, type Snapshot } from "./engine";
import { ProBadge } from "./ProPanel";

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
  videos: "Create a project, add your clips, then describe the video you want.",
  reels: "Create a project, import a long video, then ask for reels.",
  photos: "Create a photo project, add pictures, then describe the look you want.",
  automation: "Automation runs from the aieditor command or aieditor serve. You can also edit here.",
};

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
    <div className="center">
      <div className="card start">
        <div className="row between">
          <h1>AIEditor</h1>
          {onOpenPro && <ProBadge status={licence ?? null} onClick={onOpenPro} />}
        </div>
        <p className="muted">{focus ? HINT[focus] : "Create a project or open an existing one."}</p>

        <label>
          Project folder
          <div className="row">
            <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="C:\Videos\Holiday" />
            {isTauri() && (
              <button type="button" onClick={browse}>
                Browse…
              </button>
            )}
          </div>
        </label>

        <div className="row">
          <label className="grow">
            Name
            <input value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <label>
            Type
            <select value={kind} onChange={(e) => setKind(e.target.value as Project["kind"])}>
              <option value="video">Video</option>
              <option value="photo">Photo</option>
              <option value="collage">Collage</option>
            </select>
          </label>
        </div>

        {error && <p className="error">{error}</p>}
        {onSetup && (
          <button type="button" className="link small" onClick={onSetup}>
            Run setup again
          </button>
        )}

        <div className="row end">
          <button disabled={busy || !path} onClick={() => act(() => client.openProject(path))}>
            Open existing
          </button>
          <button className="primary" disabled={busy || !path || !name} onClick={() => act(() => client.createProject(path, name, kind))}>
            Create project
          </button>
        </div>
      </div>
    </div>
  );
}
