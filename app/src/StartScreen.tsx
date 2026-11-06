import { useState } from "react";
import { EngineClient, isTauri, type Project, type Snapshot } from "./engine";

async function pickFolder(): Promise<string | null> {
  if (!isTauri()) return null;
  const { open } = await import("@tauri-apps/plugin-dialog");
  const picked = await open({ directory: true, multiple: false });
  return typeof picked === "string" ? picked : null;
}

export function StartScreen({ client, onOpened }: { client: EngineClient; onOpened: (s: Snapshot) => void }) {
  const [path, setPath] = useState("");
  const [name, setName] = useState("My project");
  const [kind, setKind] = useState<Project["kind"]>("video");
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
        <h1>AIEditor</h1>
        <p className="muted">Create a project or open an existing one.</p>

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
