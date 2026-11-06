import { useCallback, useEffect, useRef, useState } from "react";
import { EngineClient, type EngineEvent, type Snapshot } from "./engine";
import { StartScreen } from "./StartScreen";
import { Editor } from "./Editor";

export default function App() {
  const [client, setClient] = useState<EngineClient | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [proxyProgress, setProxyProgress] = useState<Record<string, number>>({});
  const refreshTimer = useRef<number | undefined>(undefined);

  useEffect(() => {
    EngineClient.connect().then(setClient, (e) => setFatal(String(e.message ?? e)));
  }, []);

  const refresh = useCallback(() => {
    if (!client) return;
    // Coalesce bursts of events (e.g. batch import) into one fetch.
    window.clearTimeout(refreshTimer.current);
    refreshTimer.current = window.setTimeout(() => {
      client.project().then(setSnapshot, () => undefined);
    }, 60);
  }, [client]);

  useEffect(() => {
    if (!client) return;
    return client.events((e: EngineEvent) => {
      if (e.type === "proxy_progress") {
        setProxyProgress((p) => ({ ...p, [e.media_id as string]: e.progress as number }));
        return;
      }
      if (e.type !== "hello" || e.revision !== null) refresh();
    });
  }, [client, refresh]);

  if (fatal) {
    return (
      <div className="center">
        <div className="card">
          <h2>Engine unavailable</h2>
          <p className="muted">{fatal}</p>
        </div>
      </div>
    );
  }
  if (!client) {
    return <div className="center muted">Starting engine…</div>;
  }
  if (!snapshot) {
    return <StartScreen client={client} onOpened={setSnapshot} />;
  }
  return <Editor client={client} snapshot={snapshot} proxyProgress={proxyProgress} />;
}
