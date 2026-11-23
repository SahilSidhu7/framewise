import { useCallback, useEffect, useRef, useState } from "react";
import { EngineClient, type AppSettings, type EngineEvent, type Snapshot } from "./engine";
import { StartScreen } from "./StartScreen";
import { Editor } from "./Editor";
import { Onboarding } from "./Onboarding";

export default function App() {
  const [client, setClient] = useState<EngineClient | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [proxyProgress, setProxyProgress] = useState<Record<string, number>>({});
  const refreshTimer = useRef<number | undefined>(undefined);
  const [settings, setSettings] = useState<AppSettings | null>(null);

  useEffect(() => {
    EngineClient.connect().then(setClient, (e) => setFatal(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    if (!client) return;
    // Older engines without these routes: skip onboarding rather than block the editor.
    client.appSettings().then(setSettings, () => setSettings({ onboarded: true, focus: null }));
  }, [client]);

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
  if (!client || !settings) {
    return <div className="center muted">Starting engine…</div>;
  }
  if (!settings.onboarded) {
    return (
      <Onboarding client={client} onDone={(focus) => setSettings({ onboarded: true, focus })} />
    );
  }

  const reopenSetup = () => {
    client.saveAppSettings({ onboarded: false }).catch(() => undefined);
    setSettings({ ...settings, onboarded: false });
  };

  if (!snapshot) {
    return (
      <StartScreen client={client} onOpened={setSnapshot} focus={settings.focus} onSetup={reopenSetup} />
    );
  }
  return <Editor client={client} snapshot={snapshot} proxyProgress={proxyProgress} />;
}
