import { useState } from "react";
import type { EngineClient, LicenceStatus } from "./engine";

const FREE = ["Photo and video editing by prompt", "Captions, looks, titles and music", "Exports without a watermark", "Command line: 1 reel per video, 1 photo per job"];
const PRO = ["Automation API for n8n, Zapier and scripts", "Webhooks, watch folders and batches", "Up to 10 reels from one video", "Everything in Free"];

export function tierLabel(s: LicenceStatus | null): string {
  if (!s) return "…";
  if (s.tier === "trial") return `Pro trial · ${s.trial_days_left} day${s.trial_days_left === 1 ? "" : "s"} left`;
  if (s.tier === "pro") return "Pro";
  if (s.tier === "studio") return "Studio";
  return "Free";
}

const date = (unix: number) => new Date(unix * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });

export function ProBadge({ status, onClick }: { status: LicenceStatus | null; onClick: () => void }) {
  return (
    <button type="button" className={`pro-badge ${status?.pro ? "on" : ""}`} onClick={onClick} title="Plan and licence">
      {tierLabel(status)}
    </button>
  );
}

interface Props {
  client: EngineClient;
  status: LicenceStatus | null;
  onChange: (s: LicenceStatus) => void;
}

export function ProPanel({ client, status, onChange }: Props) {
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [done, setDone] = useState<string | null>(null);

  const act = async (fn: () => Promise<LicenceStatus>, message: string) => {
    setBusy(true);
    setError(null);
    setDone(null);
    try {
      const s = await fn();
      onChange(s);
      setDone(message);
      setKey("");
      setConfirmRemove(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const lic = status?.licence;
  const keyActive = !!lic && (status?.tier === "pro" || status?.tier === "studio");
  const trialUnused = status?.tier === "free" && status.trial_days_left === null;
  const trialOver = status?.tier === "free" && status.trial_days_left === 0;

  return (
    <div className="pro-panel">
      <div className="pro-status">
        <span className={`pill ${status?.pro ? "pill-pro" : ""}`}>{tierLabel(status)}</span>
        {keyActive && lic && (
          <span className="muted small">
            Licensed{lic.to ? ` to ${lic.to}` : ""}
            {lic.seats > 1 ? ` · ${lic.seats} seats` : ""}
            {lic.expires ? ` · renews or ends ${date(lic.expires)}` : " · never expires"}
          </span>
        )}
        {status?.tier === "trial" && <span className="muted small">All Pro features are on until the trial ends.</span>}
        {trialOver && <span className="muted small">Your free trial has ended. Editing stays free.</span>}
      </div>
      {status?.problem && <p className="error small">The saved key was not accepted: {status.problem}</p>}

      <div className="compare">
        <div>
          <h4>Free</h4>
          <ul>{FREE.map((f) => <li key={f}>{f}</li>)}</ul>
        </div>
        <div className="pro-col">
          <h4>Pro</h4>
          <ul>{PRO.map((f) => <li key={f}>{f}</li>)}</ul>
        </div>
      </div>

      {trialUnused && (
        <button className="primary big" disabled={busy} onClick={() => act(client.startTrial, "Your 14-day Pro trial has started.")}>
          Start 14-day free trial
        </button>
      )}

      {!keyActive && (
        <label className="key-field">
          Have a licence key?
          <textarea
            id="licence-key"
            rows={3}
            spellCheck={false}
            value={key}
            placeholder="AIE1-…"
            onChange={(e) => setKey(e.target.value)}
          />
          <div className="row end">
            <button disabled={busy || key.trim().length < 20} onClick={() => act(() => client.activateLicence(key.trim()), "Licence activated. Thank you!")}>
              Activate key
            </button>
          </div>
        </label>
      )}

      {keyActive &&
        (confirmRemove ? (
          <div className="row">
            <span className="small">Remove the key from this PC? You can add it again later.</span>
            <div className="spacer" />
            <button onClick={() => setConfirmRemove(false)}>Keep it</button>
            <button disabled={busy} onClick={() => act(client.deactivateLicence, "Key removed from this PC.")}>
              Remove
            </button>
          </div>
        ) : (
          <div className="row end">
            <button onClick={() => setConfirmRemove(true)}>Remove key from this PC</button>
          </div>
        ))}

      {error && <p className="error small">{error}</p>}
      {done && <p className="ok small">{done}</p>}
      <p className="muted small">Keys are checked on this computer; nothing is sent anywhere. Buying a key opens at launch.</p>
    </div>
  );
}

export function ProModal(props: Props & { onClose: () => void }) {
  return (
    <div className="modal-backdrop" onClick={props.onClose}>
      <div className="card modal pro-modal" role="dialog" aria-label="Framewise Pro" onClick={(e) => e.stopPropagation()}>
        <div className="row between">
          <h3>Framewise Pro</h3>
          <button className="icon" aria-label="Close" onClick={props.onClose}>
            ✕
          </button>
        </div>
        <ProPanel client={props.client} status={props.status} onChange={props.onChange} />
      </div>
    </div>
  );
}
