import { useEffect, useState } from "react";
import type { Benchmark, EngineClient, Focus, SystemInfo } from "./engine";
import { Shell } from "./Shell";

const STEPS = ["Welcome", "Your PC", "What you'll make"] as const;

const FOCUS: { id: Focus; title: string; text: string }[] = [
  { id: "videos", title: "Videos", text: "Holiday clips, vlogs, family movies" },
  { id: "reels", title: "Reels and shorts", text: "Turn long videos into TikToks and Reels" },
  { id: "photos", title: "Photos", text: "Fix and style pictures" },
  { id: "automation", title: "Automation", text: "Run edits from n8n, scripts or agents" },
];

type Check = { label: string; state: "ok" | "warn" | "off"; detail: string };

function checks(info: SystemInfo): Check[] {
  const mem = info.memory_gb;
  return [
    { label: "Processor", state: info.cores >= 4 ? "ok" : "warn", detail: `${info.cpu} · ${info.cores} threads` },
    {
      label: "Memory",
      state: mem === null ? "warn" : mem >= 8 ? "ok" : "warn",
      detail: mem === null ? "Could not read" : `${mem} GB${mem < 8 ? " · 8 GB or more is smoother" : ""}`,
    },
    {
      label: "Free space",
      state: info.disk_free_gb >= 10 ? "ok" : "warn",
      detail: `${Math.round(info.disk_free_gb)} GB${info.disk_free_gb < 10 ? " · exports and previews need room" : ""}`,
    },
    {
      label: "Fast exports",
      state: info.hardware_encoder ? "ok" : "off",
      detail: info.hardware_encoder ? "NVIDIA encoder found" : "Not available · exports use the processor",
    },
    {
      label: "Editing model",
      state: info.ai.editing_model.ready ? "ok" : "warn",
      detail: info.ai.editing_model.ready ? "Ready" : "Not installed yet · simple rules are used instead",
    },
    {
      label: "Speech (captions, reels)",
      state: info.ai.speech.ready ? "ok" : "warn",
      detail: info.ai.speech.ready ? "Ready" : "Not installed yet · needed for captions and reels",
    },
    {
      label: "Face finder",
      state: info.ai.face_model.ready ? "ok" : "off",
      detail: info.ai.face_model.ready ? "Ready" : "Optional · a built-in fallback is used",
    },
  ];
}

interface Props {
  client: EngineClient;
  onDone: (focus: Focus | null) => void;
}

export function Onboarding({ client, onDone }: Props) {
  const [step, setStep] = useState(0);
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [bench, setBench] = useState<Benchmark | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);
  const [focus, setFocus] = useState<Focus | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (step !== 1 || info) return;
    client.system().then(setInfo, (e) => setCheckError(String(e.message ?? e)));
    client.benchmark().then(setBench, () => undefined);
  }, [step, info, client]);

  const finish = async () => {
    setSaving(true);
    try {
      await client.saveAppSettings({ onboarded: true, focus });
    } catch {
      /* not fatal: onboarding just shows again next time */
    }
    onDone(focus);
  };

  const nav = (
    <>
      {step > 0 ? <button onClick={() => setStep(step - 1)}>Back</button> : <span />}
      {step < STEPS.length - 1 ? (
        <button className="primary" onClick={() => setStep(step + 1)}>
          {step === 0 ? "Get started" : "Continue"}
        </button>
      ) : (
        <button className="primary" disabled={saving} onClick={finish}>
          Open Framewise
        </button>
      )}
    </>
  );

  return (
    <Shell
      footer={nav}
      side={
        <ol className="steps" aria-label="Setup steps">
          {STEPS.map((s, i) => (
            <li key={s} className={i === step ? "now" : i < step ? "done" : ""} aria-current={i === step ? "step" : undefined}>
              <span className="step-n">{i < step ? "✓" : i + 1}</span>
              {s}
            </li>
          ))}
        </ol>
      }
    >
        {step === 0 && (
          <div className="ob-body">
            <h1>Edit photos and video. Free, on your own computer.</h1>
            <p className="lead">Fix and style photos by describing the result, edit video on the timeline, and turn long videos into short clips automatically.</p>
            <ul className="ticks">
              <li>Works on this computer. Your photos and videos are never uploaded.</li>
              <li>Every automatic edit can be undone with one click.</li>
              <li>Editing is free, with no watermark.</li>
            </ul>
          </div>
        )}

        {step === 1 && (
          <div className="ob-body">
            <h2>Your PC</h2>
            {checkError && <p className="error">{checkError}</p>}
            {!info && !checkError && <p className="muted">Checking processor, memory and installed components…</p>}
            {info && (
              <ul className="checks">
                {checks(info).map((c) => (
                  <li key={c.label}>
                    <span className={`dot ${c.state}`} aria-hidden="true" />
                    <span className="c-label">{c.label}</span>
                    <span className="c-detail">{c.detail}</span>
                  </li>
                ))}
              </ul>
            )}
            <div className={`speed ${bench?.tier ?? ""}`}>
              <strong>Speed test</strong>
              <span>{bench ? bench.summary : "Timing a short 1080p render…"}</span>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="ob-body">
            <h2>What will you make most?</h2>
            <p className="muted">This only sets what you see first. Everything stays available.</p>
            <div className="focus-grid">
              {FOCUS.map((f) => (
                <button key={f.id} type="button" className={`focus ${focus === f.id ? "sel" : ""}`} aria-pressed={focus === f.id} onClick={() => setFocus(f.id)}>
                  <strong>{f.title}</strong>
                  <span>{f.text}</span>
                </button>
              ))}
            </div>
          </div>
        )}
    </Shell>
  );
}
