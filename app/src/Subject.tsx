import { useEffect, useState } from "react";
import type { EditOp, EngineClient, MediaItem } from "./engine";

/** Normalised (0..1) position of a click on an <img> drawn with object-fit: contain. */
export function pointOnImage(img: HTMLImageElement, clientX: number, clientY: number): { x: number; y: number } | null {
  const r = img.getBoundingClientRect();
  const nw = img.naturalWidth || 1;
  const nh = img.naturalHeight || 1;
  const s = Math.min(r.width / nw, r.height / nh);
  const w = nw * s;
  const h = nh * s;
  const x = (clientX - (r.left + (r.width - w) / 2)) / w;
  const y = (clientY - (r.top + (r.height - h) / 2)) / h;
  if (x < 0 || x > 1 || y < 0 || y > 1) return null;
  return { x: Math.round(x * 1000) / 1000, y: Math.round(y * 1000) / 1000 };
}

/** Outline of the selected subject, drawn over the photo (same box and fit as the image). */
export function SubjectOutline({ client, item, point, aspect }: { client: EngineClient; item: MediaItem; point: string; aspect: number }) {
  const [polys, setPolys] = useState<number[][][]>([]);
  useEffect(() => {
    let live = true;
    client.subject(item.id, point).then((r) => live && setPolys(r.outline ?? []), () => live && setPolys([]));
    return () => {
      live = false;
    };
  }, [client, item.id, point]);
  if (!polys.length) return null;
  return (
    <svg className="subject-outline" viewBox={`0 0 ${aspect} 1`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      {polys.map((p, i) => (
        <polygon key={i} points={p.map(([x, y]) => `${x * aspect},${y}`).join(" ")} vectorEffect="non-scaling-stroke" />
      ))}
    </svg>
  );
}

interface Props {
  item: MediaItem;
  stack: EditOp[];
  picking: boolean;
  setPicking: (on: boolean) => void;
  run: (name: string, params: Record<string, unknown>) => Promise<unknown>;
}

/** Choose the photo's subject: automatic, or pick it with a click on the photo. */
export function PhotoSubjectControls({ item, stack, picking, setPicking, run }: Props) {
  const op = stack.find((o) => o.op === "subject");
  const point = String(op?.params.point ?? "auto");
  const auto = () =>
    op ? run("photo.set_op", { media_id: item.id, op_id: op.id, params: { point: "auto" } }) : Promise.resolve();
  return (
    <div className="subject-controls">
      <p className="muted small">
        {point === "auto" ? "Subject: found automatically." : "Subject: picked by you."} Use the sliders to light it or soften what's behind it.
      </p>
      <div className="row">
        <button className={picking ? "active" : ""} aria-pressed={picking} onClick={() => setPicking(!picking)}>
          {picking ? "Click the subject on the photo…" : "Pick on photo"}
        </button>
        <button disabled={point === "auto"} onClick={() => void auto()}>
          Automatic
        </button>
      </div>
    </div>
  );
}

/** Store a picked point on the photo's subject edit (creating it if needed). */
export async function setPhotoSubject(item: MediaItem, stack: EditOp[], run: Props["run"], x: number, y: number) {
  const op = stack.find((o) => o.op === "subject");
  const point = `${x.toFixed(3)},${y.toFixed(3)}`;
  if (op) await run("photo.set_op", { media_id: item.id, op_id: op.id, params: { point } });
  else await run("photo.add_op", { media_id: item.id, op: "subject", params: { point } });
}
