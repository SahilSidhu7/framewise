// The drawn box of a text or shape layer in layer space (fractions of the
// canvas), so the viewer handles hug the words or the shape, not the canvas.

import { useEffect, useState } from "react";

import type { Clip, EngineClient } from "../engine";

export type Bounds = [number, number, number, number];

export function useVectorBounds(client: EngineClient, clip: Clip | null | undefined, W: number, H: number): Bounds | null {
  const shape = clip?.source === "shape" ? clip.shape : null;
  const textKey = clip?.source === "text" && clip.text ? `${clip.id}|${JSON.stringify(clip.text)}|${W}x${H}` : "";
  const [text, setText] = useState<{ key: string; bounds: Bounds } | null>(null);

  useEffect(() => {
    if (!textKey || !clip) return;
    let live = true;
    // measure at a moderate size: bounds are fractions, so the size barely matters
    const s = Math.min(1, 1280 / Math.max(W, H));
    const w = Math.max(2, Math.round(W * s));
    const h = Math.max(2, Math.round(H * s));
    client
      .layerBounds(clip.id, w, h)
      .then((b) => live && setText({ key: textKey, bounds: b }))
      .catch(() => undefined);
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, textKey]);

  if (shape) return [0.5 - shape.width / 2, 0.5 - shape.height / 2, 0.5 + shape.width / 2, 0.5 + shape.height / 2];
  if (textKey) return text?.key === textKey ? text.bounds : null;
  return null;
}
