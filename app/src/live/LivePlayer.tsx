// Live preview player: plays the timeline in the browser with WebGL2 using the
// engine's shaders. Video comes from the media proxies (<video> elements, one per
// clip so two clips of one file can overlap), sound from those elements and from
// <audio> elements for music. Captions, bars, fades and the progress bar are
// drawn on a 2D overlay. When playback stops, the editor shows the engine's
// exact frame on top.

import { useEffect, useRef, useState } from "react";

import { BLEND_MODES, type Clip, type EngineClient, type LivePostOp, type Project, type Track } from "../engine";
import { masksCanvas } from "./masks";
import { LiveRenderer, srgbToLin, type Geometry, type LiveFrame, type LiveLayer, type LiveStep } from "./renderer";
import { activeClip, clipEnd, clipValue, inverseColumnMajor, layerAlpha, matrix, pictureTracks, sourceTime, trackLayer, type LayerRef } from "./timeline";

type MediaEl = HTMLVideoElement | HTMLImageElement | HTMLAudioElement;

interface Props {
  client: EngineClient;
  project: Project;
  revision: number;
  playhead: number;
  playing: boolean;
  /** Bump to move the clock to `playhead` while playing (a seek). */
  jump?: number;
  maxSide?: number;
  onTime: (t: number) => void;
  onEnded: () => void;
  onDrawn?: () => void;
  onInfo?: (missing: string[]) => void;
  onError?: (message: string) => void;
}

export const liveSupported = (() => {
  let cached: boolean | null = null;
  return () => (cached ??= LiveRenderer.supported());
})();

function outputSize(p: Project, maxSide: number): [number, number] {
  let { width: w, height: h } = p.settings;
  if (Math.max(w, h) > maxSide) {
    const s = maxSide / Math.max(w, h);
    w = Math.round((w * s) / 2) * 2;
    h = Math.round((h * s) / 2) * 2;
  }
  return [Math.max(2, w), Math.max(2, h)];
}

const hexToLinear = (hex: string): [number, number, number] =>
  [1, 3, 5].map((i) => srgbToLin(parseInt(hex.slice(i, i + 2), 16) / 255)) as [number, number, number];

export function LivePlayer({ client, project, revision, playhead, playing, jump = 0, maxSide = 1280, onTime, onEnded, onDrawn, onInfo, onError }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const renderer = useRef<LiveRenderer | null>(null);
  const els = useRef(new Map<string, MediaEl>());
  const clock = useRef<{ wall: number; t: number } | null>(null);
  const seekedAt = useRef(new Map<string, number>()); // element key -> when we last moved it (playback)
  const lastSize = useRef(new Map<string, [number, number]>()); // element key -> video size of its last frame
  const latest = useRef({ project, playhead, playing, revision });
  latest.current = { project, playhead, playing, revision };
  const [, setTick] = useState(0);
  const duration = Math.max(0, ...project.tracks.flatMap((t) => t.clips.map(clipEnd)), ...project.captions.map((c) => c.end));

  // ---- renderer
  useEffect(() => {
    try {
      renderer.current = new LiveRenderer(canvasRef.current!);
    } catch (e) {
      onError?.(e instanceof Error ? e.message : String(e));
    }
    const map = els.current;
    return () => {
      renderer.current?.dispose();
      renderer.current = null;
      for (const el of map.values()) release(el);
      map.clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- grade: refetch after every change
  useEffect(() => {
    let stop = false;
    client.liveInfo().then(
      async (info) => {
        const lattice = info.lattice ? await client.gradeLattice().catch(() => null) : null;
        if (stop || !renderer.current) return;
        renderer.current.setGrade(lattice, info.lattice_size, info.post as LivePostOp[]);
        onInfo?.(info.missing);
        draw(latest.current.playhead);
      },
      () => undefined,
    );
    return () => {
      stop = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, revision]);

  // ---- media elements: one per clip on the timeline
  useEffect(() => {
    const map = els.current;
    const want = new Set<string>();
    for (const track of project.tracks) {
      for (const clip of track.clips) {
        const item = project.media[clip.media_id];
        if ((clip.source ?? "media") !== "media" || !item) continue;
        const url = item.kind === "image" ? client.renderUrl(item.id, { max: 2048, rev: revision }) : client.mediaPlayUrl(item);
        const key = `${clip.id}|${url}`;
        want.add(key);
        if (map.has(key)) continue;
        let el: MediaEl;
        if (item.kind === "image") {
          el = new Image();
          el.onload = () => redraw();
        } else {
          el = document.createElement(item.kind === "audio" ? "audio" : "video");
          el.preload = "auto";
          if (el instanceof HTMLVideoElement) el.playsInline = true;
          el.muted = true;
          el.addEventListener("seeked", redraw);
          el.addEventListener("loadeddata", redraw);
        }
        el.crossOrigin = "anonymous";
        el.src = url;
        map.set(key, el);
      }
    }
    for (const [key, el] of map) {
      if (!want.has(key)) {
        release(el);
        map.delete(key);
        renderer.current?.dropSource(key);
      }
    }
    redraw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, project, revision]);

  const elFor = (clip: Clip): [string, MediaEl] | null => {
    for (const [key, el] of els.current) if (key.startsWith(clip.id + "|")) return [key, el];
    return null;
  };

  // ---- media sync: play the clips under the playhead, pause the rest
  function sync(t: number, live: boolean) {
    const p = latest.current.project;
    const active = new Map<string, { clip: Clip; st: number; track: Track }>();
    for (const track of p.tracks) {
      if (track.kind === "audio") {
        const c = activeClip(track, t);
        if (c) active.set(c.id, { clip: c, st: sourceTime(c, t), track });
        continue;
      }
      const tl = trackLayer(track, t);
      for (const ref of tl ? [tl.a, tl.b] : []) {
        if (ref && "clip" in ref) active.set(ref.clip.id, { clip: ref.clip, st: sourceTime(ref.clip, ref.t), track });
      }
    }
    for (const [key, el] of els.current) {
      if (el instanceof HTMLImageElement) continue;
      const clipId = key.split("|")[0];
      const a = active.get(clipId);
      if (!a) {
        if (!el.paused) el.pause();
        el.muted = true;
        continue;
      }
      const { clip, st, track } = a;
      const item = p.media[clip.media_id];
      let vol = clipValue(clip, "volume", t);
      if (clip.fade_in > 0) vol *= Math.min(1, Math.max(0, (t - clip.start) / clip.fade_in));
      if (clip.fade_out > 0) vol *= Math.min(1, Math.max(0, (clipEnd(clip) - t) / clip.fade_out));
      el.muted = !live || clip.muted || !!track.muted || !(item?.has_audio || item?.kind === "audio") || vol <= 0;
      el.volume = Math.min(1, Math.max(0, vol));
      const rate = Math.min(16, Math.max(0.0625, clip.speed));
      if (live) {
        if (el.playbackRate !== rate) el.playbackRate = rate;
        // re-sync only when far off, and give a seek a second to land before trying again
        const now = performance.now();
        if (Math.abs(el.currentTime - st) > 0.3 && now - (seekedAt.current.get(key) ?? 0) > 1000) {
          el.currentTime = st;
          seekedAt.current.set(key, now);
        }
        if (el.paused) el.play().catch(() => undefined);
      } else {
        if (!el.paused) el.pause();
        if (Math.abs(el.currentTime - st) > 0.004) el.currentTime = st;
      }
    }
  }

  // ---- one frame
  function geometry(clip: Clip, t: number, sw: number, sh: number, w: number, h: number): Geometry {
    const { m, covers } = matrix(clip, t, sw, sh, w, h);
    const scale = Math.sqrt(Math.abs(m[0] * m[4] - m[1] * m[3]));
    let pw = sw;
    let ph = sh;
    if (scale < 0.75) {
      pw = Math.max(1, Math.floor(sw * scale * 1.25));
      ph = Math.max(1, Math.floor(sh * scale * 1.25));
    }
    const cr = clip.crop ?? { left: 0, top: 0, right: 0, bottom: 0 };
    const mask = masksCanvas(clip.masks, sw, sh, m, w, h);
    return {
      sw, sh, inv: inverseColumnMajor(m),
      keep: [Math.round(pw * cr.left), Math.round(ph * cr.top), pw - Math.round(pw * cr.right), ph - Math.round(ph * cr.bottom)],
      pre: [pw / sw, ph / sh], covers, opacity: layerAlpha(clip, t), mask: mask?.canvas ?? null, maskKey: mask?.key ?? "", mip: scale < 0.75,
    };
  }

  function layerOf(ref: LayerRef | undefined, w: number, h: number): LiveLayer | null {
    if (!ref) return null;
    if ("clear" in ref) return { kind: "clear", alpha: ref.clear };
    const { clip, t } = ref;
    if (clip.source === "solid") return { kind: "solid", color: hexToLinear(clip.color ?? "#000000"), ...geometry(clip, t, w, h, w, h) };
    if (clip.source === "adjustment") return null;
    const found = elFor(clip);
    if (!found) return null;
    const [key, el] = found;
    if (el instanceof HTMLImageElement) {
      if (!el.complete || !el.naturalWidth) return null;
      return { kind: "source", el, key, fresh: false, ...geometry(clip, t, el.naturalWidth, el.naturalHeight, w, h) };
    }
    if (!(el instanceof HTMLVideoElement)) return null;
    if (el.readyState < 2 || !el.videoWidth) {
      // mid-seek: keep showing the frame we already have, if any
      const size = lastSize.current.get(key);
      return size ? { kind: "source", el, key, fresh: false, ...geometry(clip, t, size[0], size[1], w, h) } : null;
    }
    lastSize.current.set(key, [el.videoWidth, el.videoHeight]);
    return { kind: "source", el, key, fresh: true, ...geometry(clip, t, el.videoWidth, el.videoHeight, w, h) };
  }

  function frameAt(t: number): LiveFrame {
    const p = latest.current.project;
    const [w, h] = outputSize(p, maxSide);
    const steps: LiveStep[] = [];
    const tracks = pictureTracks(p);
    const skip = new Set<number>();
    tracks.forEach((track, i) => {
      if (skip.has(i) || track.hidden) return;
      const matteTrack = track.matte && i + 1 < tracks.length ? tracks[i + 1] : null;
      if (matteTrack) skip.add(i + 1);
      const tl = trackLayer(track, t);
      if (!tl) return;
      let layer: LiveLayer | null;
      if (tl.kind === "transition") {
        const a = layerOf(tl.a, w, h);
        const b = layerOf(tl.b, w, h);
        layer = a && b ? { kind: "transition", a, b, transition: tl.transition!, alphaKind: tl.alphaKind!, p: tl.p! } : b ?? a;
      } else layer = layerOf(tl.b, w, h);
      if (!layer) return;
      const step: LiveStep = { layer, blend: Math.max(0, BLEND_MODES.indexOf(tl.blend as never)) };
      if (matteTrack && track.matte) {
        const mt = trackLayer(matteTrack, t);
        step.matte = track.matte;
        step.matteLayer = mt ? layerOf(mt.b, w, h) : null;
      }
      steps.push(step);
    });
    return { w, h, seed: Math.floor(t * (p.settings.fps || 30)), steps };
  }

  function draw(t: number) {
    const r = renderer.current;
    if (!r) return;
    try {
      const frame = frameAt(t);
      r.render(frame);
      overlay(t, frame.w, frame.h);
      if (import.meta.env.DEV) (window as unknown as { __liveDraws: number }).__liveDraws = ((window as unknown as { __liveDraws?: number }).__liveDraws ?? 0) + 1;
      onDrawn?.();
    } catch (e) {
      onError?.(e instanceof Error ? e.message : String(e));
    }
  }

  function overlay(t: number, w: number, h: number) {
    const c = overlayRef.current;
    if (!c) return;
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const g = c.getContext("2d")!;
    g.clearRect(0, 0, w, h);
    const p = latest.current.project;
    const s = p.settings;
    let bar = 0;
    if (s.letterbox && w / s.letterbox < h) {
      bar = Math.round((h - w / s.letterbox) / 2);
      g.fillStyle = "#000";
      g.fillRect(0, 0, w, bar);
      g.fillRect(0, h - bar, w, bar);
    }
    for (const cap of p.captions) {
      if (cap.start > t || t >= cap.end) continue;
      const title = cap.kind === "title" || cap.style === "title";
      const size = Math.round(h * (title ? 0.085 : 0.055));
      g.font = `${cap.style === "cinematic" || title ? "600" : "800"} ${size}px Montserrat, Inter, sans-serif`;
      g.textAlign = "center";
      g.textBaseline = "middle";
      const pos = cap.position ?? (title ? "center" : "bottom");
      const y = pos === "top" || pos === "high" ? bar + h * 0.12 : pos === "center" ? h / 2 : pos === "center_low" ? h * 0.62 : h - bar - h * 0.12;
      const text = cap.style === "bold_pop" ? cap.text.toUpperCase() : cap.text;
      if (cap.style === "boxed") {
        const tw = g.measureText(text).width;
        g.fillStyle = "rgba(0,0,0,0.6)";
        g.fillRect(w / 2 - tw / 2 - size * 0.4, y - size * 0.7, tw + size * 0.8, size * 1.4);
      }
      g.lineWidth = Math.max(2, size * 0.12);
      g.strokeStyle = "rgba(0,0,0,0.75)";
      if (cap.style !== "boxed" && cap.style !== "minimal") g.strokeText(text, w / 2, y, w * 0.9);
      g.fillStyle = "#fff";
      g.fillText(text, w / 2, y, w * 0.9);
    }
    const total = duration;
    if (s.progress_bar && s.progress_bar !== "off" && total > 0) {
      const th = Math.max(3, Math.round(Math.min(w, h) * 0.008));
      const y0 = s.progress_bar === "top" ? bar : h - bar - th;
      g.fillStyle = "rgba(0,0,0,0.45)";
      g.fillRect(0, y0, w, th);
      g.fillStyle = "#fff";
      g.fillRect(0, y0, Math.round(w * Math.min(1, t / total)), th);
    }
    let fade = 1;
    if (s.fade_in > 0) fade = Math.min(fade, t / s.fade_in);
    if (s.fade_out > 0 && total > 0) fade = Math.min(fade, (total - t) / s.fade_out);
    if (fade < 1) {
      g.fillStyle = `rgba(0,0,0,${1 - Math.max(0, fade)})`;
      g.fillRect(0, 0, w, h);
    }
  }

  function redraw() {
    if (!latest.current.playing) draw(latest.current.playhead);
  }

  // ---- paused: follow the playhead (scrubbing)
  useEffect(() => {
    if (playing) return;
    sync(playhead, false);
    draw(playhead);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playhead, playing, project, revision]);

  // ---- playing: our own clock, media follows it
  useEffect(() => {
    if (!playing) {
      clock.current = null;
      for (const el of els.current.values()) if (!(el instanceof HTMLImageElement) && !el.paused) el.pause();
      return;
    }
    clock.current = { wall: performance.now(), t: latest.current.playhead >= duration - 0.05 ? 0 : latest.current.playhead };
    let raf = 0;
    const tick = () => {
      const c = clock.current;
      if (!c) return;
      const t = c.t + (performance.now() - c.wall) / 1000;
      if (t >= duration) {
        sync(duration, false);
        onEnded();
        return;
      }
      sync(t, true);
      draw(t);
      onTime(t);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    setTick((n) => n + 1);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  // ---- a seek while playing restarts the clock from the new playhead
  useEffect(() => {
    if (clock.current) clock.current = { wall: performance.now(), t: latest.current.playhead };
  }, [jump]);

  return (
    <div className="live-player">
      <canvas ref={canvasRef} />
      <canvas ref={overlayRef} className="live-overlay" />
    </div>
  );
}

function release(el: MediaEl) {
  if (!(el instanceof HTMLImageElement)) {
    el.pause();
    el.removeAttribute("src");
    el.load();
  } else el.src = "";
}

