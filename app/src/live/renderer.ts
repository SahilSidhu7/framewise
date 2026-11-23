// WebGL2 live preview. Runs the engine's own GLSL (engine/src/aieditor_engine/gpu/shaders),
// so playback looks like the export: same placement, masks, blend modes,
// transitions, mattes and grade. Differences: the grade LUT is the 65^3 lattice
// (trilinear) instead of the exact 256^3 table, grain is random per frame, and
// effects the shaders do not have (clip effects, bloom, adjustment layers) are left out.

import commonSrc from "../../../engine/src/aieditor_engine/gpu/shaders/common.glsl?raw";
import blendSrc from "../../../engine/src/aieditor_engine/gpu/shaders/blend.glsl?raw";
import layerSrc from "../../../engine/src/aieditor_engine/gpu/shaders/layer.frag?raw";
import compositeSrc from "../../../engine/src/aieditor_engine/gpu/shaders/composite.frag?raw";
import transitionSrc from "../../../engine/src/aieditor_engine/gpu/shaders/transition.frag?raw";
import matteSrc from "../../../engine/src/aieditor_engine/gpu/shaders/matte.frag?raw";
import finishSrc from "../../../engine/src/aieditor_engine/gpu/shaders/finish.frag?raw";

import type { LivePostOp } from "../engine";

const INCLUDES: Record<string, string> = { "common.glsl": commonSrc, "blend.glsl": blendSrc };
const HEADER = "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\nprecision highp sampler3D;\n";
const VERTEX = HEADER + "in vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";

export const TRANSITION_INDEX: Record<string, number> = {
  crossfade: 0, fade_black: 1, fade_white: 2, wipe_left: 3, wipe_right: 4, slide_left: 5, slide_up: 6, zoom: 7, cut: 8,
};

function withIncludes(src: string): string {
  const m = src.match(/^\/\/ needs:(.*)$/m);
  const needs = m ? m[1].split(",").map((s) => s.trim()).filter(Boolean) : [];
  return HEADER + needs.map((n) => INCLUDES[n] + "\n").join("") + src;
}

// ------------------------------------------------------------------ frame description

export interface Geometry {
  sw: number;
  sh: number;
  inv: Float32Array;
  keep: [number, number, number, number];
  pre: [number, number];
  covers: boolean;
  opacity: number;
  mask: HTMLCanvasElement | null;
  maskKey: string;
  mip: boolean;
}
export type LiveLayer =
  | ({ kind: "source"; el: TexImageSource; key: string; fresh: boolean } & Geometry)
  | ({ kind: "solid"; color: [number, number, number] } & Geometry)
  | { kind: "clear"; alpha: number }
  | { kind: "transition"; a: LiveLayer; b: LiveLayer; transition: string; alphaKind: string; p: number };

export interface LiveStep {
  layer: LiveLayer;
  blend: number;
  matte?: string;
  matteLayer?: LiveLayer | null;
}

export interface LiveFrame {
  w: number;
  h: number;
  seed: number;
  steps: LiveStep[];
}

type Uniform = { loc: WebGLUniformLocation; type: number };
interface Prog {
  prog: WebGLProgram;
  uniforms: Map<string, Uniform>;
}
interface Target {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
}

// ------------------------------------------------------------------ renderer

export class LiveRenderer {
  readonly gl: WebGL2RenderingContext;
  private progs: Record<string, Prog> = {};
  private vao: WebGLVertexArrayObject;
  private targets = new Map<string, Target>();
  private textures = new Map<string, WebGLTexture>();
  private size = { w: 0, h: 0 };
  private dummy2d: WebGLTexture;
  private dummy3d: WebGLTexture;
  private enc: WebGLTexture;
  private lin8: WebGLTexture;
  private noise: WebGLTexture | null = null;
  private lut: WebGLTexture | null = null;
  private lutSize = 0;
  private post: LivePostOp[] = [];
  private fields: WebGLTexture[][] = []; // per post op: one texture (vignette) or a few to cycle (grain)

  static supported(): boolean {
    const gl = document.createElement("canvas").getContext("webgl2");
    return !!gl && !!gl.getExtension("EXT_color_buffer_float") && gl.getParameter(gl.MAX_TEXTURE_SIZE) >= 16384;
  }

  constructor(readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
    if (!gl || !gl.getExtension("EXT_color_buffer_float")) throw new Error("WebGL2 with float render targets is not available");
    this.gl = gl;
    const sources: Record<string, string> = {
      layer: layerSrc, composite: compositeSrc, transition: transitionSrc, matte: matteSrc, finish: finishSrc,
    };
    for (const [name, src] of Object.entries(sources)) this.progs[name] = this.program(withIncludes(src), name);
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.dummy2d = this.floatTexture(1, 1, 4, new Float32Array(4));
    this.dummy3d = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_3D, this.dummy3d);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB8, 1, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array(3));
    // linear -> sRGB * 255 table and u8 -> linear table (same as the engine's)
    const enc = new Float32Array(16384);
    for (let i = 0; i < 16384; i++) enc[i] = linToSrgb(i / 16383) * 255;
    this.enc = this.floatTexture(16384, 1, 1, enc);
    const lin8 = new Float32Array(256);
    for (let i = 0; i < 256; i++) lin8[i] = srgbToLin(i / 255);
    this.lin8 = this.floatTexture(256, 1, 1, lin8);
  }

  private program(fragment: string, name: string): Prog {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`${name} shader: ${gl.getShaderInfoLog(s)}`);
      return s;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fragment));
    gl.bindAttribLocation(prog, 0, "pos");
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`${name} program: ${gl.getProgramInfoLog(prog)}`);
    const uniforms = new Map<string, Uniform>();
    const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS) as number;
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(prog, i)!;
      uniforms.set(info.name.replace(/\[0\]$/, ""), { loc: gl.getUniformLocation(prog, info.name)!, type: info.type });
    }
    return { prog, uniforms };
  }

  private floatTexture(w: number, h: number, comps: 1 | 3 | 4, data: Float32Array | null): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const [internal, format] = comps === 1 ? [gl.R32F, gl.RED] : comps === 3 ? [gl.RGB32F, gl.RGB] : [gl.RGBA32F, gl.RGBA];
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, gl.FLOAT, data);
    this.nearest(gl.TEXTURE_2D);
    return tex;
  }

  private nearest(target: number) {
    const gl = this.gl;
    gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  private target(name: string): Target {
    const { w, h } = this.size;
    let t = this.targets.get(name);
    if (!t) {
      const gl = this.gl;
      const tex = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
      this.nearest(gl.TEXTURE_2D);
      const fbo = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      t = { fbo, tex };
      this.targets.set(name, t);
    }
    return t;
  }

  /** Set the drawing size (render targets, dither and effect fields follow). */
  resize(w: number, h: number) {
    if (w === this.size.w && h === this.size.h) return;
    const gl = this.gl;
    for (const t of this.targets.values()) {
      gl.deleteFramebuffer(t.fbo);
      gl.deleteTexture(t.tex);
    }
    this.targets.clear();
    this.size = { w, h };
    this.canvas.width = w;
    this.canvas.height = h;
    const nw = w + 32;
    const nh = h + 32;
    const noise = new Float32Array(nw * nh * 3);
    for (let i = 0; i < noise.length; i++) noise[i] = Math.random() - Math.random() + 0.5;
    if (this.noise) gl.deleteTexture(this.noise);
    this.noise = this.floatTexture(nw, nh, 3, noise);
    this.buildFields();
  }

  /** The grade: the baked 65^3 lattice (or null) and the effects after it. */
  setGrade(lattice: Uint8Array | null, size: number, post: LivePostOp[]) {
    const gl = this.gl;
    if (this.lut) gl.deleteTexture(this.lut);
    this.lut = null;
    this.lutSize = 0;
    if (lattice && lattice.length === size * size * size * 3) {
      this.lut = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_3D, this.lut);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB8, size, size, size, 0, gl.RGB, gl.UNSIGNED_BYTE, lattice);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
      this.lutSize = size;
    }
    this.post = post.slice(0, 4);
    this.buildFields();
  }

  private buildFields() {
    const gl = this.gl;
    for (const list of this.fields) for (const t of list) gl.deleteTexture(t);
    this.fields = [];
    const { w, h } = this.size;
    if (!w || !h) return;
    for (const op of this.post) {
      if (op.op === "vignette") this.fields.push([this.floatTexture(w, h, 1, vignetteGain(w, h, op.amount, op.midpoint, op.feather))]);
      else this.fields.push([0, 1, 2, 3].map(() => this.floatTexture(w, h, 1, grainField(w, h, op.size))));
    }
  }

  private sourceTexture(key: string, el: TexImageSource, fresh: boolean, mip: boolean): WebGLTexture {
    const gl = this.gl;
    let tex = this.textures.get(key);
    const isNew = !tex;
    if (!tex) {
      tex = gl.createTexture()!;
      this.textures.set(key, tex);
    }
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (isNew || fresh) {
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.SRGB8_ALPHA8, gl.RGBA, gl.UNSIGNED_BYTE, el);
      if (mip) gl.generateMipmap(gl.TEXTURE_2D);
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mip ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  /** Forget a source texture (its element went away). */
  dropSource(key: string) {
    const tex = this.textures.get(key);
    if (tex) this.gl.deleteTexture(tex);
    this.textures.delete(key);
  }

  private maskTexture(key: string, canvas: HTMLCanvasElement): WebGLTexture {
    const gl = this.gl;
    const k = "mask:" + key;
    let tex = this.textures.get(k);
    if (!tex) {
      tex = gl.createTexture()!;
      this.textures.set(k, tex);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, gl.RED, gl.UNSIGNED_BYTE, canvas);
      this.nearest(gl.TEXTURE_2D);
      if (this.textures.size > 64) {
        for (const old of [...this.textures.keys()].filter((x) => x.startsWith("mask:")).slice(0, 16)) this.dropSource(old);
      }
    }
    return tex;
  }

  private draw(name: string, target: Target | null, values: Record<string, unknown>) {
    const gl = this.gl;
    const p = this.progs[name];
    gl.useProgram(p.prog);
    let unit = 0;
    for (const [key, value] of Object.entries(values)) {
      const u = p.uniforms.get(key);
      if (!u || value === undefined) continue;
      switch (u.type) {
        case gl.SAMPLER_2D:
        case gl.SAMPLER_3D:
          gl.activeTexture(gl.TEXTURE0 + unit);
          gl.bindTexture(u.type === gl.SAMPLER_3D ? gl.TEXTURE_3D : gl.TEXTURE_2D, value as WebGLTexture);
          gl.uniform1i(u.loc, unit++);
          break;
        case gl.FLOAT:
          gl.uniform1f(u.loc, value as number);
          break;
        case gl.INT:
        case gl.BOOL:
          gl.uniform1i(u.loc, value as number);
          break;
        case gl.FLOAT_VEC2:
          gl.uniform2fv(u.loc, value as number[]);
          break;
        case gl.FLOAT_VEC3:
          gl.uniform3fv(u.loc, value as number[]);
          break;
        case gl.FLOAT_VEC4:
          gl.uniform4fv(u.loc, value as number[]);
          break;
        case gl.INT_VEC2:
          gl.uniform2iv(u.loc, value as number[]);
          break;
        case gl.INT_VEC4:
          gl.uniform4iv(u.loc, value as number[]);
          break;
        case gl.FLOAT_MAT3:
          gl.uniformMatrix3fv(u.loc, false, value as Float32Array);
          break;
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
    gl.viewport(0, 0, this.size.w, this.size.h);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private layer(l: LiveLayer, name: string): Target {
    const gl = this.gl;
    const out = this.target(name);
    if (l.kind === "clear") {
      gl.bindFramebuffer(gl.FRAMEBUFFER, out.fbo);
      gl.viewport(0, 0, this.size.w, this.size.h);
      gl.clearColor(0, 0, 0, l.alpha);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return out;
    }
    if (l.kind === "transition") {
      const a = this.layer(l.a, name + "_a");
      const b = this.layer(l.b, name + "_b");
      const e = l.p * l.p * (3 - 2 * l.p);
      const { w, h } = this.size;
      this.draw("transition", out, {
        ta: a.tex, tb: b.tex, kind: TRANSITION_INDEX[l.transition] ?? 8, alpha_kind: TRANSITION_INDEX[l.alphaKind] ?? 0,
        p: l.p, e, edge: [Math.floor(w * e), Math.floor(h * e)], size: [w, h],
      });
      return out;
    }
    const src = l.kind === "source" ? this.sourceTexture(l.key, l.el, l.fresh, l.mip) : this.dummy2d;
    const mask = l.mask ? this.maskTexture(l.maskKey, l.mask) : this.dummy2d;
    this.draw("layer", out, {
      src, solid: l.kind === "solid" ? 1 : 0, color: l.kind === "solid" ? l.color : [0, 0, 0], inv: l.inv,
      src_size: [l.sw, l.sh], keep: l.keep, pre: l.pre, covers: l.covers ? 1 : 0, opacity: l.opacity,
      has_mask: l.mask ? 1 : 0, mask, height: this.size.h,
    });
    return out;
  }

  render(frame: LiveFrame) {
    this.resize(frame.w, frame.h);
    const gl = this.gl;
    let cur = this.target("canvas0");
    let other = this.target("canvas1");
    gl.bindFramebuffer(gl.FRAMEBUFFER, cur.fbo);
    gl.viewport(0, 0, frame.w, frame.h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    for (const step of frame.steps) {
      let layer = this.layer(step.layer, "layer");
      if (step.matte) {
        const m = step.matteLayer ? this.layer(step.matteLayer, "matte") : null;
        const out = this.target("matted");
        this.draw("matte", out, {
          layer: layer.tex, matte: m ? m.tex : this.dummy2d, use_matte: m ? 1 : 0,
          fallback: step.matte.endsWith("inverted") ? 1 : 0, luma: step.matte.startsWith("luma") ? 1 : 0,
          inverted: step.matte.endsWith("inverted") ? 1 : 0,
        });
        layer = out;
      }
      this.draw("composite", other, { base: cur.tex, layer: layer.tex, mode: step.blend });
      [cur, other] = [other, cur];
    }
    const kinds = [0, 0, 0, 0];
    const amounts = [0, 0, 0, 0];
    const fields: Record<string, WebGLTexture> = {};
    for (let i = 0; i < 4; i++) {
      const op = this.post[i];
      const list = this.fields[i];
      fields[`op${i}`] = list ? list[frame.seed % list.length] : this.dummy2d;
      if (op) {
        kinds[i] = op.op === "vignette" ? 1 : 2;
        amounts[i] = op.op === "grain" ? op.amount : 0;
      }
    }
    this.draw("finish", null, {
      canvas: cur.tex, enc: this.enc, lin8: this.lin8, noise: this.noise!, offset: [(frame.seed * 13) % 32, (frame.seed * 7) % 32],
      use_lut: this.lut ? 1 : 0, lut: this.lut ?? this.dummy3d, lut_size: this.lutSize, height: frame.h,
      n_ops: this.post.length, op_kind: kinds, op_amount: amounts, ...fields,
    });
  }

  /** Free GPU memory. The context itself stays usable: React may mount the same canvas again. */
  dispose() {
    const gl = this.gl;
    for (const t of this.targets.values()) {
      gl.deleteFramebuffer(t.fbo);
      gl.deleteTexture(t.tex);
    }
    this.targets.clear();
    for (const tex of this.textures.values()) gl.deleteTexture(tex);
    this.textures.clear();
    for (const list of this.fields) for (const tex of list) gl.deleteTexture(tex);
    this.fields = [];
    this.size = { w: 0, h: 0 };
  }
}

// ------------------------------------------------------------------ helpers (ports of the engine's)

export function srgbToLin(x: number): number {
  x = Math.max(x, 0);
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
}

export function linToSrgb(x: number): number {
  x = Math.max(x, 0);
  return x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055;
}

/** imaging/ops.py _vignette_gain */
function vignetteGain(w: number, h: number, amount: number, midpoint: number, feather: number): Float32Array {
  const out = new Float32Array(w * h);
  const aspect = w / h;
  const start = 0.25 + 0.6 * midpoint;
  const e0 = start - feather * 0.5;
  const e1 = start + feather * 0.6;
  for (let j = 0; j < h; j++) {
    const y = h > 1 ? -1 + (2 * j) / (h - 1) : 0;
    for (let i = 0; i < w; i++) {
      const x = w > 1 ? -1 + (2 * i) / (w - 1) : 0;
      const r = (Math.sqrt((x * Math.min(aspect, 1)) ** 2 + (y / Math.max(aspect, 1)) ** 2) / Math.SQRT2) * 1.4;
      const t = Math.min(1, Math.max(0, (r - e0) / (e1 - e0)));
      const m = t * t * (3 - 2 * t);
      out[j * w + i] = amount >= 0 ? 1 - amount * 0.85 * m : 1 + -amount * 0.6 * m;
    }
  }
  return out;
}

/** A grain field like imaging/ops.py grain_noise (random, bilinear instead of bicubic). */
function grainField(w: number, h: number, size: number): Float32Array {
  const cell = Math.max(1, (size * Math.min(w, h)) / 1000);
  const gw = Math.max(2, Math.floor(w / cell));
  const gh = Math.max(2, Math.floor(h / cell));
  const grid = new Float32Array(gw * gh);
  for (let i = 0; i < grid.length; i++) {
    const u = Math.max(1e-9, Math.random());
    grid[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }
  const out = new Float32Array(w * h);
  for (let j = 0; j < h; j++) {
    const gy = Math.min(gh - 1.001, Math.max(0, ((j + 0.5) * gh) / h - 0.5));
    const y0 = Math.floor(gy);
    const fy = gy - y0;
    for (let i = 0; i < w; i++) {
      const gx = Math.min(gw - 1.001, Math.max(0, ((i + 0.5) * gw) / w - 0.5));
      const x0 = Math.floor(gx);
      const fx = gx - x0;
      const a = grid[y0 * gw + x0] * (1 - fx) + grid[y0 * gw + x0 + 1] * fx;
      const b = grid[(y0 + 1) * gw + x0] * (1 - fx) + grid[(y0 + 1) * gw + x0 + 1] * fx;
      out[j * w + i] = a * (1 - fy) + b * fy;
    }
  }
  return out;
}
