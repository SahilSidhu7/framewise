// Linear canvas -> dithered 8-bit sRGB, the baked grade LUT, then (optionally)
// the grade's spatial effects and a second encode.
// Mirrors video/fastgrade.py:GradePlan.finish: encode_u8 (16384-step table +
// the same triangular dither texture), the 256^3 table lookup, u8 -> linear,
// imaging/render.py:apply_flat for vignette and grain (domain switches
// included), encode_u8 again.
// needs: common.glsl

#define OP_VIGNETTE 1
#define OP_GRAIN 2

uniform sampler2D canvas;  // GPU rendered
uniform sampler2D enc;     // 16384 x 1, linear -> sRGB * 255 (R32F)
uniform sampler2D lin8;    // 256 x 1, u8 sRGB -> linear (R32F)
uniform sampler2D noise;   // (h + 32) x (w + 32) RGB32F, top row first
uniform ivec2 offset;      // dither offset for this frame (x, y)
uniform int use_lut;
uniform sampler3D lut;     // 256^3 RGB8: x = blue, y = green, z = red
uniform int lut_size;      // 0 or 256: exact table (engine); e.g. 65: the lattice, trilinear (app preview)
uniform int height;
uniform int n_ops;         // spatial grade effects after the LUT (0..4)
uniform ivec4 op_kind;
uniform vec4 op_amount;    // grain amount
uniform sampler2D op0;     // per op: vignette gain or grain field, canvas sized, top row first
uniform sampler2D op1;
uniform sampler2D op2;
uniform sampler2D op3;

out vec4 frag;

float encode1(float x, float n) {
    int i = int(clamp(x, 0.0, 1.0) * 16383.0);
    float v = texelFetch(enc, ivec2(i, 0), 0).r + n;
    return floor(clamp(v, 0.0, 255.0));
}

vec3 encode(vec3 c, vec3 n) { return vec3(encode1(c.r, n.r), encode1(c.g, n.g), encode1(c.b, n.b)); }

float field(int i, ivec2 top) {
    if (i == 0) return texelFetch(op0, top, 0).r;
    if (i == 1) return texelFetch(op1, top, 0).r;
    if (i == 2) return texelFetch(op2, top, 0).r;
    return texelFetch(op3, top, 0).r;
}

void main() {
    ivec2 top = top_pixel(height);
    vec3 c = texelFetch(canvas, ivec2(gl_FragCoord.xy), 0).rgb;
    vec3 n = texelFetch(noise, top + offset, 0).rgb;
    vec3 u = encode(c, n);
    if (use_lut == 1) {
        if (lut_size > 0 && lut_size < 256) {
            float ls = float(lut_size);
            vec3 q = (u.bgr / 255.0 * (ls - 1.0) + 0.5) / ls;  // texture x = blue
            u = floor(texture(lut, q).rgb * 255.0 + 0.5);
        } else {
            u = floor(texelFetch(lut, ivec3(int(u.b), int(u.g), int(u.r)), 0).rgb * 255.0 + 0.5);
        }
    }
    if (n_ops > 0) {
        vec3 x = vec3(texelFetch(lin8, ivec2(int(u.r), 0), 0).r, texelFetch(lin8, ivec2(int(u.g), 0), 0).r,
                      texelFetch(lin8, ivec2(int(u.b), 0), 0).r);
        bool display = false;
        for (int i = 0; i < 4; i++) {
            if (i >= n_ops) break;
            float f = field(i, top);
            if (op_kind[i] == OP_VIGNETTE) {  // linear domain
                if (display) { x = srgb_to_lin(x); display = false; }
                x = max(x * f, 0.0);
            } else {  // OP_GRAIN, display domain
                if (!display) { x = lin_to_srgb(x); display = true; }
                float l = clamp(dot(x, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
                x += f * (op_amount[i] * 0.12 * (0.35 + 2.6 * l * (1.0 - l)));
            }
        }
        if (display) x = srgb_to_lin(x);
        u = encode(x, n);
    }
    frag = vec4(u / 255.0, 1.0);
}
