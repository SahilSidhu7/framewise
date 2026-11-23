// Transition between the outgoing (a) and incoming (b) layer of one track.
// Mirrors video/compose.py:_blend for the picture and, for alpha, uses
// `alpha_kind` (the CPU uses the same kind, or crossfade for the fades).
// needs: common.glsl

#define T_CROSSFADE 0
#define T_FADE_BLACK 1
#define T_FADE_WHITE 2
#define T_WIPE_LEFT 3
#define T_WIPE_RIGHT 4
#define T_SLIDE_LEFT 5
#define T_SLIDE_UP 6
#define T_ZOOM 7
#define T_CUT 8

uniform sampler2D ta;  // GPU rendered layers (bottom-up)
uniform sampler2D tb;
uniform int kind;
uniform int alpha_kind;
uniform float p;       // progress 0..1
uniform float e;       // eased progress (computed in double precision by the engine)
uniform ivec2 edge;    // int(w * e), int(h * e): wipe/slide offsets in pixels
uniform ivec2 size;    // canvas w, h

out vec4 frag;

// a GPU-rendered texel at a top-based pixel
vec4 at(sampler2D t, ivec2 top) { return texelFetch(t, ivec2(top.x, size.y - 1 - top.y), 0); }

// bilinear sample of a GPU-rendered layer at a top-based position, zero outside (cv2 BORDER_CONSTANT)
vec4 sample0(sampler2D t, vec2 top) {
    vec2 f = floor(top);
    vec2 w = top - f;
    ivec2 i = ivec2(f);
    vec4 acc = vec4(0.0);
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            ivec2 k = i + ivec2(dx, dy);
            if (k.x < 0 || k.y < 0 || k.x >= size.x || k.y >= size.y) continue;
            float wt = (dx == 1 ? w.x : 1.0 - w.x) * (dy == 1 ? w.y : 1.0 - w.y);
            acc += at(t, k) * wt;
        }
    }
    return acc;
}

vec4 mixed(int k, ivec2 top) {
    int w = size.x, h = size.y;
    vec4 A = at(ta, top);
    vec4 B = at(tb, top);
    if (k == T_CROSSFADE) return A * (1.0 - e) + B * e;
    if (k == T_FADE_BLACK || k == T_FADE_WHITE) {
        vec4 mid = k == T_FADE_WHITE ? vec4(1.0) : vec4(0.0);
        return p < 0.5 ? A * (1.0 - e * 2.0) + mid * (e * 2.0) : mid * (2.0 - e * 2.0) + B * (e * 2.0 - 1.0);
    }
    if (k == T_WIPE_LEFT) return top.x >= w - edge.x ? B : A;
    if (k == T_WIPE_RIGHT) return top.x < edge.x ? B : A;
    if (k == T_SLIDE_LEFT) {
        int off = edge.x;
        return top.x < w - off ? at(ta, ivec2(top.x + off, top.y)) : at(tb, ivec2(top.x - (w - off), top.y));
    }
    if (k == T_SLIDE_UP) {
        int off = edge.y;
        return top.y < h - off ? at(ta, ivec2(top.x, top.y + off)) : at(tb, ivec2(top.x, top.y - (h - off)));
    }
    if (k == T_ZOOM) {
        float z = 1.0 + 0.25 * e;
        vec2 c = vec2(float(w), float(h)) / 2.0;
        vec2 src = (vec2(top) - (1.0 - z) * c) / z;
        return sample0(ta, src) * (1.0 - e) + B * e;
    }
    return p >= 0.5 ? B : A;  // T_CUT
}

void main() {
    ivec2 top = ivec2(int(gl_FragCoord.x), size.y - 1 - int(gl_FragCoord.y));
    vec4 img = mixed(kind, top);
    float a = kind == alpha_kind ? img.a : mixed(alpha_kind, top).a;
    frag = vec4(img.rgb, a);
}
