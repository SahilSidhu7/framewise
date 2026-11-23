// The 26 Photoshop / W3C blend modes. Mirrors video/layers.py:blend_rgb exactly;
// mode numbers are the positions in core/project.py:BLEND_MODES.

#define M_NORMAL 0
#define M_DARKEN 1
#define M_MULTIPLY 2
#define M_COLOR_BURN 3
#define M_LINEAR_BURN 4
#define M_DARKER_COLOR 5
#define M_LIGHTEN 6
#define M_SCREEN 7
#define M_COLOR_DODGE 8
#define M_ADD 9
#define M_LIGHTER_COLOR 10
#define M_OVERLAY 11
#define M_SOFT_LIGHT 12
#define M_HARD_LIGHT 13
#define M_VIVID_LIGHT 14
#define M_LINEAR_LIGHT 15
#define M_PIN_LIGHT 16
#define M_HARD_MIX 17
#define M_DIFFERENCE 18
#define M_EXCLUSION 19
#define M_SUBTRACT 20
#define M_DIVIDE 21
#define M_HUE 22
#define M_SATURATION 23
#define M_COLOR 24
#define M_LUMINOSITY 25

vec3 pick(bvec3 cond, vec3 a, vec3 b) { return mix(b, a, vec3(cond)); }

float lum(vec3 c) { return dot(c, vec3(0.3, 0.59, 0.11)); }

vec3 clip_color(vec3 c) {
    float l = lum(c);
    float n = min(c.r, min(c.g, c.b));
    float x = max(c.r, max(c.g, c.b));
    vec3 o = n < 0.0 ? l + (c - l) * l / max(l - n, 1e-6) : c;
    return x > 1.0 ? l + (o - l) * (1.0 - l) / max(x - l, 1e-6) : o;
}

vec3 set_lum(vec3 c, float l) { return clip_color(c + (l - lum(c))); }
float sat(vec3 c) { return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b)); }

vec3 set_sat(vec3 c, float s) {
    float lo = min(c.r, min(c.g, c.b));
    float rng = max(c.r, max(c.g, c.b)) - lo;
    return rng > 1e-6 ? (c - lo) * s / max(rng, 1e-6) : vec3(0.0);
}

vec3 color_burn(vec3 cb, vec3 cs) {
    vec3 v = 1.0 - min(vec3(1.0), (1.0 - cb) / max(cs, vec3(1e-6)));
    return pick(greaterThanEqual(cb, vec3(1.0)), vec3(1.0), pick(lessThanEqual(cs, vec3(0.0)), vec3(0.0), v));
}

vec3 color_dodge(vec3 cb, vec3 cs) {
    vec3 v = min(vec3(1.0), cb / max(1.0 - cs, vec3(1e-6)));
    return pick(lessThanEqual(cb, vec3(0.0)), vec3(0.0), pick(greaterThanEqual(cs, vec3(1.0)), vec3(1.0), v));
}

vec3 hard_light(vec3 cb, vec3 cs) {
    return pick(lessThanEqual(cs, vec3(0.5)), cb * 2.0 * cs, 1.0 - (1.0 - cb) * (1.0 - (2.0 * cs - 1.0)));
}

vec3 blend_rgb(vec3 cb, vec3 cs, int mode) {
    if (mode == M_NORMAL) return cs;
    if (mode == M_DARKEN) return min(cb, cs);
    if (mode == M_MULTIPLY) return cb * cs;
    if (mode == M_COLOR_BURN) return color_burn(cb, cs);
    if (mode == M_LINEAR_BURN) return max(vec3(0.0), cb + cs - 1.0);
    if (mode == M_DARKER_COLOR) return lum(cs) < lum(cb) ? cs : cb;
    if (mode == M_LIGHTEN) return max(cb, cs);
    if (mode == M_SCREEN) return cb + cs - cb * cs;
    if (mode == M_COLOR_DODGE) return color_dodge(cb, cs);
    if (mode == M_ADD) return cb + cs;
    if (mode == M_LIGHTER_COLOR) return lum(cs) > lum(cb) ? cs : cb;
    if (mode == M_OVERLAY) return hard_light(cs, cb);
    if (mode == M_SOFT_LIGHT) {
        vec3 d = pick(lessThanEqual(cb, vec3(0.25)), ((16.0 * cb - 12.0) * cb + 4.0) * cb, sqrt(max(cb, vec3(0.0))));
        return pick(lessThanEqual(cs, vec3(0.5)), cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb), cb + (2.0 * cs - 1.0) * (d - cb));
    }
    if (mode == M_HARD_LIGHT) return hard_light(cb, cs);
    if (mode == M_VIVID_LIGHT) {
        vec3 burn = color_burn(cb, clamp(2.0 * cs, 0.0, 1.0));
        vec3 dodge = color_dodge(cb, clamp(2.0 * cs - 1.0, 0.0, 1.0));
        return pick(lessThanEqual(cs, vec3(0.5)), burn, dodge);
    }
    if (mode == M_LINEAR_LIGHT) return clamp(cb + 2.0 * cs - 1.0, 0.0, 1.0);
    if (mode == M_PIN_LIGHT) return pick(lessThanEqual(cs, vec3(0.5)), min(cb, 2.0 * cs), max(cb, 2.0 * cs - 1.0));
    if (mode == M_HARD_MIX) return vec3(greaterThanEqual(cb + cs, vec3(1.0)));
    if (mode == M_DIFFERENCE) return abs(cb - cs);
    if (mode == M_EXCLUSION) return cb + cs - 2.0 * cb * cs;
    if (mode == M_SUBTRACT) return max(vec3(0.0), cb - cs);
    if (mode == M_DIVIDE) {
        vec3 v = min(vec3(1.0), cb / max(cs, vec3(1e-6)));
        return pick(lessThanEqual(cs, vec3(0.0)), vec3(greaterThan(cb, vec3(0.0))), v);
    }
    if (mode == M_HUE) return set_lum(set_sat(cs, sat(cb)), lum(cb));
    if (mode == M_SATURATION) return set_lum(set_sat(cb, sat(cs)), lum(cb));
    if (mode == M_COLOR) return set_lum(cs, lum(cb));
    return set_lum(cb, lum(cs));  // M_LUMINOSITY
}

// One premultiplied layer (img, a) over an opaque linear canvas. Mirrors layers.py:composite.
vec3 composite(vec3 canvas, vec3 img, float a, int mode) {
    if (mode == M_NORMAL) return canvas * (1.0 - a) + img;
    if (mode == M_ADD) return canvas + img;
    if (mode == M_SCREEN) return canvas + img - clamp(canvas, 0.0, 1.0) * clamp(img, 0.0, 1.0);
    if (mode == M_MULTIPLY) return canvas * (1.0 - a) + canvas * img;
    vec3 cb = lin_to_srgb(clamp(canvas, 0.0, 1.0));
    vec3 cs = lin_to_srgb(clamp(img / max(a, 1e-4), 0.0, 1.0));
    vec3 mixed = srgb_to_lin(clamp(blend_rgb(cb, cs, mode), 0.0, 1.0));
    return canvas * (1.0 - a) + mixed * a;
}
