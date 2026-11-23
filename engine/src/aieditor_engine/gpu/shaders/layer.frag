// One layer placed on the canvas: sample the source through the inverse of the
// clip's affine (fit/fill, motion, transform, anchor), cut it by its crop and
// masks, apply opacity. Output is premultiplied (rgb * a, a).
// Mirrors video/compose.py:_place + clip_frame (bilinear, edge-replicated
// picture, antialiased coverage like cv2.warpAffine of a mask of ones).
// needs: common.glsl

uniform sampler2D src;      // CPU upload, top row first; sRGB textures decode to linear
uniform int solid;          // 1: a flat colour instead of `src`
uniform vec3 color;         // linear
uniform mat3 inv;           // canvas pixel -> source pixel
uniform vec2 src_size;      // source pixels
uniform vec4 keep;          // kept source pixels: first col, first row, end col, end row (in `pre` units)
uniform vec2 pre;           // source px -> the pixels the CPU measures edges in (it pre-shrinks small layers)
uniform int covers;         // 1: the layer fills the canvas (no edges to antialias)
uniform float opacity;
uniform int has_mask;
uniform sampler2D mask;     // canvas sized coverage, top row first
uniform int height;

out vec4 frag;

void main() {
    ivec2 p = top_pixel(height);
    vec2 s = (inv * vec3(vec2(p), 1.0)).xy;
    vec3 c = solid == 1 ? color : texture(src, (s + 0.5) / src_size).rgb;
    float cov = 1.0;
    if (covers == 0) {
        vec2 e = s * pre;  // compose._place scales the matrix columns only
        float cx = clamp(min(e.x - keep.x + 1.0, keep.z - e.x), 0.0, 1.0);
        float cy = clamp(min(e.y - keep.y + 1.0, keep.w - e.y), 0.0, 1.0);
        cov = cx * cy;
    }
    if (has_mask == 1) cov *= texelFetch(mask, p, 0).r;
    float a = cov * opacity;
    frag = vec4(c * a, a);
}
