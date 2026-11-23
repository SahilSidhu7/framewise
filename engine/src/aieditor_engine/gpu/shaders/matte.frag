// Cut a layer by the track above it (mirrors video/layers.py:matte_value).
// needs: common.glsl

uniform sampler2D layer;  // GPU rendered
uniform sampler2D matte;  // GPU rendered
uniform int use_matte;    // 0: no matte clip at this time -> constant `fallback`
uniform float fallback;
uniform int luma;
uniform int inverted;

out vec4 frag;

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    vec4 l = texelFetch(layer, q, 0);
    float v = fallback;
    if (use_matte == 1) {
        vec4 m = texelFetch(matte, q, 0);
        v = luma == 1 ? dot(lin_to_srgb(clamp(m.rgb, 0.0, 1.0)), vec3(0.2126, 0.7152, 0.0722)) : m.a;
        if (inverted == 1) v = 1.0 - v;
    }
    frag = l * v;
}
