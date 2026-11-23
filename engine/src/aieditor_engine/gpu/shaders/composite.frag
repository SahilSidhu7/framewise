// canvas + one premultiplied layer -> new canvas, with the layer's blend mode.
// needs: common.glsl, blend.glsl

uniform sampler2D base;   // GPU rendered (bottom-up)
uniform sampler2D layer;  // GPU rendered (bottom-up)
uniform int mode;

out vec4 frag;

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    vec3 canvas = texelFetch(base, q, 0).rgb;
    vec4 l = texelFetch(layer, q, 0);
    frag = vec4(composite(canvas, l.rgb, l.a, mode), 1.0);
}
