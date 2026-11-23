// Shared by every Framewise shader. No #version line: the engine prepends
// "#version 330 core", the app's WebGL2 preview "#version 300 es" + precision.
//
// Pixel conventions: canvas pixel centres are integers with y = 0 at the TOP
// (like numpy rows). Textures uploaded from the CPU keep that row order;
// textures the GPU rendered are bottom-up (OpenGL), so they are read with
// gl_FragCoord directly.

float srgb_to_lin1(float x) {
    x = max(x, 0.0);
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

float lin_to_srgb1(float x) {
    x = max(x, 0.0);
    return x <= 0.0031308 ? x * 12.92 : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

vec3 srgb_to_lin(vec3 c) { return vec3(srgb_to_lin1(c.r), srgb_to_lin1(c.g), srgb_to_lin1(c.b)); }
vec3 lin_to_srgb(vec3 c) { return vec3(lin_to_srgb1(c.r), lin_to_srgb1(c.g), lin_to_srgb1(c.b)); }

// top-based integer pixel of this fragment on a canvas `size` pixels high
ivec2 top_pixel(int height) {
    return ivec2(int(gl_FragCoord.x), height - 1 - int(gl_FragCoord.y));
}
