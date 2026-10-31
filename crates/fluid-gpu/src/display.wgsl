// Display compositing on the GPU: the layer stack over the paper, as wet-compose's
// `flatten` / `composite_isolated` and wet-doc's `Document::render` do on the CPU (blend
// formulas: W3C Compositing and Blending Level 1; pigment mixing: wet-pigment's
// Kubelka–Munk mixer), plus the app's view overlays (greyscale, stencil film, selection
// dimming), written premultiplied to an RGBA8 texture for the window.
//
// Passes, each with its own uniforms:
//   k_clear        paper colour and relief shading; clip base alpha 1
//   k_layer        per layer: mask and clipping, impasto shading, relief height, gloss,
//                  metallic coverage, then source-over with a blend mode (onto the
//                  painting, or into the open group)
//   k_group_clear  start an isolated group (transparent)
//   k_group_out    composite the finished group onto the painting
//   k_hmax, k_shadow   soft shadows from the summed relief
//   k_metal        metallic sheen and darkening
//   k_out          shading applied, rounded to 8 bits, view overlays, deckle alpha

struct DCfg {
    w: u32, h: u32, mode: u32, flags: u32,
    opacity: f32, impasto: f32, gloss: f32, gloss_k: f32,
    bg: vec4<f32>,
    shadow_k: f32, shadow_tan: f32, metal_ref: f32, metal_str: f32,
    metal_zoom: f32, env_w: u32, env_h: u32, _p0: u32,
};
// k_layer flags
const F_COMPOSITE: u32 = 1u;   // blend the layer's colour
const F_SHADE: u32 = 2u;       // multiply in its impasto shading
const F_CLIPPED: u32 = 4u;     // alpha × the base alpha below
const F_BASE: u32 = 8u;        // its alpha becomes the base alpha for clipped layers above
const F_MASK: u32 = 16u;       // alpha × its mask
const F_GROUP: u32 = 32u;      // composite into the open group
const F_HEIGHT: u32 = 64u;     // add its relief to the height sum
const F_METAL: u32 = 128u;     // add its coverage to the metallic coverage
const F_GLOSS: u32 = 256u;     // add its specular highlight
const F_PIGMENT: u32 = 512u;   // the Pigment mode mixes with Kubelka–Munk (else Normal)
// k_clear
const F_PAPER: u32 = 1u;
// k_out
const F_GREY: u32 = 1u;
const F_STENCIL: u32 = 2u;
const F_SELECT: u32 = 4u;
const F_DECKLE: u32 = 8u;

@group(0) @binding(0) var<uniform> cfg: DCfg;
// Composited colour (rgb) and the product of the shading factors (a).
@group(0) @binding(1) var<storage, read_write> ACC: array<vec4<f32>>;
// A layer: straight RGBA per cell, then relief heights, then wetness (GpuLayer's
// `rendered` layout).
@group(0) @binding(2) var<storage, read> L: array<f32>;
// Document planes: paper shading, stencil film, selection (0–255), deckle; then the
// Kubelka–Munk tables (128 slots) and the metallic reflection map.
@group(0) @binding(3) var<storage, read> DOC: array<f32>;
@group(0) @binding(4) var OUT: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read> MASK: array<f32>;
// Per cell: clip base alpha, relief height sum, metallic coverage, additive highlight.
@group(0) @binding(6) var<storage, read_write> AUX: array<vec4<f32>>;
// The open group: straight RGBA.
@group(0) @binding(7) var<storage, read_write> GACC: array<vec4<f32>>;
// Largest relief height (f32 bits), for the shadow reach.
@group(0) @binding(8) var<storage, read_write> HMAX: atomic<u32>;

fn ncells() -> u32 { return cfg.w * cfg.h; }
fn doc_plane(k: u32, c: u32) -> f32 { return DOC[k * ncells() + c]; }
fn km(i: u32) -> f32 { return DOC[4u * ncells() + i]; }

fn height(x: u32, y: u32) -> f32 {
    let cx = min(x, cfg.w - 1u); let cy = min(y, cfg.h - 1u);
    return L[4u * ncells() + cy * cfg.w + cx];
}
fn hsum(x: u32, y: u32) -> f32 {
    let cx = min(x, cfg.w - 1u); let cy = min(y, cfg.h - 1u);
    return AUX[cy * cfg.w + cx].y;
}
fn below(v: u32) -> u32 { return select(v - 1u, 0u, v == 0u); }

// ---- blend modes, in MixMode order
fn color_dodge(b: f32, s: f32) -> f32 {
    if (b <= 0.0) { return 0.0; }
    if (s >= 1.0) { return 1.0; }
    return min(b / (1.0 - s), 1.0);
}
fn color_burn(b: f32, s: f32) -> f32 {
    if (b >= 1.0) { return 1.0; }
    if (s <= 0.0) { return 0.0; }
    return 1.0 - min((1.0 - b) / s, 1.0);
}
fn hard_light(b: f32, s: f32) -> f32 {
    if (s <= 0.5) { return b * 2.0 * s; }
    let t = 2.0 * s - 1.0;
    return b + t - b * t;
}
fn soft_light(b: f32, s: f32) -> f32 {
    if (s <= 0.5) { return b - (1.0 - 2.0 * s) * b * (1.0 - b); }
    var d: f32;
    if (b <= 0.25) { d = ((16.0 * b - 12.0) * b + 4.0) * b; } else { d = sqrt(b); }
    return b + (2.0 * s - 1.0) * (d - b);
}
fn vivid_light(b: f32, s: f32) -> f32 {
    if (s <= 0.5) { return color_burn(b, 2.0 * s); }
    return color_dodge(b, 2.0 * s - 1.0);
}
fn separable(m: u32, b: f32, s: f32) -> f32 {
    switch m {
        case 1u: { return min(b + s, 1.0); }
        case 2u: { return max(b - s, 0.0); }
        case 3u: { return max(b + s - 1.0, 0.0); }
        case 4u: { return min(b, s); }
        case 5u: { return max(b, s); }
        case 6u: { return b * s; }
        case 7u: { return b + s - b * s; }
        case 8u: { return color_dodge(b, s); }
        case 9u: { return color_burn(b, s); }
        case 10u: { return hard_light(s, b); }
        case 11u: { return hard_light(b, s); }
        case 12u: { return soft_light(b, s); }
        case 13u: { return vivid_light(b, s); }
        case 14u: { return clamp(b + 2.0 * s - 1.0, 0.0, 1.0); }
        case 15u: { if (s <= 0.5) { return min(b, 2.0 * s); } return max(b, 2.0 * s - 1.0); }
        case 16u: { return select(0.0, 1.0, vivid_light(b, s) >= 0.5); }
        case 17u: { return abs(b - s); }
        case 18u: { return b + s - 2.0 * b * s; }
        case 26u: { return 1.0 - (1.0 - b) * s; }
        default: { return s; }
    }
}
fn lum(c: vec3<f32>) -> f32 { return 0.3 * c.x + 0.59 * c.y + 0.11 * c.z; }
fn clip_color(c: vec3<f32>) -> vec3<f32> {
    let l = lum(c);
    let n = min(min(c.x, c.y), c.z);
    let x = max(max(c.x, c.y), c.z);
    var o = c;
    if (n < 0.0) { o = l + (o - l) * l / (l - n); }
    if (x > 1.0) { o = l + (o - l) * (1.0 - l) / (x - l); }
    return o;
}
fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> { return clip_color(c + (l - lum(c))); }
fn sat(c: vec3<f32>) -> f32 { return max(max(c.x, c.y), c.z) - min(min(c.x, c.y), c.z); }
fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    var imax = 0u; var imin = 0u;
    for (var i = 1u; i < 3u; i++) {
        if (c[i] > c[imax]) { imax = i; }
        if (c[i] < c[imin]) { imin = i; }
    }
    if (imax == imin) { return vec3<f32>(0.0); }
    let imid = 3u - imax - imin;
    var o = vec3<f32>(0.0);
    o[imid] = (c[imid] - c[imin]) * s / (c[imax] - c[imin]);
    o[imax] = s;
    return o;
}
fn blend(m: u32, cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    switch m {
        case 0u, 23u, 24u, 25u: { return cs; }
        case 19u: { return set_lum(set_sat(cs, sat(cb)), lum(cb)); }
        case 20u: { return set_lum(set_sat(cb, sat(cs)), lum(cb)); }
        case 21u: { return set_lum(cs, lum(cb)); }
        case 22u: { return set_lum(cb, lum(cs)); }
        default: { return vec3<f32>(separable(m, cb.x, cs.x), separable(m, cb.y, cs.y), separable(m, cb.z, cs.z)); }
    }
}

// ---- wet-pigment KmMixer::mix2 (tables: basis 0..48, sensor 48..96, correction 96..105)
const BANDS: u32 = 16u;
const R_MIN: f32 = 1e-3;
const TINT: f32 = 3.3;
fn km_refl(c: vec3<f32>, i: u32) -> f32 {
    return clamp(c.x * km(i) + c.y * km(BANDS + i) + c.z * km(2u * BANDS + i), R_MIN, 1.0);
}
fn km_mix2(a: vec3<f32>, b: vec3<f32>, t0: f32) -> vec3<f32> {
    let t = clamp(t0, 0.0, 1.0);
    if (t <= 0.0) { return a; }
    if (t >= 1.0) { return b; }
    let sa = exp(TINT * lum(a)); let sb = exp(TINT * lum(b));
    let tb = t * sb / (t * sb + (1.0 - t) * sa);
    let total = (1.0 - tb) + tb;
    var raw = vec3<f32>(0.0);
    for (var i = 0u; i < BANDS; i++) {
        let ra = km_refl(a, i); let rb = km_refl(b, i);
        let q = (1.0 - tb) / total * (1.0 - ra) * (1.0 - ra) / (2.0 * ra)
              + tb / total * (1.0 - rb) * (1.0 - rb) / (2.0 * rb);
        let r = clamp(1.0 + q - sqrt(q * q + 2.0 * q), R_MIN, 1.0);
        raw += r * vec3<f32>(km(48u + i), km(48u + BANDS + i), km(48u + 2u * BANDS + i));
    }
    var o = vec3<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        o[k] = clamp(km(96u + 3u * k) * raw.x + km(96u + 3u * k + 1u) * raw.y + km(96u + 3u * k + 2u) * raw.z, 0.0, 1.0);
    }
    return o;
}

// composite_pixel: straight colours and alphas in, straight colour and alpha out.
fn composite(cb: vec3<f32>, ab: f32, cs: vec3<f32>, a: f32, mode: u32, pigment: bool) -> vec4<f32> {
    let ao = a + ab * (1.0 - a);
    if (ao <= 0.0) { return vec4<f32>(0.0); }
    let pig = pigment && mode == 23u;
    var mixed = blend(mode, cb, cs);
    if (pig) { mixed = cs; }  // KmMixer::mix2(cb, cs, 1) = cs
    var co = clamp(((1.0 - a) * ab * cb + (1.0 - ab) * a * cs + a * ab * mixed) / ao, vec3<f32>(0.0), vec3<f32>(1.0));
    if (pig && ab > 0.0) { co = km_mix2(cb, cs, a / ao); }
    return vec4<f32>(co, ao);
}

@compute @workgroup_size(8, 8)
fn k_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let c = id.y * cfg.w + id.x;
    var s = 1.0;
    if ((cfg.flags & F_PAPER) != 0u) { s = doc_plane(0u, c); }
    ACC[c] = vec4<f32>(cfg.bg.xyz, s);
    AUX[c] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    if (c == 0u) { atomicStore(&HMAX, 0u); }
}

@compute @workgroup_size(8, 8)
fn k_group_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    GACC[id.y * cfg.w + id.x] = vec4<f32>(0.0);
}

@compute @workgroup_size(8, 8)
fn k_layer(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let x = id.x; let y = id.y; let c = y * cfg.w + x; let n = ncells();
    let f = cfg.flags;
    var acc = ACC[c];
    var aux = AUX[c];
    let raw_alpha = L[4u * c + 3u];
    var alpha = raw_alpha;
    if ((f & F_MASK) != 0u) { alpha *= MASK[c] / 255.0; }
    if ((f & F_CLIPPED) != 0u) { alpha *= aux.x; }
    if ((f & F_BASE) != 0u) { aux.x = alpha; }
    let h0 = height(x, y);
    if ((f & F_SHADE) != 0u) {
        // render.rs impasto_shade: light from the upper left
        let dx = height(x + 1u, y) - height(below(x), y);
        let dy = height(x, y + 1u) - height(x, below(y));
        acc.w *= clamp(1.0 - cfg.impasto * 0.12 * (dx + dy), 0.6, 1.4);
    }
    if ((f & F_HEIGHT) != 0u) { aux.y += h0; }
    if ((f & F_METAL) != 0u) { aux.z = 1.0 - (1.0 - aux.z) * (1.0 - raw_alpha * cfg.opacity); }
    if ((f & F_GLOSS) != 0u && h0 > 0.0) {
        // render.rs gloss_highlight: Blinn–Phong, light from the upper left
        let l = normalize(vec3<f32>(-1.0, -1.0, 1.5));
        let hv = normalize(l + vec3<f32>(0.0, 0.0, 1.0));
        let k = cfg.gloss_k;
        let dx = k * (height(x + 1u, y) - height(below(x), y));
        let dy = k * (height(x, y + 1u) - height(x, below(y)));
        let nl = sqrt(dx * dx + dy * dy + 1.0);
        let ndoth = max((-dx * hv.x - dy * hv.y + hv.z) / nl, 0.0);
        let wet = L[5u * n + c];
        aux.w += cfg.gloss * (0.4 + 0.6 * wet) * pow(ndoth, 40.0);
    }
    if ((f & F_COMPOSITE) != 0u) {
        let a = alpha * cfg.opacity;
        if (a > 0.0) {
            let cs = vec3<f32>(L[4u * c], L[4u * c + 1u], L[4u * c + 2u]);
            let pig = (f & F_PIGMENT) != 0u;
            if ((f & F_GROUP) != 0u) {
                let g = GACC[c];
                GACC[c] = composite(g.xyz, g.w, cs, a, cfg.mode, pig);
            } else {
                acc = vec4<f32>(composite(acc.xyz, 1.0, cs, a, cfg.mode, pig).xyz, acc.w);
            }
        }
    }
    ACC[c] = acc;
    AUX[c] = aux;
}

@compute @workgroup_size(8, 8)
fn k_group_out(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let c = id.y * cfg.w + id.x;
    let g = GACC[c];
    let a = g.w * cfg.opacity;
    if (a > 0.0) {
        let acc = ACC[c];
        ACC[c] = vec4<f32>(composite(acc.xyz, 1.0, g.xyz, a, cfg.mode, (cfg.flags & F_PIGMENT) != 0u).xyz, acc.w);
    }
}

@compute @workgroup_size(8, 8)
fn k_hmax(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    // non-negative floats order like their bits
    atomicMax(&HMAX, bitcast<u32>(max(AUX[id.y * cfg.w + id.x].y, 0.0)));
}

// shadow.rs soft_shadows: march toward the light, keep the strongest softened occlusion.
@compute @workgroup_size(8, 8)
fn k_shadow(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let c = id.y * cfg.w + id.x;
    let tan = cfg.shadow_tan;
    let max_h = bitcast<f32>(atomicLoad(&HMAX));
    let reach = u32(clamp(ceil(max_h / tan), 1.0, 64.0));
    let d = 0.70710678;
    let h0 = AUX[c].y;
    var occ = 0.0;
    for (var t = 1u; t <= reach; t++) {
        let tf = f32(t);
        let sx = f32(id.x) - d * tf; let sy = f32(id.y) - d * tf;
        if (sx < 0.0 || sy < 0.0) { break; }
        let s = AUX[u32(sy) * cfg.w + u32(sx)].y;
        occ = max(occ, clamp((s - h0 - tf * tan) / 0.3, 0.0, 1.0));
        if (occ >= 1.0) { break; }
    }
    ACC[c].w *= 1.0 - cfg.shadow_k * occ;
}

fn env(u: f32, v: f32) -> f32 {
    if (cfg.env_w > 0u && cfg.env_h > 0u) {
        let w = f32(cfg.env_w); let h = f32(cfg.env_h);
        let x = min(u32(u - w * floor(u / w)), cfg.env_w - 1u);
        let y = min(u32(v - h * floor(v / h)), cfg.env_h - 1u);
        return DOC[4u * ncells() + 128u + y * cfg.env_w + x];
    }
    let s = sin(0.013 * u + 0.021 * v) * 0.5 + sin(0.031 * v - 0.007 * u) * 0.3;
    return clamp(0.5 + 0.5 * s, 0.0, 1.0);
}

// metallic.rs MetallicSettings::shade
@compute @workgroup_size(8, 8)
fn k_metal(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let x = id.x; let y = id.y; let c = y * cfg.w + x;
    let m = AUX[c].z;
    if (m <= 0.0) { return; }
    let dx = hsum(x + 1u, y) - hsum(below(x), y);
    let dy = hsum(x, y + 1u) - hsum(x, below(y));
    let e = env((f32(x) + 40.0 * dx) / cfg.metal_zoom, (f32(y) + 40.0 * dy) / cfg.metal_zoom);
    ACC[c].w *= 1.0 - m * cfg.metal_ref * 0.35;
    AUX[c].w += m * cfg.metal_ref * cfg.metal_str * 0.6 * e * e;
}

@compute @workgroup_size(8, 8)
fn k_out(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let c = id.y * cfg.w + id.x;
    let acc = ACC[c];
    var px = clamp(round((acc.xyz * acc.w + AUX[c].w) * 255.0), vec3<f32>(0.0), vec3<f32>(255.0));
    var a = 255.0;
    if ((cfg.flags & F_DECKLE) != 0u) { a = round(doc_plane(3u, c) * 255.0); }
    // the app's view overlays (ui.rs decorate), on 8-bit values
    if ((cfg.flags & F_GREY) != 0u) {
        px = vec3<f32>(round(0.299 * px.x + 0.587 * px.y + 0.114 * px.z));
    }
    if ((cfg.flags & F_STENCIL) != 0u) {
        let k = doc_plane(1u, c);
        px = floor(px * (1.0 - 0.35 * k) + vec3<f32>(150.0, 90.0, 200.0) * 0.35 * k);
    }
    if ((cfg.flags & F_SELECT) != 0u) {
        px = floor(px * (1.0 - 0.35 * (1.0 - doc_plane(2u, c) / 255.0)));
    }
    // premultiplied, as egui's Color32 is
    if (a < 255.0) { px = round(px * a / 255.0); }
    textureStore(OUT, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(px, a) / 255.0);
}
