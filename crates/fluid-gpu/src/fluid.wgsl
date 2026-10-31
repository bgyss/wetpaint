// GPU port of the wet-fluid reference kernels.
// Mirrors engine/fluid/src/kernels/*; f16 planes are stored as f32 but quantised to f16
// on every write so results track the CPU reference.

struct Cfg {
    w: u32, h: u32, n: u32, vel: u32,
    wd: u32, mid: u32, gran_on: u32, _p0: u32,
    vmax: f32, wthr: f32, gx: f32, gy: f32,
    pthr: f32, kp: f32, ks: f32, kw: f32,
    wclamp: f32, rate: f32, offset: f32, k_abs: f32,
    k_rewet: f32, gran_strength: f32, gran_contrast: f32, s0: f32,
    kh: f32, bx0: f32, bx1: f32, by0: f32,
    by1: f32, _p1: f32, _p2: f32, _p3: f32,
    // drop spawning in full velocity diffusion (that pass only) and the drip settings
    seed0: u32, seed1: u32, seed2: u32, seed3: u32,
    drops_on: u32, drip_on: u32, _p4: u32, _p5: u32,
    d_width0: f32, d_speed: f32, d_taper: f32, d_minw: f32,
    d_dx: f32, d_dy: f32, trail_water: f32, d_pick: f32,
    trail_share: f32, _p6: f32, _p7: f32, _p8: f32,
    // the canvas rectangle k_pack / k_unpack move
    rx0: u32, ry0: u32, rw: u32, rh: u32,
};

@group(0) @binding(0) var<uniform> cfg: Cfg;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;   // channel planes
@group(0) @binding(2) var<storage, read_write> M: array<f32>;   // maps
@group(0) @binding(3) var<storage, read> G: array<f32>;         // granulation tile 1024²
@group(0) @binding(4) var<storage, read> LUT: array<f32>;       // opacity table, 3001
@group(0) @binding(5) var<storage, read_write> R: array<atomic<i32>, 16>; // rect; R[10]: drop spawn cell
// The running drip (drip.rs; one at a time): active, x, y, width, taper, min width, load,
// r, g, b, density.
@group(0) @binding(6) var<storage, read_write> D: array<f32, 16>;
// The layer rendered for display (k_render): straight RGBA per cell, then relief heights,
// then wetness (gloss).
@group(0) @binding(7) var<storage, read_write> O: array<f32>;

// Plane bases in S (double-buffered planes occupy base, base+1).
const P_U: u32 = 0u;  const P_V: u32 = 2u;  const P_W: u32 = 4u;
const P_DR: u32 = 6u; const P_DG: u32 = 8u; const P_DB: u32 = 10u;
const P_DA: u32 = 12u; const P_DS: u32 = 14u;
const P_MR: u32 = 16u; const P_MG: u32 = 18u; const P_MB: u32 = 20u;
const P_MA: u32 = 22u; const P_MS: u32 = 24u;
const P_US: u32 = 26u; const P_VS: u32 = 27u; const P_ACC: u32 = 28u;
const P_DIV: u32 = 30u; const P_PR: u32 = 31u;
// Maps in M.
const M_ACT: u32 = 0u; const M_OPEN: u32 = 1u; const M_FAC: u32 = 2u;
const M_ABS: u32 = 3u; const M_H: u32 = 4u; const M_ST: u32 = 5u;
const MARGIN: i32 = 2;
const BIG: i32 = 0x3fffffff;

fn ncells() -> u32 { return cfg.w * cfg.h; }
fn ld(p: u32, i: u32) -> f32 { return S[p * ncells() + i]; }
fn stf(p: u32, i: u32, v: f32) { S[p * ncells() + i] = v; }
// Store to an f16 plane: round to f16 like the CPU reference.
fn sth(p: u32, i: u32, v: f32) { S[p * ncells() + i] = unpack2x16float(pack2x16float(vec2<f32>(v, 0.0))).x; }
fn mp(m: u32, i: u32) -> f32 { return M[m * ncells() + i]; }
fn cur(base: u32, par: u32) -> u32 { return base + par; }
fn nxt(base: u32, par: u32) -> u32 { return base + 1u - par; }

fn rect_valid() -> bool { return atomicLoad(&R[8]) != 0; }
fn in_rect(x: u32, y: u32) -> bool {
    if (!rect_valid()) { return false; }
    let xi = i32(x); let yi = i32(y);
    return xi >= atomicLoad(&R[4]) && xi <= atomicLoad(&R[6]) && yi >= atomicLoad(&R[5]) && yi <= atomicLoad(&R[7]);
}

struct Foot { base: u32, w: vec4<f32> };
// Bilinear footprint with the same clamp as grid::bilinear_footprint.
fn footprint(px0: f32, py0: f32) -> Foot {
    let px = clamp(px0, 0.0, f32(cfg.w - 2u) + 0.999);
    let py = clamp(py0, 0.0, f32(cfg.h - 2u) + 0.999);
    let ix = u32(px); let iy = u32(py);
    let fx = px - f32(ix); let fy = py - f32(iy);
    return Foot(iy * cfg.w + ix, vec4<f32>((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy));
}
fn foot_cell(f: Foot, k: u32) -> u32 {
    switch k { case 0u: { return f.base; } case 1u: { return f.base + 1u; }
               case 2u: { return f.base + cfg.w; } default: { return f.base + cfg.w + 1u; } }
}
fn bilinear(p: u32, px: f32, py: f32) -> f32 {
    let f = footprint(px, py);
    var s = 0.0;
    for (var k = 0u; k < 4u; k++) { s += f.w[k] * ld(p, foot_cell(f, k)); }
    return s;
}

fn lut(x: f32) -> f32 {
    if (!(x > 0.01)) { return 0.0; }
    if (x >= 3000.0) { return 1.0; }
    let i = u32(x); let t = x - f32(i); let a = LUT[i];
    return a + t * (LUT[i + 1u] - a);
}
fn mixc(c1: vec3<f32>, a1: f32, c2: vec3<f32>, a2: f32) -> vec3<f32> {
    let a1p = a1 * (1.0 - a2);
    let t = a1p + a2;
    if (t <= 0.0) { return c2; }
    return vec3<f32>((c1.x * a1p + c2.x * a2) / t, (c1.y * a1p + c2.y * a2) / t, (c1.z * a1p + c2.z * a2) / t);
}

// ---------------------------------------------------------------- stage 0: bounding box
@compute @workgroup_size(8, 8)
fn k_bbox(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let c = id.y * cfg.w + id.x;
    let live = ld(cur(P_W, cfg.wd), c) > 0.0 || ld(cur(P_DA, cfg.wd), c) > 0.0
        || ld(cur(P_U, cfg.vel), c) != 0.0 || ld(cur(P_V, cfg.vel), c) != 0.0;
    if (live) {
        atomicMin(&R[0], i32(id.x)); atomicMin(&R[1], i32(id.y));
        atomicMax(&R[2], i32(id.x)); atomicMax(&R[3], i32(id.y));
    }
}

@compute @workgroup_size(1)
fn k_bbox_final() {
    let x0 = atomicLoad(&R[0]);
    if (x0 == BIG) { atomicStore(&R[8], 0); return; }
    atomicStore(&R[4], max(x0 - MARGIN, MARGIN));
    atomicStore(&R[5], max(atomicLoad(&R[1]) - MARGIN, MARGIN));
    atomicStore(&R[6], min(atomicLoad(&R[2]) + MARGIN, i32(cfg.w) - 1 - MARGIN));
    atomicStore(&R[7], min(atomicLoad(&R[3]) + MARGIN, i32(cfg.h) - 1 - MARGIN));
    atomicStore(&R[8], 1);
}

// ---------------------------------------------------------------- drying
@compute @workgroup_size(8, 8)
fn k_dry(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let x = id.x; let y = id.y; let c = y * cfg.w + x; let w = cfg.w;
    let wd = cfg.wd; let md = cfg.mid;
    let wat = ld(cur(P_W, wd), c);
    var dc = vec3<f32>(ld(cur(P_DR, wd), c), ld(cur(P_DG, wd), c), ld(cur(P_DB, wd), c));
    var mc = vec3<f32>(ld(cur(P_MR, md), c), ld(cur(P_MG, md), c), ld(cur(P_MB, md), c));
    var da = ld(cur(P_DA, wd), c); var ds = ld(cur(P_DS, wd), c);
    var ma = ld(cur(P_MA, md), c); var ms = ld(cur(P_MS, md), c);
    var w2 = wat;
    if (in_rect(x, y) && wat > 0.0) {
        let g = select(mp(M_ABS, c) / 255.0, 0.0, ma >= 1.0);
        w2 = cfg.rate * wat - cfg.offset;
        if (g > 0.0 && w2 < 0.25) { w2 -= pow(1.0 - g, 3.0) * cfg.k_abs * cfg.offset; }
        w2 = select(0.0, min(w2, 100.0), w2 >= 1e-4);

        let lost = 1.0 - w2 / wat;
        let d = da * lost; let d_s = ds * lost;
        let la = lut(ma);
        if (ma <= 0.01 || la <= 0.0) { mc = dc; } else { mc = mixc(mc, la, dc, lut(d)); }
        ma = min(ma + d, 100000.0);
        da = min(max(da - d, 0.0), 100000.0);
        ms += d_s;
        ds = max(ds - d_s, 0.0);

        if (w2 > 0.0 && w2 < 0.25 && cfg.offset != 1000.0 && g < 0.98 && mp(M_OPEN, c) != 0.0) {
            let nb = array<u32, 4>(c - w, c - 1u, c + 1u, c + w);
            var has0 = false; var ksum = 0.0; var anylow = false;
            var mn = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
            for (var k = 0u; k < 4u; k++) {
                let kv = mp(M_OPEN, nb[k]);
                ksum += kv;
                if (kv == 0.0) { has0 = true; }
                mn[k] = ld(cur(P_MA, md), nb[k]);
                if (mn[k] < 250.0) { anylow = true; }
            }
            if (has0 && ma > 100.0 && anylow) {
                let m = ksum * 0.0013058825;
                let cap = (8000.0 * w2 + 1000.0) * m * m + 250.0;
                ma = (min(mn[0], cap) + min(mn[1], cap) + min(mn[2], cap) + min(mn[3], cap) + 2.5 * min(ma, cap)) / 6.5;
            }
        }

        if (cfg.gran_on != 0u && w2 > 0.01 && w2 < 0.02 && cfg.gran_strength > 0.0 && ma > 1.0) {
            let tv = G[(y & 1023u) * 1024u + (x & 1023u)] / 255.0;
            let t3 = tv * tv * tv;
            let tt = t3 + cfg.gran_contrast * (tv - t3);
            if (ma < 4000.0 * tt) {
                ma += min((da + ma) * 0.01, 1.0) * max(1.0 - 0.0005 * ma, 0.0) * cfg.gran_strength * tt;
            }
        }

        if (w2 > 0.001 && ma > 0.0) {
            let q = min(w2 * cfg.k_rewet, 1.0);
            let lift = ma * q;
            if (lut(lift) > 0.001) {
                let lda = lut(da);
                if (lda <= 0.0) { dc = mc; } else { dc = mixc(dc, lda, mc, lut(lift)); }
                da += lift;
                ma = max(ma - lift, 0.0);
                let ls = ms * q;
                ds += ls;
                ms -= ls;
            }
        }
        sth(nxt(P_W, wd), c, w2);
        sth(nxt(P_DR, wd), c, dc.x); sth(nxt(P_DG, wd), c, dc.y); sth(nxt(P_DB, wd), c, dc.z);
        sth(nxt(P_MR, md), c, mc.x); sth(nxt(P_MG, md), c, mc.y); sth(nxt(P_MB, md), c, mc.z);
    } else {
        stf(nxt(P_W, wd), c, wat);
        stf(nxt(P_DR, wd), c, dc.x); stf(nxt(P_DG, wd), c, dc.y); stf(nxt(P_DB, wd), c, dc.z);
        stf(nxt(P_MR, md), c, mc.x); stf(nxt(P_MG, md), c, mc.y); stf(nxt(P_MB, md), c, mc.z);
    }
    stf(nxt(P_DA, wd), c, da); stf(nxt(P_DS, wd), c, ds);
    stf(nxt(P_MA, md), c, ma); stf(nxt(P_MS, md), c, ms);
}

// ---------------------------------------------------------------- diffusion
fn avg5(p: u32, c: u32, center: f32) -> f32 {
    return (center + ld(p, c - 1u) + ld(p, c + 1u) + ld(p, c - cfg.w) + ld(p, c + cfg.w)) * 0.2;
}
fn lookahead(c: u32, uv: vec2<f32>) -> vec2<f32> {
    let m2 = uv.x * uv.x + uv.y * uv.y;
    if (m2 <= 0.0) { return uv; }
    let s = 1.5 / sqrt(m2);
    let k = i32(c) + i32(s * uv.x) + i32(s * uv.y) * i32(cfg.w);
    if (k > 0 && u32(k) < ncells()) {
        let l = mp(M_OPEN, u32(k));
        if (l == 0.0) { return vec2<f32>(0.0, 0.0); }
        return uv * min(l / 255.0, 1.0);
    }
    return uv;
}
fn guard(x: u32, y: u32, uv: vec2<f32>) -> vec2<f32> {
    let px = f32(x) - uv.x; let py = f32(y) - uv.y;
    return vec2<f32>(select(0.0, uv.x, px >= cfg.bx0 && px <= cfg.bx1),
                     select(0.0, uv.y, py >= cfg.by0 && py <= cfg.by1));
}
fn damping(c: u32) -> f32 { return min(mp(M_FAC, c) * 4.0 / 255.0, 1.0); }

@compute @workgroup_size(8, 8)
fn k_diffuse_fast(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let x = id.x; let y = id.y; let c = y * cfg.w + x;
    let pu = cur(P_U, cfg.vel); let pv = cur(P_V, cfg.vel);
    var u = ld(pu, c); var v = ld(pv, c);
    if (!in_rect(x, y) || mp(M_ACT, c) == 0.0) { stf(P_US, c, u); stf(P_VS, c, v); return; }
    let wat = ld(cur(P_W, cfg.wd), c);
    if (wat > cfg.wthr) { u = avg5(pu, c, u); v = avg5(pv, c, v); }
    var uv = lookahead(c, vec2<f32>(u, v));
    uv = clamp(uv, vec2<f32>(-cfg.vmax), vec2<f32>(cfg.vmax));
    let f = damping(c);
    uv = vec2<f32>((uv.x + cfg.gx * wat) * f, (uv.y + cfg.gy * wat) * f);
    uv = guard(x, y, uv);
    if (wat <= 0.0) { uv *= 0.5; }
    sth(P_US, c, uv.x); sth(P_VS, c, uv.y);
}

@compute @workgroup_size(8, 8)
fn k_diffuse_complete(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let x = id.x; let y = id.y; let c = y * cfg.w + x; let w = cfg.w;
    let pu = cur(P_U, cfg.vel); let pv = cur(P_V, cfg.vel);
    if (!in_rect(x, y) || mp(M_ACT, c) == 0.0) { stf(P_US, c, ld(pu, c)); stf(P_VS, c, ld(pv, c)); return; }
    let pw = cur(P_W, cfg.wd);
    let wat = ld(pw, c);
    let pp = cur(P_MS, cfg.mid); let pq = cur(P_DS, cfg.wd);
    var fx = 0.0; var fy = 0.0;
    if (ld(pp, c) + ld(pq, c) >= cfg.pthr) {
        fx = (ld(pp, c - 1u) - ld(pp, c + 1u)) * cfg.kp;
        fy = (ld(pp, c - w) - ld(pp, c + w)) * cfg.kp;
    } else {
        var s = min(2.0 * wat, 1.0);
        if (wat > 0.05 && wat < 0.15) { s += s * cfg.ks * cfg.ks * (2.25 - 15.0 * wat); }
        let g = cfg.ks * s / 255.0;
        fx = g * (mp(M_H, c - 1u) - mp(M_H, c + 1u));
        fy = g * (mp(M_H, c - w) - mp(M_H, c + w));
    }
    let ax = clamp((ld(pw, c - 1u) - ld(pw, c + 1u)) * cfg.kw, -cfg.wclamp, cfg.wclamp);
    let ay = clamp((ld(pw, c - w) - ld(pw, c + w)) * cfg.kw, -cfg.wclamp, cfg.wclamp);
    var u = ld(pu, c) + ax + fx;
    var v = ld(pv, c) + ay + fy;
    if (wat > cfg.wthr) { u = avg5(pu, c, u); v = avg5(pv, c, v); }
    // drops: the first hashed wet cell in row-major order spawns one
    if (cfg.drops_on != 0u && wat >= 20.0) {
        let hsh = (cfg.seed0 * x + cfg.seed1 * y + cfg.seed2) ^ cfg.seed3;
        if (hsh % 1000u > 998u) { atomicMin(&R[10], i32(c)); }
    }
    var uv = lookahead(c, vec2<f32>(u, v));
    uv = vec2<f32>(clamp(uv.x + cfg.gx * wat, -cfg.vmax, cfg.vmax), clamp(uv.y + cfg.gy * wat, -cfg.vmax, cfg.vmax));
    uv *= damping(c);
    uv = guard(x, y, uv);
    if (wat <= 0.0) { uv *= 0.5; }
    sth(P_US, c, uv.x); sth(P_VS, c, uv.y);
}

// ---------------------------------------------------------------- advection
@compute @workgroup_size(8, 8)
fn k_advect(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let x = id.x; let y = id.y; let c = y * cfg.w + x;
    let ou = nxt(P_U, cfg.vel); let ov = nxt(P_V, cfg.vel);
    if (!in_rect(x, y) || mp(M_ACT, c) == 0.0) { stf(ou, c, ld(P_US, c)); stf(ov, c, ld(P_VS, c)); return; }
    let px = f32(x) - ld(P_US, c);
    let py = f32(y) - ld(P_VS, c);
    sth(ou, c, bilinear(P_US, px, py));
    sth(ov, c, bilinear(P_VS, px, py));
}

// ---------------------------------------------------------------- projection
@compute @workgroup_size(8, 8)
fn k_divergence(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let x = id.x; let y = id.y; let c = y * cfg.w + x; let w = cfg.w;
    if (!in_rect(x, y) || mp(M_ACT, c) == 0.0) { return; }
    let pu = cur(P_U, cfg.vel); let pv = cur(P_V, cfg.vel);
    sth(P_DIV, c, (ld(pu, c - 1u) + ld(pv, c - w)) - (ld(pu, c + 1u) + ld(pv, c + w)));
}

@compute @workgroup_size(8, 8)
fn k_relax(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let x = id.x; let y = id.y; let c = y * cfg.w + x; let w = cfg.w;
    if (!in_rect(x, y) || mp(M_ACT, c) == 0.0) { return; }
    let n = ld(P_DIV, c - 1u) + ld(P_DIV, c + 1u) + ld(P_DIV, c - w) + ld(P_DIV, c + w);
    sth(P_PR, c, 0.25 * (ld(P_DIV, c) + 0.25 * n));
}

@compute @workgroup_size(8, 8)
fn k_project(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let x = id.x; let y = id.y; let c = y * cfg.w + x; let w = cfg.w;
    let pu = cur(P_U, cfg.vel); let pv = cur(P_V, cfg.vel);
    let ou = nxt(P_U, cfg.vel); let ov = nxt(P_V, cfg.vel);
    if (!in_rect(x, y) || mp(M_ACT, c) == 0.0) { stf(ou, c, ld(pu, c)); stf(ov, c, ld(pv, c)); return; }
    let k = mp(M_FAC, c) * mp(M_OPEN, c) / (255.0 * 510.0);
    var u = ld(pu, c) + (ld(P_PR, c - 1u) - ld(P_PR, c + 1u)) * k;
    var v = ld(pv, c) + (ld(P_PR, c - w) - ld(P_PR, c + w)) * k;
    let px = f32(x) - u; let py = f32(y) - v;
    if (!(px >= cfg.bx0 && px <= cfg.bx1)) { u = 0.0; }
    if (!(py >= cfg.by0 && py <= cfg.by1)) { v = 0.0; }
    sth(ou, c, u); sth(ov, c, v);
}

// ---------------------------------------------------------------- transport
fn step_len(c: u32) -> f32 {
    if (ld(cur(P_DS, cfg.wd), c) < 5.0) { return cfg.s0 - (mp(M_H, c) + 128.0) * cfg.kh / 255.0; }
    return 1.5;
}

// Pass 1 as a gather: A[j] = Σ of the bilinear weights that cells near j put on j.
// Exact while back-trace displacements stay below 3 cells per axis.
@compute @workgroup_size(8, 8)
fn k_transport_weights(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let j = id.y * cfg.w + id.x;
    var a = 0.0;
    for (var dy = -3; dy <= 3; dy++) {
        for (var dx = -3; dx <= 3; dx++) {
            let sx = i32(id.x) + dx; let sy = i32(id.y) + dy;
            if (sx < 0 || sy < 0 || sx >= i32(cfg.w) || sy >= i32(cfg.h)) { continue; }
            let x = u32(sx); let y = u32(sy); let c = y * cfg.w + x;
            if (!in_rect(x, y) || mp(M_ACT, c) == 0.0) { continue; }
            let s = step_len(c);
            let f = footprint(f32(x) - s * ld(cur(P_U, cfg.vel), c), f32(y) - s * ld(cur(P_V, cfg.vel), c));
            for (var k = 0u; k < 4u; k++) { if (foot_cell(f, k) == j) { a += f.w[k]; } }
        }
    }
    stf(P_ACC, j, a);
}

fn gather_foot(px: f32, py: f32) -> Foot {
    var f = footprint(px, py);
    for (var k = 0u; k < 4u; k++) { f.w[k] = f.w[k] / max(ld(P_ACC, foot_cell(f, k)), 1.0); }
    return f;
}
fn clamp_pos(px: f32, py: f32) -> vec2<f32> {
    return vec2<f32>(clamp(px, 0.0, f32(cfg.w - 1u)), clamp(py, 0.0, f32(cfg.h - 1u)));
}

@compute @workgroup_size(8, 8)
fn k_transport_gather(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let x = id.x; let y = id.y; let c = y * cfg.w + x;
    let wd = cfg.wd;
    let pw = cur(P_W, wd); let pda = cur(P_DA, wd); let pds = cur(P_DS, wd);
    let rgb_in = array<u32, 3>(cur(P_DR, wd), cur(P_DG, wd), cur(P_DB, wd));
    let rgb_out = array<u32, 3>(nxt(P_DR, wd), nxt(P_DG, wd), nxt(P_DB, wd));
    // copy-through defaults
    stf(nxt(P_W, wd), c, ld(pw, c));
    stf(nxt(P_DA, wd), c, ld(pda, c));
    stf(nxt(P_DS, wd), c, ld(pds, c));
    for (var k = 0u; k < 3u; k++) { stf(rgb_out[k], c, ld(rgb_in[k], c)); }
    if (!in_rect(x, y) || mp(M_ACT, c) == 0.0) { return; }

    let s = step_len(c);
    let du0 = s * ld(cur(P_U, cfg.vel), c);
    let dv0 = s * ld(cur(P_V, cfg.vel), c);
    let q = ld(pw, c);
    let slow = du0 * du0 + dv0 * dv0 < 0.02;
    var p = vec2<f32>(f32(x) - du0, f32(y) - dv0);
    if (q < 0.002 && slow) {
        let m = 10.0 - 4500.0 * q;
        p = clamp_pos(f32(x) - du0 * m, f32(y) - dv0 * m);
    }
    let f = gather_foot(p.x, p.y);
    let r = max(1.0 - ld(P_ACC, c), 0.0);
    if (mp(M_OPEN, c) != 0.0) {
        // CPU sums Σ first then adds r·X; keep that order for closeness
        var sa2 = 0.0; var ss2 = 0.0;
        for (var k = 0u; k < 4u; k++) {
            let cc = foot_cell(f, k);
            sa2 += f.w[k] * ld(pda, cc); ss2 += f.w[k] * ld(pds, cc);
        }
        stf(nxt(P_DA, wd), c, min(sa2 + r * ld(pda, c), 35000.0));
        stf(nxt(P_DS, wd), c, ss2 + r * ld(pds, c));
    }
    var hf = f;
    if (q < 0.05 && ld(pda, c) > 15.0 && slow) {
        let m = 4.0 - 60.0 * q;
        let rp = clamp_pos(f32(x) - du0 * m, f32(y) - dv0 * m);
        hf = gather_foot(rp.x, rp.y);
    }
    var sw = 0.0;
    for (var k = 0u; k < 4u; k++) { sw += hf.w[k] * ld(pw, foot_cell(hf, k)); }
    sth(nxt(P_W, wd), c, sw + r * ld(pw, c));
    var wa = vec4<f32>(0.0);
    var tot = 0.0;
    for (var k = 0u; k < 4u; k++) { wa[k] = hf.w[k] * ld(pda, foot_cell(hf, k)); tot += wa[k]; }
    let wself = r * ld(pda, c);
    tot += wself;
    if (tot > 0.0) {
        for (var ch = 0u; ch < 3u; ch++) {
            var acc = 0.0;
            for (var k = 0u; k < 4u; k++) { acc += wa[k] * ld(rgb_in[ch], foot_cell(hf, k)); }
            sth(rgb_out[ch], c, (acc + wself * ld(rgb_in[ch], c)) / tot);
        }
    }
}

// ---------------------------------------------------------------- wet mask upkeep
@compute @workgroup_size(8, 8)
fn k_wet_mask(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let c = id.y * cfg.w + id.x;
    if (in_rect(id.x, id.y) && ld(cur(P_W, cfg.wd), c) <= 0.0) { M[M_OPEN * ncells() + c] = 0.0; }
}

// ---------------------------------------------------------------- drips (drip.rs)
// One workgroup walks the disc around the drip; per-thread partial sums are reduced in
// workgroup memory. The disc and the go flag are broadcast with workgroupUniformLoad so
// every barrier stays in uniform control flow.
var<workgroup> WI: array<i32, 6>;    // go, x0, y0, x1, y1, unused
var<workgroup> WF: array<f32, 4>;    // centre x, y, radius, per-cell pigment
var<workgroup> RS: array<vec4<f32>, 256>;
var<workgroup> RC: array<u32, 256>;

fn reduce(li: u32) {
    for (var s = 128u; s > 0u; s = s >> 1u) {
        workgroupBarrier();
        if (li < s) { RS[li] += RS[li + s]; RC[li] += RC[li + s]; }
    }
    workgroupBarrier();
}

// drip.rs `disc`: the cells within `r` of (x, y), bounds truncated toward zero
fn disc_bounds(x: f32, y: f32, r: f32) {
    WI[1] = max(i32(x - r), 0); WI[3] = min(i32(x + r), i32(cfg.w) - 1);
    WI[2] = max(i32(y - r), 0); WI[4] = min(i32(y + r), i32(cfg.h) - 1);
    WF[0] = x; WF[1] = y; WF[2] = r;
}

// Spawn: with no drip running, a drop at this step's spawn cell takes up the suspended
// pigment of the disc under it.
@compute @workgroup_size(256)
fn k_drip_spawn(@builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        WI[0] = 0;
        if (cfg.drip_on == 0u) {
            D[0] = 0.0;
        } else {
            let sp = atomicLoad(&R[10]);
            if (D[0] == 0.0 && sp != BIG) {
                WI[0] = 1;
                disc_bounds(f32(u32(sp) % cfg.w), f32(u32(sp) / cfg.w), 0.5 * cfg.d_width0);
            }
        }
    }
    let b = workgroupUniformLoad(&WI);
    if (b[0] == 0) { return; }
    let f = workgroupUniformLoad(&WF);
    let bw = u32(b[3] - b[1] + 1); let total = bw * u32(b[4] - b[2] + 1);
    let pa = cur(P_DA, cfg.wd);
    var acc = vec4<f32>(0.0); var cells = 0u;
    for (var k = li; k < total; k += 256u) {
        let cx = b[1] + i32(k % bw); let cy = b[2] + i32(k / bw);
        let ox = f32(cx) - f[0]; let oy = f32(cy) - f[1];
        if (ox * ox + oy * oy <= f[2] * f[2]) {
            cells += 1u;
            let c = u32(cy) * cfg.w + u32(cx);
            let a = ld(pa, c) * cfg.d_pick;
            if (a > 0.0) {
                acc += vec4<f32>(a, a * ld(cur(P_DR, cfg.wd), c), a * ld(cur(P_DG, cfg.wd), c), a * ld(cur(P_DB, cfg.wd), c));
                stf(pa, c, ld(pa, c) - a);
            }
        }
    }
    RS[li] = acc; RC[li] = cells;
    reduce(li);
    if (li == 0u) {
        let t = RS[0];
        var rgb = vec3<f32>(0.0);
        if (t.x > 0.0) { rgb = t.yzw / t.x; }
        D[0] = 1.0; D[1] = f[0]; D[2] = f[1]; D[3] = cfg.d_width0; D[4] = cfg.d_taper; D[5] = cfg.d_minw;
        D[6] = t.x; D[7] = rgb.x; D[8] = rgb.y; D[9] = rgb.z; D[10] = t.x / f32(max(RC[0], 1u));
    }
}

fn fresh(c: u32) -> bool { return mp(M_OPEN, c) == 0.0 && mp(M_ST, c) > 0.0; }

// Move: the drip runs downhill, narrowing, and wets the paper it newly reaches with water
// and a share of its pigment until the load runs out.
@compute @workgroup_size(256)
fn k_drip_move(@builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        WI[0] = 0;
        if (cfg.drip_on != 0u && D[0] != 0.0) {
            let x = D[1] + cfg.d_dx * cfg.d_speed;
            let y = D[2] + cfg.d_dy * cfg.d_speed;
            let wd = max(D[3] - D[4], D[5]);
            if (x < 0.0 || y < 0.0 || x >= f32(cfg.w) || y >= f32(cfg.h)) {
                D[0] = 0.0;
            } else {
                D[1] = x; D[2] = y; D[3] = wd;
                WI[0] = 1;
                disc_bounds(x, y, 0.5 * wd);
            }
        }
    }
    let b = workgroupUniformLoad(&WI);
    if (b[0] == 0) { return; }
    let f = workgroupUniformLoad(&WF);
    let bw = u32(b[3] - b[1] + 1); let total = bw * u32(b[4] - b[2] + 1);
    var n = 0u;
    for (var k = li; k < total; k += 256u) {
        let cx = b[1] + i32(k % bw); let cy = b[2] + i32(k / bw);
        let ox = f32(cx) - f[0]; let oy = f32(cy) - f[1];
        if (ox * ox + oy * oy <= f[2] * f[2] && fresh(u32(cy) * cfg.w + u32(cx))) { n += 1u; }
    }
    RS[li] = vec4<f32>(0.0); RC[li] = n;
    reduce(li);
    if (li == 0u) {
        let count = RC[0];
        var per = 0.0;
        if (count > 0u) { per = min(cfg.trail_share * D[10], D[6] / f32(count)); }
        WF[3] = per;
        D[6] = D[6] - per * f32(count);
    }
    let per = workgroupUniformLoad(&WF)[3];
    let pw = cur(P_W, cfg.wd); let pa = cur(P_DA, cfg.wd);
    let col = vec3<f32>(D[7], D[8], D[9]);
    for (var k = li; k < total; k += 256u) {
        let cx = b[1] + i32(k % bw); let cy = b[2] + i32(k / bw);
        let ox = f32(cx) - f[0]; let oy = f32(cy) - f[1];
        let c = u32(cy) * cfg.w + u32(cx);
        if (ox * ox + oy * oy <= f[2] * f[2] && fresh(c)) {
            // WetLayer::deposit
            let st = mp(M_ST, c);
            let water = max(cfg.trail_water - ld(pw, c), 0.0) * st;
            let amount = per * st;
            M[M_OPEN * ncells() + c] = 255.0;
            sth(pw, c, ld(pw, c) + water);
            if (amount > 0.0) {
                let old_a = ld(pa, c);
                let old = vec3<f32>(ld(cur(P_DR, cfg.wd), c), ld(cur(P_DG, cfg.wd), c), ld(cur(P_DB, cfg.wd), c));
                let nc = mixc(old, lut(old_a), col, lut(amount));
                sth(cur(P_DR, cfg.wd), c, nc.x); sth(cur(P_DG, cfg.wd), c, nc.y); sth(cur(P_DB, cfg.wd), c, nc.z);
                stf(pa, c, old_a + amount);
            }
        }
    }
}

// ---------------------------------------------------------------- display (render.rs)
// render_layer: the suspended pigment over the settled, coverage from the opacity LUT;
// and the paint relief height and wetness impasto_shade and gloss_highlight read.
@compute @workgroup_size(8, 8)
fn k_render(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.w || id.y >= cfg.h) { return; }
    let c = id.y * cfg.w + id.x; let n = ncells();
    let wd = cfg.wd; let md = cfg.mid;
    let lm = lut(ld(cur(P_MA, md), c)); let ldd = lut(ld(cur(P_DA, wd), c));
    let alpha = ldd + lm * (1.0 - ldd);
    var px = vec4<f32>(0.0);
    if (alpha > 0.0) {
        let mc = vec3<f32>(ld(cur(P_MR, md), c), ld(cur(P_MG, md), c), ld(cur(P_MB, md), c));
        let dc = vec3<f32>(ld(cur(P_DR, wd), c), ld(cur(P_DG, wd), c), ld(cur(P_DB, wd), c));
        px = vec4<f32>(mixc(mc, lm, dc, ldd) / 255.0, alpha);
    }
    O[4u * c] = px.x; O[4u * c + 1u] = px.y; O[4u * c + 2u] = px.z; O[4u * c + 3u] = px.w;
    O[4u * n + c] = min((ld(cur(P_MS, md), c) + ld(cur(P_DS, wd), c)) / 400.0, 4.0);
    O[5u * n + c] = clamp(ld(cur(P_W, wd), c) / 2.0, 0.0, 1.0);
}

// ---------------------------------------------------------------- region transfer
// A rectangle of the planes a brush stroke reads and writes (water, velocity, both pigment
// layers' colours, amounts and thicknesses: the current half of each) and the wet mask,
// packed plane by plane, so a stroke on the CPU moves only the cells under the brush.
@group(1) @binding(0) var<storage, read_write> X: array<f32>;
const REGION_PLANES: u32 = 13u;
fn region_slot(k: u32) -> u32 {
    switch k {
        case 0u: { return P_W + cfg.wd; }
        case 1u: { return P_U + cfg.vel; }
        case 2u: { return P_V + cfg.vel; }
        case 3u: { return P_DR + cfg.wd; }
        case 4u: { return P_DG + cfg.wd; }
        case 5u: { return P_DB + cfg.wd; }
        case 6u: { return P_MR + cfg.mid; }
        case 7u: { return P_MG + cfg.mid; }
        case 8u: { return P_MB + cfg.mid; }
        case 9u: { return P_DA + cfg.wd; }
        case 10u: { return P_DS + cfg.wd; }
        case 11u: { return P_MA + cfg.mid; }
        default: { return P_MS + cfg.mid; }
    }
}

@compute @workgroup_size(8, 8)
fn k_pack(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.rw || id.y >= cfg.rh) { return; }
    let c = (cfg.ry0 + id.y) * cfg.w + cfg.rx0 + id.x;
    let r = id.y * cfg.rw + id.x; let rn = cfg.rw * cfg.rh;
    for (var k = 0u; k < REGION_PLANES; k++) { X[k * rn + r] = ld(region_slot(k), c); }
    X[REGION_PLANES * rn + r] = mp(M_OPEN, c);
}

@compute @workgroup_size(8, 8)
fn k_unpack(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= cfg.rw || id.y >= cfg.rh) { return; }
    let c = (cfg.ry0 + id.y) * cfg.w + cfg.rx0 + id.x;
    let r = id.y * cfg.rw + id.x; let rn = cfg.rw * cfg.rh;
    for (var k = 0u; k < REGION_PLANES; k++) { stf(region_slot(k), c, X[k * rn + r]); }
    M[M_OPEN * ncells() + c] = X[REGION_PLANES * rn + r];
}
