//! fast velocity diffusion and full velocity diffusion .

use crate::grid::{PlaneF, PlaneH, PlaneI8, PlaneU8, Rect, Shared, ld};
use crate::params::DiffuseConsts;

/// Inputs shared by both diffusion kernels. Wiring per `look = K`, `factor = F`,
/// `mask = M`.
pub struct DiffuseIn<'a> {
    pub u: &'a PlaneH,
    pub v: &'a PlaneH,
    pub water: &'a PlaneH,
    pub mask: &'a PlaneU8,
    pub look: &'a PlaneU8,
    pub factor: &'a PlaneU8,
    /// Valid region for back-traces, `[xmin, xmax] × [ymin, ymax]`.
    pub bounds: [f32; 4],
}

/// Extra inputs of the complete kernel: `p = MidS`, `q = DensS`, paper height `h`.
pub struct ForceIn<'a> {
    pub p: &'a PlaneF,
    pub q: &'a PlaneF,
    pub h: &'a PlaneI8,
}

#[inline]
fn avg5(p: &PlaneH, c: usize, w: usize, center: f32) -> f32 {
    (center + ld(p, c - 1) + ld(p, c + 1) + ld(p, c - w) + ld(p, c + w)) * 0.2
}

/// scale (or stop) the flow by the lookahead map 1.5 cells ahead.
#[inline]
fn lookahead(look: &PlaneU8, c: usize, w: usize, u: f32, v: f32) -> (f32, f32) {
    let m2 = u * u + v * v;
    if m2 <= 0.0 {
        return (u, v);
    }
    let s = 1.5 / m2.sqrt();
    let k = c as i64 + (s * u) as i64 + (s * v) as i64 * w as i64;
    if k > 0 && (k as usize) < look.data.len() {
        let l = look.data[k as usize];
        if l == 0 {
            return (0.0, 0.0);
        }
        let f = (l as f32 / 255.0).min(1.0);
        return (u * f, v * f);
    }
    (u, v)
}

/// zero a component whose back-trace leaves the valid region.
#[inline]
fn guard(x: usize, y: usize, u: f32, v: f32, b: [f32; 4]) -> (f32, f32) {
    let px = x as f32 - u;
    let py = y as f32 - v;
    let u = if px >= b[0] && px <= b[1] { u } else { 0.0 };
    let v = if py >= b[2] && py <= b[3] { v } else { 0.0 };
    (u, v)
}

#[inline]
fn factor(f: &PlaneU8, c: usize) -> f32 {
    (f.data[c] as f32 * 4.0 / 255.0).min(1.0)
}

pub fn diffuse_fast(
    rect: Rect,
    k: &DiffuseConsts,
    i: &DiffuseIn<'_>,
    u_out: &mut PlaneH,
    v_out: &mut PlaneH,
) {
    u_out.copy_from(i.u);
    v_out.copy_from(i.v);
    let w = i.u.w;
    // SAFETY: each row task writes only its own row's cells.
    let (uo, vo) = unsafe { (Shared::new(u_out), Shared::new(v_out)) };
    rect.par_rows(|y| {
        for x in rect.x0..=rect.x1 {
            let c = i.u.idx(x, y);
            if i.mask.data[c] == 0 {
                continue;
            }
            let wat = ld(i.water, c);
            let (mut u, mut v) = (ld(i.u, c), ld(i.v, c));
            if wat > k.wthr {
                u = avg5(i.u, c, w, u);
                v = avg5(i.v, c, w, v);
            }
            (u, v) = lookahead(i.look, c, w, u, v);
            u = u.clamp(-k.vmax, k.vmax);
            v = v.clamp(-k.vmax, k.vmax);
            let f = factor(i.factor, c);
            u = (u + k.gx * wat) * f;
            v = (v + k.gy * wat) * f;
            (u, v) = guard(x, y, u, v, i.bounds);
            if wat <= 0.0 {
                u *= 0.5;
                v *= 0.5;
            }
            uo.st(c, u);
            vo.st(c, v);
        }
    });
}

/// `spawn_drop` is called at most once per call when a drop should be created ;
/// pass `None` to disable drops. `seed` supplies the per-step hash constants.
pub fn diffuse_complete(
    rect: Rect,
    k: &DiffuseConsts,
    i: &DiffuseIn<'_>,
    f: &ForceIn<'_>,
    mut spawn_drop: Option<&mut dyn FnMut(usize, usize)>,
    seed: [u32; 4],
    u_out: &mut PlaneH,
    v_out: &mut PlaneH,
) {
    u_out.copy_from(i.u);
    v_out.copy_from(i.v);
    let w = i.u.w;
    let drops_on = spawn_drop.is_some();
    // first spawn point in row-major order (deterministic regardless of thread timing)
    let first_drop = std::sync::Mutex::new(None::<(usize, usize)>);
    // SAFETY: each row task writes only its own row's cells.
    let (uo, vo) = unsafe { (Shared::new(u_out), Shared::new(v_out)) };
    rect.par_rows(|y| {
        let mut row_drop = None;
        for x in rect.x0..=rect.x1 {
            let c = i.u.idx(x, y);
            if i.mask.data[c] == 0 {
                continue;
            }
            let wat = ld(i.water, c);
            // 1. thickness-pressure or paper-driven force
            let (fx, fy) = if f.p.data[c] + f.q.data[c] >= k.pthr {
                (
                    (f.p.data[c - 1] - f.p.data[c + 1]) * k.kp,
                    (f.p.data[c - w] - f.p.data[c + w]) * k.kp,
                )
            } else {
                let mut s = (2.0 * wat).min(1.0);
                if wat > 0.05 && wat < 0.15 {
                    s += s * k.ks * k.ks * (2.25 - 15.0 * wat);
                }
                let g = k.ks * s / 255.0;
                let h = |j: usize| f.h.data[j] as f32;
                (g * (h(c - 1) - h(c + 1)), g * (h(c - w) - h(c + w)))
            };
            // 2. water-gradient force
            let ax = ((ld(i.water, c - 1) - ld(i.water, c + 1)) * k.kw).clamp(-k.wclamp, k.wclamp);
            let ay = ((ld(i.water, c - w) - ld(i.water, c + w)) * k.kw).clamp(-k.wclamp, k.wclamp);
            // 3. + 4.
            let mut u = ld(i.u, c) + ax + fx;
            let mut v = ld(i.v, c) + ay + fy;
            if wat > k.wthr {
                u = avg5(i.u, c, w, u);
                v = avg5(i.v, c, w, v);
            }
            // 5. drops (collected per row, resolved after the pass)
            if drops_on && row_drop.is_none() && wat >= 20.0 {
                let hsh = (seed[0].wrapping_mul(x as u32))
                    .wrapping_add(seed[1].wrapping_mul(y as u32))
                    .wrapping_add(seed[2])
                    ^ seed[3];
                if hsh % 1000 > 998 {
                    row_drop = Some((x, y));
                }
            }
            // 6. lookahead, gravity, clamp, damping
            (u, v) = lookahead(i.look, c, w, u, v);
            u = (u + k.gx * wat).clamp(-k.vmax, k.vmax);
            v = (v + k.gy * wat).clamp(-k.vmax, k.vmax);
            let fac = factor(i.factor, c);
            u *= fac;
            v *= fac;
            // 7.
            (u, v) = guard(x, y, u, v, i.bounds);
            if wat <= 0.0 {
                u *= 0.5;
                v *= 0.5;
            }
            uo.st(c, u);
            vo.st(c, v);
        }
        if let Some(d) = row_drop {
            let mut fd = first_drop.lock().unwrap();
            if fd.is_none_or(|(fx, fy)| (d.1, d.0) < (fy, fx)) {
                *fd = Some(d);
            }
        }
    });
    if let (Some(cb), Some((x, y))) = (spawn_drop.as_mut(), first_drop.into_inner().unwrap()) {
        cb(x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::FluidParams;
    use half::f16;

    struct Fixture {
        u: PlaneH,
        v: PlaneH,
        water: PlaneH,
        open: PlaneU8,
    }

    fn fixture() -> Fixture {
        let n = 24;
        Fixture {
            u: PlaneH::new(n, n, f16::from_f32(1.0)),
            v: PlaneH::new(n, n, f16::ZERO),
            water: PlaneH::new(n, n, f16::from_f32(10.0)),
            open: PlaneU8::new(n, n, 255),
        }
    }

    fn rect() -> Rect {
        Rect {
            x0: 2,
            y0: 2,
            x1: 21,
            y1: 21,
        }
    }

    #[test]
    fn fast_clamps_to_vmax() {
        let fx = fixture();
        let k = FluidParams::default().diffuse_consts();
        let i = DiffuseIn {
            u: &fx.u,
            v: &fx.v,
            water: &fx.water,
            mask: &fx.open,
            look: &fx.open,
            factor: &fx.open,
            bounds: [0.0, 23.0, 0.0, 23.0],
        };
        let (mut uo, mut vo) = (fx.u.clone(), fx.v.clone());
        diffuse_fast(rect(), &k, &i, &mut uo, &mut vo);
        let c = uo.idx(10, 10);
        assert!((ld(&uo, c) - k.vmax).abs() < 1e-3);
    }

    #[test]
    fn blocked_lookahead_stops_flow() {
        let mut fx = fixture();
        fx.open.set(12, 10, 0); // obstacle 1.5 cells to the right of (10,10)
        let k = FluidParams::default().diffuse_consts();
        let i = DiffuseIn {
            u: &fx.u,
            v: &fx.v,
            water: &fx.water,
            mask: &fx.open,
            look: &fx.open,
            factor: &fx.open,
            bounds: [0.0, 23.0, 0.0, 23.0],
        };
        let (mut uo, mut vo) = (fx.u.clone(), fx.v.clone());
        diffuse_fast(rect(), &k, &i, &mut uo, &mut vo);
        assert_eq!(ld(&uo, uo.idx(11, 10)), 0.0);
    }

    #[test]
    fn complete_paper_slope_pushes_downhill() {
        let mut fx = fixture();
        fx.u.fill(f16::ZERO);
        fx.water.fill(f16::from_f32(0.1));
        let n = fx.u.w;
        let mut h = PlaneI8::new(n, n, 0);
        for y in 0..n {
            for x in 0..n {
                h.set(x, y, (x as i32 * 4 - 48) as i8); // height rises to the right
            }
        }
        let zeros = PlaneF::new(n, n, 0.0);
        let k = FluidParams::default().diffuse_consts();
        let i = DiffuseIn {
            u: &fx.u,
            v: &fx.v,
            water: &fx.water,
            mask: &fx.open,
            look: &fx.open,
            factor: &fx.open,
            bounds: [0.0, 23.0, 0.0, 23.0],
        };
        let fi = ForceIn {
            p: &zeros,
            q: &zeros,
            h: &h,
        };
        let (mut uo, mut vo) = (fx.u.clone(), fx.v.clone());
        diffuse_complete(rect(), &k, &i, &fi, None, [0; 4], &mut uo, &mut vo);
        assert!(ld(&uo, uo.idx(10, 10)) < 0.0, "flow toward lower paper");
    }
}
