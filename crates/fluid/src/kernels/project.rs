//! projection — one-pass pressure projection.

use crate::grid::{PlaneH, PlaneU8, Rect, Shared, ld};

pub struct ProjectIo<'a> {
    pub u: &'a PlaneH,
    pub v: &'a PlaneH,
    pub mask: &'a PlaneU8,
    /// Gain maps: `k = a·b / (255·510)`.
    pub gain_a: &'a PlaneU8,
    pub gain_b: &'a PlaneU8,
    pub bounds: [f32; 4],
    /// Scratch planes: divergence `D` and relaxed pressure `Pp`.
    pub div: &'a mut PlaneH,
    pub pressure: &'a mut PlaneH,
    pub u_out: &'a mut PlaneH,
    pub v_out: &'a mut PlaneH,
}

pub fn velocity_spread(rect: Rect, io: ProjectIo<'_>) {
    let w = io.u.w;
    let (u_in, v_in, mask) = (io.u, io.v, io.mask);
    // A. divergence
    {
        // SAFETY: each row task writes only its own row's cells.
        let div = unsafe { Shared::new(io.div) };
        rect.par_rows(|y| {
            for x in rect.x0..=rect.x1 {
                let c = u_in.idx(x, y);
                if mask.data[c] == 0 {
                    continue;
                }
                let d = (ld(u_in, c - 1) + ld(v_in, c - w)) - (ld(u_in, c + 1) + ld(v_in, c + w));
                div.st(c, d);
            }
        });
    }
    // B. single relaxation pass, p = d
    {
        let d = &*io.div;
        // SAFETY: as above.
        let pr = unsafe { Shared::new(io.pressure) };
        rect.par_rows(|y| {
            for x in rect.x0..=rect.x1 {
                let c = u_in.idx(x, y);
                if mask.data[c] == 0 {
                    continue;
                }
                let n = ld(d, c - 1) + ld(d, c + 1) + ld(d, c - w) + ld(d, c + w);
                pr.st(c, 0.25 * (ld(d, c) + 0.25 * n));
            }
        });
    }
    // C. gradient subtraction
    io.u_out.copy_from(io.u);
    io.v_out.copy_from(io.v);
    let p = &*io.pressure;
    let (ga, gb, bounds) = (io.gain_a, io.gain_b, io.bounds);
    // SAFETY: as above.
    let (uo, vo) = unsafe { (Shared::new(io.u_out), Shared::new(io.v_out)) };
    rect.par_rows(|y| {
        for x in rect.x0..=rect.x1 {
            let c = u_in.idx(x, y);
            if mask.data[c] == 0 {
                continue;
            }
            let k = ga.data[c] as f32 * gb.data[c] as f32 / (255.0 * 510.0);
            let mut u = ld(u_in, c) + (ld(p, c - 1) - ld(p, c + 1)) * k;
            let mut v = ld(v_in, c) + (ld(p, c - w) - ld(p, c + w)) * k;
            let (px, py) = (x as f32 - u, y as f32 - v);
            if !(px >= bounds[0] && px <= bounds[1]) {
                u = 0.0;
            }
            if !(py >= bounds[2] && py <= bounds[3]) {
                v = 0.0;
            }
            uo.st(c, u);
            vo.st(c, v);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn reduces_divergence_of_a_source() {
        let n = 32;
        // Radial outflow from the centre: positive divergence.
        let mut u = PlaneH::new(n, n, f16::ZERO);
        let mut v = PlaneH::new(n, n, f16::ZERO);
        for y in 0..n {
            for x in 0..n {
                let (dx, dy) = (x as f32 - 16.0, y as f32 - 16.0);
                let r = (dx * dx + dy * dy).sqrt().max(1.0);
                if r < 6.0 {
                    u.set(x, y, f16::from_f32(0.2 * dx / r));
                    v.set(x, y, f16::from_f32(0.2 * dy / r));
                }
            }
        }
        let open = PlaneU8::new(n, n, 255);
        let mut div = PlaneH::new(n, n, f16::ZERO);
        let mut pr = div.clone();
        let (mut uo, mut vo) = (u.clone(), v.clone());
        let rect = Rect {
            x0: 2,
            y0: 2,
            x1: n - 3,
            y1: n - 3,
        };
        velocity_spread(
            rect,
            ProjectIo {
                u: &u,
                v: &v,
                mask: &open,
                gain_a: &open,
                gain_b: &open,
                bounds: [0.0, (n - 1) as f32, 0.0, (n - 1) as f32],
                div: &mut div,
                pressure: &mut pr,
                u_out: &mut uo,
                v_out: &mut vo,
            },
        );
        let total_div = |u: &PlaneH, v: &PlaneH| -> f32 {
            rect.cells()
                .map(|(x, y)| {
                    let c = u.idx(x, y);
                    ((ld(u, c + 1) - ld(u, c - 1)) + (ld(v, c + n) - ld(v, c - n))).abs()
                })
                .sum()
        };
        assert!(total_div(&uo, &vo) < total_div(&u, &v));
    }
}
