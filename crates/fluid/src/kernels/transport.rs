//! transport — mass-conserving transport of water and the suspended (Dens)
//! layer.

use crate::grid::{
    PlaneF, PlaneH, PlaneI8, PlaneU8, Rect, Shared, bilinear_footprint, footprint_cells, ld,
};

pub struct TransportIn<'a> {
    pub u: &'a PlaneH,
    pub v: &'a PlaneH,
    /// `M1` (activity) and `M2` (gates the f32 amounts).
    pub m1: &'a PlaneU8,
    pub m2: &'a PlaneU8,
    pub h: &'a PlaneI8,
    pub water: &'a PlaneH,
    pub rgb: [&'a PlaneH; 3],
    pub amount: &'a PlaneF,    // DensA
    pub thickness: &'a PlaneF, // DensS
    pub s0: f32,
    pub kh: f32,
}

pub struct TransportOut<'a> {
    pub water: &'a mut PlaneH,
    pub rgb: [&'a mut PlaneH; 3],
    pub amount: &'a mut PlaneF,
    pub thickness: &'a mut PlaneF,
}

const AMOUNT_CAP: f32 = 35000.0;

#[inline]
fn step_len(i: &TransportIn<'_>, c: usize) -> f32 {
    if i.thickness.data[c] < 5.0 {
        i.s0 - (i.h.data[c] as f32 + 128.0) * i.kh / 255.0
    } else {
        1.5
    }
}

/// Pass 1: scatter outgoing bilinear weights into `acc_a`. `acc_b` is a
/// second accumulator; this implementation only writes `acc_a` and reads `acc_a + acc_b` (both are zeroed before this pass).
///
/// Parallel scatter: rows are split into bands of at least `BAND` rows. A source writes
/// only to rows within `REACH` of its own, so all even bands can scatter concurrently,
/// then all odd bands. The per-cell result equals the sequential scatter up to float
/// summation order.
pub fn transport_weights(rect: Rect, i: &TransportIn<'_>, acc_a: &mut PlaneF) {
    use rayon::prelude::*;
    const REACH: usize = 2;
    const BAND: usize = 2 * REACH + 2;
    let (w, h) = (i.u.w, i.u.h);
    let rows = rect.y1 + 1 - rect.y0;
    let bands = rows.div_ceil(BAND.max(rows / (4 * rayon::current_num_threads()).max(1)));
    let band_h = rows.div_ceil(bands.max(1));
    // SAFETY: within a phase, bands are at least BAND rows apart, so concurrent tasks
    // touch disjoint cells (each source writes within REACH rows of itself).
    let acc = unsafe { Shared::new(acc_a) };
    let scatter_band = |b: usize| {
        let y0 = rect.y0 + b * band_h;
        let y1 = (y0 + band_h).min(rect.y1 + 1);
        for y in y0..y1 {
            for x in rect.x0..=rect.x1 {
                let c = y * w + x;
                if i.m1.data[c] == 0 {
                    continue;
                }
                let s = step_len(i, c);
                let (px, py) = (x as f32 - s * ld(i.u, c), y as f32 - s * ld(i.v, c));
                let (base, wt) = bilinear_footprint(px, py, w, h);
                for (k, cell) in footprint_cells(base, w).into_iter().enumerate() {
                    acc.add(cell, wt[k]);
                }
            }
        }
    };
    let n_bands = rows.div_ceil(band_h.max(1));
    if band_h < BAND {
        (0..n_bands).for_each(scatter_band);
    } else {
        (0..n_bands).into_par_iter().filter(|b| b % 2 == 0).for_each(scatter_band);
        (0..n_bands).into_par_iter().filter(|b| b % 2 == 1).for_each(scatter_band);
    }
}

/// Pass 2: gather.
pub fn transport_gather(
    rect: Rect,
    i: &TransportIn<'_>,
    acc_a: &PlaneF,
    acc_b: &PlaneF,
    o: TransportOut<'_>,
) {
    let (w, h) = (i.u.w, i.u.h);
    o.water.copy_from(i.water);
    for k in 0..3 {
        o.rgb[k].copy_from(i.rgb[k]);
    }
    o.amount.copy_from(i.amount);
    o.thickness.copy_from(i.thickness);
    let total = |j: usize| acc_a.data[j] + acc_b.data[j];
    let clamp_pos =
        |px: f32, py: f32| (px.clamp(0.0, (w - 1) as f32), py.clamp(0.0, (h - 1) as f32));
    // Normalised gather weights for a back-traced position.
    let gather_weights = |px: f32, py: f32| {
        let (b, wt) = bilinear_footprint(px, py, w, h);
        let cells = footprint_cells(b, w);
        let mut nw = [0.0f32; 4];
        for k in 0..4 {
            nw[k] = wt[k] / total(cells[k]).max(1.0);
        }
        (cells, nw)
    };
    // SAFETY: each row task writes only its own row's cells.
    let (ow, oa, ot) = unsafe { (Shared::new(o.water), Shared::new(o.amount), Shared::new(o.thickness)) };
    let [r0, r1, r2] = o.rgb;
    // SAFETY: as above.
    let orgb = unsafe { [Shared::new(r0), Shared::new(r1), Shared::new(r2)] };
    rect.par_rows(|y| {
        for x in rect.x0..=rect.x1 {
            let c = i.u.idx(x, y);
            if i.m1.data[c] == 0 {
                continue;
            }
            let s = step_len(i, c);
            let (du0, dv0) = (s * ld(i.u, c), s * ld(i.v, c));
            let q = ld(i.water, c);
            let slow = du0 * du0 + dv0 * dv0 < 0.02;
            let (mut px, mut py) = (x as f32 - du0, y as f32 - dv0);
            if q < 0.002 && slow {
                let m = 10.0 - 4500.0 * q;
                (px, py) = clamp_pos(x as f32 - du0 * m, y as f32 - dv0 * m);
            }
            let (cells, nw) = gather_weights(px, py);
            let r = (1.0 - total(c)).max(0.0);
            let gather_f = |p: &PlaneF, cells: &[usize; 4], nw: &[f32; 4]| {
                (0..4).map(|k| nw[k] * p.data[cells[k]]).sum::<f32>() + r * p.data[c]
            };
            if i.m2.data[c] != 0 {
                oa.set(c, gather_f(i.amount, &cells, &nw).min(AMOUNT_CAP));
                ot.set(c, gather_f(i.thickness, &cells, &nw));
            }
            // f16 planes, with the faint-film re-trace
            let (hcells, hnw) = if q < 0.05 && i.amount.data[c] > 15.0 && slow {
                let m = 4.0 - 60.0 * q;
                let (rx, ry) = clamp_pos(x as f32 - du0 * m, y as f32 - dv0 * m);
                gather_weights(rx, ry)
            } else {
                (cells, nw)
            };
            let gather_h =
                |p: &PlaneH| (0..4).map(|k| hnw[k] * ld(p, hcells[k])).sum::<f32>() + r * ld(p, c);
            ow.st(c, gather_h(i.water));
            // Colour is transported amount-weighted (Σ wᵢ·Aᵢ·Cᵢ / Σ wᵢ·Aᵢ) rather than gathered
            // like an amount: a plain gather does not preserve colour (a cell nobody pulls
            // from would keep its colour *and* gather more).
            let wa: [f32; 4] = std::array::from_fn(|k| hnw[k] * i.amount.data[hcells[k]]);
            let wself = r * i.amount.data[c];
            let total_a = wa.iter().sum::<f32>() + wself;
            if total_a > 0.0 {
                for k in 0..3 {
                    let p = i.rgb[k];
                    let col = ((0..4).map(|j| wa[j] * ld(p, hcells[j])).sum::<f32>()
                        + wself * ld(p, c))
                        / total_a;
                    orgb[k].st(c, col);
                }
            }

        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn conserves_water_and_amount() {
        let n = 40;
        let mut water = PlaneH::new(n, n, f16::ZERO);
        let mut amount = PlaneF::new(n, n, 0.0);
        for y in 15..25 {
            for x in 15..25 {
                water.set(x, y, f16::from_f32(8.0));
                amount.set(x, y, 100.0);
            }
        }
        let u = PlaneH::new(n, n, f16::from_f32(0.3));
        let v = PlaneH::new(n, n, f16::from_f32(-0.2));
        let open = PlaneU8::new(n, n, 255);
        let h = PlaneI8::new(n, n, 0);
        let thick = PlaneF::new(n, n, 10.0);
        let rgb = PlaneH::new(n, n, f16::from_f32(128.0));
        let i = TransportIn {
            u: &u,
            v: &v,
            m1: &open,
            m2: &open,
            h: &h,
            water: &water,
            rgb: [&rgb, &rgb, &rgb],
            amount: &amount,
            thickness: &thick,
            s0: 1.5,
            kh: 0.5,
        };
        let rect = Rect {
            x0: 2,
            y0: 2,
            x1: n - 3,
            y1: n - 3,
        };
        let mut a = PlaneF::new(n, n, 0.0);
        let b = PlaneF::new(n, n, 0.0);
        transport_weights(rect, &i, &mut a);
        let (mut wo, mut r0, mut r1, mut r2) =
            (water.clone(), rgb.clone(), rgb.clone(), rgb.clone());
        let (mut ao, mut to) = (amount.clone(), thick.clone());
        transport_gather(
            rect,
            &i,
            &a,
            &b,
            TransportOut {
                water: &mut wo,
                rgb: [&mut r0, &mut r1, &mut r2],
                amount: &mut ao,
                thickness: &mut to,
            },
        );
        let sum_h = |p: &PlaneH| p.data.iter().map(|v| v.to_f32()).sum::<f32>();
        let sum_f = |p: &PlaneF| p.data.iter().sum::<f32>();
        assert!((sum_h(&wo) - sum_h(&water)).abs() / sum_h(&water) < 2e-3);
        assert!((sum_f(&ao) - sum_f(&amount)).abs() / sum_f(&amount) < 1e-4);
        // the blob actually moved (+x, -y velocity ⇒ content shifts right and up)
        assert!(ao.get(25, 17) > 0.0);
    }
}
