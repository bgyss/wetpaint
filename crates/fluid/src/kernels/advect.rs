//! advection — semi-Lagrangian self-advection of velocity.

use crate::grid::{PlaneF, PlaneH, PlaneU8, Rect, Shared, bilinear_h, ld};

pub struct AdvectIo<'a> {
    pub u_in: &'a PlaneH,
    pub v_in: &'a PlaneH,
    pub mask: &'a PlaneU8,
    pub u_out: &'a mut PlaneH,
    pub v_out: &'a mut PlaneH,
    pub acc_a: &'a mut PlaneF,
    pub acc_b: &'a mut PlaneF,
}

/// Cells outside `rect` (or with `mask == 0`) copy through unchanged.
pub fn calculate_velocity(rect: Rect, io: AdvectIo<'_>) {
    io.u_out.copy_from(io.u_in);
    io.v_out.copy_from(io.v_in);
    // SAFETY: each row task writes only its own row's cells.
    let (uo, vo, aa, ab) = unsafe {
        (Shared::new(io.u_out), Shared::new(io.v_out), Shared::new(io.acc_a), Shared::new(io.acc_b))
    };
    let (u_in, v_in, mask) = (io.u_in, io.v_in, io.mask);
    rect.par_rows(|y| {
        for x in rect.x0..=rect.x1 {
            let c = u_in.idx(x, y);
            if mask.data[c] == 0 {
                continue;
            }
            let px = x as f32 - ld(u_in, c);
            let py = y as f32 - ld(v_in, c);
            uo.st(c, bilinear_h(u_in, px, py));
            vo.st(c, bilinear_h(v_in, px, py));
            aa.set(c, 0.0);
            ab.set(c, 0.0);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn uniform_field_is_unchanged() {
        let (w, h) = (16, 16);
        let u = PlaneH::new(w, h, f16::from_f32(0.25));
        let v = PlaneH::new(w, h, f16::from_f32(-0.125));
        let mask = PlaneU8::new(w, h, 255);
        let (mut uo, mut vo) = (PlaneH::new(w, h, f16::ZERO), PlaneH::new(w, h, f16::ZERO));
        let (mut a, mut b) = (PlaneF::new(w, h, 1.0), PlaneF::new(w, h, 1.0));
        let rect = Rect {
            x0: 2,
            y0: 2,
            x1: 13,
            y1: 13,
        };
        calculate_velocity(
            rect,
            AdvectIo {
                u_in: &u,
                v_in: &v,
                mask: &mask,
                u_out: &mut uo,
                v_out: &mut vo,
                acc_a: &mut a,
                acc_b: &mut b,
            },
        );
        let c = uo.idx(8, 8);
        assert!((ld(&uo, c) - 0.25).abs() < 1e-3);
        assert!((ld(&vo, c) + 0.125).abs() < 1e-3);
        assert_eq!(a.data[c], 0.0);
        assert_eq!(a.data[0], 1.0, "outside rect untouched");
    }
}
