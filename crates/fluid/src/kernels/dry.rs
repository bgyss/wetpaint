//! drying — evaporation, deposition, edge limiter, granulation, re-wetting
//!.

use crate::grid::{PlaneF, PlaneH, PlaneU8, Rect, Shared, ld};
use crate::lut::OpacityLut;
use crate::mix::{PigmentMixer, Rgb, mix};
use crate::params::DryConsts;

/// One pigment layer's planes (read side).
pub struct PigmentIn<'a> {
    pub rgb: [&'a PlaneH; 3],
    pub amount: &'a PlaneF,
    pub thickness: &'a PlaneF,
}

pub struct PigmentOut<'a> {
    pub rgb: [&'a mut PlaneH; 3],
    pub amount: &'a mut PlaneF,
    pub thickness: &'a mut PlaneF,
}

pub struct DryIn<'a> {
    pub water: &'a PlaneH,
    pub dens: PigmentIn<'a>,
    pub mid: PigmentIn<'a>,
    /// Absorbency `G`, wet mask `K`, granulation tile `T` (1024×1024, or `None`).
    pub absorb: &'a PlaneU8,
    pub wet: &'a PlaneU8,
    pub gran: Option<&'a PlaneU8>,
    pub lut: &'a OpacityLut,
    pub pigment: Option<&'a dyn PigmentMixer>,
}

pub struct DryOut<'a> {
    pub water: &'a mut PlaneH,
    pub dens: PigmentOut<'a>,
    pub mid: PigmentOut<'a>,
}

const AMOUNT_MAX: f32 = 100000.0;

fn rgb_at(p: &[&PlaneH; 3], c: usize) -> Rgb {
    [ld(p[0], c), ld(p[1], c), ld(p[2], c)]
}

fn copy_layer(i: &PigmentIn<'_>, o: &mut PigmentOut<'_>) {
    for k in 0..3 {
        o.rgb[k].copy_from(i.rgb[k]);
    }
    o.amount.copy_from(i.amount);
    o.thickness.copy_from(i.thickness);
}

pub fn dry_density(rect: Rect, k: &DryConsts, i: &DryIn<'_>, mut o: DryOut<'_>) {
    o.water.copy_from(i.water);
    copy_layer(&i.dens, &mut o.dens);
    copy_layer(&i.mid, &mut o.mid);
    let w = i.water.w;
    // SAFETY: each row task writes only its own row's cells.
    let (ow, oda, ods, oma, oms) = unsafe {
        (
            Shared::new(o.water),
            Shared::new(o.dens.amount),
            Shared::new(o.dens.thickness),
            Shared::new(o.mid.amount),
            Shared::new(o.mid.thickness),
        )
    };
    let [dr, dg, db] = o.dens.rgb;
    let [mr, mg, mb] = o.mid.rgb;
    // SAFETY: as above.
    let (odc, omc) = unsafe {
        (
            [Shared::new(dr), Shared::new(dg), Shared::new(db)],
            [Shared::new(mr), Shared::new(mg), Shared::new(mb)],
        )
    };
    rect.par_rows(|y| {
        for x in rect.x0..=rect.x1 {
            let c = i.water.idx(x, y);
            let wat = ld(i.water, c);
            if wat <= 0.0 {
                continue;
            }
            let (mut da, mut ds) = (i.dens.amount.data[c], i.dens.thickness.data[c]);
            let (mut ma, mut ms) = (i.mid.amount.data[c], i.mid.thickness.data[c]);
            let mut dc = rgb_at(&i.dens.rgb, c);
            let mut mc = rgb_at(&i.mid.rgb, c);

            // evaporation
            let g = if ma >= 1.0 {
                0.0
            } else {
                i.absorb.data[c] as f32 / 255.0
            };
            let mut w2 = k.rate * wat - k.offset;
            if g > 0.0 && w2 < 0.25 {
                w2 -= (1.0 - g).powi(3) * k.k_abs * k.offset;
            }
            w2 = if w2 >= 1e-4 { w2.min(100.0) } else { 0.0 };

            // deposition
            let lost = 1.0 - w2 / wat;
            let d = da * lost;
            let d_s = ds * lost;
            let la = i.lut.eval(ma);
            mc = if ma <= 0.01 || la <= 0.0 {
                dc
            } else {
                mix(mc, la, dc, i.lut.eval(d), i.pigment)
            };
            ma = (ma + d).min(AMOUNT_MAX);
            da = (da - d).max(0.0).min(AMOUNT_MAX);
            ms += d_s;
            ds = (ds - d_s).max(0.0);

            // edge darkening at the wet-region boundary
            if w2 > 0.0
                && w2 < 0.25
                && k.offset != 1000.0
                && g < 0.98
                && i.wet.data[c] != 0
                && x > 0
                && y > 0
            {
                let nb = [c - w, c - 1, c + 1, c + w];
                let km: [u8; 4] = nb.map(|j| i.wet.data[j]);
                if km.contains(&0) && ma > 100.0 {
                    let mn: [f32; 4] = nb.map(|j| i.mid.amount.data[j]);
                    if mn.iter().any(|&m| m < 250.0) {
                        let m = km.iter().map(|&v| v as f32).sum::<f32>() * 0.0013058825;
                        let cap = (8000.0 * w2 + 1000.0) * m * m + 250.0;
                        ma = (mn.iter().map(|&v| v.min(cap)).sum::<f32>() + 2.5 * ma.min(cap)) / 6.5;
                    }
                }
            }

            // granulation
            if let Some(t) = i.gran {
                if k.gran_enabled && w2 > 0.01 && w2 < 0.02 && k.gran_strength > 0.0 && ma > 1.0 {
                    let tv = t.data[(y & 1023) * t.w + (x & 1023)] as f32 / 255.0;
                    let t3 = tv * tv * tv;
                    let tt = t3 + k.gran_contrast * (tv - t3);
                    if ma < 4000.0 * tt {
                        ma += ((da + ma) * 0.01).min(1.0)
                            * (1.0 - 0.0005 * ma).max(0.0)
                            * k.gran_strength
                            * tt;
                    }
                }
            }

            // re-wetting
            if w2 > 0.001 && ma > 0.0 {
                let q = (w2 * k.k_rewet).min(1.0);
                let lift = ma * q;
                if i.lut.eval(lift) > 0.001 {
                    let ld_a = i.lut.eval(da);
                    dc = if ld_a <= 0.0 {
                        mc
                    } else {
                        mix(dc, ld_a, mc, i.lut.eval(lift), i.pigment)
                    };
                    da += lift;
                    ma = (ma - lift).max(0.0);
                    let ls = ms * q;
                    ds += ls;
                    ms -= ls;
                }
            }

            ow.st(c, w2);
            for ch in 0..3 {
                odc[ch].st(c, dc[ch]);
                omc[ch].st(c, mc[ch]);
            }
            oda.set(c, da);
            ods.set(c, ds);
            oma.set(c, ma);
            oms.set(c, ms);

        }
    });
}
