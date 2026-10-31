//! A wet layer: channel storage and the per-step driver.

use half::f16;

use crate::grid::{Buffered, PlaneF, PlaneH, PlaneI8, PlaneU8, Rect, ld, st};
use crate::kernels::advect::{AdvectIo, calculate_velocity};
use crate::kernels::diffuse::{DiffuseIn, ForceIn, diffuse_complete, diffuse_fast};
use crate::kernels::dry::{DryIn, DryOut, PigmentIn, PigmentOut, dry_density};
use crate::kernels::project::{ProjectIo, velocity_spread};
use crate::kernels::transport::{TransportIn, TransportOut, transport_gather, transport_weights};
use crate::lut::OpacityLut;
use crate::mix::{PigmentMixer, Rgb, mix};
use crate::params::FluidParams;

/// A pigment layer (Dens = suspended, Mid = deposited): colour 0–255, amount, thickness.
pub struct Pigment {
    pub rgb: [Buffered<f16>; 3],
    pub amount: Buffered<f32>,
    pub thickness: Buffered<f32>,
}

impl Pigment {
    fn new(w: usize, h: usize) -> Self {
        let c = || Buffered::new(w, h, f16::from_f32(255.0));
        Self {
            rgb: [c(), c(), c()],
            amount: Buffered::new(w, h, 0.0),
            thickness: Buffered::new(w, h, 0.0),
        }
    }

    fn swap(&mut self) {
        self.rgb.iter_mut().for_each(Buffered::swap);
        self.amount.swap();
        self.thickness.swap();
    }
}

/// Per-layer u8/i8 maps. Defaults are neutral: everything open, no absorbency, flat paper, no granulation tile.
pub struct Maps {
    /// Activity mask `M`.
    pub active: PlaneU8,
    /// `K`: wetted-paper mask, used as lookahead/obstacle map (flow cannot enter dry
    /// paper), projection gain and wet-region boundary for edge darkening. Set by the
    /// brush, cleared where a cell has no water left.
    pub open: PlaneU8,
    /// `F`: damping/gain factor map.
    pub factor: PlaneU8,
    /// `G`: paper absorbency.
    pub absorb: PlaneU8,
    /// `H`: signed paper height.
    pub height: PlaneI8,
    /// `T`: 1024×1024 granulation tile.
    pub gran: Option<PlaneU8>,
    /// Stencil / selection mask (255 = paintable, 0 = protected). Painting and wetting are
    /// scaled by it, so protected cells stay dry and the flow cannot enter them.
    pub stencil: Option<PlaneU8>,
}

impl Maps {
    /// Stencil factor of a cell in 0..1.
    pub fn stencil_at(&self, c: usize) -> f32 {
        self.stencil.as_ref().map_or(1.0, |s| s.data[c] as f32 / 255.0)
    }
}

pub struct WetLayer {
    pub w: usize,
    pub h: usize,
    pub u: Buffered<f16>,
    pub v: Buffered<f16>,
    pub water: Buffered<f16>,
    pub dens: Pigment,
    pub mid: Pigment,
    pub maps: Maps,
    // single-buffered scratch (public so other backends can mirror the full state)
    pub us: PlaneH,
    pub vs: PlaneH,
    pub acc_a: PlaneF,
    pub acc_b: PlaneF,
    pub div: PlaneH,
    pub pressure: PlaneH,
    pub params: FluidParams,
    pub lut: OpacityLut,
    pub pigment: Option<Box<dyn PigmentMixer>>,
    pub step_index: u64,
    /// Cells where a drop was spawned during the last step.
    pub drops: Vec<(usize, usize)>,
    /// Active drips (`drip.rs`).
    pub drips: Vec<crate::drip::Drip>,
}

/// Margin kept between the active rectangle and the grid edge (stencils read ±1).
const MARGIN: usize = 2;

impl WetLayer {
    pub fn new(w: usize, h: usize) -> Self {
        assert!(w > 2 * MARGIN + 2 && h > 2 * MARGIN + 2);
        let hz = || PlaneH::new(w, h, f16::ZERO);
        Self {
            w,
            h,
            u: Buffered::new(w, h, f16::ZERO),
            v: Buffered::new(w, h, f16::ZERO),
            water: Buffered::new(w, h, f16::ZERO),
            dens: Pigment::new(w, h),
            mid: Pigment::new(w, h),
            maps: Maps {
                active: PlaneU8::new(w, h, 255),
                open: PlaneU8::new(w, h, 0),
                factor: PlaneU8::new(w, h, 255),
                absorb: PlaneU8::new(w, h, 0),
                height: PlaneI8::new(w, h, 0),
                gran: None,
                stencil: None,
            },
            us: hz(),
            vs: hz(),
            acc_a: PlaneF::new(w, h, 0.0),
            acc_b: PlaneF::new(w, h, 0.0),
            div: hz(),
            pressure: hz(),
            params: FluidParams::default(),
            lut: OpacityLut::default(),
            pigment: None,
            step_index: 0,
            drops: Vec::new(),
            drips: Vec::new(),
        }
    }

    fn bounds(&self) -> [f32; 4] {
        [0.0, (self.w - 1) as f32, 0.0, (self.h - 1) as f32]
    }

    /// Stage 0 (bounding box): uses the bounding
    /// box of cells with water, suspended pigment or motion, grown by `MARGIN`, clamped so
    /// stencils stay in the grid.
    pub fn active_rect(&self) -> Option<Rect> {
        let (wat, da, u, v) = (
            self.water.cur(),
            self.dens.amount.cur(),
            self.u.cur(),
            self.v.cur(),
        );
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for y in 0..self.h {
            for x in 0..self.w {
                let c = y * self.w + x;
                let live =
                    ld(wat, c) > 0.0 || da.data[c] > 0.0 || ld(u, c) != 0.0 || ld(v, c) != 0.0;
                if live {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        if x0 == usize::MAX {
            return None;
        }
        Some(Rect {
            x0: x0.saturating_sub(MARGIN).max(MARGIN),
            y0: y0.saturating_sub(MARGIN).max(MARGIN),
            x1: (x1 + MARGIN).min(self.w - 1 - MARGIN),
            y1: (y1 + MARGIN).min(self.h - 1 - MARGIN),
        })
    }

    /// One simulation step plus drip motion. Returns false when there was
    /// nothing to do.
    pub fn step(&mut self) -> bool {
        let busy = self.fluid_step();
        crate::drip::advance(self);
        busy || !self.drips.is_empty()
    }

    fn fluid_step(&mut self) -> bool {
        let n = self.step_index;
        self.step_index += 1;
        self.drops.clear();
        let Some(rect) = self.active_rect() else {
            return false;
        };
        let t = self.params.timing();
        let (n_dry, n_full) = (t.n_dry as u64, t.n_full as u64);

        // 1–2. drying
        if n % n_dry == 0 {
            self.stage_dry(rect);
        }
        // 3–4. diffusion into scratch velocity
        self.stage_diffuse(rect, n % n_full == 0, n);
        // 5. advection into U/V
        {
            let (u_out, v_out) = (self.u.next_mut(), self.v.next_mut());
            calculate_velocity(
                rect,
                AdvectIo {
                    u_in: &self.us,
                    v_in: &self.vs,
                    mask: &self.maps.active,
                    u_out,
                    v_out,
                    acc_a: &mut self.acc_a,
                    acc_b: &mut self.acc_b,
                },
            );
            self.u.swap();
            self.v.swap();
        }
        // 6. projection every third step
        if n % 3 == 0 && self.params.project_cycles > 0 {
            let bounds = self.bounds();
            let (u, uo) = self.u.split();
            let (v, vo) = self.v.split();
            velocity_spread(
                rect,
                ProjectIo {
                    u,
                    v,
                    mask: &self.maps.active,
                    gain_a: &self.maps.factor,
                    gain_b: &self.maps.open,
                    bounds,
                    div: &mut self.div,
                    pressure: &mut self.pressure,
                    u_out: uo,
                    v_out: vo,
                },
            );
            self.u.swap();
            self.v.swap();
        }
        // 7. transport
        self.stage_transport(rect);
        self.update_wet_mask(rect);
        true
    }

    /// Clear the wetted mask where a cell has dried out (see `Maps::open`).
    fn update_wet_mask(&mut self, rect: Rect) {
        let wat = self.water.cur();
        for (x, y) in rect.cells() {
            let c = wat.idx(x, y);
            if ld(wat, c) <= 0.0 {
                self.maps.open.data[c] = 0;
            }
        }
    }

    fn stage_dry(&mut self, rect: Rect) {
        let k = self.params.dry_consts();
        let (water, water_o) = self.water.split();
        let [dr, dg, db] = &mut self.dens.rgb;
        let [mr, mg, mb] = &mut self.mid.rgb;
        let ((dr, dro), (dg, dgo), (db, dbo)) = (dr.split(), dg.split(), db.split());
        let ((mr, mro), (mg, mgo), (mb, mbo)) = (mr.split(), mg.split(), mb.split());
        let ((da, dao), (ds, dso)) = (self.dens.amount.split(), self.dens.thickness.split());
        let ((ma, mao), (ms, mso)) = (self.mid.amount.split(), self.mid.thickness.split());
        dry_density(
            rect,
            &k,
            &DryIn {
                water,
                dens: PigmentIn {
                    rgb: [dr, dg, db],
                    amount: da,
                    thickness: ds,
                },
                mid: PigmentIn {
                    rgb: [mr, mg, mb],
                    amount: ma,
                    thickness: ms,
                },
                absorb: &self.maps.absorb,
                wet: &self.maps.open,
                gran: self.maps.gran.as_ref(),
                lut: &self.lut,
                pigment: self.pigment.as_deref(),
            },
            DryOut {
                water: water_o,
                dens: PigmentOut {
                    rgb: [dro, dgo, dbo],
                    amount: dao,
                    thickness: dso,
                },
                mid: PigmentOut {
                    rgb: [mro, mgo, mbo],
                    amount: mao,
                    thickness: mso,
                },
            },
        );
        self.water.swap();
        self.dens.swap();
        self.mid.swap();
    }

    fn stage_diffuse(&mut self, rect: Rect, complete: bool, n: u64) {
        let k = self.params.diffuse_consts();
        let i = DiffuseIn {
            u: self.u.cur(),
            v: self.v.cur(),
            water: self.water.cur(),
            mask: &self.maps.active,
            look: &self.maps.open,
            factor: &self.maps.factor,
            bounds: self.bounds(),
        };
        if complete {
            let f = ForceIn {
                p: self.mid.thickness.cur(),
                q: self.dens.thickness.cur(),
                h: &self.maps.height,
            };
            let drops_on = self.params.create_drops && n % 3 == 0 && self.params.tilted();
            let mut found: Option<(usize, usize)> = None;
            let mut cb = |x: usize, y: usize| found = Some((x, y));
            let seed = step_seed(n);
            diffuse_complete(
                rect,
                &k,
                &i,
                &f,
                if drops_on {
                    Some(&mut cb as &mut dyn FnMut(usize, usize))
                } else {
                    None
                },
                seed,
                &mut self.us,
                &mut self.vs,
            );
            self.drops.extend(found);
        } else {
            diffuse_fast(rect, &k, &i, &mut self.us, &mut self.vs);
        }
    }

    fn stage_transport(&mut self, rect: Rect) {
        let p = &self.params.placeholders;
        let (s0, kh) = (p.transport_s0, p.transport_kh);
        let (water, water_o) = self.water.split();
        let [dr, dg, db] = &mut self.dens.rgb;
        let ((dr, dro), (dg, dgo), (db, dbo)) = (dr.split(), dg.split(), db.split());
        let ((da, dao), (ds, dso)) = (self.dens.amount.split(), self.dens.thickness.split());
        let i = TransportIn {
            u: self.u.cur(),
            v: self.v.cur(),
            m1: &self.maps.active,
            m2: &self.maps.open,
            h: &self.maps.height,
            water,
            rgb: [dr, dg, db],
            amount: da,
            thickness: ds,
            s0,
            kh,
        };
        // advection only zeroes the accumulators inside `rect`, but the scatter
        // writes one cell beyond it; clear fully so no weight survives between steps.
        self.acc_a.fill(0.0);
        self.acc_b.fill(0.0);
        transport_weights(rect, &i, &mut self.acc_a);
        transport_gather(
            rect,
            &i,
            &self.acc_a,
            &self.acc_b,
            TransportOut {
                water: water_o,
                rgb: [dro, dgo, dbo],
                amount: dao,
                thickness: dso,
            },
        );
        self.water.swap();
        self.dens.rgb.iter_mut().for_each(Buffered::swap);
        self.dens.amount.swap();
        self.dens.thickness.swap();
    }

    /// Paint a round dab (a minimal brush for tests and demos):
    /// adds water and suspended pigment with a smooth falloff, mixing the colour into
    /// what is already suspended.
    pub fn dab(&mut self, cx: f32, cy: f32, radius: f32, water: f32, color: Rgb, amount: f32) {
        let r2 = radius * radius;
        let (x0, x1) = (
            (cx - radius).max(0.0) as usize,
            ((cx + radius) as usize).min(self.w - 1),
        );
        let (y0, y1) = (
            (cy - radius).max(0.0) as usize,
            ((cy + radius) as usize).min(self.h - 1),
        );
        for y in y0..=y1 {
            for x in x0..=x1 {
                let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                if d2 > r2 {
                    continue;
                }
                let fall = 1.0 - d2 / r2;
                self.deposit(y * self.w + x, water * fall, color, amount * fall);
            }
        }
    }

    /// Inject into one cell: opens the wet mask, adds `water`, and adds `amount` of
    /// suspended pigment of `color`, mixed into what is already suspended.
    pub fn deposit(&mut self, c: usize, water: f32, color: Rgb, amount: f32) {
        let k = self.maps.stencil_at(c);
        if k <= 0.0 {
            return;
        }
        let (water, amount) = (water * k, amount * k);
        self.maps.open.data[c] = 255;
        let wat = self.water.cur_mut();
        let nw = ld(wat, c) + water;
        st(wat, c, nw);
        if amount <= 0.0 {
            return;
        }
        let old_a = self.dens.amount.cur().data[c];
        let old: Rgb = [0, 1, 2].map(|k| ld(self.dens.rgb[k].cur(), c));
        let new = mix(
            old,
            self.lut.eval(old_a),
            color,
            self.lut.eval(amount),
            self.pigment.as_deref(),
        );
        for k in 0..3 {
            st(self.dens.rgb[k].cur_mut(), c, new[k]);
        }
        self.dens.amount.cur_mut().data[c] = old_a + amount;
    }

    /// Wet the whole layer (`FluidActionType::TYPE_WET`): open the wet mask everywhere and
    /// raise water to at least `film` (a calibrated wetting film is ≈ 1.2). Stencilled
    /// cells stay as they are, partially stencilled ones get a proportional film.
    pub fn wet_all(&mut self, film: f32) {
        for c in 0..self.w * self.h {
            let k = self.maps.stencil_at(c);
            if k <= 0.0 {
                continue;
            }
            self.maps.open.data[c] = 255;
            let wat = self.water.cur_mut();
            if ld(wat, c) < film * k {
                st(wat, c, film * k);
            }
        }
    }

    /// Remove all water but keep the layer marked wet (`TYPE_DRY_FAST`), except under a
    /// stencil (mostly protected cells keep their water).
    pub fn fast_dry(&mut self) {
        for c in 0..self.w * self.h {
            if self.maps.stencil_at(c) >= 0.5 {
                for p in [&mut self.water, &mut self.u, &mut self.v] {
                    st(p.cur_mut(), c, 0.0);
                }
            }
        }
    }

    /// Dry the layer completely (`TYPE_DRY`): settle all suspended pigment into the
    /// deposited layer, remove water and motion, close the wet mask.
    pub fn dry_all(&mut self) {
        for c in 0..self.w * self.h {
            // a stencil keeps the cells it covers wet
            if self.maps.stencil_at(c) < 0.5 {
                continue;
            }
            let da = self.dens.amount.cur().data[c];
            if da > 0.0 {
                let ma = self.mid.amount.cur().data[c];
                let dc: Rgb = [0, 1, 2].map(|k| ld(self.dens.rgb[k].cur(), c));
                let mc: Rgb = [0, 1, 2].map(|k| ld(self.mid.rgb[k].cur(), c));
                let la = self.lut.eval(ma);
                let new = if ma <= 0.01 || la <= 0.0 {
                    dc
                } else {
                    mix(mc, la, dc, self.lut.eval(da), self.pigment.as_deref())
                };
                for k in 0..3 {
                    st(self.mid.rgb[k].cur_mut(), c, new[k]);
                }
                self.mid.amount.cur_mut().data[c] = ma + da;
                let ds = self.dens.thickness.cur().data[c];
                self.mid.thickness.cur_mut().data[c] += ds;
                self.dens.amount.cur_mut().data[c] = 0.0;
                self.dens.thickness.cur_mut().data[c] = 0.0;
            }
        }
        self.fast_dry();
        for c in 0..self.w * self.h {
            if self.maps.stencil_at(c) >= 0.5 {
                self.maps.open.data[c] = 0;
            }
        }
    }

    /// Clear paint, water and motion (`TYPE_CLEAR`).
    pub fn clear(&mut self) {
        for p in [&mut self.dens, &mut self.mid] {
            for b in p.rgb.iter_mut() {
                b.cur_mut().fill(f16::from_f32(255.0));
            }
            p.amount.cur_mut().fill(0.0);
            p.thickness.cur_mut().fill(0.0);
        }
        for p in [&mut self.water, &mut self.u, &mut self.v] {
            p.cur_mut().fill(f16::ZERO);
        }
        self.maps.open.data.fill(0);
    }

    /// Total water and pigment (suspended + deposited), for conservation checks.
    pub fn totals(&self) -> (f64, f64) {
        let wat: f64 = self
            .water
            .cur()
            .data
            .iter()
            .map(|v| v.to_f32() as f64)
            .sum();
        let pig: f64 = self
            .dens
            .amount
            .cur()
            .data
            .iter()
            .chain(self.mid.amount.cur().data.iter())
            .map(|&v| v as f64)
            .sum();
        (wat, pig)
    }
}

/// Per-step hash constants for drop spawning (any decorrelated sequence will do).
/// The hash constants of step `n` (drop spawning in full velocity diffusion).
pub fn step_seed(n: u64) -> [u32; 4] {
    let mut s = n.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    let mut next = || {
        s ^= s >> 33;
        s = s.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        s ^= s >> 33;
        s as u32
    };
    [next() | 1, next() | 1, next(), next()]
}
