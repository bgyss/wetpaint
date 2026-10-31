//! Simulation parameters and the derived per-kernel constants.

/// User/engine parameters.
#[derive(Clone, Debug)]
pub struct FluidParams {
    // Velocity diffusion.
    pub diffuse_uv: f32,         // kw
    pub max_diffuse_uv: f32,     // wclamp
    pub max_limit_uv: f32,       // vmax
    pub mass_limit: f32,         // wthr
    pub plaster_flow_limit: f32, // pthr
    pub paper_struct: f32,
    pub paint_struct: f32,
    pub project_cycles: u32, // > 0 enables projection
    pub tilt_active: bool,
    pub tilt_x: f32,
    pub tilt_y: f32,
    pub dryme_edge: f32,
    // "Visual settings" (0..10 sliders).
    pub absorbency: f32,
    pub border_dry: f32,
    pub paper_influence: f32,
    pub rewet: f32,
    // Granulation.
    pub gran_enabled: bool,
    pub gran_strength: f32,
    pub gran_contrast: f32,
    /// Forces full drying (drying `offset = 1000`).
    pub dry_everything: bool,
    /// Drip simulation (on/off, drop size and length on the 0–10 UI scale). Off by default.
    pub create_drops: bool,
    pub drops_size: f32,
    pub drops_length: f32,
    pub placeholders: Placeholders,
}

/// Water unit scale of the brush/film injection relative to the original calibration
/// (water amounts are tens to hundreds; drops spawn at ≥ 20). A 10× larger scale fits the
/// tilt behaviour better and widens wet-in-wet spread. Gravity and the evaporation offset are divided / multiplied by it so drying
/// times and tilt speeds at tilt 1 stay calibrated.
pub const WATER_SCALE: f32 = 10.0;

/// Tuned constants, fitted jointly with the `wet-brush` injection (centre-heavy water
/// `water_rate` 0.16) to tilt, dry-paper, drying and spreading measurements.
#[derive(Clone, Debug)]
pub struct Placeholders {
    /// Tilt → gravity scale (`gx = tilt_x · tilt_scale`).
    pub tilt_scale: f32,
    /// transport step-length scalars `s0`, `kh`.
    pub transport_s0: f32,
    pub transport_kh: f32,
}

impl Default for Placeholders {
    fn default() -> Self {
        Self {
            tilt_scale: 0.00440,
            transport_s0: 2.11,
            transport_kh: 0.413,
        }
    }
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            diffuse_uv: 0.5,
            max_diffuse_uv: 0.2,
            max_limit_uv: 0.4,
            mass_limit: 0.05,
            plaster_flow_limit: 200.0,
            paper_struct: 0.5,
            paint_struct: 7e-5,
            project_cycles: 1,
            tilt_active: false,
            tilt_x: 0.0,
            tilt_y: 0.0,
            dryme_edge: 60.0,
            absorbency: 5.0,
            border_dry: 5.0,
            paper_influence: 5.0,
            rewet: 5.0,
            gran_enabled: false,
            gran_strength: 0.5,
            gran_contrast: 0.5,
            dry_everything: false,
            create_drops: false,
            drops_size: 5.0,
            drops_length: 5.0,
            placeholders: Placeholders::default(),
        }
    }
}

/// Step scheduling and drying inputs derived from the tilt state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    /// drying runs when `n % n_dry == 0`.
    pub n_dry: u32,
    /// full velocity diffusion runs when `n % n_full == 0` (else Fast).
    pub n_full: u32,
    /// Evaporation offset scale and rate floor.
    pub evap_offset: f32,
    pub evap_rate_floor: f32,
    /// Re-wet gain.
    pub rewet_gain: f32,
}

/// The edge-drying response curve used for `k_abs`, `ks`, `kp`: `0.2b` below 5,
/// `0.6b - 2` from 5.
pub fn border_curve(b: f32) -> f32 {
    if b < 5.0 { 0.2 * b } else { 0.6 * b - 2.0 }
}

/// Constants for velocity diffusion.
#[derive(Clone, Copy, Debug)]
pub struct DiffuseConsts {
    pub vmax: f32,
    pub wthr: f32,
    pub gx: f32,
    pub gy: f32,
    pub pthr: f32,
    pub kp: f32,
    pub ks: f32,
    pub kw: f32,
    pub wclamp: f32,
}

/// Constants for drying .
#[derive(Clone, Copy, Debug)]
pub struct DryConsts {
    pub rate: f32,
    pub offset: f32,
    pub k_abs: f32,
    pub k_rewet: f32,
    pub gran_enabled: bool,
    pub gran_strength: f32,
    pub gran_contrast: f32,
}

impl FluidParams {
    pub fn diffuse_consts(&self) -> DiffuseConsts {
        let f = border_curve(self.paper_influence);
        DiffuseConsts {
            vmax: self.max_limit_uv,
            wthr: self.mass_limit,
            gx: self.effective_tilt().0 * self.placeholders.tilt_scale / WATER_SCALE,
            gy: self.effective_tilt().1 * self.placeholders.tilt_scale / WATER_SCALE,
            pthr: self.plaster_flow_limit,
            kp: f * self.paint_struct,
            ks: f * self.paper_struct,
            kw: self.diffuse_uv,
            wclamp: self.max_diffuse_uv,
        }
    }

    pub fn dry_consts(&self) -> DryConsts {
        let y = if self.absorbency == 0.0 {
            0.025
        } else {
            self.absorbency / 10.0
        };
        let t = self.timing();
        let r = self.rewet;
        let g = if r <= 5.0 {
            (r / 5.0).powi(2)
        } else {
            (r / 5.0).powi(6)
        };
        let s = self.gran_strength;
        DryConsts {
            rate: 1.0 - y * (1.0 - t.evap_rate_floor),
            offset: if self.dry_everything {
                1000.0
            } else {
                y * t.evap_offset * WATER_SCALE
            },
            k_abs: border_curve(self.border_dry) * self.dryme_edge,
            k_rewet: g * t.rewet_gain,
            gran_enabled: self.gran_enabled,
            gran_strength: 100.0 * s * (0.85 * s + 0.15),
            gran_contrast: self.gran_contrast,
        }
    }

    /// Tilt as seen by the kernels: zero unless `tilt_active`.
    pub fn effective_tilt(&self) -> (f32, f32) {
        if self.tilt_active {
            (self.tilt_x, self.tilt_y)
        } else {
            (0.0, 0.0)
        }
    }

    pub fn tilted(&self) -> bool {
        let (x, y) = self.effective_tilt();
        x != 0.0 || y != 0.0
    }

    pub fn timing(&self) -> Timing {
        let n_dry = if self.tilted() { 2 } else { 4 };
        Timing {
            n_dry,
            n_full: 4,
            evap_offset: n_dry as f32 * 0.0005,
            evap_rate_floor: if self.tilted() { 0.99 } else { 0.98 },
            rewet_gain: n_dry as f32 * 0.002,
        }
    }
}
