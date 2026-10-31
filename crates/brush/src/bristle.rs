//! Bristle brushes.
//!
//! A bristle preset paints with individual bristles dragged along the stroke instead of
//! stamping the shape: near-opaque parallel lines covering
//! `particle_thin_density + particle_thick_density` of the width (≈ 84 % for a round
//! bristle brush). Line widths vary, line starts are staggered, and a press without
//! movement paints nothing.

use wet_preset::BrushPreset;

use crate::stroke::Rng;

/// Bristles per unit of density are this many times the "density = covered fraction"
/// count, so bristles overlap (fitted; best of 1–4).
const COUNT_GAIN: f32 = 3.0;

/// Bristle settings from a preset's `particle_*`, `thin_*`, `thick_*` and `bristles_*` keys.
#[derive(Clone, Debug)]
pub struct BristleParams {
    /// Fraction of the brush width covered by thin and by thick bristles.
    pub thin_density: f32,
    pub thick_density: f32,
    /// Thick bristle width in pixels at a 60 px brush, and its relative jitter.
    pub thick_size: f32,
    pub thick_size_jitter: f32,
    /// Lateral wobble: wavelength in pixels at a 60 px brush, amplitude relative to width.
    pub wave_length: f32,
    pub wave_jitter: f32,
    /// Bristles start after a random delay of up to `random_span` brush widths.
    pub random_span: f32,
    /// Edge bristles (`particle_edge_enabled`, `particle_edge_size`): one of this many
    /// pixels at each side of the brush, straddling its border: bristle strokes are wider
    /// than the brush by about 1.3 edge sizes whatever the brush size, and are not wider
    /// without edges.
    pub edge_size: Option<f32>,
    /// Thick bristles enabled with zero density (a ribbon): one thick bristle still runs
    /// down the middle, `RIBBON_PX_PER_SIZE · particle_thick_size` pixels wide whatever the
    /// brush size (4 / 8 / 16 / 30 px at thick size 25 / 50 / 100 / 200).
    pub ribbon_px: Option<f32>,
}

/// Ribbon bristle width per unit `particle_thick_size`.
const RIBBON_PX_PER_SIZE: f32 = 0.155;

impl BristleParams {
    /// `None` unless the preset is a bristle brush (`brush_mode` PARTICLE with bristle groups
    /// enabled). TEXTURE-mode presets (pencils, many oils) also carry particle settings,
    /// but for them particles are a texture overlay, not the paint source.
    pub fn from_preset(p: &BrushPreset) -> Option<Self> {
        if p.parameters.get("brush_mode").and_then(|v| v.as_str()) != Some("PARTICLE") {
            return None;
        }
        let b = |k: &str| p.parameters.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
        let f = |k: &str, d: f32| p.param_f64(k).map_or(d, |v| v as f32);
        let (thin, thick) = (b("particle_thin_enabled"), b("particle_thick_enabled"));
        if !thin && !thick {
            return None;
        }
        Some(Self {
            thin_density: if thin { f("particle_thin_density", 0.5) } else { 0.0 },
            thick_density: if thick { f("particle_thick_density", 0.3) } else { 0.0 },
            thick_size: f("particle_thick_size", 5.0),
            thick_size_jitter: f("particle_thick_size_jitter", 0.5),
            wave_length: f("thin_wave_length", 500.0),
            wave_jitter: f("thin_wave_jitter", 0.5),
            random_span: f("bristles_random_span", 0.5),
            edge_size: b("particle_edge_enabled").then(|| f("particle_edge_size", 15.0)),
            ribbon_px: (thick && f("particle_thick_density", 0.3) <= 0.0)
                .then(|| RIBBON_PX_PER_SIZE * f("particle_thick_size", 5.0)),
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Bristle {
    /// Offset across the brush in −1..1 (fraction of the radius).
    offset: f32,
    /// Width relative to the brush diameter.
    width: f32,
    /// Edge bristle: fixed width in pixels, centred just outside the brush border on
    /// the side given by the sign of `offset`.
    edge_px: Option<f32>,
    /// Fixed width in pixels, centred on `offset` (the ribbon bristle).
    fixed_px: Option<f32>,
    /// Stroke distance (in brush diameters) before the bristle touches the paper.
    start: f32,
    phase: f32,
}

/// The bristles of one stroke.
#[derive(Clone, Debug, Default)]
pub struct BristleSet {
    bristles: Vec<Bristle>,
    /// Distance travelled so far, in pixels.
    pub distance: f32,
}

impl BristleSet {
    /// Lay out bristles for a stroke. Reference sizes are for a 60 px brush.
    pub fn new(p: &BristleParams, rng: &mut Rng) -> Self {
        const REF: f32 = 60.0;
        let mut bristles = Vec::new();
        let mut add = |density: f32, width_px: f32, jitter: f32, rng: &mut Rng| {
            if density <= 0.0 {
                return;
            }
            let n = (COUNT_GAIN * density * REF / width_px).round().max(1.0) as usize;
            for _ in 0..n {
                let w = (width_px * (1.0 + jitter * rng.signed())).max(0.8) / REF;
                bristles.push(Bristle {
                    offset: rng.signed(),
                    width: w,
                    edge_px: None,
                    fixed_px: None,
                    start: p.random_span * rng.next_f32() * 0.5,
                    phase: rng.next_f32() * std::f32::consts::TAU,
                });
            }
        };
        add(p.thick_density, p.thick_size, p.thick_size_jitter, rng);
        add(p.thin_density, 2.0, 0.25, rng);
        // stratify: spread the bristles evenly across the width (shuffled order, jittered
        // within their slot), as the observed strokes have small, evenly spread gaps
        let n = bristles.len();
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.next_f32() * (i + 1) as f32) as usize % (i + 1);
            order.swap(i, j);
        }
        for (slot, &i) in order.iter().enumerate() {
            let centre = (slot as f32 + 0.5 + 0.35 * rng.signed()) / n as f32;
            bristles[i].offset = 2.0 * centre - 1.0;
        }
        if let Some(e) = p.edge_size {
            for side in [-1.0, 1.0] {
                bristles.push(Bristle {
                    offset: side,
                    width: 0.0,
                    edge_px: Some(e),
                    fixed_px: None,
                    start: p.random_span * rng.next_f32() * 0.5,
                    phase: rng.next_f32() * std::f32::consts::TAU,
                });
            }
        }
        if let Some(px) = p.ribbon_px {
            bristles.push(Bristle { offset: 0.0, width: 0.0, edge_px: None, fixed_px: Some(px), start: 0.0, phase: 0.0 });
        }
        Self { bristles, distance: 0.0 }
    }

    /// Footprints for a segment from `a` to `b` with brush diameter `size`: calls
    /// `stamp(x, y, radius)` for small discs along every active bristle's path.
    pub fn segment(
        &mut self,
        p: &BristleParams,
        a: (f32, f32),
        b: (f32, f32),
        size: f32,
        mut stamp: impl FnMut(f32, f32, f32),
    ) {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = (dx * dx + dy * dy).sqrt();
        if len <= 0.0 {
            return;
        }
        let (nx, ny) = (-dy / len, dx / len);
        let r = 0.5 * size;
        let steps = (len / 0.5).ceil().max(1.0) as usize;
        let wave_len = (p.wave_length * size / 60.0).max(1.0);
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let d = self.distance + len * t;
            let (cx, cy) = (a.0 + dx * t, a.1 + dy * t);
            for br in &self.bristles {
                if d < br.start * size {
                    continue;
                }
                let bw = br.edge_px.or(br.fixed_px).unwrap_or(br.width * size);
                let wobble = p.wave_jitter * bw * 0.5 * (std::f32::consts::TAU * d / wave_len + br.phase).sin();
                let o = match br.edge_px {
                    // outer extent r + 0.65·e: the stroke is about 1.3 edge sizes wider
                    Some(e) => br.offset * (r + 0.15 * e) + wobble,
                    None => br.offset * (r - 0.5 * bw).max(0.0) + wobble,
                };
                stamp(cx + nx * o, cy + ny * o, 0.5 * bw);
            }
        }
        self.distance += len;
    }
}
