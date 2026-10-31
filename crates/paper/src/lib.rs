//! Procedural papers: the height (`H`) and absorbency (`G`) maps the fluid simulation
//! reads, plus a display shading term.
//!
//! Papers are generated from seeded value noise (fractal sums of smoothly interpolated
//! lattice noise) plus directional fibre noise.

use wet_fluid::WetLayer;

/// Paper families, roughly matching the usual watercolour paper surfaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// Smooth (fine grain, low relief).
    HotPressed,
    /// The default medium tooth.
    ColdPressed,
    /// Pronounced, larger bumps.
    Rough,
}

/// Paper generation settings.
#[derive(Clone, Copy, Debug)]
pub struct PaperSpec {
    pub surface: Surface,
    pub seed: u64,
    /// Feature scale multiplier.
    pub scale: f32,
    /// Height amplitude: the map spans about ±`relief`·127.
    pub relief: f32,
    /// Absorbency: the base level and how much it follows the valleys (0–1).
    pub absorbency: f32,
    pub absorbency_variation: f32,
}

impl Default for PaperSpec {
    fn default() -> Self {
        Self {
            surface: Surface::ColdPressed,
            seed: 1,
            scale: 1.0,
            relief: 1.0,
            absorbency: 0.0,
            absorbency_variation: 0.0,
        }
    }
}

/// A generated paper.
#[derive(Clone, Debug)]
pub struct Paper {
    pub w: usize,
    pub h: usize,
    /// Height in −1..1 (positive = ridge).
    pub height: Vec<f32>,
    /// Absorbency 0..1.
    pub absorb: Vec<f32>,
}

fn hash(x: i64, y: i64, seed: u64) -> f32 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B1_85EB_CA87)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
}

/// Smoothly interpolated lattice noise with anisotropic cell size `(sx, sy)` pixels.
fn value_noise(x: f32, y: f32, sx: f32, sy: f32, seed: u64) -> f32 {
    let (fx, fy) = (x / sx, y / sy);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let s = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (ux, uy) = (s(tx), s(ty));
    let (xi, yi) = (x0 as i64, y0 as i64);
    let a = hash(xi, yi, seed);
    let b = hash(xi + 1, yi, seed);
    let c = hash(xi, yi + 1, seed);
    let d = hash(xi + 1, yi + 1, seed);
    let top = a + (b - a) * ux;
    let bot = c + (d - c) * ux;
    top + (bot - top) * uy
}

impl Paper {
    pub fn generate(w: usize, h: usize, spec: &PaperSpec) -> Self {
        // (base feature size in px, octaves, fibre weight)
        let (base, octaves, fibre) = match spec.surface {
            Surface::HotPressed => (3.0, 3, 0.35),
            Surface::ColdPressed => (7.0, 4, 0.25),
            Surface::Rough => (14.0, 5, 0.15),
        };
        let base = base * spec.scale.max(0.05);
        let mut height = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (xf, yf) = (x as f32, y as f32);
                let (mut v, mut amp, mut size, mut norm) = (0.0, 1.0, base, 0.0);
                for o in 0..octaves {
                    v += amp * value_noise(xf, yf, size, size, spec.seed + o as u64);
                    norm += amp;
                    amp *= 0.5;
                    size *= 0.5;
                }
                v /= norm;
                // paper fibres: thin streaks at a few orientations
                let f1 = value_noise(xf, yf, base * 4.0, base * 0.35, spec.seed + 101);
                let f2 = value_noise(yf, xf, base * 4.0, base * 0.35, spec.seed + 202);
                height[y * w + x] = v * (1.0 - fibre) + fibre * 0.5 * (f1 + f2);
            }
        }
        // normalise to unit standard deviation × relief / 2.5, clamped to ±1
        let n = height.len() as f32;
        let mean = height.iter().sum::<f32>() / n;
        let sd = (height.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt().max(1e-6);
        for v in &mut height {
            *v = ((*v - mean) / sd * spec.relief / 2.5).clamp(-1.0, 1.0);
        }
        let absorb = height
            .iter()
            .map(|&v| (spec.absorbency + spec.absorbency_variation * (-v) * 0.5).clamp(0.0, 1.0))
            .collect();
        Self { w, h, height, absorb }
    }

    /// Write the maps into a layer (`H` as signed bytes, `G` as 0–255).
    pub fn apply(&self, layer: &mut WetLayer) {
        assert_eq!((layer.w, layer.h), (self.w, self.h));
        for c in 0..self.w * self.h {
            layer.maps.height.data[c] = (self.height[c] * 127.0).round() as i8;
            layer.maps.absorb.data[c] = (self.absorb[c] * 255.0).round() as u8;
        }
    }

    /// Lambertian-style shading factor for display (light from the upper left), 1 = flat.
    pub fn shade(&self, strength: f32) -> Vec<f32> {
        let (w, h) = (self.w, self.h);
        let at = |x: usize, y: usize| self.height[y.min(h - 1) * w + x.min(w - 1)];
        use rayon::prelude::*;
        (0..w * h)
            .into_par_iter()
            .map(|c| {
                let (x, y) = (c % w, c / w);
                let dx = at(x + 1, y) - at(x.saturating_sub(1), y);
                let dy = at(x, y + 1) - at(x, y.saturating_sub(1));
                1.0 + strength * (-(dx + dy) * 0.5)
            })
            .collect()
    }
}

/// Average deckle border width in pixels, the same at every canvas size.
pub const DECKLE_WIDTH_PX: f32 = 12.0;

/// The deckle border width for a `w`×`h` canvas, or `None` on small canvases
/// (128 px and below).
pub fn deckle_width(w: usize, h: usize) -> Option<f32> {
    (w.min(h) > 128).then_some(DECKLE_WIDTH_PX)
}

/// Deckled (torn) paper edge: 1 on the sheet, 0 beyond its ragged border. The border
/// sits `width` pixels in on average and wanders by about ±40 % of that along each side
/// (two noise octaves), with a 1.5 px soft rim.
pub fn deckle_mask(w: usize, h: usize, seed: u64, width: f32) -> Vec<f32> {
    let edge = |t: f32, side: u64| {
        let n = 0.65 * value_noise(t, 0.0, 6.0 * width.max(1.0), 1.0, seed + side)
            + 0.35 * value_noise(t, 0.0, 1.5 * width.max(1.0), 1.0, seed + side + 10);
        width * (1.0 + 0.4 * n)
    };
    (0..w * h)
        .map(|c| {
            let (x, y) = ((c % w) as f32, (c / w) as f32);
            let inset = [
                x - edge(y, 1),
                y - edge(x, 2),
                (w as f32 - 1.0 - x) - edge(y, 3),
                (h as f32 - 1.0 - y) - edge(x, 4),
            ]
            .into_iter()
            .fold(f32::MAX, f32::min);
            (inset / 1.5 + 0.5).clamp(0.0, 1.0)
        })
        .collect()
}

/// A tileable granulation texture (`size`×`size`, 0–255): fine pigment-clump speckle
/// built from small-scale lattice noise that wraps at the tile edge.
pub fn granulation_tile(size: usize, seed: u64, clump: f32) -> Vec<u8> {
    let noise = |x: f32, y: f32, cell: f32, s: u64| {
        // wrap the lattice so the tile repeats seamlessly
        let n = (size as f32 / cell).round().max(1.0) as i64;
        let (fx, fy) = (x / size as f32 * n as f32, y / size as f32 * n as f32);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let sm = |t: f32| t * t * (3.0 - 2.0 * t);
        let (ux, uy) = (sm(tx), sm(ty));
        let at = |i: i64, j: i64| hash(i.rem_euclid(n), j.rem_euclid(n), s);
        let (xi, yi) = (x0 as i64, y0 as i64);
        let top = at(xi, yi) + (at(xi + 1, yi) - at(xi, yi)) * ux;
        let bot = at(xi, yi + 1) + (at(xi + 1, yi + 1) - at(xi, yi + 1)) * ux;
        top + (bot - top) * uy
    };
    let c = clump.max(0.5);
    let mut v: Vec<f32> = (0..size * size)
        .map(|i| {
            let (x, y) = ((i % size) as f32, (i / size) as f32);
            0.55 * noise(x, y, 1.5 * c, seed) + 0.3 * noise(x, y, 3.0 * c, seed + 1)
                + 0.15 * noise(x, y, 8.0 * c, seed + 2)
        })
        .collect();
    let (lo, hi) = v.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    for x in &mut v {
        // contrast shaping: most of the tile mid-grey, clumps bright
        let t = (*x - lo) / (hi - lo).max(1e-6);
        *x = t * t * (3.0 - 2.0 * t);
    }
    v.into_iter().map(|t| (t * 255.0).round() as u8).collect()
}

/// Install a granulation tile into a layer.
pub fn apply_granulation(layer: &mut WetLayer, tile: &[u8], size: usize) {
    let mut plane = wet_fluid::grid::PlaneU8::new(size, size, 0);
    plane.data.copy_from_slice(tile);
    layer.maps.gran = Some(plane);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_normalised() {
        let spec = PaperSpec::default();
        let a = Paper::generate(64, 64, &spec);
        let b = Paper::generate(64, 64, &spec);
        assert_eq!(a.height, b.height);
        let n = a.height.len() as f32;
        let mean = a.height.iter().sum::<f32>() / n;
        let sd = (a.height.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
        assert!(mean.abs() < 0.02, "{mean}");
        assert!((sd - 0.4).abs() < 0.05, "{sd}");
        let c = Paper::generate(64, 64, &PaperSpec { seed: 2, ..spec });
        assert_ne!(a.height, c.height);
    }
}
