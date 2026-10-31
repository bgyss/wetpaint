//! Kubelka–Munk pigment mixing.
//!
//! Method (our own construction, after the Kubelka–Munk two-flux model, Kubelka & Munk
//! 1931, and its use for paint mixing in Duncan 1940 / Haase & Meyer 1992):
//! 1. upsample an RGB colour to a reflectance spectrum on `BANDS` wavelengths as
//!    `R(λ) = r·S_r(λ) + g·S_g(λ) + b·S_b(λ)`, where the three smooth, overlapping basis
//!    bumps sum to 1 (white is flat), and the overlap gives blue and yellow pigments a
//!    shared green reflectance;
//! 2. mix in K/S space, linear in concentration: `K/S = (1 − R)² / 2R`;
//! 3. convert back with narrower sensor curves, followed by the inverse of the 3×3
//!    round-trip matrix so unmixed colours come back exactly.

use wet_compose::PigmentMix;
use wet_fluid::mix::PigmentMixer;

pub mod ryb;
pub use ryb::RybMixer;

const BANDS: usize = 16;
const R_MIN: f32 = 1e-3;
const TINT: f32 = 3.3;

fn luminance(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

/// Gaussian bump on band index space.
fn bump(i: usize, centre: f32, width: f32) -> f32 {
    let d = (i as f32 - centre) / width;
    (-0.5 * d * d).exp()
}

/// A Kubelka–Munk mixer with precomputed basis, sensor and correction tables.
#[derive(Clone, Debug)]
pub struct KmMixer {
    basis: [[f32; BANDS]; 3],
    sensor: [[f32; BANDS]; 3],
    correction: [[f32; 3]; 3],
}

impl Default for KmMixer {
    fn default() -> Self {
        Self::new()
    }
}

impl KmMixer {
    pub fn new() -> Self {
        // band 0 = short (blue) end, band 15 = long (red) end
        let raw = [
            |i| bump(i, 13.0, 3.2) + 0.5 * bump(i, 0.0, 1.5), // red
            |i| bump(i, 7.5, 2.6),  // green
            |i| bump(i, 2.0, 2.2),  // blue
        ];
        let mut basis = [[0.0; BANDS]; 3];
        for i in 0..BANDS {
            let v = [raw[0](i), raw[1](i), raw[2](i)];
            let s: f32 = v.iter().sum();
            for k in 0..3 {
                basis[k][i] = v[k] / s; // partition of unity: white → flat 1
            }
        }
        let sensor_raw = [
            |i| bump(i, 13.0, 1.8),
            |i| bump(i, 7.5, 1.6),
            |i| bump(i, 2.0, 1.8),
        ];
        let mut sensor = [[0.0; BANDS]; 3];
        for k in 0..3 {
            let s: f32 = (0..BANDS).map(sensor_raw[k]).sum();
            for i in 0..BANDS {
                sensor[k][i] = sensor_raw[k](i) / s; // flat 1 → 1
            }
        }
        // round-trip matrix M[k][j] = Σ sensor_k · basis_j, then invert
        let mut m = [[0.0f32; 3]; 3];
        for k in 0..3 {
            for j in 0..3 {
                m[k][j] = (0..BANDS).map(|i| sensor[k][i] * basis[j][i]).sum();
            }
        }
        Self { basis, sensor, correction: invert3(m) }
    }

    /// The mixer's tables for a GPU port: basis (3×16, by colour then band), sensor (3×16)
    /// and the 3×3 correction matrix (row-major), 105 values.
    pub fn tables(&self) -> Vec<f32> {
        let mut t: Vec<f32> = self.basis.iter().chain(&self.sensor).flatten().copied().collect();
        t.extend(self.correction.iter().flatten());
        t
    }

    fn spectrum(&self, c: [f32; 3]) -> [f32; BANDS] {
        let mut r = [0.0; BANDS];
        for i in 0..BANDS {
            r[i] = (0..3).map(|k| c[k] * self.basis[k][i]).sum::<f32>().clamp(R_MIN, 1.0);
        }
        r
    }

    fn to_rgb(&self, r: &[f32; BANDS]) -> [f32; 3] {
        let raw: [f32; 3] =
            [0, 1, 2].map(|k| (0..BANDS).map(|i| self.sensor[k][i] * r[i]).sum::<f32>());
        [0, 1, 2].map(|k| {
            (0..3).map(|j| self.correction[k][j] * raw[j]).sum::<f32>().clamp(0.0, 1.0)
        })
    }

    /// Mix colours (0..1 RGB) with concentration weights.
    pub fn mix_many(&self, colors: &[([f32; 3], f32)]) -> [f32; 3] {
        let total: f32 = colors.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            return [1.0; 3];
        }
        let mut ks = [0.0f32; BANDS];
        for (c, w) in colors {
            let r = self.spectrum(*c);
            for i in 0..BANDS {
                ks[i] += w / total * (1.0 - r[i]).powi(2) / (2.0 * r[i]);
            }
        }
        // invert K/S → R: R = 1 + K/S − sqrt((K/S)² + 2·K/S)
        let r = ks.map(|q| (1.0 + q - (q * q + 2.0 * q).sqrt()).clamp(R_MIN, 1.0));
        self.to_rgb(&r)
    }

    /// Two-colour mix, `t` = concentration of `b`.
    pub fn mix2(&self, a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
        let t = t.clamp(0.0, 1.0);
        if t <= 0.0 {
            return a;
        }
        if t >= 1.0 {
            return b;
        }
        let (sa, sb) = ((TINT * luminance(a)).exp(), (TINT * luminance(b)).exp());
        let tb = t * sb / (t * sb + (1.0 - t) * sa);
        self.mix_many(&[(a, 1.0 - tb), (b, tb)])
    }
}

fn invert3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv_det = 1.0 / det;
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [c(1, 1, 2, 2) * inv_det, -c(0, 1, 2, 2) * inv_det, c(0, 1, 1, 2) * inv_det],
        [-c(1, 0, 2, 2) * inv_det, c(0, 0, 2, 2) * inv_det, -c(0, 0, 1, 2) * inv_det],
        [c(1, 0, 2, 1) * inv_det, -c(0, 0, 2, 1) * inv_det, c(0, 0, 1, 1) * inv_det],
    ]
}

impl PigmentMix for KmMixer {
    fn mix(&self, a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
        self.mix2(a, b, t)
    }
}

/// The fluid engine works on a 0–255 scale.
impl PigmentMixer for KmMixer {
    fn lerp(&self, c1: [f32; 3], c2: [f32; 3], t: f32) -> [f32; 3] {
        self.mix2(c1.map(|v| v / 255.0), c2.map(|v| v / 255.0), t).map(|v| v * 255.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_colours_round_trip() {
        let m = KmMixer::new();
        for c in [[1.0, 1.0, 1.0], [0.8, 0.2, 0.1], [0.1, 0.3, 0.9], [0.5, 0.5, 0.5]] {
            let r = m.mix_many(&[(c, 1.0)]);
            for k in 0..3 {
                assert!((r[k] - c[k]).abs() < 0.03, "{c:?} -> {r:?}");
            }
        }
    }

    #[test]
    fn blue_and_yellow_make_green() {
        let m = KmMixer::new();
        let g = m.mix2([0.05, 0.15, 0.85], [0.95, 0.85, 0.05], 0.5);
        assert!(g[1] > g[0] && g[1] > g[2], "{g:?}");
    }

    #[test]
    fn mixing_is_subtractive() {
        let m = KmMixer::new();
        // red + blue: darker than their additive average
        let (a, b) = ([0.9, 0.1, 0.1], [0.1, 0.1, 0.9]);
        let km = m.mix2(a, b, 0.5);
        let lin: f32 = (0..3).map(|k| 0.5 * (a[k] + b[k])).sum();
        assert!(km.iter().sum::<f32>() < lin);
    }
}
