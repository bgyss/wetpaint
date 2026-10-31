//! Red–yellow–blue colour mixing ("Natural Colors": blue and yellow make green, as with
//! paints). Our implementation of the published computational
//! RYB model of Sugita & Takahashi (2015): colours are converted to RYB by removing their
//! white and black parts and splitting green into yellow and blue, interpolated there, and
//! converted back.

use wet_compose::PigmentMix;
use wet_fluid::mix::PigmentMixer;

/// RGB (0…1) → RYB (0…1).
pub fn rgb_to_ryb(c: [f32; 3]) -> [f32; 3] {
    let white = c[0].min(c[1]).min(c[2]);
    let [r, g, b] = c.map(|v| v - white);
    let max_g = r.max(g).max(b);
    let rg = r.min(g);
    let mut ryb = [r - rg, (g + rg) / 2.0, (b + g - rg) / 2.0];
    let max_y = ryb[0].max(ryb[1]).max(ryb[2]);
    if max_y > 1e-6 {
        let n = max_g / max_y;
        ryb = ryb.map(|v| v * n);
    }
    let black = (1.0 - c[0]).min(1.0 - c[1]).min(1.0 - c[2]);
    ryb.map(|v| (v + black).clamp(0.0, 1.0))
}

/// RYB (0…1) → RGB (0…1).
pub fn ryb_to_rgb(c: [f32; 3]) -> [f32; 3] {
    let black = c[0].min(c[1]).min(c[2]);
    let [r, y, b] = c.map(|v| v - black);
    let max_y = r.max(y).max(b);
    let yb = y.min(b);
    let mut rgb = [r + y - yb, y + yb, 2.0 * (b - yb)];
    let max_g = rgb[0].max(rgb[1]).max(rgb[2]);
    if max_g > 1e-6 {
        let n = max_y / max_g;
        rgb = rgb.map(|v| v * n);
    }
    let white = (1.0 - c[0]).min(1.0 - c[1]).min(1.0 - c[2]);
    rgb.map(|v| (v + white).clamp(0.0, 1.0))
}

/// Mixes colours linearly in RYB space.
#[derive(Clone, Copy, Debug, Default)]
pub struct RybMixer;

impl RybMixer {
    /// Interpolate from `a` to `b` by `t` (0…1 colours).
    pub fn mix2(&self, a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
        let (ya, yb) = (rgb_to_ryb(a), rgb_to_ryb(b));
        ryb_to_rgb([0, 1, 2].map(|k| ya[k] + (yb[k] - ya[k]) * t.clamp(0.0, 1.0)))
    }
}

impl PigmentMix for RybMixer {
    fn mix(&self, a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
        self.mix2(a, b, t)
    }
}

/// The fluid engine works on a 0–255 scale.
impl PigmentMixer for RybMixer {
    fn lerp(&self, c1: [f32; 3], c2: [f32; 3], t: f32) -> [f32; 3] {
        self.mix2(c1.map(|v| v / 255.0), c2.map(|v| v / 255.0), t).map(|v| v * 255.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_primaries_and_greys() {
        for c in [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.8, 0.4, 0.1]] {
            let back = ryb_to_rgb(rgb_to_ryb(c));
            for k in 0..3 {
                assert!((back[k] - c[k]).abs() < 1e-3, "{c:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn blue_and_yellow_make_green() {
        let g = RybMixer.mix2([0.0, 0.0, 1.0], [1.0, 1.0, 0.0], 0.5);
        assert!(g[1] > g[0] && g[1] > g[2], "{g:?}");
    }
}
