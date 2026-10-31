//! Metallic Material layers (settings: reflectivity, reflection map, scale and strength):
//! metallic paint darkens slightly
//! and picks up a sheen from a reflection map sampled at positions shifted by the impasto
//! relief, so the highlights follow the brushstrokes.

/// Metallic settings, 0–10 like the other visual settings.
#[derive(Clone, Debug, PartialEq)]
pub struct MetallicSettings {
    pub reflectivity: f32,
    pub scale: f32,
    pub strength: f32,
    /// Grayscale reflection (environment) map, 0–1, tiled; `None` uses a smooth
    /// procedural map.
    pub map: Option<(usize, usize, Vec<f32>)>,
}

impl Default for MetallicSettings {
    fn default() -> Self {
        Self { reflectivity: 5.0, scale: 5.0, strength: 5.0, map: None }
    }
}

impl MetallicSettings {
    /// Use a PNG (grayscale or colour, by luminance) as the reflection map.
    pub fn load_map(&mut self, png: &[u8]) -> bool {
        match crate::edit::decode_gray(png) {
            Some((w, h, d)) => {
                self.map = Some((w, h, d.iter().map(|&v| v as f32 / 255.0).collect()));
                true
            }
            None => false,
        }
    }

    /// Reflection map value at `(u, v)` in map pixels (tiled).
    fn env(&self, u: f32, v: f32) -> f32 {
        match &self.map {
            Some((w, h, d)) if *w > 0 && *h > 0 => {
                let (x, y) = (u.rem_euclid(*w as f32) as usize, v.rem_euclid(*h as f32) as usize);
                d[y.min(h - 1) * w + x.min(w - 1)]
            }
            _ => {
                // broad soft bands, like a studio environment
                let s = (0.013 * u + 0.021 * v).sin() * 0.5 + (0.031 * v - 0.007 * u).sin() * 0.3;
                (0.5 + 0.5 * s).clamp(0.0, 1.0)
            }
        }
    }

    /// Per-pixel (multiplier, additive sheen) for metallic coverage `m` (0–1) and relief
    /// height `h` (impasto height units).
    pub fn shade(&self, m: &[f32], h: &[f32], w: usize, hgt: usize) -> Vec<(f32, f32)> {
        let k_ref = self.reflectivity.clamp(0.0, 10.0) / 10.0;
        let k_str = self.strength.clamp(0.0, 10.0) / 10.0;
        let zoom = 0.5 + self.scale.clamp(1.0, 10.0) / 5.0;
        let at = |x: usize, y: usize| h[y.min(hgt - 1) * w + x.min(w - 1)];
        (0..w * hgt)
            .map(|c| {
                if m[c] <= 0.0 {
                    return (1.0, 0.0);
                }
                let (x, y) = (c % w, c / w);
                let dx = at(x + 1, y) - at(x.saturating_sub(1), y);
                let dy = at(x, y + 1) - at(x, y.saturating_sub(1));
                // reflected lookup shifts with the surface slope
                let e = self.env((x as f32 + 40.0 * dx) / zoom, (y as f32 + 40.0 * dy) / zoom);
                let mult = 1.0 - m[c] * k_ref * 0.35;
                let sheen = m[c] * k_ref * k_str * 0.6 * e.powf(2.0);
                (mult, sheen)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metallic_paint_gets_a_sheen_and_plain_paper_does_not() {
        let (w, h) = (16, 16);
        let mut m = vec![0.0; w * h];
        m[5 * w + 5] = 1.0;
        let s = MetallicSettings::default().shade(&m, &vec![0.0; w * h], w, h);
        assert_eq!(s[0], (1.0, 0.0));
        assert!(s[5 * w + 5].0 < 1.0);
        let none = MetallicSettings { reflectivity: 0.0, ..Default::default() }.shade(&m, &vec![0.0; w * h], w, h);
        assert_eq!(none[5 * w + 5], (1.0, 0.0));
    }
}
