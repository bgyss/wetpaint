//! Soft shadows: thick paint casts soft shadows, with a strength and a light altitude
//! (0–10; higher altitude, more pronounced shadows). The impasto heightfield is marched toward a light in the upper left (as for
//! the relief shading), and the strongest occlusion along the ray, softened, darkens the
//! pixel.

/// Height from paint thickness, as for the relief shading (`render::impasto_shade`).
pub fn height(thickness: f32) -> f32 {
    (thickness / 400.0).min(4.0)
}

/// Shadow factor per pixel (1 = lit) for heightfield `h` (`w`×`h_px`), `strength` and
/// `altitude` in 0–10.
pub fn soft_shadows(h: &[f32], w: usize, h_px: usize, strength: f32, altitude: f32) -> Vec<f32> {
    let k = 0.06 * strength.clamp(0.0, 10.0);
    if k <= 0.0 {
        return vec![1.0; w * h_px];
    }
    // slope of the light ray: steep (short shadows) at altitude 0, shallow at 10
    let tan = 0.6 - 0.05 * altitude.clamp(0.0, 10.0);
    let max_h = h.iter().copied().fold(0.0f32, f32::max);
    let reach = ((max_h / tan).ceil() as usize).clamp(1, 64);
    const SOFT: f32 = 0.3;
    let d = std::f32::consts::FRAC_1_SQRT_2;
    (0..w * h_px)
        .map(|c| {
            let (x, y) = ((c % w) as f32, (c / w) as f32);
            let h0 = h[c];
            let mut occ = 0.0f32;
            for t in 1..=reach {
                let tf = t as f32;
                let (sx, sy) = (x - d * tf, y - d * tf);
                if sx < 0.0 || sy < 0.0 {
                    break;
                }
                let s = h[sy as usize * w + sx as usize];
                occ = occ.max(((s - h0 - tf * tan) / SOFT).clamp(0.0, 1.0));
                if occ >= 1.0 {
                    break;
                }
            }
            1.0 - k * occ
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ridge_shades_the_lower_right_side() {
        let (w, h) = (32, 32);
        let mut hf = vec![0.0; w * h];
        for y in 0..h {
            for x in 14..16 {
                hf[y * w + x] = 3.0;
            }
        }
        let s = soft_shadows(&hf, w, h, 10.0, 5.0);
        assert!(s[16 * w + 18] < 0.7, "behind the ridge");
        assert_eq!(s[16 * w + 10], 1.0, "in front of the ridge");
        let longer = soft_shadows(&hf, w, h, 10.0, 10.0);
        assert!(longer[16 * w + 24] < s[16 * w + 24] || s[16 * w + 24] < 1.0);
    }
}
