//! Layer colour mixing rule `mix(c1, a1, c2, a2)`.

/// Colour on a 0–255 scale.
pub type Rgb = [f32; 3];

/// Pigment-style mixer used when pigment mixing is on (`wet-pigment` provides a
/// Kubelka–Munk one); this trait keeps it pluggable.
pub trait PigmentMixer: Send + Sync {
    /// Interpolate from `c1` to `c2` by `t ∈ [0, 1]` (0–255 colours).
    fn lerp(&self, c1: Rgb, c2: Rgb, t: f32) -> Rgb;
}

/// Combine layer 1 (`c1`, coverage `a1`) with layer 2 landing on top (`c2`, `a2`).
pub fn mix(c1: Rgb, a1: f32, c2: Rgb, a2: f32, pigment: Option<&dyn PigmentMixer>) -> Rgb {
    let a1p = a1 * (1.0 - a2);
    let t = a1p + a2;
    if t <= 0.0 {
        return c2;
    }
    match pigment {
        Some(m) => m.lerp(c1, c2, a2 / t),
        None => [
            (c1[0] * a1p + c2[0] * a2) / t,
            (c1[1] * a1p + c2[1] * a2) / t,
            (c1[2] * a1p + c2[2] * a2) / t,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_top_wins() {
        assert_eq!(
            mix([0.0; 3], 1.0, [200.0, 100.0, 50.0], 1.0, None),
            [200.0, 100.0, 50.0]
        );
    }

    #[test]
    fn half_over_full() {
        let c = mix([0.0; 3], 1.0, [255.0; 3], 0.5, None);
        assert!((c[0] - 127.5).abs() < 1e-4);
    }
}
