//! Layer compositing: 27 colour mix modes and a layer stack. Formulas follow the W3C *Compositing and Blending Level 1* spec for the
//! modes it defines. The remaining modes use the usual definitions from public image
//! editors' documentation. All colours are straight (non-premultiplied) RGB in 0..1.

use rayon::prelude::*;

pub type Rgb = [f32; 3];

/// Blend modes, in their stable numeric order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixMode {
    Normal,
    Additive,
    Subtractive,
    LinearBurn,
    Darken,
    Lighten,
    Multiply,
    Screen,
    ColorDodge,
    ColorBurn,
    Overlay,
    HardLight,
    SoftLight,
    VividLight,
    LinearLight,
    PinLight,
    HardMix,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
    /// Pigment (subtractive, Kubelka–Munk) mixing. Uses the mixer supplied to
    /// [`composite_pixel`], or Normal if there is none.
    Pigment,
    /// Height modes apply to the impasto/structure channel. For colour they act as Normal.
    Height,
    LinearHeight,
    /// Inverse multiply: `1 − (1 − cb)·cs`.
    InvMultiply,
}

impl MixMode {
    pub const ALL: [MixMode; 27] = [
        Self::Normal, Self::Additive, Self::Subtractive, Self::LinearBurn, Self::Darken,
        Self::Lighten, Self::Multiply, Self::Screen, Self::ColorDodge, Self::ColorBurn,
        Self::Overlay, Self::HardLight, Self::SoftLight, Self::VividLight, Self::LinearLight,
        Self::PinLight, Self::HardMix, Self::Difference, Self::Exclusion, Self::Hue,
        Self::Saturation, Self::Color, Self::Luminosity, Self::Pigment, Self::Height,
        Self::LinearHeight, Self::InvMultiply,
    ];

    /// Mode names (as stored in documents and presets), in the order of [`Self::ALL`].
    const NAMES: [&'static str; 27] = [
        "NORMAL", "ADDITIVE", "SUBTRACTIVE", "LINEAR_BURN", "DARKEN", "LIGHTEN",
        "MULTIPLY", "SCREEN", "COLOR_DODGE", "COLOR_BURN", "OVERLAY", "HARD_LIGHT",
        "SOFT_LIGHT", "VIVID_LIGHT", "LINEAR_LIGHT", "PIN_LIGHT", "HARD_MIX",
        "DIFFERENCE", "EXCLUSION", "HUE", "SATURATION", "COLOR", "LUMINOSITY", "PIGMENT",
        "HEIGHT", "LINEAR_HEIGHT", "INV_MULTIPLY",
    ];

    /// Parse a mode name (`NORMAL`, `COLOR_DODGE`, `PIGMENT`, …).
    pub fn from_name(s: &str) -> Option<Self> {
        Self::NAMES.iter().position(|n| *n == s).map(|i| Self::ALL[i])
    }

    /// The enum name [`Self::from_name`] reads.
    pub fn name(self) -> &'static str {
        Self::NAMES[Self::ALL.iter().position(|&m| m == self).unwrap_or(0)]
    }
}

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn color_dodge(b: f32, s: f32) -> f32 {
    if b <= 0.0 {
        0.0
    } else if s >= 1.0 {
        1.0
    } else {
        (b / (1.0 - s)).min(1.0)
    }
}

fn color_burn(b: f32, s: f32) -> f32 {
    if b >= 1.0 {
        1.0
    } else if s <= 0.0 {
        0.0
    } else {
        1.0 - ((1.0 - b) / s).min(1.0)
    }
}

fn hard_light(b: f32, s: f32) -> f32 {
    if s <= 0.5 { b * 2.0 * s } else { let t = 2.0 * s - 1.0; b + t - b * t }
}

fn soft_light(b: f32, s: f32) -> f32 {
    if s <= 0.5 {
        b - (1.0 - 2.0 * s) * b * (1.0 - b)
    } else {
        let d = if b <= 0.25 { ((16.0 * b - 12.0) * b + 4.0) * b } else { b.sqrt() };
        b + (2.0 * s - 1.0) * (d - b)
    }
}

fn vivid_light(b: f32, s: f32) -> f32 {
    if s <= 0.5 { color_burn(b, 2.0 * s) } else { color_dodge(b, 2.0 * s - 1.0) }
}

/// Separable blend function `B(cb, cs)` for one channel.
fn separable(mode: MixMode, b: f32, s: f32) -> f32 {
    use MixMode::*;
    match mode {
        Additive => (b + s).min(1.0),
        Subtractive => (b - s).max(0.0),
        LinearBurn => (b + s - 1.0).max(0.0),
        Darken => b.min(s),
        Lighten => b.max(s),
        Multiply => b * s,
        Screen => b + s - b * s,
        ColorDodge => color_dodge(b, s),
        ColorBurn => color_burn(b, s),
        Overlay => hard_light(s, b),
        HardLight => hard_light(b, s),
        SoftLight => soft_light(b, s),
        VividLight => vivid_light(b, s),
        LinearLight => clamp01(b + 2.0 * s - 1.0),
        PinLight => if s <= 0.5 { b.min(2.0 * s) } else { b.max(2.0 * s - 1.0) },
        HardMix => if vivid_light(b, s) >= 0.5 { 1.0 } else { 0.0 },
        Difference => (b - s).abs(),
        Exclusion => b + s - 2.0 * b * s,
        InvMultiply => 1.0 - (1.0 - b) * s,
        _ => s,
    }
}

// Non-separable helpers (W3C §10.3).
fn lum(c: Rgb) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn clip_color(c: Rgb) -> Rgb {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    let mut out = c;
    if n < 0.0 {
        out = out.map(|v| l + (v - l) * l / (l - n));
    }
    if x > 1.0 {
        out = out.map(|v| l + (v - l) * (1.0 - l) / (x - l));
    }
    out
}

fn set_lum(c: Rgb, l: f32) -> Rgb {
    let d = l - lum(c);
    clip_color(c.map(|v| v + d))
}

fn sat(c: Rgb) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn set_sat(c: Rgb, s: f32) -> Rgb {
    let (mut imax, mut imin) = (0, 0);
    for i in 1..3 {
        if c[i] > c[imax] { imax = i; }
        if c[i] < c[imin] { imin = i; }
    }
    if imax == imin {
        return [0.0; 3];
    }
    let imid = 3 - imax - imin;
    let mut out = [0.0; 3];
    out[imid] = (c[imid] - c[imin]) * s / (c[imax] - c[imin]);
    out[imax] = s;
    out
}

/// Blend function `B(cb, cs)` on a whole colour.
pub fn blend(mode: MixMode, cb: Rgb, cs: Rgb) -> Rgb {
    use MixMode::*;
    match mode {
        Normal | Pigment | Height | LinearHeight => cs,
        Hue => set_lum(set_sat(cs, sat(cb)), lum(cb)),
        Saturation => set_lum(set_sat(cb, sat(cs)), lum(cb)),
        Color => set_lum(cs, lum(cb)),
        Luminosity => set_lum(cb, lum(cs)),
        m => [0, 1, 2].map(|i| separable(m, cb[i], cs[i])),
    }
}

/// Subtractive pigment mixing of two colours: `t` = fraction of the second.
/// `Sync` so compositing can run on several threads.
pub trait PigmentMix: Sync {
    fn mix(&self, a: Rgb, b: Rgb, t: f32) -> Rgb;
}

/// Source-over compositing with a blend mode (W3C §5.8): straight colours and alphas
/// in, straight colour and alpha out.
pub fn composite_pixel(
    cb: Rgb,
    ab: f32,
    cs: Rgb,
    a_s: f32,
    mode: MixMode,
    pigment: Option<&dyn PigmentMix>,
) -> (Rgb, f32) {
    let ao = a_s + ab * (1.0 - a_s);
    if ao <= 0.0 {
        return ([0.0; 3], 0.0);
    }
    let mixed = match (mode, pigment) {
        (MixMode::Pigment, Some(p)) => p.mix(cb, cs, 1.0),
        _ => blend(mode, cb, cs),
    };
    let co = [0, 1, 2].map(|i| {
        let premul =
            (1.0 - a_s) * ab * cb[i] + (1.0 - ab) * a_s * cs[i] + a_s * ab * mixed[i];
        clamp01(premul / ao)
    });
    let co = match (mode, pigment) {
        // pigment glazing: mix by coverage instead of the source-over weights
        (MixMode::Pigment, Some(p)) if ab > 0.0 => p.mix(cb, cs, a_s / ao),
        _ => co,
    };
    (co, ao)
}

/// One raster layer (straight RGBA, 0..1).
#[derive(Clone, Debug)]
pub struct Layer {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<[f32; 4]>,
    pub opacity: f32,
    pub mode: MixMode,
    pub visible: bool,
}

impl Layer {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, rgba: vec![[0.0; 4]; w * h], opacity: 1.0, mode: MixMode::Normal, visible: true }
    }
}

/// Composite visible layers bottom to top over an opaque background.
pub fn flatten(background: Rgb, layers: &[Layer], pigment: Option<&dyn PigmentMix>) -> Vec<Rgb> {
    let n = layers.first().map_or(0, |l| l.w * l.h);
    let mut out = vec![background; n];
    for layer in layers.iter().filter(|l| l.visible && l.opacity > 0.0) {
        out.par_iter_mut().zip(&layer.rgba).for_each(|(o, p)| {
            let a = p[3] * layer.opacity;
            if a > 0.0 {
                *o = composite_pixel(*o, 1.0, [p[0], p[1], p[2]], a, layer.mode, pigment).0;
            }
        });
    }
    out
}

/// Composite visible layers bottom to top onto a transparent backdrop (an isolated
/// layer group): straight RGBA out.
pub fn composite_isolated(layers: &[Layer], pigment: Option<&dyn PigmentMix>) -> Vec<[f32; 4]> {
    let n = layers.first().map_or(0, |l| l.w * l.h);
    let mut out = vec![[0.0f32; 4]; n];
    for layer in layers.iter().filter(|l| l.visible && l.opacity > 0.0) {
        out.par_iter_mut().zip(&layer.rgba).for_each(|(o, p)| {
            let a = p[3] * layer.opacity;
            if a > 0.0 {
                let (c, ao) = composite_pixel([o[0], o[1], o[2]], o[3], [p[0], p[1], p[2]], a, layer.mode, pigment);
                *o = [c[0], c[1], c[2], ao];
            }
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Rgb, b: Rgb) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn separable_reference_values() {
        let (b, s) = ([0.2, 0.5, 0.8], [0.6, 0.5, 0.3]);
        assert!(close(blend(MixMode::Multiply, b, s), [0.12, 0.25, 0.24]));
        assert!(close(blend(MixMode::Screen, b, s), [0.68, 0.75, 0.86]));
        assert!(close(blend(MixMode::Darken, b, s), [0.2, 0.5, 0.3]));
        assert!(close(blend(MixMode::Difference, b, s), [0.4, 0.0, 0.5]));
        assert!(close(blend(MixMode::Exclusion, b, s), [0.56, 0.5, 0.62]));
        assert!(close(blend(MixMode::LinearBurn, b, s), [0.0, 0.0, 0.1]));
        assert!(close(blend(MixMode::Additive, b, s), [0.8, 1.0, 1.0]));
        // overlay = hard light with the arguments swapped
        assert!(close(blend(MixMode::Overlay, b, s), blend(MixMode::HardLight, s, b)));
    }

    #[test]
    fn non_separable_preserve_their_component() {
        let (b, s) = ([0.8, 0.3, 0.1], [0.1, 0.4, 0.9]);
        assert!((lum(blend(MixMode::Color, b, s)) - lum(b)).abs() < 1e-4);
        assert!((lum(blend(MixMode::Luminosity, b, s)) - lum(s)).abs() < 1e-4);
        assert!((sat(blend(MixMode::Saturation, b, s)) - sat(s)).abs() < 1e-3);
    }

    #[test]
    fn source_over_alpha() {
        let (c, a) = composite_pixel([1.0, 1.0, 1.0], 1.0, [0.0, 0.0, 0.0], 0.25, MixMode::Normal, None);
        assert!(close(c, [0.75; 3]));
        assert_eq!(a, 1.0);
        let (c, a) = composite_pixel([0.0; 3], 0.0, [0.2, 0.4, 0.6], 0.5, MixMode::Multiply, None);
        assert!(close(c, [0.2, 0.4, 0.6]));
        assert_eq!(a, 0.5);
        assert_eq!(MixMode::from_name("PIGMENT"), Some(MixMode::Pigment));
        assert_eq!(MixMode::from_name("INV_MULTIPLY"), Some(MixMode::InvMultiply));
    }
}
