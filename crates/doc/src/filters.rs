//! Layer filters: colour adjustments, blurs, sharpening and
//! colour-to-alpha, applied to a paint layer's pigment, optionally through the selection.
//!
//! Each adjustment uses its standard image-editing definition. Colour filters change the pigment colours of both the deposited
//! and the suspended paint; spatial filters move amount-weighted colour and pigment amount
//! together so colours do not bleed into empty paper; colour-to-alpha lowers pigment
//! amounts through the opacity LUT.

use half::f16;
use wet_fluid::WetLayer;
use wet_fluid::lut::OpacityLut;

/// Which colour channels a tonal adjustment acts on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Channel {
    Rgb,
    Red,
    Green,
    Blue,
}

impl Channel {
    fn applies(self, k: usize) -> bool {
        match self {
            Channel::Rgb => true,
            Channel::Red => k == 0,
            Channel::Green => k == 1,
            Channel::Blue => k == 2,
        }
    }
}

/// A filter and its settings. Colour values are 0–255; fractions are −1…1 or 0…1 as noted.
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    /// Brightness and contrast, each −1…1.
    BrightnessContrast { brightness: f32, contrast: f32 },
    /// Input black/white points, midtone gamma (1 = linear) and output black/white points.
    Levels { channel: Channel, in_black: f32, in_white: f32, gamma: f32, out_black: f32, out_white: f32 },
    /// Tone curve through `points` (x, y in 0…1, smooth interpolation).
    Curves { channel: Channel, points: Vec<(f32, f32)> },
    /// Hue shift in degrees, saturation and lightness −1…1.
    HueSaturation { hue: f32, saturation: f32, lightness: f32 },
    /// Cyan–red, magenta–green and yellow–blue shifts (−1…1) for shadows, midtones and
    /// highlights.
    ColorBalance { shadows: [f32; 3], midtones: [f32; 3], highlights: [f32; 3], preserve_luminosity: bool },
    /// Photo filter: multiply by `color` with `strength` 0…1, keeping the luminosity.
    ColorFilter { color: [f32; 3], strength: f32 },
    /// Recolour to one hue (degrees) and saturation (0…1), blended by `strength` 0…1.
    Colorize { hue: f32, saturation: f32, strength: f32 },
    /// Shades of grey from luminosity (true) or the channel average.
    Desaturate { luminosity: bool },
    Invert,
    /// Map grey levels onto a gradient (`stops`: position 0…1, colour).
    GradientMap { stops: Vec<(f32, [f32; 3])> },
    /// Gaussian blur; `radius` in pixels covers three standard deviations.
    GaussianBlur { radius: f32 },
    /// Disc ("lens") blur of diameter `size` px; colours brighter than `threshold` (0–255)
    /// weigh `1 + brightness` times more, as specular highlights do.
    LensBlur { size: f32, threshold: f32, brightness: f32 },
    /// Unsharp mask: `radius` px blur, `strength` 0…1 (× 2 boost at 1).
    Sharpen { radius: f32, strength: f32 },
    /// A quick, light sharpen (radius 1, strength 0.35).
    SubtleSharpen,
    /// White (or black) becomes transparent: the colour-to-alpha decomposition.
    WhiteToAlpha,
    BlackToAlpha,
}

impl Filter {
    /// The filter's menu name.
    pub fn label(&self) -> &'static str {
        match self {
            Self::BrightnessContrast { .. } => "Brightness / Contrast",
            Self::Levels { .. } => "Levels",
            Self::Curves { .. } => "Curves",
            Self::HueSaturation { .. } => "Hue / Saturation",
            Self::ColorBalance { .. } => "Color Balance",
            Self::ColorFilter { .. } => "Color Filter",
            Self::Colorize { .. } => "Colorize",
            Self::Desaturate { .. } => "Desaturate",
            Self::Invert => "Invert",
            Self::GradientMap { .. } => "Gradient Map",
            Self::GaussianBlur { .. } => "Gaussian Blur",
            Self::LensBlur { .. } => "Lens Blur",
            Self::Sharpen { .. } => "Sharpen",
            Self::SubtleSharpen => "Subtle Sharpen",
            Self::WhiteToAlpha => "White to Alpha",
            Self::BlackToAlpha => "Black to Alpha",
        }
    }
}

/// Apply `f` to `layer`; `selection` (0–255 per cell) limits and feathers the effect.
pub fn apply(layer: &mut WetLayer, f: &Filter, selection: Option<&[u8]>) {
    let n = layer.w * layer.h;
    let sel = |c: usize| selection.map_or(1.0, |s| s[c] as f32 / 255.0);
    match f {
        Filter::GaussianBlur { radius } => spatial(layer, selection, |p, w, h| gaussian(p, w, h, *radius)),
        Filter::LensBlur { size, threshold, brightness } => lens_blur(layer, selection, *size, *threshold, *brightness),
        Filter::Sharpen { radius, strength } => sharpen(layer, selection, *radius, *strength),
        Filter::SubtleSharpen => sharpen(layer, selection, 1.0, 0.35),
        Filter::WhiteToAlpha | Filter::BlackToAlpha => {
            let white = matches!(f, Filter::WhiteToAlpha);
            for c in 0..n {
                let s = sel(c);
                if s > 0.0 {
                    to_alpha(layer, c, white, s);
                }
            }
        }
        _ => {
            let map = color_map(f);
            for c in 0..n {
                let s = sel(c);
                if s <= 0.0 {
                    continue;
                }
                for rgb in [&mut layer.mid.rgb, &mut layer.dens.rgb] {
                    let old = [0, 1, 2].map(|k| rgb[k].cur().data[c].to_f32());
                    let new = map(old);
                    for k in 0..3 {
                        let v = old[k] + s * (new[k] - old[k]);
                        rgb[k].cur_mut().data[c] = f16::from_f32(v.clamp(0.0, 255.0));
                    }
                }
            }
        }
    }
}

/// The per-colour transform of a colour filter (0–255 in and out).
fn color_map(f: &Filter) -> Box<dyn Fn([f32; 3]) -> [f32; 3] + '_> {
    match f {
        Filter::BrightnessContrast { brightness, contrast } => {
            let (b, k) = (brightness * 255.0, (1.0 + contrast.clamp(-1.0, 0.99)) / (1.0 - contrast.clamp(-1.0, 0.99)).max(0.01));
            Box::new(move |c| c.map(|v| (v - 127.5) * k + 127.5 + b))
        }
        Filter::Levels { channel, in_black, in_white, gamma, out_black, out_white } => Box::new(move |c| {
            let mut o = c;
            for (k, v) in o.iter_mut().enumerate() {
                if channel.applies(k) {
                    let t = ((*v - in_black) / (in_white - in_black).max(1e-3)).clamp(0.0, 1.0);
                    *v = out_black + (out_white - out_black) * t.powf(1.0 / gamma.max(0.01));
                }
            }
            o
        }),
        Filter::Curves { channel, points } => Box::new(move |c| {
            let mut o = c;
            for (k, v) in o.iter_mut().enumerate() {
                if channel.applies(k) {
                    *v = 255.0 * curve(points, *v / 255.0);
                }
            }
            o
        }),
        Filter::HueSaturation { hue, saturation, lightness } => Box::new(move |c| {
            let (h, s, l) = to_hsl(c);
            let s = if *saturation >= 0.0 { s + (1.0 - s) * saturation * s.signum() } else { s * (1.0 + saturation) };
            let l = if *lightness >= 0.0 { l + (1.0 - l) * lightness } else { l * (1.0 + lightness) };
            from_hsl((h + hue).rem_euclid(360.0), s.clamp(0.0, 1.0), l.clamp(0.0, 1.0))
        }),
        Filter::ColorBalance { shadows, midtones, highlights, preserve_luminosity } => Box::new(move |c| {
            let l = luma(c) / 255.0;
            // tonal weights: shadows fall off over the lower third, highlights over the upper
            let ws = (1.0 - l / 0.5).clamp(0.0, 1.0);
            let wh = ((l - 0.5) / 0.5).clamp(0.0, 1.0);
            let wm = 1.0 - ws - wh;
            let mut o = [0, 1, 2].map(|k| c[k] + 255.0 * 0.5 * (ws * shadows[k] + wm * midtones[k] + wh * highlights[k]));
            if *preserve_luminosity {
                let d = luma(c) - luma(o);
                o = o.map(|v| v + d);
            }
            o
        }),
        Filter::ColorFilter { color, strength } => Box::new(move |c| {
            let m = [0, 1, 2].map(|k| c[k] * color[k] / 255.0);
            let d = luma(c) - luma(m);
            [0, 1, 2].map(|k| c[k] + strength * (m[k] + d - c[k]))
        }),
        Filter::Colorize { hue, saturation, strength } => Box::new(move |c| {
            let l = luma(c) / 255.0;
            let t = from_hsl(hue.rem_euclid(360.0), saturation.clamp(0.0, 1.0), l);
            [0, 1, 2].map(|k| c[k] + strength * (t[k] - c[k]))
        }),
        Filter::Desaturate { luminosity } => Box::new(move |c| {
            let g = if *luminosity { luma(c) } else { (c[0] + c[1] + c[2]) / 3.0 };
            [g; 3]
        }),
        Filter::Invert => Box::new(|c| c.map(|v| 255.0 - v)),
        Filter::GradientMap { stops } => Box::new(move |c| gradient(stops, luma(c) / 255.0)),
        _ => Box::new(|c| c),
    }
}

/// Rec. 601 luma, 0–255.
fn luma(c: [f32; 3]) -> f32 {
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

/// Smooth curve through sorted control points (identity without points).
fn curve(points: &[(f32, f32)], x: f32) -> f32 {
    let mut p: Vec<(f32, f32)> = points.to_vec();
    if p.is_empty() {
        return x;
    }
    p.sort_by(|a, b| a.0.total_cmp(&b.0));
    if x <= p[0].0 {
        return p[0].1;
    }
    if x >= p[p.len() - 1].0 {
        return p[p.len() - 1].1;
    }
    let i = p.windows(2).position(|w| x <= w[1].0).unwrap_or(0);
    let ((x0, y0), (x1, y1)) = (p[i], p[i + 1]);
    let t = ((x - x0) / (x1 - x0).max(1e-6)).clamp(0.0, 1.0);
    y0 + (y1 - y0) * t * t * (3.0 - 2.0 * t)
}

/// Colour at position `t` along a gradient of sorted stops.
fn gradient(stops: &[(f32, [f32; 3])], t: f32) -> [f32; 3] {
    let mut s = stops.to_vec();
    match s.len() {
        0 => return [255.0 * t; 3],
        1 => return s[0].1,
        _ => {}
    }
    s.sort_by(|a, b| a.0.total_cmp(&b.0));
    if t <= s[0].0 {
        return s[0].1;
    }
    if t >= s[s.len() - 1].0 {
        return s[s.len() - 1].1;
    }
    let i = s.windows(2).position(|w| t <= w[1].0).unwrap_or(0);
    let ((t0, c0), (t1, c1)) = (s[i], s[i + 1]);
    let u = ((t - t0) / (t1 - t0).max(1e-6)).clamp(0.0, 1.0);
    [0, 1, 2].map(|k| c0[k] + (c1[k] - c0[k]) * u)
}

fn to_hsl(c: [f32; 3]) -> (f32, f32, f32) {
    let [r, g, b] = c.map(|v| (v / 255.0).clamp(0.0, 1.0));
    let (max, min) = (r.max(g).max(b), r.min(g).min(b));
    let l = 0.5 * (max + min);
    let d = max - min;
    if d < 1e-6 {
        return (0.0, 0.0, l);
    }
    let s = d / (1.0 - (2.0 * l - 1.0).abs()).max(1e-6);
    let h = if max == r {
        60.0 * ((g - b) / d).rem_euclid(6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    (h, s, l)
}

fn from_hsl(h: f32, s: f32, l: f32) -> [f32; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
    let m = l - 0.5 * c;
    let (r, g, b) = match (h / 60.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r, g, b].map(|v| 255.0 * (v + m))
}

/// Colour-to-alpha at cell `c`: split each pigment layer's colour into white (or black)
/// plus a more saturated colour at lower coverage; `s` blends with the selection.
fn to_alpha(layer: &mut WetLayer, c: usize, white: bool, s: f32) {
    let lut = layer.lut.clone();
    for (rgb, amount) in [(&mut layer.mid.rgb, &mut layer.mid.amount), (&mut layer.dens.rgb, &mut layer.dens.amount)] {
        let a0 = amount.cur().data[c];
        if a0 <= 0.0 {
            continue;
        }
        let col = [0, 1, 2].map(|k| rgb[k].cur().data[c].to_f32() / 255.0);
        let alpha = if white {
            col.iter().map(|v| 1.0 - v).fold(0.0f32, f32::max)
        } else {
            col.iter().copied().fold(0.0f32, f32::max)
        };
        let alpha = 1.0 - s * (1.0 - alpha);
        let new_col = if alpha > 1e-4 {
            if white {
                col.map(|v| 1.0 - (1.0 - v) / alpha)
            } else {
                col.map(|v| v / alpha)
            }
        } else {
            col
        };
        for k in 0..3 {
            rgb[k].cur_mut().data[c] = f16::from_f32((255.0 * new_col[k]).clamp(0.0, 255.0));
        }
        amount.cur_mut().data[c] = OpacityLut::inverse(lut.eval(a0) * alpha).min(a0);
    }
}

/// Apply a spatial filter `op` to a layer: pigment amount and amount-weighted colour of
/// both pigment layers (and paint thickness), blended through the selection.
fn spatial(layer: &mut WetLayer, selection: Option<&[u8]>, op: impl Fn(&[f32], usize, usize) -> Vec<f32>) {
    let (w, h) = (layer.w, layer.h);
    let n = w * h;
    let sel = |c: usize| selection.map_or(1.0, |s| s[c] as f32 / 255.0);
    for (rgb, amount, thick) in [
        (&mut layer.mid.rgb, &mut layer.mid.amount, &mut layer.mid.thickness),
        (&mut layer.dens.rgb, &mut layer.dens.amount, &mut layer.dens.thickness),
    ] {
        let a: Vec<f32> = amount.cur().data.clone();
        let a2 = op(&a, w, h);
        for k in 0..3 {
            let weighted: Vec<f32> = (0..n).map(|c| rgb[k].cur().data[c].to_f32() * a[c]).collect();
            let wb = op(&weighted, w, h);
            for c in 0..n {
                let s = sel(c);
                if s <= 0.0 || a2[c] <= 1e-6 {
                    continue;
                }
                let old = rgb[k].cur().data[c].to_f32();
                let new = (wb[c] / a2[c]).clamp(0.0, 255.0);
                rgb[k].cur_mut().data[c] = f16::from_f32(old + s * (new - old));
            }
        }
        let t: Vec<f32> = thick.cur().data.clone();
        let t2 = op(&t, w, h);
        for c in 0..n {
            let s = sel(c);
            amount.cur_mut().data[c] = (a[c] + s * (a2[c] - a[c])).max(0.0);
            thick.cur_mut().data[c] = (t[c] + s * (t2[c] - t[c])).max(0.0);
        }
    }
}

/// Separable Gaussian blur with standard deviation `radius / 3` (clamped edges).
pub fn gaussian(p: &[f32], w: usize, h: usize, radius: f32) -> Vec<f32> {
    let sigma = (radius / 3.0).max(1e-3);
    let r = radius.ceil().max(0.0) as i64;
    if r == 0 {
        return p.to_vec();
    }
    let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let norm: f32 = k.iter().sum();
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (j, kv) in k.iter().enumerate() {
                    let d = j as i64 - r;
                    let (sx, sy) = if horizontal {
                        ((x as i64 + d).clamp(0, w as i64 - 1) as usize, y)
                    } else {
                        (x, (y as i64 + d).clamp(0, h as i64 - 1) as usize)
                    };
                    acc += kv * src[sy * w + sx];
                }
                out[y * w + x] = acc / norm;
            }
        }
        out
    };
    pass(&pass(p, true), false)
}

/// Disc blur of diameter `size` where bright colours weigh more (specular highlights).
fn lens_blur(layer: &mut WetLayer, selection: Option<&[u8]>, size: f32, threshold: f32, brightness: f32) {
    let r = (0.5 * size).max(0.5);
    // per-cell weight from the displayed brightness of the deposited colour
    let n = layer.w * layer.h;
    let bright: Vec<f32> = (0..n)
        .map(|c| {
            let col = [0, 1, 2].map(|k| layer.mid.rgb[k].cur().data[c].to_f32());
            if luma(col) > threshold { 1.0 + brightness } else { 1.0 }
        })
        .collect();
    spatial(layer, selection, |p, w, h| {
        let ri = r.ceil() as i64;
        let mut out = vec![0.0; w * h];
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let (mut acc, mut wsum) = (0.0, 0.0);
                for dy in -ri..=ri {
                    for dx in -ri..=ri {
                        if (dx * dx + dy * dy) as f32 > r * r {
                            continue;
                        }
                        let (sx, sy) = ((x + dx).clamp(0, w as i64 - 1), (y + dy).clamp(0, h as i64 - 1));
                        let c = (sy * w as i64 + sx) as usize;
                        acc += bright[c] * p[c];
                        wsum += bright[c];
                    }
                }
                out[(y * w as i64 + x) as usize] = acc / wsum.max(1e-6);
            }
        }
        out
    });
}

/// Unsharp mask on pigment amounts and colours.
fn sharpen(layer: &mut WetLayer, selection: Option<&[u8]>, radius: f32, strength: f32) {
    let k = 2.0 * strength.clamp(0.0, 1.0);
    spatial(layer, selection, |p, w, h| {
        let b = gaussian(p, w, h, radius.max(0.5));
        p.iter().zip(&b).map(|(v, bv)| v + k * (v - bv)).collect()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn painted() -> WetLayer {
        let mut l = WetLayer::new(32, 32);
        for y in 8..24 {
            for x in 8..24 {
                let c = y * 32 + x;
                l.mid.amount.cur_mut().data[c] = 600.0;
                for (k, v) in [200.0, 60.0, 40.0].into_iter().enumerate() {
                    l.mid.rgb[k].cur_mut().data[c] = f16::from_f32(v);
                }
            }
        }
        l
    }

    fn rgb(l: &WetLayer, c: usize) -> [f32; 3] {
        [0, 1, 2].map(|k| l.mid.rgb[k].cur().data[c].to_f32())
    }

    #[test]
    fn invert_and_desaturate() {
        let mut l = painted();
        apply(&mut l, &Filter::Invert, None);
        let c = 16 * 32 + 16;
        assert_eq!(rgb(&l, c), [55.0, 195.0, 215.0]);
        apply(&mut l, &Filter::Desaturate { luminosity: false }, None);
        let g = rgb(&l, c);
        assert!((g[0] - g[1]).abs() < 0.5 && (g[1] - g[2]).abs() < 0.5);
    }

    #[test]
    fn hue_shift_turns_red_towards_green() {
        let mut l = painted();
        apply(&mut l, &Filter::HueSaturation { hue: 120.0, saturation: 0.0, lightness: 0.0 }, None);
        let c = rgb(&l, 16 * 32 + 16);
        assert!(c[1] > c[0] && c[1] > c[2], "{c:?}");
    }

    #[test]
    fn selection_limits_the_filter() {
        let mut l = painted();
        let mut sel = vec![0u8; 32 * 32];
        for c in 0..32 * 16 {
            sel[c] = 255;
        }
        apply(&mut l, &Filter::Invert, Some(&sel));
        assert_eq!(rgb(&l, 10 * 32 + 16), [55.0, 195.0, 215.0]);
        assert_eq!(rgb(&l, 20 * 32 + 16), [200.0, 60.0, 40.0]);
    }

    #[test]
    fn blur_spreads_pigment_without_changing_colour() {
        let mut l = painted();
        let total: f32 = l.mid.amount.cur().data.iter().sum();
        apply(&mut l, &Filter::GaussianBlur { radius: 4.0 }, None);
        let after: f32 = l.mid.amount.cur().data.iter().sum();
        assert!((after - total).abs() / total < 0.02);
        let edge = 7 * 32 + 16;
        assert!(l.mid.amount.cur().data[edge] > 0.0);
        let c = rgb(&l, edge);
        assert!((c[0] - 200.0).abs() < 1.0 && (c[1] - 60.0).abs() < 1.0, "{c:?}");
    }

    #[test]
    fn white_to_alpha_thins_light_paint() {
        let mut l = painted();
        let c = 16 * 32 + 16;
        for k in 0..3 {
            l.mid.rgb[k].cur_mut().data[c] = f16::from_f32(240.0);
        }
        let a0 = l.mid.amount.cur().data[c];
        apply(&mut l, &Filter::WhiteToAlpha, None);
        assert!(l.mid.amount.cur().data[c] < 0.2 * a0);
        // a strong colour keeps most of its paint (its white share goes: alpha 0.84 for
        // 200/60/40) and becomes more saturated
        let red = 16 * 32 + 10;
        assert!(l.mid.amount.cur().data[red] > 0.6 * a0);
        let c = rgb(&l, red);
        assert!(c[1] < 60.0 && c[2] < 40.0, "{c:?}");
    }

    #[test]
    fn levels_and_curves() {
        let mut l = painted();
        apply(&mut l, &Filter::Levels { channel: Channel::Red, in_black: 0.0, in_white: 100.0, gamma: 1.0, out_black: 0.0, out_white: 255.0 }, None);
        let c = rgb(&l, 16 * 32 + 16);
        assert_eq!(c[0], 255.0);
        assert_eq!(c[1], 60.0);
        apply(&mut l, &Filter::Curves { channel: Channel::Rgb, points: vec![(0.0, 1.0), (1.0, 0.0)] }, None);
        assert_eq!(rgb(&l, 16 * 32 + 16)[0], 0.0);
    }
}
