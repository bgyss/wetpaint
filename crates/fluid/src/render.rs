//! Provisional display compositing.
//!
//! Stacks the deposited layer and then the suspended layer over the paper colour using
//! the opacity LUT coverages and the [`mix`] rule.

use crate::grid::ld;
use crate::layer::WetLayer;
use crate::mix::{Rgb, mix};
use rayon::prelude::*;

/// Render the layer on its own: straight colour (0..1) and coverage alpha (the union of the
/// settled and suspended coverages), for compositing in a layer stack.
pub fn render_layer(layer: &WetLayer) -> Vec<[f32; 4]> {
    let mid_rgb = layer.mid.rgb.each_ref().map(|b| b.cur());
    let dens_rgb = layer.dens.rgb.each_ref().map(|b| b.cur());
    let (ma, da) = (layer.mid.amount.cur(), layer.dens.amount.cur());
    let pig = layer.pigment.as_deref();
    (0..layer.w * layer.h)
        .into_par_iter()
        .map(|c| {
            let (lm, ld_) = (layer.lut.eval(ma.data[c]), layer.lut.eval(da.data[c]));
            let alpha = ld_ + lm * (1.0 - ld_);
            if alpha <= 0.0 {
                return [0.0; 4];
            }
            let mc: Rgb = mid_rgb.map(|p| ld(p, c));
            let dc: Rgb = dens_rgb.map(|p| ld(p, c));
            let col = mix(mc, lm, dc, ld_, pig);
            [col[0] / 255.0, col[1] / 255.0, col[2] / 255.0, alpha]
        })
        .collect()
}

/// Render to straight RGBA8 over an opaque `paper` colour.
pub fn render_rgba(layer: &WetLayer, paper: Rgb) -> Vec<u8> {
    let mut out = Vec::with_capacity(layer.w * layer.h * 4);
    let mid_rgb = layer.mid.rgb.each_ref().map(|b| b.cur());
    let dens_rgb = layer.dens.rgb.each_ref().map(|b| b.cur());
    let (ma, da) = (layer.mid.amount.cur(), layer.dens.amount.cur());
    let pig = layer.pigment.as_deref();
    for c in 0..layer.w * layer.h {
        let mc: Rgb = mid_rgb.map(|p| ld(p, c));
        let dc: Rgb = dens_rgb.map(|p| ld(p, c));
        let col = mix(paper, 1.0, mc, layer.lut.eval(ma.data[c]), pig);
        let col = mix(col, 1.0, dc, layer.lut.eval(da.data[c]), pig);
        out.extend(col.map(|v| v.round().clamp(0.0, 255.0) as u8));
        out.push(255);
    }
    out
}

/// Impasto lighting from paint thickness (settled + suspended), light from the upper
/// left: a multiplicative shading factor per cell, 1 on flat paint. `strength` scales
/// the relief (UI "impasto depth" 0–10 maps to `strength = depth / 5`).
/// Specular highlight (0–1, to add to the colour) of the paint relief, for a glossy
/// "real shader" look: Blinn–Phong with light from the upper left and the viewer above.
/// Wet paint shines more than dry.
pub fn gloss_highlight(layer: &WetLayer, strength: f32, gloss: f32) -> Vec<f32> {
    let (w, h) = (layer.w, layer.h);
    let (mt, dt) = (layer.mid.thickness.cur(), layer.dens.thickness.cur());
    let height: Vec<f32> = (0..w * h).into_par_iter().map(|c| ((mt.data[c] + dt.data[c]) / 400.0).min(4.0)).collect();
    let at = |x: usize, y: usize| height[y.min(h - 1) * w + x.min(w - 1)];
    let half = {
        let (l, v) = ([-1.0f32, -1.0, 1.5], [0.0f32, 0.0, 1.0]);
        let ln = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        let hv = [l[0] / ln + v[0], l[1] / ln + v[1], l[2] / ln + v[2]];
        let n = (hv[0] * hv[0] + hv[1] * hv[1] + hv[2] * hv[2]).sqrt();
        hv.map(|c| c / n)
    };
    let water = layer.water.cur();
    (0..w * h)
        .into_par_iter()
        .map(|c| {
            if height[c] <= 0.0 {
                return 0.0;
            }
            let (x, y) = (c % w, c / w);
            let k = strength * 0.5;
            let (dx, dy) = (k * (at(x + 1, y) - at(x.saturating_sub(1), y)), k * (at(x, y + 1) - at(x, y.saturating_sub(1))));
            let n = (dx * dx + dy * dy + 1.0).sqrt();
            let ndoth = ((-dx * half[0] - dy * half[1] + half[2]) / n).max(0.0);
            let wet = (water.data[c].to_f32() / 2.0).clamp(0.0, 1.0);
            gloss * (0.4 + 0.6 * wet) * ndoth.powf(40.0)
        })
        .collect()
}

pub fn impasto_shade(layer: &WetLayer, strength: f32) -> Vec<f32> {
    let (w, h) = (layer.w, layer.h);
    let (mt, dt) = (layer.mid.thickness.cur(), layer.dens.thickness.cur());
    // thickness → height with a soft response so heavy paint does not explode the slope
    let height: Vec<f32> = (0..w * h).into_par_iter().map(|c| ((mt.data[c] + dt.data[c]) / 400.0).min(4.0)).collect();
    let at = |x: usize, y: usize| height[y.min(h - 1) * w + x.min(w - 1)];
    (0..w * h)
        .into_par_iter()
        .map(|c| {
            let (x, y) = (c % w, c / w);
            let dx = at(x + 1, y) - at(x.saturating_sub(1), y);
            let dy = at(x, y + 1) - at(x, y.saturating_sub(1));
            (1.0 - strength * 0.12 * (dx + dy)).clamp(0.6, 1.4)
        })
        .collect()
}
