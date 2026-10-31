//! Tools and media: what a dab does to a cell.

use wet_fluid::{
    WetLayer,
    grid::{ld, st},
    mix::{Rgb, mix},
};

/// Paint media.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Medium {
    Watercolor,
    Ink,
    Oil,
    ExpressOil,
    Pastel,
    Pencil,
    Marker,
    Airbrush,
}

/// Where a medium's paint goes and how it behaves.
#[derive(Clone, Copy, Debug)]
pub struct MediumModel {
    /// Wet media deposit suspended pigment with water; the rest deposit straight into the
    /// settled layer and never flow.
    pub wet: bool,
    /// Target amount at opacity 1 (coverage through the opacity LUT).
    pub amount: f32,
    /// Paint thickness added per unit amount (oils: keeps paint above the flow limit).
    pub thickness: f32,
    /// Fraction of pre-existing wet paint picked up under a fully covering stroke.
    pub pick: f32,
    /// Target amount per unit `splat_opacity` (alternative model, see
    /// `Calibration::splat_scales_target`): `amount / splat_opacity` of the reference brush.
    pub amount_per_splat: f32,
    /// Whether `splat_opacity` scales the target amount (watercolour, pencil, marker) or
    /// only the deposit rate (ink, airbrush, oils, pastel). Chosen per medium.
    pub splat_target: bool,
    /// The opacity slider acts as `min(1, gain·opacity)` (oils: low-opacity strokes cover
    /// more than their opacity).
    pub opacity_gain: f32,
}

/// Oil opacity response: coverage at opacity `o` is the linear response at ≈ 1.45·`o`.
const OIL_OPACITY_GAIN: f32 = 1.45;

impl Medium {
    const IDS: [(Self, &str); 8] = [
        (Self::Watercolor, "watercolor"),
        (Self::Ink, "ink"),
        (Self::Oil, "oil"),
        (Self::ExpressOil, "express_oil"),
        (Self::Pastel, "pastel"),
        (Self::Pencil, "pencil"),
        (Self::Marker, "marker"),
        (Self::Airbrush, "airbrush"),
    ];

    /// The medium's preset tool id (`watercolor`, `express_oil`, …).
    pub fn id(self) -> &'static str {
        Self::IDS.iter().find(|(m, _)| *m == self).map_or("watercolor", |(_, id)| id)
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::IDS.iter().find(|(_, s)| *s == id).map(|(m, _)| *m)
    }

    /// Per-medium defaults, fitted to the fresh-stroke coverage at opacity 100 of one
    /// reference brush per medium.
    pub fn model(self) -> MediumModel {
        // (wet, amount, thickness, pick, splat opacity of the reference preset, splat_target)
        let m = |wet, amount: f32, thickness, pick, splat: f32, splat_target| MediumModel {
            wet,
            amount,
            thickness,
            pick,
            amount_per_splat: amount / splat,
            splat_target,
            opacity_gain: 1.0,
        };
        match self {
            Self::Watercolor => m(true, 215.0, 0.0, 0.29, 0.5, true),
            // ink, oil, express oil and airbrush fitted at sizes 30 and 60
            Self::Ink => m(true, 1400.0, 0.0, 0.29, 1.0, false),
            Self::Oil => MediumModel { opacity_gain: OIL_OPACITY_GAIN, ..m(false, 1560.0, 0.5, 0.0, 6.0, false) },
            Self::ExpressOil => MediumModel { opacity_gain: OIL_OPACITY_GAIN, ..m(false, 1440.0, 0.5, 0.0, 20.0, false) },
            // splat opacity scales the target for pastels too
            Self::Pastel => m(false, 3000.0, 0.0, 0.0, 4.0, true),
            // pencil, marker and airbrush amounts fitted with tip prefiltering
            Self::Pencil => m(false, 1625.0, 0.0, 0.0, 1.4, true),
            Self::Marker => m(false, 2100.0, 0.0, 0.0, 1.3, true),
            Self::Airbrush => m(false, 528.0, 0.0, 0.0, 0.5, false),
        }
    }
}

/// What the brush does (`BrushID`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tool {
    Paint(Medium),
    /// Adds water and opens the wet mask (re-wets dried paint).
    Water,
    /// Removes water under the brush.
    Dry,
    /// Removes paint (suspended and settled) under the brush.
    Eraser,
    /// Averages paint within the dab.
    Blend,
    /// Carries paint along the stroke.
    Smudge,
    /// Pushes the water in the stroke direction.
    Blow,
    /// Copies paint from a fixed offset.
    Clone { dx: f32, dy: f32 },
    /// Warps the paint under the dab (Smudge & Liquify group).
    Liquify(Liquify),
}

/// Liquify modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Liquify {
    /// Drags paint along the stroke.
    Push,
    /// Rotates paint around the dab centre.
    Twirl,
    /// Pulls paint towards the centre.
    Pucker,
    /// Pushes paint away from the centre.
    Bloat,
    /// Restores the paint present when the stroke began.
    Reconstruct,
}

/// Bilinear sample of a cell state (amounts interpolated, colours amount-weighted).
pub fn sample_paint(l: &WetLayer, x: f32, y: f32) -> CellPaint {
    let (w, h) = (l.w, l.h);
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let top = lerp_paint(&read_cell(l, y0 * w + x0), &read_cell(l, y0 * w + x1), fx);
    let bot = lerp_paint(&read_cell(l, y1 * w + x0), &read_cell(l, y1 * w + x1), fx);
    lerp_paint(&top, &bot, fy)
}

impl Tool {
    /// The tool's preset id: the medium's for paint tools, else `water`, `dry`, `eraser`,
    /// `blend`, `smudge`, `blow`, `clone` or `liquify`.
    pub fn id(self) -> &'static str {
        match self {
            Self::Paint(m) => m.id(),
            Self::Water => "water",
            Self::Dry => "dry",
            Self::Eraser => "eraser",
            Self::Blend => "blend",
            Self::Smudge => "smudge",
            Self::Blow => "blow",
            Self::Clone { .. } => "clone",
            Self::Liquify(_) => "liquify",
        }
    }

    /// The tool for a preset id; unknown ids paint watercolour.
    pub fn from_id(id: &str) -> Self {
        match id {
            "water" => Self::Water,
            "dry" => Self::Dry,
            "eraser" => Self::Eraser,
            "blend" => Self::Blend,
            "smudge" => Self::Smudge,
            "blow" => Self::Blow,
            "clone" => Self::Clone { dx: 0.0, dy: 0.0 },
            "liquify" => Self::Liquify(Liquify::Push),
            _ => Self::Paint(Medium::from_id(id).unwrap_or(Medium::Watercolor)),
        }
    }
}

/// Snapshot of one cell's paint, used by the moving tools.
#[derive(Clone, Copy, Debug, Default)]
pub struct CellPaint {
    pub dens_rgb: Rgb,
    pub dens_a: f32,
    pub mid_rgb: Rgb,
    pub mid_a: f32,
}

pub fn read_cell(l: &WetLayer, c: usize) -> CellPaint {
    CellPaint {
        dens_rgb: [0, 1, 2].map(|k| ld(l.dens.rgb[k].cur(), c)),
        dens_a: l.dens.amount.cur().data[c],
        mid_rgb: [0, 1, 2].map(|k| ld(l.mid.rgb[k].cur(), c)),
        mid_a: l.mid.amount.cur().data[c],
    }
}

pub fn write_cell(l: &mut WetLayer, c: usize, p: &CellPaint) {
    for k in 0..3 {
        st(l.dens.rgb[k].cur_mut(), c, p.dens_rgb[k]);
        st(l.mid.rgb[k].cur_mut(), c, p.mid_rgb[k]);
    }
    l.dens.amount.cur_mut().data[c] = p.dens_a.max(0.0);
    l.mid.amount.cur_mut().data[c] = p.mid_a.max(0.0);
}

/// Linear interpolation of two cell states by `t` (amounts and amount-weighted colours).
pub fn lerp_paint(a: &CellPaint, b: &CellPaint, t: f32) -> CellPaint {
    let lerp_layer = |ca: Rgb, aa: f32, cb: Rgb, ab: f32| {
        let amount = aa + (ab - aa) * t;
        let (wa, wb) = (aa * (1.0 - t), ab * t);
        let rgb = if wa + wb > 0.0 {
            [0, 1, 2].map(|k| (ca[k] * wa + cb[k] * wb) / (wa + wb))
        } else {
            ca
        };
        (rgb, amount)
    };
    let (dens_rgb, dens_a) = lerp_layer(a.dens_rgb, a.dens_a, b.dens_rgb, b.dens_a);
    let (mid_rgb, mid_a) = lerp_layer(a.mid_rgb, a.mid_a, b.mid_rgb, b.mid_a);
    CellPaint { dens_rgb, dens_a, mid_rgb, mid_a }
}

/// Add `amount` of settled paint of `color` (dry media, oils).
pub fn deposit_settled(l: &mut WetLayer, c: usize, color: Rgb, amount: f32, thickness: f32) {
    let amount = amount * l.maps.stencil_at(c);
    if amount <= 0.0 {
        return;
    }
    let old_a = l.mid.amount.cur().data[c];
    let old: Rgb = [0, 1, 2].map(|k| ld(l.mid.rgb[k].cur(), c));
    let new = mix(old, l.lut.eval(old_a), color, l.lut.eval(amount), l.pigment.as_deref());
    for k in 0..3 {
        st(l.mid.rgb[k].cur_mut(), c, new[k]);
    }
    l.mid.amount.cur_mut().data[c] = old_a + amount;
    l.mid.thickness.cur_mut().data[c] += thickness * amount;
}
