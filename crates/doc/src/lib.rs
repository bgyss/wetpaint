//! A painting document: a stack of wet layers over one paper, with undo and an open,
//! lossless file format (`.wetdoc`).
//!
//! Every paint layer is simulated separately (one fluid state per layer) and composited
//! with the layer blend modes. `.wetdoc` is an 8-byte magic, a little-endian `u32`
//! manifest length, a JSON manifest, then the raw little-endian planes listed in the
//! manifest.

mod edit;
pub use edit::iterative_path;
pub mod filters;
pub mod guides;
pub mod metallic;
pub mod palette;
pub mod shadow;
pub mod visual;
pub mod psd;

use std::io::{Read, Write};

use half::f16;
use rayon::prelude::*;
use serde_json::{Value, json};
use wet_compose::{Layer, MixMode, PigmentMix, flatten};
use wet_fluid::{WetLayer, grid::Plane, render::render_layer};
use wet_color::{Converter, RgbSpace};
use wet_paper::{Paper, PaperSpec, Surface};

const MAGIC: &[u8; 8] = b"WETDOC1\n";
const GRAN_SIZE: usize = 1024;

fn gran_tile() -> &'static [u8] {
    static TILE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    TILE.get_or_init(|| wet_paper::granulation_tile(GRAN_SIZE, 9, 1.0))
}

/// One paint layer.
pub struct PaintLayer {
    pub name: String,
    pub wet: WetLayer,
    pub mode: MixMode,
    pub opacity: f32,
    pub visible: bool,
    pub locked: bool,
    /// Clipped to the alpha of the layer below (clipping mask).
    pub clip: bool,
    /// Layer mask (0–255 per pixel) multiplying the layer's alpha.
    pub mask: Option<Vec<u8>>,
    /// A masking-fluid layer: its paint is not shown but protects the other layers
    /// from painting (see [`Document::update_stencils`]).
    pub masking_fluid: bool,
    /// Index into [`Document::groups`]. A group's layers must be adjacent in the stack.
    pub group: Option<usize>,
    /// The reference layer strokes take their colour from ([`Document::tracing_color`]).
    pub tracing: bool,
    /// Pigment mixing (Kubelka–Munk) within the layer and with the layers below; the
    /// layer's blend mode is not used while it is on. Set with
    /// [`Document::set_layer_pigments`].
    pub pigments: bool,
    /// A Metallic Material layer: its paint renders with the metallic sheen
    /// ([`Document::metallic`]).
    pub metallic: bool,
}

/// A layer group: its layers are composited together (isolated), then onto the layers
/// below with the group's mode and opacity.
#[derive(Clone, Debug)]
pub struct LayerGroup {
    pub name: String,
    pub mode: MixMode,
    pub opacity: f32,
    pub visible: bool,
}

impl PaintLayer {
    fn new(name: &str, wet: WetLayer) -> Self {
        Self {
            name: name.into(),
            wet,
            mode: MixMode::Normal,
            opacity: 1.0,
            visible: true,
            locked: false,
            clip: false,
            mask: None,
            masking_fluid: false,
            group: None,
            tracing: false,
            pigments: false,
            metallic: false,
        }
    }
}

/// Paper settings shared by all layers.
#[derive(Clone, Copy, Debug)]
pub struct PaperSettings {
    pub spec: PaperSpec,
    pub color: [f32; 3],
    pub visible: bool,
    /// Display shading and tone (a blank canvas shows ≈ 244).
    pub shading: f32,
    pub tone: f32,
    /// Torn sheet edges: transparent and unpaintable beyond a ragged border.
    pub deckled: bool,
}

impl Default for PaperSettings {
    fn default() -> Self {
        Self { spec: PaperSpec::default(), color: [255.0; 3], visible: true, shading: 0.094, tone: 0.963, deckled: false }
    }
}

/// An undo step as [`Document::history`] lists it: its label and the layer it changed.
pub type HistoryStep<'a> = (&'a str, usize);

/// Saved state of one layer for undo.
#[derive(Clone)]
struct Snapshot {
    layer: usize,
    planes: LayerPlanes,
    /// What the edit was, for a history list ("Watercolor stroke", "Fill", …).
    label: String,
}

/// Copies of the state planes of a `WetLayer`.
#[derive(Clone, Default)]
struct LayerPlanes {
    h16: Vec<Vec<f16>>,
    f32s: Vec<Vec<f32>>,
    open: Vec<u8>,
}

fn h16_planes(l: &mut WetLayer) -> Vec<&mut Plane<f16>> {
    let [dr, dg, db] = &mut l.dens.rgb;
    let [mr, mg, mb] = &mut l.mid.rgb;
    vec![
        l.water.cur_mut(), l.u.cur_mut(), l.v.cur_mut(),
        dr.cur_mut(), dg.cur_mut(), db.cur_mut(),
        mr.cur_mut(), mg.cur_mut(), mb.cur_mut(),
    ]
}

fn f32_planes(l: &mut WetLayer) -> Vec<&mut Plane<f32>> {
    vec![
        l.dens.amount.cur_mut(), l.dens.thickness.cur_mut(),
        l.mid.amount.cur_mut(), l.mid.thickness.cur_mut(),
    ]
}

const H16_NAMES: [&str; 9] = ["water", "u", "v", "dens_r", "dens_g", "dens_b", "mid_r", "mid_g", "mid_b"];
const F32_NAMES: [&str; 4] = ["dens_amount", "dens_thickness", "mid_amount", "mid_thickness"];

impl LayerPlanes {
    fn capture(l: &mut WetLayer) -> Self {
        Self {
            h16: h16_planes(l).into_iter().map(|p| p.data.clone()).collect(),
            f32s: f32_planes(l).into_iter().map(|p| p.data.clone()).collect(),
            open: l.maps.open.data.clone(),
        }
    }

    fn restore(&self, l: &mut WetLayer) {
        for (p, d) in h16_planes(l).into_iter().zip(&self.h16) {
            p.data.clone_from(d);
        }
        for (p, d) in f32_planes(l).into_iter().zip(&self.f32s) {
            p.data.clone_from(d);
        }
        l.maps.open.data.clone_from(&self.open);
    }
}

/// The document.
/// A stencil placed on the canvas: an image whose covered part blocks paint, water,
/// wetting and drying on every paint layer.
#[derive(Clone, Debug)]
pub struct Stencil {
    /// Coverage per stencil pixel (255 = blocks), `w × h`.
    pub mask: Vec<u8>,
    pub w: usize,
    pub h: usize,
    /// Canvas position of the stencil centre, rotation (radians), scale (canvas pixels
    /// per stencil pixel).
    pub centre: (f32, f32),
    pub rotation: f32,
    pub scale: f32,
    pub flip_h: bool,
    pub flip_v: bool,
    /// Reverse which part blocks.
    pub invert: bool,
    /// Also block the canvas outside the stencil's rectangle.
    pub border: bool,
    /// Repeat the stencil over the whole canvas.
    pub tile: bool,
    pub visible: bool,
}

impl Stencil {
    /// A stencil from a coverage image, centred at `centre` at 1:1 scale.
    pub fn new(mask: Vec<u8>, w: usize, h: usize, centre: (f32, f32)) -> Self {
        Self {
            mask,
            w,
            h,
            centre,
            rotation: 0.0,
            scale: 1.0,
            flip_h: false,
            flip_v: false,
            invert: false,
            border: false,
            tile: false,
            visible: true,
        }
    }

    /// How much the stencil blocks canvas pixel `(x, y)` (0–1, bilinear).
    pub fn coverage(&self, x: f32, y: f32) -> f32 {
        let (dx, dy) = (x - self.centre.0, y - self.centre.1);
        let (sn, cs) = self.rotation.sin_cos();
        let mut u = (dx * cs + dy * sn) / self.scale.max(1e-6);
        let mut v = (-dx * sn + dy * cs) / self.scale.max(1e-6);
        if self.flip_h {
            u = -u;
        }
        if self.flip_v {
            v = -v;
        }
        let (mut px, mut py) = (u + 0.5 * self.w as f32 - 0.5, v + 0.5 * self.h as f32 - 0.5);
        let (wf, hf) = (self.w as f32, self.h as f32);
        let inside = px > -0.5 && py > -0.5 && px < wf - 0.5 && py < hf - 0.5;
        let c = if inside || self.tile {
            if self.tile {
                px = px.rem_euclid(wf);
                py = py.rem_euclid(hf);
            }
            let at = |xi: i64, yi: i64| {
                let (xi, yi) = if self.tile {
                    (xi.rem_euclid(self.w as i64), yi.rem_euclid(self.h as i64))
                } else {
                    (xi.clamp(0, self.w as i64 - 1), yi.clamp(0, self.h as i64 - 1))
                };
                self.mask[yi as usize * self.w + xi as usize] as f32 / 255.0
            };
            let (x0, y0) = (px.floor(), py.floor());
            let (fx, fy) = (px - x0, py - y0);
            let (xi, yi) = (x0 as i64, y0 as i64);
            let top = at(xi, yi) * (1.0 - fx) + at(xi + 1, yi) * fx;
            let bot = at(xi, yi + 1) * (1.0 - fx) + at(xi + 1, yi + 1) * fx;
            top * (1.0 - fy) + bot * fy
        } else {
            0.0
        };
        let c = if self.invert { 1.0 - c } else { c };
        // the border blocks outside the rectangle whatever the inversion
        if !inside && !self.tile && self.border { 1.0 } else { c }
    }
}

/// Scale each cell's pigment coverage by its mask value (or by its complement), by
/// reducing the amounts through the inverse opacity LUT.
fn scale_coverage(wet: &mut WetLayer, mask: &[u8], complement: bool) {
    let lut = wet.lut.clone();
    for p in [&mut wet.mid, &mut wet.dens] {
        let amount = p.amount.cur_mut();
        for (a, &m) in amount.data.iter_mut().zip(mask) {
            let k = if complement { 255 - m } else { m };
            if k < 255 && *a > 0.0 {
                *a = wet_fluid::lut::OpacityLut::inverse(lut.eval(*a) * k as f32 / 255.0).min(*a);
            }
        }
    }
}

/// A structure placed on the canvas: a relief image (light = raised) added to the paper
/// height, so it shows in the paper shading and steers the wet paint like paper grain.
/// Placement (move, rotate, scale, flip, invert, tile) works as for stencils.
#[derive(Clone, Debug)]
pub struct Structure {
    pub image: Stencil,
    /// Height added at full image value (paper height spans −1..1).
    pub strength: f32,
}

/// What the fill tool lays down.
#[derive(Clone, Debug)]
pub enum Fill {
    Color([f32; 3]),
    /// Two-colour gradient from `from` to `to` (canvas points), linear or radial.
    Gradient { from: (f32, f32), to: (f32, f32), colors: [[f32; 3]; 2], radial: bool },
    /// A tiled straight-RGBA8 pattern image at `scale`.
    Pattern { rgba: Vec<u8>, w: usize, h: usize, scale: f32 },
}

impl Fill {
    /// Colour (0–255) and alpha (0–1) at canvas pixel `(x, y)`.
    fn color_at(&self, x: f32, y: f32, _w: usize, _h: usize) -> ([f32; 3], f32) {
        match self {
            Fill::Color(c) => (*c, 1.0),
            Fill::Gradient { from, to, colors, radial } => {
                let (dx, dy) = (to.0 - from.0, to.1 - from.1);
                let len2 = (dx * dx + dy * dy).max(1e-6);
                let t = if *radial {
                    (((x - from.0).powi(2) + (y - from.1).powi(2)) / len2).sqrt()
                } else {
                    ((x - from.0) * dx + (y - from.1) * dy) / len2
                }
                .clamp(0.0, 1.0);
                ([0, 1, 2].map(|k| colors[0][k] + (colors[1][k] - colors[0][k]) * t), 1.0)
            }
            Fill::Pattern { rgba, w, h, scale } => {
                let px = ((x / scale.max(1e-3)) as usize) % w.max(&1);
                let py = ((y / scale.max(1e-3)) as usize) % h.max(&1);
                let i = 4 * (py * w + px);
                ([rgba[i] as f32, rgba[i + 1] as f32, rgba[i + 2] as f32], rgba[i + 3] as f32 / 255.0)
            }
        }
    }
}

/// Fill tool options.
#[derive(Clone, Copy, Debug)]
pub struct FillOptions {
    /// 0–255, maximum channel difference from the clicked colour.
    pub tolerance: u8,
    /// 0–1 coverage of the fill.
    pub opacity: f32,
    pub contiguous: bool,
    /// Also wet the filled pixels (the fill is then suspended, wet paint).
    pub wet: bool,
}

impl Default for FillOptions {
    fn default() -> Self {
        Self { tolerance: 32, opacity: 1.0, contiguous: true, wet: false }
    }
}

/// The projective map (row-major 3×3 with h₂₂ = 1, as 8 values) taking the four points
/// `from` to `to`, or `None` for a degenerate quad.
pub fn homography(from: [(f32, f32); 4], to: [(f32, f32); 4]) -> Option<[f32; 8]> {
    let mut m = [[0.0f64; 9]; 8];
    for k in 0..4 {
        let ((x, y), (u, v)) = ((from[k].0 as f64, from[k].1 as f64), (to[k].0 as f64, to[k].1 as f64));
        m[2 * k] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        m[2 * k + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    // Gaussian elimination with partial pivoting
    for col in 0..8 {
        let piv = (col..8).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        for r in 0..8 {
            if r != col {
                let f = m[r][col] / m[col][col];
                for c in col..9 {
                    m[r][c] -= f * m[col][c];
                }
            }
        }
    }
    Some(std::array::from_fn(|i| (m[i][8] / m[i][i]) as f32))
}

/// Where the colour picker samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickSource {
    /// The active layer only (its paint over transparency).
    CurrentLayer,
    /// All visible layers, without the paper.
    AllLayers,
    /// The finished picture, paper included.
    AllLayersAndCanvas,
}

/// How a new selection combines with the current one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectOp {
    Replace,
    Add,
    Subtract,
    Intersect,
}

pub struct Document {
    pub w: usize,
    pub h: usize,
    pub layers: Vec<PaintLayer>,
    pub active: usize,
    pub paper: PaperSettings,
    paper_maps: Paper,
    /// Paper relief shading for the display, by shading strength: it only changes with the
    /// paper, so it is kept between frames (cleared when the relief is rebuilt).
    paper_shade: std::sync::Mutex<Option<(f32, std::sync::Arc<Vec<f32>>)>>,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    pub undo_limit: usize,
    /// Impasto relief strength (UI impasto depth / 5).
    pub impasto: f32,
    /// Glossy relief highlights ("real shader"): highlight strength, 0 = off.
    pub gloss: f32,
    /// SoftShadows strength and light altitude, 0–10 each (0 strength = off).
    pub soft_shadows: f32,
    /// Metallic Material settings for metallic layers.
    pub metallic: metallic::MetallicSettings,
    pub shadow_altitude: f32,
    /// Active selection (0–255; painting is limited to it), or `None` for everything.
    pub selection: Option<Vec<u8>>,
    /// The selection before the last deselect or selection change (Restore Last Selection).
    pub last_selection: Option<Vec<u8>>,
    pub groups: Vec<LayerGroup>,
    /// Stencils on the canvas (applied by [`Document::update_stencils`]).
    pub stencils: Vec<Stencil>,
    /// Structures on the canvas (applied by [`Document::apply_structures`]).
    pub structures: Vec<Structure>,
    /// Sheet coverage when the paper is deckled (1 on the sheet).
    deckle: Option<Vec<f32>>,
    /// Colour profile the document's RGB values are in (`None`: not colour managed).
    pub profile: Option<RgbSpace>,
    /// Grid, guides and snapping (non-printing).
    pub guides: guides::Guides,
    /// Reference images shown beside the canvas (not painted, not exported).
    pub references: Vec<ReferenceImage>,
}

/// A reference image (straight RGBA8) for the reference panel.
#[derive(Clone, Debug)]
pub struct ReferenceImage {
    pub name: String,
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
    pub visible: bool,
}

impl Document {
    pub fn new(w: usize, h: usize) -> Self {
        let paper = PaperSettings::default();
        let mut d = Self {
            w,
            h,
            layers: Vec::new(),
            active: 0,
            paper,
            paper_maps: Paper::generate(w, h, &paper.spec),
            paper_shade: Default::default(),
            undo: Vec::new(),
            redo: Vec::new(),
            undo_limit: 32,
            impasto: 1.0,
            selection: None,
            last_selection: None,
            groups: Vec::new(),
            stencils: Vec::new(),
            structures: Vec::new(),
            deckle: None,
            profile: None,
            guides: guides::Guides::default(),
            references: Vec::new(),
            gloss: 0.0,
            soft_shadows: 0.0,
            metallic: metallic::MetallicSettings::default(),
            shadow_altitude: 5.0,
        };
        d.add_layer("Layer 1");
        d
    }

    /// Regenerate the paper and apply it to every layer.
    pub fn set_paper(&mut self, paper: PaperSettings) {
        self.paper = paper;
        self.deckle = paper
            .deckled
            .then(|| wet_paper::deckle_width(self.w, self.h))
            .flatten()
            .map(|dw| wet_paper::deckle_mask(self.w, self.h, paper.spec.seed, dw));
        self.apply_structures();
        self.update_stencils();
    }

    /// Rebuild the paper relief from the paper settings plus the visible structures and
    /// hand it to every layer (call after changing `structures`).
    pub fn apply_structures(&mut self) {
        self.paper_maps = Paper::generate(self.w, self.h, &self.paper.spec);
        *self.paper_shade.lock().expect("paper shade") = None;
        for st in self.structures.iter().filter(|s| s.image.visible) {
            for (c, hgt) in self.paper_maps.height.iter_mut().enumerate() {
                let v = st.image.coverage((c % self.w) as f32, (c / self.w) as f32);
                *hgt = (*hgt + st.strength * v).clamp(-1.0, 1.0);
            }
        }
        for l in &mut self.layers {
            self.paper_maps.apply(&mut l.wet);
        }
    }

    /// Add a layer above the active one and make it active.
    pub fn add_layer(&mut self, name: &str) -> usize {
        let mut wet = WetLayer::new(self.w, self.h);
        self.paper_maps.apply(&mut wet);
        wet_paper::apply_granulation(&mut wet, gran_tile(), GRAN_SIZE);
        if let Some(l) = self.layers.get(self.active) {
            wet.params = l.wet.params.clone();
        }
        let idx = if self.layers.is_empty() { 0 } else { self.active + 1 };
        self.layers.insert(
            idx,
            PaintLayer::new(name, wet),
        );
        self.active = idx;
        self.undo.clear();
        self.redo.clear();
        idx
    }

    /// Create a group holding the adjacent layers `first..=last`; returns its index.
    pub fn group_layers(&mut self, name: &str, first: usize, last: usize) -> usize {
        let g = self.groups.len();
        self.groups.push(LayerGroup { name: name.into(), mode: MixMode::Normal, opacity: 1.0, visible: true });
        for l in self.layers.iter_mut().take(last + 1).skip(first) {
            l.group = Some(g);
        }
        g
    }

    /// Whether a layer is shown (its own and its group's visibility).
    pub fn layer_shown(&self, l: &PaintLayer) -> bool {
        l.visible && l.group.and_then(|g| self.groups.get(g)).is_none_or(|g| g.visible)
    }

    pub fn remove_layer(&mut self, idx: usize) {
        if self.layers.len() > 1 && idx < self.layers.len() {
            self.layers.remove(idx);
            self.active = self.active.min(self.layers.len() - 1);
            self.undo.clear();
            self.redo.clear();
        }
    }

    pub fn move_layer(&mut self, from: usize, to: usize) {
        if from < self.layers.len() && to < self.layers.len() {
            let l = self.layers.remove(from);
            self.layers.insert(to, l);
            self.active = to;
            self.undo.clear();
            self.redo.clear();
        }
    }

    pub fn active_layer(&mut self) -> &mut WetLayer {
        &mut self.layers[self.active].wet
    }

    /// Record the active layer's state before an edit (call at stroke start).
    pub fn checkpoint(&mut self) {
        self.checkpoint_layer(self.active);
    }

    /// [`Self::checkpoint`] with a name for the edit, shown by [`Self::history`].
    pub fn checkpoint_as(&mut self, label: &str) {
        self.checkpoint_layer_as(self.active, label);
    }

    /// Record layer `idx`'s state before an edit.
    pub fn checkpoint_layer(&mut self, idx: usize) {
        self.checkpoint_layer_as(idx, "Edit");
    }

    pub fn checkpoint_layer_as(&mut self, idx: usize, label: &str) {
        let planes = LayerPlanes::capture(&mut self.layers[idx].wet);
        self.push_undo(Snapshot { layer: idx, planes, label: label.into() });
    }

    /// [`Self::checkpoint_layer_as`] with the layer's state captured elsewhere (a copy from
    /// the GPU): water, u, v, settled r g b, suspended r g b (`h16`); settled amount and
    /// thickness, suspended amount and thickness (`f32s`); the wet mask.
    pub fn checkpoint_planes(&mut self, idx: usize, label: &str, h16: Vec<Vec<f16>>, f32s: Vec<Vec<f32>>, open: Vec<u8>) {
        assert!(h16.len() == H16_NAMES.len() && f32s.len() == F32_NAMES.len(), "checkpoint planes");
        self.push_undo(Snapshot { layer: idx, planes: LayerPlanes { h16, f32s, open }, label: label.into() });
    }

    fn push_undo(&mut self, s: Snapshot) {
        self.undo.push(s);
        if self.undo.len() > self.undo_limit {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn swap_snapshot(&mut self, s: Snapshot) -> Snapshot {
        let l = &mut self.layers[s.layer].wet;
        let now = Snapshot { layer: s.layer, planes: LayerPlanes::capture(l), label: s.label };
        s.planes.restore(l);
        now
    }

    /// The undoable edits, oldest first, and the redoable ones, next first: each as its
    /// label and layer index.
    pub fn history(&self) -> (Vec<HistoryStep<'_>>, Vec<HistoryStep<'_>>) {
        fn f(s: &Snapshot) -> (&str, usize) {
            (s.label.as_str(), s.layer)
        }
        (self.undo.iter().map(f).collect(), self.redo.iter().rev().map(f).collect())
    }

    pub fn undo(&mut self) -> bool {
        let Some(s) = self.undo.pop() else { return false };
        let now = self.swap_snapshot(s);
        self.redo.push(now);
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(s) = self.redo.pop() else { return false };
        let now = self.swap_snapshot(s);
        self.undo.push(now);
        true
    }

    /// Recompute each layer's stencil from the masking-fluid layers: painted masking fluid
    /// protects the other layers (stencil = 1 − min(1, 3·coverage)). Call before
    /// painting.
    pub fn update_stencils(&mut self) {
        let n = self.w * self.h;
        let mut protect = vec![0.0f32; n];
        let mut any = false;
        for l in self.layers.iter().filter(|l| l.masking_fluid && l.visible) {
            any = true;
            for (c, p) in protect.iter_mut().enumerate() {
                let a = l.wet.lut.eval(l.wet.mid.amount.cur().data[c] + l.wet.dens.amount.cur().data[c]);
                // masking fluid protects fully once moderately covered
                *p = p.max((3.0 * a).min(1.0));
            }
        }
        if let Some(d) = &self.deckle {
            any = true;
            for (p, &k) in protect.iter_mut().zip(d) {
                *p = p.max(1.0 - k);
            }
        }
        for s in self.stencils.iter().filter(|s| s.visible) {
            any = true;
            for (c, p) in protect.iter_mut().enumerate() {
                *p = p.max(s.coverage((c % self.w) as f32, (c / self.w) as f32));
            }
        }
        let sel = self.selection.as_ref();
        for l in self.layers.iter_mut().filter(|l| !l.masking_fluid) {
            l.wet.maps.stencil = (any || sel.is_some()).then(|| {
                let mut st = wet_fluid::grid::PlaneU8::new(self.w, self.h, 255);
                for (c, p) in protect.iter().enumerate() {
                    let s = sel.map_or(1.0, |m| m[c] as f32 / 255.0);
                    st.data[c] = ((1.0 - p.min(1.0)) * s * 255.0).round() as u8;
                }
                st
            });
        }
    }

    /// Give a layer a mask: the selection's shape when there is one, else all white.
    pub fn add_layer_mask(&mut self, idx: usize) {
        let m = self.selection.clone().unwrap_or_else(|| vec![255; self.w * self.h]);
        if let Some(l) = self.layers.get_mut(idx) {
            l.mask = Some(m);
        }
    }

    /// Bake a layer's mask into its paint (the visible result is unchanged) and drop the
    /// mask: each pigment amount is reduced so its coverage is scaled by the mask.
    pub fn apply_layer_mask(&mut self, idx: usize) {
        let Some(l) = self.layers.get_mut(idx) else { return };
        let Some(mask) = l.mask.take() else { return };
        scale_coverage(&mut l.wet, &mask, false);
    }

    /// Select everything.
    pub fn select_all(&mut self) {
        self.selection = Some(vec![255; self.w * self.h]);
    }

    /// Select the painted (non-transparent) pixels of a layer, by coverage.
    pub fn select_visible(&mut self, idx: usize, op: SelectOp) {
        let Some(l) = self.layers.get(idx) else { return };
        let l = &l.wet;
        let m = (0..self.w * self.h)
            .map(|c| (l.lut.eval(l.mid.amount.cur().data[c] + l.dens.amount.cur().data[c]).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        self.apply_selection(m, op);
    }

    /// Copy (or cut) the selected part of a layer into a new layer above it; returns the
    /// new layer's index, or `None` without a selection.
    pub fn copy_selection(&mut self, idx: usize, cut: bool) -> Option<usize> {
        let sel = self.selection.clone()?;
        let planes = LayerPlanes::capture(&mut self.layers.get_mut(idx)?.wet);
        self.active = idx;
        let name = format!("{} {}", self.layers[idx].name, if cut { "cut" } else { "copy" });
        let new = self.add_layer(&name);
        let wet = &mut self.layers[new].wet;
        planes.restore(wet);
        scale_coverage(wet, &sel, false);
        for (c, &m) in sel.iter().enumerate() {
            if m == 0 {
                wet.maps.open.data[c] = 0;
                wet.water.cur_mut().data[c] = f16::ZERO;
            }
        }
        if cut {
            scale_coverage(&mut self.layers[idx].wet, &sel, true);
        }
        Some(new)
    }

    /// A stencil from the current selection (the selected area stays paintable).
    pub fn stencil_from_selection(&self) -> Option<Stencil> {
        let sel = self.selection.as_ref()?;
        let mask = sel.iter().map(|&v| 255 - v).collect();
        Some(Stencil::new(mask, self.w, self.h, (self.w as f32 / 2.0, self.h as f32 / 2.0)))
    }

    /// A stencil from a layer's paint coverage (painted areas block).
    pub fn stencil_from_layer(&self, idx: usize) -> Option<Stencil> {
        let l = &self.layers.get(idx)?.wet;
        let mask = (0..self.w * self.h)
            .map(|c| {
                let a = l.lut.eval(l.mid.amount.cur().data[c] + l.dens.amount.cur().data[c]);
                (a.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect();
        Some(Stencil::new(mask, self.w, self.h, (self.w as f32 / 2.0, self.h as f32 / 2.0)))
    }

    /// Enlarge or crop the canvas to `new_w × new_h`, placing the old canvas's top-left
    /// corner at `offset` (may be negative to crop). New area is blank paper.
    pub fn resize_canvas(&mut self, new_w: usize, new_h: usize, offset: (i64, i64)) {
        self.reshape(new_w, new_h, |x, y| {
            let (sx, sy) = (x as i64 - offset.0, y as i64 - offset.1);
            Some((sx as f32, sy as f32))
        }, false);
    }

    /// Resample the whole image to `new_w × new_h` (bilinear; masks nearest).
    pub fn resize_image(&mut self, new_w: usize, new_h: usize) {
        let (kx, ky) = (self.w as f32 / new_w as f32, self.h as f32 / new_h as f32);
        self.reshape(new_w, new_h, |x, y| Some(((x as f32 + 0.5) * kx - 0.5, (y as f32 + 0.5) * ky - 0.5)), true);
    }

    /// Rebuild every layer at a new size, reading old cell `map(x, y)` (bilinear when
    /// `smooth`, else nearest; cells mapping outside the old canvas stay blank).
    fn reshape(&mut self, new_w: usize, new_h: usize, map: impl Fn(usize, usize) -> Option<(f32, f32)>, smooth: bool) {
        let (ow, oh) = (self.w, self.h);
        let src_of = |x: usize, y: usize| -> Option<(f32, f32)> {
            let (sx, sy) = map(x, y)?;
            (sx > -0.5 && sy > -0.5 && sx < ow as f32 - 0.5 && sy < oh as f32 - 0.5).then_some((sx, sy))
        };
        let sample = |d: &dyn Fn(usize) -> f32, sx: f32, sy: f32| -> f32 {
            if !smooth {
                let (xi, yi) = (sx.round().clamp(0.0, ow as f32 - 1.0) as usize, sy.round().clamp(0.0, oh as f32 - 1.0) as usize);
                return d(yi * ow + xi);
            }
            let (x0, y0) = (sx.floor().max(0.0), sy.floor().max(0.0));
            let (fx, fy) = ((sx - x0).clamp(0.0, 1.0), (sy - y0).clamp(0.0, 1.0));
            let (xi, yi) = (x0 as usize, y0 as usize);
            let (xj, yj) = ((xi + 1).min(ow - 1), (yi + 1).min(oh - 1));
            let top = d(yi * ow + xi) * (1.0 - fx) + d(yi * ow + xj) * fx;
            let bot = d(yj * ow + xi) * (1.0 - fx) + d(yj * ow + xj) * fx;
            top * (1.0 - fy) + bot * fy
        };
        let cells: Vec<Option<(f32, f32)>> = (0..new_w * new_h).map(|c| src_of(c % new_w, c / new_w)).collect();
        let near = |sx: f32, sy: f32| sy.round().clamp(0.0, oh as f32 - 1.0) as usize * ow + sx.round().clamp(0.0, ow as f32 - 1.0) as usize;
        self.w = new_w;
        self.h = new_h;
        for l in &mut self.layers {
            let old = LayerPlanes::capture(&mut l.wet);
            let mut wet = WetLayer::new(new_w, new_h);
            wet.params = l.wet.params.clone();
            wet.pigment = l.wet.pigment.take();
            wet.step_index = l.wet.step_index;
            let mut new = LayerPlanes::capture(&mut wet);
            for (dst, src) in new.h16.iter_mut().zip(&old.h16) {
                for (c, cell) in cells.iter().enumerate() {
                    if let Some((sx, sy)) = *cell {
                        dst[c] = f16::from_f32(sample(&|i| src[i].to_f32(), sx, sy));
                    }
                }
            }
            for (dst, src) in new.f32s.iter_mut().zip(&old.f32s) {
                for (c, cell) in cells.iter().enumerate() {
                    if let Some((sx, sy)) = *cell {
                        dst[c] = sample(&|i| src[i], sx, sy);
                    }
                }
            }
            for (c, cell) in cells.iter().enumerate() {
                if let Some((sx, sy)) = *cell {
                    new.open[c] = old.open[near(sx, sy)];
                }
            }
            new.restore(&mut wet);
            l.mask = l.mask.take().map(|m| {
                cells.iter().map(|cell| cell.map_or(255, |(sx, sy)| m[near(sx, sy)])).collect()
            });
            l.wet = wet;
        }
        self.selection = None;
        self.undo.clear();
        self.redo.clear();
        let paper = self.paper;
        self.set_paper(paper);
        for l in &mut self.layers {
            wet_paper::apply_granulation(&mut l.wet, gran_tile(), GRAN_SIZE);
        }
    }

    /// Tag the document with a profile (or untag it) without changing its values.
    pub fn assign_profile(&mut self, profile: Option<RgbSpace>) {
        self.profile = profile;
    }

    /// Convert the document's colours (paint and paper) from its profile to `to` and tag
    /// it with `to`. An untagged document is treated as sRGB.
    pub fn convert_to_profile(&mut self, to: RgbSpace) {
        let cv = Converter::new(self.profile.unwrap_or(RgbSpace::Srgb), to);
        let conv = |c: [f32; 3]| cv.convert(c.map(|v| v / 255.0)).map(|v| v * 255.0);
        for l in &mut self.layers {
            for p in [&mut l.wet.mid, &mut l.wet.dens] {
                for c in 0..self.w * self.h {
                    let rgb = [0, 1, 2].map(|k| p.rgb[k].cur().data[c].to_f32());
                    let out = conv(rgb);
                    for k in 0..3 {
                        p.rgb[k].cur_mut().data[c] = f16::from_f32(out[k]);
                    }
                }
            }
        }
        self.paper.color = conv(self.paper.color);
        self.profile = Some(to);
    }

    /// Soft proof: the flattened canvas as it would look in `proof` (e.g. an installed
    /// ICC profile via [`wet_color::Space::from_icc`]), plus a per-pixel gamut warning.
    /// The document is read in its profile (sRGB when untagged).
    pub fn soft_proof(&self, proof: &wet_color::Space) -> (Vec<u8>, Vec<bool>) {
        let working: wet_color::Space = self.profile.unwrap_or(RgbSpace::Srgb).into();
        let img = self.render(None);
        let mut out = img.clone();
        let mut warn = vec![false; self.w * self.h];
        for (i, px) in img.chunks_exact(4).enumerate() {
            let (c, outside) = wet_color::soft_proof(&working, proof, [0, 1, 2].map(|k| px[k] as f32 / 255.0));
            for k in 0..3 {
                out[4 * i + k] = (c[k] * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            warn[i] = outside;
        }
        (out, warn)
    }

    /// Export the flattened canvas as a PNG, with the document's ICC profile embedded.
    pub fn export_png(&self, out: impl Write) -> std::io::Result<()> {
        let mut info = png::Info::with_size(self.w as u32, self.h as u32);
        info.color_type = png::ColorType::Rgba;
        info.bit_depth = png::BitDepth::Eight;
        info.icc_profile = self.profile.map(|p| wet_color::icc_profile(p).into());
        let enc = png::Encoder::with_info(out, info).map_err(std::io::Error::other)?;
        enc.write_header()
            .and_then(|mut w| w.write_image_data(&self.render(None)))
            .map_err(std::io::Error::other)
    }

    /// Import a straight RGBA8 image (`w × h`, placed at the top-left) as a new layer of
    /// dry, settled paint above the active one; alpha becomes coverage.
    pub fn import_image(&mut self, name: &str, rgba: &[u8], w: usize, h: usize) -> usize {
        let idx = self.add_layer(name);
        let wet = &mut self.layers[idx].wet;
        for y in 0..h.min(self.h) {
            for x in 0..w.min(self.w) {
                let (i, c) = (4 * (y * w + x), y * self.w + x);
                let a = rgba[i + 3] as f32 / 255.0;
                if a <= 0.0 {
                    continue;
                }
                for k in 0..3 {
                    wet.mid.rgb[k].cur_mut().data[c] = f16::from_f32(rgba[i + k] as f32);
                }
                wet.mid.amount.cur_mut().data[c] = wet_fluid::lut::OpacityLut::inverse(a.min(0.9999));
            }
        }
        idx
    }

    /// The colour a stroke starting at `(x, y)` takes when a tracing layer is set: that
    /// layer's own colour there (paper where it is unpainted).
    pub fn tracing_color(&self, x: usize, y: usize) -> Option<[f32; 3]> {
        let l = self.layers.iter().find(|l| l.tracing)?;
        if x >= self.w || y >= self.h {
            return None;
        }
        let px = render_layer(&l.wet)[y * self.w + x];
        let bg = self.paper.color.map(|v| v / 255.0);
        Some([0, 1, 2].map(|k| (px[k] * px[3] + bg[k] * (1.0 - px[3])) * 255.0))
    }

    /// Pick a colour at `(x, y)`, averaged over a `size × size` square, weighted by paint
    /// coverage for the layer sources. `None` where there is nothing to pick.
    pub fn pick_color(&self, (x, y): (usize, usize), source: PickSource, size: usize) -> Option<[f32; 3]> {
        if x >= self.w || y >= self.h {
            return None;
        }
        let r = (size.max(1) / 2) as i64;
        let cells: Vec<usize> = (-r..=r)
            .flat_map(|dy| (-r..=r).map(move |dx| (x as i64 + dx, y as i64 + dy)))
            .filter(|&(px, py)| px >= 0 && py >= 0 && (px as usize) < self.w && (py as usize) < self.h)
            .map(|(px, py)| py as usize * self.w + px as usize)
            .collect();
        let average = |px: &dyn Fn(usize) -> [f32; 4]| -> Option<[f32; 3]> {
            let (mut acc, mut wsum) = ([0.0f32; 3], 0.0);
            for &c in &cells {
                let p = px(c);
                for k in 0..3 {
                    acc[k] += p[k] * p[3];
                }
                wsum += p[3];
            }
            (wsum > 1e-4).then(|| acc.map(|v| v / wsum * 255.0))
        };
        match source {
            PickSource::AllLayersAndCanvas => {
                let img = self.render(None);
                average(&|c| [img[4 * c] as f32 / 255.0, img[4 * c + 1] as f32 / 255.0, img[4 * c + 2] as f32 / 255.0, 1.0])
            }
            PickSource::CurrentLayer => {
                let px = render_layer(&self.layers.get(self.active)?.wet);
                average(&|c| px[c])
            }
            PickSource::AllLayers => {
                let stack: Vec<Layer> = self
                    .layers
                    .iter()
                    .filter(|l| self.layer_shown(l) && !l.masking_fluid)
                    .map(|l| Layer { w: self.w, h: self.h, rgba: render_layer(&l.wet), opacity: l.opacity, mode: l.mode, visible: true })
                    .collect();
                let flat = wet_compose::composite_isolated(&stack, None);
                average(&|c| flat[c])
            }
        }
    }

    /// Turn pigment mixing on or off for a layer (paint mixing inside the layer and its
    /// compositing over the layers below).
    pub fn set_layer_pigments(&mut self, idx: usize, on: bool) {
        if let Some(l) = self.layers.get_mut(idx) {
            l.pigments = on;
            l.wet.pigment = on.then(|| Box::new(wet_pigment::KmMixer::new()) as Box<dyn wet_fluid::mix::PigmentMixer>);
        }
    }

    /// Natural Colors (Edit menu): red–yellow–blue mixing within every layer that does not
    /// use pigment (Kubelka–Munk) mixing; off restores plain RGB mixing.
    pub fn set_natural_colors(&mut self, on: bool) {
        for l in self.layers.iter_mut().filter(|l| !l.pigments) {
            l.wet.pigment = on.then(|| Box::new(wet_pigment::RybMixer) as Box<dyn wet_fluid::mix::PigmentMixer>);
        }
    }

    /// Import the layers of an 8-bit RGB PSD (raw or RLE) as new layers above the active
    /// one, keeping names, opacity, blend mode, visibility and clipping. Returns how many
    /// layers were imported, or `None` if the file is not a supported PSD.
    pub fn import_psd(&mut self, data: &[u8]) -> Option<usize> {
        let psd = psd::read_psd(data)?;
        let mut n = 0;
        for l in &psd.layers {
            // place the layer's pixels on a canvas-sized image at its offset
            let mut full = vec![0u8; self.w * self.h * 4];
            for y in 0..l.h {
                let cy = l.top + y as i32;
                if cy < 0 || cy as usize >= self.h {
                    continue;
                }
                for x in 0..l.w {
                    let cx = l.left + x as i32;
                    if cx < 0 || cx as usize >= self.w {
                        continue;
                    }
                    let (si, di) = (4 * (y * l.w + x), 4 * (cy as usize * self.w + cx as usize));
                    full[di..di + 4].copy_from_slice(&l.rgba[si..si + 4]);
                }
            }
            let idx = self.import_image(&l.name, &full, self.w, self.h);
            let pl = &mut self.layers[idx];
            pl.opacity = l.opacity;
            pl.mode = l.mode;
            pl.visible = l.visible;
            pl.clip = l.clip;
            n += 1;
        }
        Some(n)
    }

    /// Combine a new selection mask with the current one.
    pub fn apply_selection(&mut self, mask: Vec<u8>, op: SelectOp) {
        let cur = self.selection.take();
        if cur.is_some() {
            self.last_selection = cur.clone();
        }
        self.selection = Some(match (op, cur) {
            (SelectOp::Replace, _) | (SelectOp::Add | SelectOp::Intersect, None) => mask,
            (SelectOp::Subtract, None) => mask.iter().map(|&m| 255 - m).collect(),
            (SelectOp::Add, Some(c)) => c.iter().zip(&mask).map(|(&a, &b)| a.max(b)).collect(),
            (SelectOp::Subtract, Some(c)) => c.iter().zip(&mask).map(|(&a, &b)| a.min(255 - b)).collect(),
            (SelectOp::Intersect, Some(c)) => c.iter().zip(&mask).map(|(&a, &b)| a.min(b)).collect(),
        });
    }

    /// Invert the selection (selecting everything else); no selection stays "everything".
    pub fn invert_selection(&mut self) {
        if let Some(m) = &mut self.selection {
            m.iter_mut().for_each(|v| *v = 255 - *v);
        }
    }

    /// Lasso / polygonal lasso: the closed polygon through `points` (even-odd rule),
    /// anti-aliased with 4×4 supersampling.
    pub fn select_polygon(&mut self, points: &[(f32, f32)], op: SelectOp) {
        let n = points.len();
        let inside = |x: f32, y: f32| {
            let mut c = false;
            for i in 0..n {
                let (a, b) = (points[i], points[(i + n - 1) % n]);
                if (a.1 > y) != (b.1 > y) && x < a.0 + (y - a.1) * (b.0 - a.0) / (b.1 - a.1) {
                    c = !c;
                }
            }
            c
        };
        let (mut bx0, mut by0, mut bx1, mut by1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for &(x, y) in points {
            (bx0, by0, bx1, by1) = (bx0.min(x), by0.min(y), bx1.max(x), by1.max(y));
        }
        let mut m = vec![0u8; self.w * self.h];
        if n >= 3 {
            for (c, v) in m.iter_mut().enumerate() {
                let (x, y) = ((c % self.w) as f32, (c / self.w) as f32);
                if x + 1.0 < bx0 || x - 1.0 > bx1 || y + 1.0 < by0 || y - 1.0 > by1 {
                    continue;
                }
                let hits = (0..16)
                    .filter(|k| inside(x - 0.375 + 0.25 * (k % 4) as f32, y - 0.375 + 0.25 * (k / 4) as f32))
                    .count();
                *v = (hits * 255 / 16) as u8;
            }
        }
        self.apply_selection(m, op);
    }

    /// Magic wand: select pixels of the flattened canvas whose colour is within
    /// `tolerance` (0–255, max channel difference) of the pixel at `(x, y)`; only the
    /// region connected to it when `contiguous`.
    pub fn select_color(&mut self, at: (usize, usize), tolerance: u8, contiguous: bool, op: SelectOp) {
        let m = self.color_region(at, tolerance, contiguous);
        self.apply_selection(m, op);
    }

    /// Pixels of the flattened canvas within `tolerance` of the colour at `(x, y)`
    /// (connected to it when `contiguous`), as a 0/255 mask.
    fn color_region(&self, (x, y): (usize, usize), tolerance: u8, contiguous: bool) -> Vec<u8> {
        let rgba = self.render(None);
        let (w, h) = (self.w, self.h);
        let px = |c: usize| [rgba[4 * c], rgba[4 * c + 1], rgba[4 * c + 2]];
        let seed = px(y.min(h - 1) * w + x.min(w - 1));
        let close = |c: usize| px(c).iter().zip(&seed).all(|(&a, &b)| a.abs_diff(b) <= tolerance);
        let mut m = vec![0u8; w * h];
        if contiguous {
            let start = y.min(h - 1) * w + x.min(w - 1);
            let mut stack = vec![start];
            m[start] = 255;
            while let Some(c) = stack.pop() {
                let (cx, cy) = (c % w, c / w);
                let mut push = |n: usize| {
                    if m[n] == 0 && close(n) {
                        m[n] = 255;
                        stack.push(n);
                    }
                };
                if cx > 0 { push(c - 1); }
                if cx + 1 < w { push(c + 1); }
                if cy > 0 { push(c - w); }
                if cy + 1 < h { push(c + w); }
            }
        } else {
            for (c, v) in m.iter_mut().enumerate() {
                if close(c) {
                    *v = 255;
                }
            }
        }
        m
    }

    /// Fill tool: fill the region similar in colour to `(x, y)` on the active layer
    /// (limited to the selection) with settled paint of `fill`'s colour, gradient or
    /// pattern at `opacity` coverage; with `wet` the filled pixels are also wetted.
    pub fn fill(&mut self, at: (usize, usize), fill: &Fill, opts: FillOptions) {
        if at.0 >= self.w || at.1 >= self.h {
            return;
        }
        let region = self.color_region(at, opts.tolerance, opts.contiguous);
        let sel = self.selection.clone();
        let (w, h) = (self.w, self.h);
        let wet = &mut self.layers[self.active].wet;
        for (c, &r) in region.iter().enumerate() {
            let k = r as f32 / 255.0 * sel.as_ref().map_or(1.0, |s| s[c] as f32 / 255.0) * wet.maps.stencil_at(c);
            if k <= 0.0 {
                continue;
            }
            let (x, y) = ((c % w) as f32, (c / w) as f32);
            let (color, alpha) = fill.color_at(x, y, w, h);
            let cov = (opts.opacity * alpha * k).clamp(0.0, 0.9999);
            if cov <= 0.0 {
                continue;
            }
            let amount = wet_fluid::lut::OpacityLut::inverse(cov);
            if opts.wet {
                wet.deposit(c, 1.2 * wet_fluid::params::WATER_SCALE, color, amount);
            } else {
                // lay the colour over what is there with the given coverage
                let old_a = wet.mid.amount.cur().data[c];
                let old: [f32; 3] = [0, 1, 2].map(|i| wet.mid.rgb[i].cur().data[c].to_f32());
                let la = wet.lut.eval(old_a);
                let a = 1.0 - (1.0 - la) * (1.0 - cov);
                let mixed = [0, 1, 2].map(|i| if a > 0.0 { (old[i] * la * (1.0 - cov) + color[i] * cov) / a } else { color[i] });
                for i in 0..3 {
                    wet.mid.rgb[i].cur_mut().data[c] = f16::from_f32(mixed[i]);
                }
                wet.mid.amount.cur_mut().data[c] = wet_fluid::lut::OpacityLut::inverse(a.min(0.9999));
            }
        }
    }

    /// Select an axis-aligned rectangle or ellipse (inclusive corners), with 1 px
    /// anti-aliasing on the ellipse edge.
    pub fn select_shape(&mut self, (x0, y0): (f32, f32), (x1, y1): (f32, f32), ellipse: bool) {
        let (ax, bx) = (x0.min(x1), x0.max(x1));
        let (ay, by) = (y0.min(y1), y0.max(y1));
        let (cx, cy, rx, ry) = ((ax + bx) / 2.0, (ay + by) / 2.0, ((bx - ax) / 2.0).max(0.5), ((by - ay) / 2.0).max(0.5));
        let m = (0..self.w * self.h)
            .map(|c| {
                let (x, y) = ((c % self.w) as f32, (c / self.w) as f32);
                let inside = if ellipse {
                    let d = (((x - cx) / rx).powi(2) + ((y - cy) / ry).powi(2)).sqrt();
                    ((1.0 - d) * rx.min(ry) + 0.5).clamp(0.0, 1.0)
                } else if x >= ax && x <= bx && y >= ay && y <= by {
                    1.0
                } else {
                    0.0
                };
                (inside * 255.0).round() as u8
            })
            .collect();
        self.selection = Some(m);
    }

    /// Transform a layer's content: `scale` and `rotation` (radians) about the canvas point
    /// `centre`, then translation by `offset`. Every fluid plane is resampled through the
    /// inverse map, bilinearly for continuous planes and nearest for masks. Velocity is
    /// cleared.
    pub fn transform_layer(&mut self, idx: usize, offset: (f32, f32), scale: f32, rotation: f32, centre: (f32, f32)) {
        let (sn, cs) = rotation.sin_cos();
        self.remap_layer(idx, |x, y| {
            // destination -> source: undo translation, rotation and scale
            let (dx, dy) = (x - centre.0 - offset.0, y - centre.1 - offset.1);
            let (rx, ry) = (dx * cs + dy * sn, -dx * sn + dy * cs);
            (centre.0 + rx / scale, centre.1 + ry / scale)
        });
    }

    /// Perspective: move the layer's corners (top-left, top-right, bottom-right,
    /// bottom-left of the canvas) to `corners`, through the projective map they define.
    pub fn perspective_layer(&mut self, idx: usize, corners: [(f32, f32); 4]) {
        let src = self.canvas_corners();
        // the inverse map: destination quad -> canvas rectangle
        let Some(hm) = homography(corners, src) else { return };
        self.remap_layer(idx, |x, y| {
            let d = hm[6] * x + hm[7] * y + 1.0;
            ((hm[0] * x + hm[1] * y + hm[2]) / d, (hm[3] * x + hm[4] * y + hm[5]) / d)
        });
    }

    /// Deform: move the layer's corners to `corners` with a bilinear (not projective)
    /// map, as when dragging the four corner handles freely.
    pub fn deform_layer(&mut self, idx: usize, corners: [(f32, f32); 4]) {
        let (w, h) = ((self.w - 1) as f32, (self.h - 1) as f32);
        let [a, b, c, d] = corners;
        let fwd = |u: f32, v: f32| {
            let top = (a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u);
            let bot = (d.0 + (c.0 - d.0) * u, d.1 + (c.1 - d.1) * u);
            (top.0 + (bot.0 - top.0) * v, top.1 + (bot.1 - top.1) * v)
        };
        let guess = homography(corners, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        self.remap_layer(idx, |x, y| {
            // invert the bilinear map by Newton steps from the projective estimate
            let (mut u, mut v) = guess.map_or((x / w, y / h), |hm| {
                let dd = hm[6] * x + hm[7] * y + 1.0;
                ((hm[0] * x + hm[1] * y + hm[2]) / dd, (hm[3] * x + hm[4] * y + hm[5]) / dd)
            });
            for _ in 0..8 {
                let (px, py) = fwd(u, v);
                let (ex, ey) = (px - x, py - y);
                let e = 1e-3;
                let (pu, pv) = (fwd(u + e, v), fwd(u, v + e));
                let (j00, j10, j01, j11) = ((pu.0 - px) / e, (pu.1 - py) / e, (pv.0 - px) / e, (pv.1 - py) / e);
                let det = j00 * j11 - j01 * j10;
                if det.abs() < 1e-9 {
                    break;
                }
                u -= (j11 * ex - j01 * ey) / det;
                v -= (-j10 * ex + j00 * ey) / det;
            }
            (u * w, v * h)
        });
    }

    /// Warp: an `n × n`-cell grid over the canvas whose `(n + 1)²` points (row-major)
    /// move by `offsets`; the layer follows the grid bilinearly within each cell.
    pub fn warp_layer(&mut self, idx: usize, n: usize, offsets: &[(f32, f32)]) {
        let n = n.max(1);
        if offsets.len() != (n + 1) * (n + 1) {
            return;
        }
        let (cw, ch) = ((self.w - 1) as f32 / n as f32, (self.h - 1) as f32 / n as f32);
        let disp = |x: f32, y: f32| {
            let (gx, gy) = ((x / cw).clamp(0.0, n as f32 - 1e-4), (y / ch).clamp(0.0, n as f32 - 1e-4));
            let (i, j) = (gx as usize, gy as usize);
            let (fx, fy) = (gx - i as f32, gy - j as f32);
            let at = |i: usize, j: usize| offsets[j * (n + 1) + i];
            let lerp = |a: (f32, f32), b: (f32, f32), t: f32| (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            lerp(lerp(at(i, j), at(i + 1, j), fx), lerp(at(i, j + 1), at(i + 1, j + 1), fx), fy)
        };
        self.remap_layer(idx, |x, y| {
            // invert p' = p + D(p) by fixed-point iteration
            let mut q = (x, y);
            for _ in 0..8 {
                let d = disp(q.0, q.1);
                q = (x - d.0, y - d.1);
            }
            q
        });
    }

    fn canvas_corners(&self) -> [(f32, f32); 4] {
        let (w, h) = ((self.w - 1) as f32, (self.h - 1) as f32);
        [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    }

    /// Resample a layer through `inv` (destination pixel -> source position): bilinear
    /// for continuous planes (colours weighted by amount), nearest for masks; velocity is
    /// cleared.
    pub fn remap_layer(&mut self, idx: usize, inv: impl Fn(f32, f32) -> (f32, f32)) {
        let (w, h) = (self.w, self.h);
        let Some(layer) = self.layers.get_mut(idx) else { return };
        let l = &mut layer.wet;
        let src: Vec<(f32, f32)> = (0..w * h).map(|c| inv((c % w) as f32, (c / w) as f32)).collect();
        let sample = |data: &[f32], (sx, sy): (f32, f32)| -> f32 {
            if sx < 0.0 || sy < 0.0 || sx > (w - 1) as f32 || sy > (h - 1) as f32 {
                return 0.0;
            }
            let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
            let at = |x: usize, y: usize| data[y * w + x];
            (at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx) * (1.0 - fy) + (at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx) * fy
        };
        let mut planes = LayerPlanes::capture(l);
        let warp = |p: &[f32]| -> Vec<f32> { src.iter().map(|&q| sample(p, q)).collect() };
        // colour planes sample amount-weighted so colours do not bleed into empty space
        let f32s: Vec<Vec<f32>> = planes.f32s.iter().map(|p| warp(p)).collect();
        let to32 = |p: &Vec<f16>| p.iter().map(|v| v.to_f32()).collect::<Vec<f32>>();
        let mut h16: Vec<Vec<f16>> = Vec::new();
        for (k, p) in planes.h16.iter().enumerate() {
            let out = match k {
                1 | 2 => vec![f16::ZERO; w * h], // velocity
                3..=5 | 6..=8 => {
                    let amount = if k <= 5 { &planes.f32s[0] } else { &planes.f32s[2] };
                    let weighted: Vec<f32> = to32(p).iter().zip(amount).map(|(c, a)| c * a).collect();
                    let (num, den) = (warp(&weighted), warp(amount));
                    num.iter().zip(den).map(|(n, d)| f16::from_f32(if d > 1e-6 { n / d } else { 255.0 })).collect()
                }
                _ => warp(&to32(p)).into_iter().map(f16::from_f32).collect(),
            };
            h16.push(out);
        }
        let open = src
            .iter()
            .map(|&(sx, sy)| {
                let (x, y) = (sx.round(), sy.round());
                if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 { 0 } else { planes.open[y as usize * w + x as usize] }
            })
            .collect();
        planes = LayerPlanes { h16, f32s, open };
        planes.restore(l);
        if let Some(m) = &mut layer.mask {
            let old = m.clone();
            for (c, &(sx, sy)) in src.iter().enumerate() {
                let (x, y) = (sx.round(), sy.round());
                m[c] = if x < 0.0 || y < 0.0 || x >= w as f32 || y >= h as f32 { 0 } else { old[y as usize * w + x as usize] };
            }
        }
    }

    /// Apply a filter to layer `idx`, through the selection if
    /// there is one. Undoable.
    pub fn apply_filter(&mut self, idx: usize, filter: &filters::Filter) {
        if idx >= self.layers.len() {
            return;
        }
        self.checkpoint_layer_as(idx, filter.label());
        let sel = self.selection.clone();
        filters::apply(&mut self.layers[idx].wet, filter, sel.as_deref());
    }

    /// One simulation step on every layer.
    pub fn step(&mut self) {
        for l in &mut self.layers {
            l.wet.step();
        }
    }

    /// Composite all visible layers over the paper (straight RGBA8, opaque).
    /// The colour under the layers in the display (0–1): the paper, toned when shown.
    pub fn display_background(&self) -> [f32; 3] {
        let tone = if self.paper.visible { self.paper.tone } else { 1.0 };
        self.paper.color.map(|c| c * tone / 255.0)
    }

    /// The paper relief shading the display multiplies in, per cell (`None`: off). Kept
    /// between calls until the paper changes.
    pub fn paper_shading(&self) -> Option<std::sync::Arc<Vec<f32>>> {
        (self.paper.visible && self.paper.shading > 0.0).then(|| {
            let mut cache = self.paper_shade.lock().expect("paper shade");
            match &*cache {
                Some((k, s)) if *k == self.paper.shading => s.clone(),
                _ => {
                    let s = std::sync::Arc::new(self.paper_maps.shade(self.paper.shading));
                    *cache = Some((self.paper.shading, s.clone()));
                    s
                }
            }
        })
    }

    /// Sheet coverage when the paper is deckled (the display's alpha), `None` otherwise.
    pub fn deckle_map(&self) -> Option<&[f32]> {
        self.deckle.as_deref()
    }

    pub fn render(&self, pigment: Option<&dyn PigmentMix>) -> Vec<u8> {
        // pigment layers need a mixer even when the caller passes none
        let own_mixer;
        let pigment = match pigment {
            None if self.layers.iter().any(|l| l.pigments) => {
                own_mixer = wet_pigment::KmMixer::new();
                Some(&own_mixer as &dyn PigmentMix)
            }
            p => p,
        };
        let mut stack: Vec<Layer> = Vec::with_capacity(self.layers.len());
        let mut below_alpha: Option<Vec<f32>> = None;
        for l in &self.layers {
            let mut rgba = render_layer(&l.wet);
            if let Some(m) = &l.mask {
                for (p, &k) in rgba.iter_mut().zip(m) {
                    p[3] *= k as f32 / 255.0;
                }
            }
            if l.clip {
                if let Some(base) = &below_alpha {
                    for (p, &a) in rgba.iter_mut().zip(base) {
                        p[3] *= a;
                    }
                }
            } else {
                // the alpha non-clipped layers offer to clipped layers above them
                below_alpha = Some(rgba.iter().map(|p| p[3]).collect());
            }
            stack.push(Layer {
                w: self.w,
                h: self.h,
                rgba,
                opacity: l.opacity,
                mode: if l.pigments { MixMode::Pigment } else { l.mode },
                visible: l.visible && !l.masking_fluid,
            });
        }
        // runs of adjacent layers in one group composite in isolation first
        let group_of = |i: usize| self.layers[i].group.filter(|&g| g < self.groups.len());
        let mut folded: Vec<Layer> = Vec::with_capacity(stack.len());
        let mut it = stack.into_iter().enumerate().peekable();
        while let Some((i, layer)) = it.next() {
            let Some(g) = group_of(i) else {
                folded.push(layer);
                continue;
            };
            let mut members = vec![layer];
            while let Some((_, l)) = it.next_if(|(j, _)| group_of(*j) == Some(g)) {
                members.push(l);
            }
            let grp = &self.groups[g];
            folded.push(Layer {
                w: self.w,
                h: self.h,
                rgba: wet_compose::composite_isolated(&members, pigment),
                opacity: grp.opacity,
                mode: grp.mode,
                visible: grp.visible,
            });
        }
        let flat = flatten(self.display_background(), &folded, pigment);
        let shade = self.paper_shading();
        let impasto: Vec<Vec<f32>> = self
            .layers
            .iter()
            .filter(|l| self.layer_shown(l))
            .map(|l| wet_fluid::render::impasto_shade(&l.wet, self.impasto))
            .collect();
        let gloss: Vec<Vec<f32>> = if self.gloss > 0.0 {
            self.layers
                .iter()
                .filter(|l| self.layer_shown(l))
                .map(|l| wet_fluid::render::gloss_highlight(&l.wet, self.impasto.max(0.2), self.gloss))
                .collect()
        } else {
            Vec::new()
        };
        let shadows = (self.soft_shadows > 0.0).then(|| {
            let mut hf = vec![0.0f32; self.w * self.h];
            for l in self.layers.iter().filter(|l| self.layer_shown(l)) {
                let (mt, dt) = (l.wet.mid.thickness.cur(), l.wet.dens.thickness.cur());
                for (c, v) in hf.iter_mut().enumerate() {
                    *v += shadow::height(mt.data[c] + dt.data[c]);
                }
            }
            shadow::soft_shadows(&hf, self.w, self.h, self.soft_shadows, self.shadow_altitude)
        });
        let metal = self.layers.iter().any(|l| l.metallic && self.layer_shown(l)).then(|| {
            let n = self.w * self.h;
            let (mut m, mut hf) = (vec![0.0f32; n], vec![0.0f32; n]);
            for l in self.layers.iter().filter(|l| self.layer_shown(l)) {
                let (ma, da) = (l.wet.mid.amount.cur(), l.wet.dens.amount.cur());
                let (mt, dt) = (l.wet.mid.thickness.cur(), l.wet.dens.thickness.cur());
                for c in 0..n {
                    hf[c] += shadow::height(mt.data[c] + dt.data[c]);
                    if l.metallic {
                        let cov = 1.0 - (1.0 - l.wet.lut.eval(ma.data[c])) * (1.0 - l.wet.lut.eval(da.data[c]));
                        m[c] = 1.0 - (1.0 - m[c]) * (1.0 - cov * l.opacity);
                    }
                }
            }
            self.metallic.shade(&m, &hf, self.w, self.h)
        });
        let mut out = vec![0u8; self.w * self.h * 4];
        out.par_chunks_exact_mut(4).zip(flat.par_iter()).enumerate().for_each(|(i, (px, c))| {
            let s = shade.as_ref().map_or(1.0, |s| s[i])
                * impasto.iter().map(|m| m[i]).product::<f32>()
                * shadows.as_ref().map_or(1.0, |m| m[i]);
            let (mm, sheen) = metal.as_ref().map_or((1.0, 0.0), |m| m[i]);
            let s = s * mm;
            let g: f32 = gloss.iter().map(|m| m[i]).sum::<f32>() + sheen;
            for k in 0..3 {
                px[k] = ((c[k] * s + g) * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            px[3] = self.deckle.as_ref().map_or(255, |d| (d[i] * 255.0).round() as u8);
        });
        out
    }

    /// Export a layered PSD: each layer's own rendering, groups as PSD folders, plus the
    /// flattened canvas.
    pub fn export_psd(&self, out: impl Write) -> std::io::Result<()> {
        let mut recs: Vec<psd::PsdLayer> = Vec::new();
        let mut open: Option<usize> = None;
        fn group_start(grp: &LayerGroup) -> psd::PsdLayer<'_> {
            psd::PsdLayer {
                name: &grp.name,
                rgba: Vec::new(),
                opacity: grp.opacity,
                mode: grp.mode,
                visible: grp.visible,
                clip: false,
                section: psd::Section::GroupStart,
            }
        }
        for l in self.layers.iter().filter(|l| !l.masking_fluid) {
            let g = l.group.filter(|&g| g < self.groups.len());
            if g != open {
                if let Some(o) = open {
                    recs.push(group_start(&self.groups[o]));
                }
                if g.is_some() {
                    recs.push(psd::PsdLayer {
                        name: "</Layer group>",
                        rgba: Vec::new(),
                        opacity: 1.0,
                        mode: MixMode::Normal,
                        visible: true,
                        clip: false,
                        section: psd::Section::GroupEnd,
                    });
                }
                open = g;
            }
            let mut px = render_layer(&l.wet);
            if let Some(m) = &l.mask {
                for (p, &k) in px.iter_mut().zip(m) {
                    p[3] *= k as f32 / 255.0;
                }
            }
            recs.push(psd::PsdLayer {
                name: &l.name,
                rgba: px.iter().flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)).collect(),
                opacity: l.opacity,
                mode: l.mode,
                visible: l.visible,
                clip: l.clip,
                section: psd::Section::Pixels,
            });
        }
        if let Some(o) = open {
            recs.push(group_start(&self.groups[o]));
        }
        let icc = self.profile.map(wet_color::icc_profile);
        psd::write_psd(out, self.w, self.h, &recs, &self.render(None), icc.as_deref())
    }

    /// Serialise to the `.wetdoc` format.
    pub fn save(&mut self, mut out: impl Write) -> std::io::Result<()> {
        let mut blob: Vec<u8> = Vec::new();
        let mut layers = Vec::new();
        for l in &mut self.layers {
            let mut planes = Vec::new();
            let p = LayerPlanes::capture(&mut l.wet);
            for (name, data) in H16_NAMES.iter().zip(&p.h16) {
                planes.push(json!({"name": name, "type": "f16", "offset": blob.len()}));
                data.iter().for_each(|v| blob.extend(v.to_le_bytes()));
            }
            for (name, data) in F32_NAMES.iter().zip(&p.f32s) {
                planes.push(json!({"name": name, "type": "f32", "offset": blob.len()}));
                data.iter().for_each(|v| blob.extend(v.to_le_bytes()));
            }
            planes.push(json!({"name": "wet_mask", "type": "u8", "offset": blob.len()}));
            blob.extend(&p.open);
            let mask_off = l.mask.as_ref().map(|m| {
                let off = blob.len();
                blob.extend(m);
                off
            });
            layers.push(json!({
                "name": l.name,
                "mode": format!("{:?}", l.mode),
                "opacity": l.opacity,
                "visible": l.visible,
                "locked": l.locked,
                "clip": l.clip,
                "masking_fluid": l.masking_fluid,
                "group": l.group,
                "tracing": l.tracing,
                "pigments": l.pigments,
                "metallic": l.metallic,
                "step_index": l.wet.step_index,
                "planes": planes,
                "layer_mask_offset": mask_off,
            }));
        }
        let stencils: Vec<Value> = self
            .stencils
            .iter()
            .map(|st| {
                let off = blob.len();
                blob.extend(&st.mask);
                json!({
                    "w": st.w, "h": st.h, "offset": off, "centre": [st.centre.0, st.centre.1],
                    "rotation": st.rotation, "scale": st.scale, "flip_h": st.flip_h, "flip_v": st.flip_v,
                    "invert": st.invert, "border": st.border, "tile": st.tile, "visible": st.visible,
                })
            })
            .collect();
        let references: Vec<Value> = self
            .references
            .iter()
            .map(|r| {
                let off = blob.len();
                blob.extend(&r.rgba);
                json!({"name": r.name, "w": r.w, "h": r.h, "offset": off, "visible": r.visible})
            })
            .collect();
        let ps = self.paper;
        let manifest = json!({
            "format": "wetdoc", "version": 1, "width": self.w, "height": self.h,
            "active": self.active,
            "profile": self.profile.map(|p| p.name()),
            "paper": {
                "surface": format!("{:?}", ps.spec.surface), "seed": ps.spec.seed,
                "scale": ps.spec.scale, "relief": ps.spec.relief,
                "absorbency": ps.spec.absorbency, "absorbency_variation": ps.spec.absorbency_variation,
                "color": ps.color, "visible": ps.visible, "shading": ps.shading, "tone": ps.tone,
                "deckled": ps.deckled,
            },
            "layers": layers,
            "stencils": stencils,
            "references": references,
            "groups": self.groups.iter().map(|g| json!({
                "name": g.name, "mode": format!("{:?}", g.mode), "opacity": g.opacity, "visible": g.visible,
            })).collect::<Vec<_>>(),
        });
        let m = serde_json::to_vec(&manifest)?;
        out.write_all(MAGIC)?;
        out.write_all(&(m.len() as u32).to_le_bytes())?;
        out.write_all(&m)?;
        out.write_all(&blob)
    }

    /// Load a `.wetdoc` document.
    pub fn load(mut input: impl Read) -> std::io::Result<Self> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_owned());
        let mut head = [0u8; 12];
        input.read_exact(&mut head)?;
        if &head[..8] != MAGIC {
            return Err(bad("not a wetdoc file"));
        }
        let mlen = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
        let mut m = vec![0u8; mlen];
        input.read_exact(&mut m)?;
        let man: Value = serde_json::from_slice(&m)?;
        let mut blob = Vec::new();
        input.read_to_end(&mut blob)?;
        let u = |v: &Value| v.as_u64().ok_or_else(|| bad("bad manifest"));
        let (w, h) = (u(&man["width"])? as usize, u(&man["height"])? as usize);
        let p = &man["paper"];
        let f = |v: &Value, d: f64| v.as_f64().unwrap_or(d) as f32;
        let surface = match p["surface"].as_str() {
            Some("HotPressed") => Surface::HotPressed,
            Some("Rough") => Surface::Rough,
            _ => Surface::ColdPressed,
        };
        let spec = PaperSpec {
            surface,
            seed: p["seed"].as_u64().unwrap_or(1),
            scale: f(&p["scale"], 1.0),
            relief: f(&p["relief"], 1.0),
            absorbency: f(&p["absorbency"], 0.0),
            absorbency_variation: f(&p["absorbency_variation"], 0.0),
        };
        let color = [0, 1, 2].map(|i| f(&p["color"][i], 255.0));
        let paper = PaperSettings {
            spec,
            color,
            visible: p["visible"].as_bool().unwrap_or(true),
            shading: f(&p["shading"], 0.094),
            tone: f(&p["tone"], 0.963),
            deckled: p["deckled"].as_bool().unwrap_or(false),
        };
        let mut doc = Self::new(w, h);
        doc.layers.clear();
        doc.set_paper(paper);
        let n = w * h;
        for lm in man["layers"].as_array().ok_or_else(|| bad("no layers"))? {
            let mut wet = WetLayer::new(w, h);
            doc.paper_maps.apply(&mut wet);
            wet_paper::apply_granulation(&mut wet, gran_tile(), GRAN_SIZE);
            let mut planes = LayerPlanes::default();
            let find = |name: &str| {
                lm["planes"].as_array().and_then(|a| a.iter().find(|p| p["name"] == name)).and_then(|p| p["offset"].as_u64())
            };
            for name in H16_NAMES {
                let off = find(name).ok_or_else(|| bad("missing plane"))? as usize;
                let bytes = blob.get(off..off + 2 * n).ok_or_else(|| bad("truncated"))?;
                planes.h16.push(bytes.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]])).collect());
            }
            for name in F32_NAMES {
                let off = find(name).ok_or_else(|| bad("missing plane"))? as usize;
                let bytes = blob.get(off..off + 4 * n).ok_or_else(|| bad("truncated"))?;
                planes.f32s.push(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect());
            }
            let off = find("wet_mask").ok_or_else(|| bad("missing plane"))? as usize;
            planes.open = blob.get(off..off + n).ok_or_else(|| bad("truncated"))?.to_vec();
            planes.restore(&mut wet);
            wet.step_index = lm["step_index"].as_u64().unwrap_or(0);
            let mode = MixMode::ALL
                .into_iter()
                .find(|m| format!("{m:?}") == lm["mode"].as_str().unwrap_or("Normal"))
                .unwrap_or(MixMode::Normal);
            let mut layer = PaintLayer::new(lm["name"].as_str().unwrap_or("Layer"), wet);
            layer.mode = mode;
            layer.opacity = f(&lm["opacity"], 1.0);
            layer.visible = lm["visible"].as_bool().unwrap_or(true);
            layer.locked = lm["locked"].as_bool().unwrap_or(false);
            layer.clip = lm["clip"].as_bool().unwrap_or(false);
            layer.masking_fluid = lm["masking_fluid"].as_bool().unwrap_or(false);
            layer.group = lm["group"].as_u64().map(|g| g as usize);
            layer.tracing = lm["tracing"].as_bool().unwrap_or(false);
            layer.metallic = lm["metallic"].as_bool().unwrap_or(false);
            if lm["pigments"].as_bool().unwrap_or(false) {
                layer.pigments = true;
                layer.wet.pigment = Some(Box::new(wet_pigment::KmMixer::new()));
            }
            if let Some(off) = lm["layer_mask_offset"].as_u64() {
                let off = off as usize;
                layer.mask = Some(blob.get(off..off + n).ok_or_else(|| bad("truncated"))?.to_vec());
            }
            doc.layers.push(layer);
        }
        if doc.layers.is_empty() {
            return Err(bad("no layers"));
        }
        doc.active = (man["active"].as_u64().unwrap_or(0) as usize).min(doc.layers.len() - 1);
        let mode_of = |v: &Value| MixMode::ALL.into_iter().find(|m| format!("{m:?}") == v.as_str().unwrap_or("Normal"));
        for sm in man["stencils"].as_array().into_iter().flatten() {
            let (sw, sh) = (u(&sm["w"])? as usize, u(&sm["h"])? as usize);
            let off = u(&sm["offset"])? as usize;
            let mask = blob.get(off..off + sw * sh).ok_or_else(|| bad("truncated"))?.to_vec();
            let b = |k: &str| sm[k].as_bool().unwrap_or(false);
            let mut st = Stencil::new(mask, sw, sh, (f(&sm["centre"][0], 0.0), f(&sm["centre"][1], 0.0)));
            st.rotation = f(&sm["rotation"], 0.0);
            st.scale = f(&sm["scale"], 1.0);
            (st.flip_h, st.flip_v, st.invert, st.border, st.tile) = (b("flip_h"), b("flip_v"), b("invert"), b("border"), b("tile"));
            st.visible = sm["visible"].as_bool().unwrap_or(true);
            doc.stencils.push(st);
        }
        for rm in man["references"].as_array().into_iter().flatten() {
            let (rw, rh) = (u(&rm["w"])? as usize, u(&rm["h"])? as usize);
            let off = u(&rm["offset"])? as usize;
            doc.references.push(ReferenceImage {
                name: rm["name"].as_str().unwrap_or("Reference").into(),
                w: rw,
                h: rh,
                rgba: blob.get(off..off + 4 * rw * rh).ok_or_else(|| bad("truncated"))?.to_vec(),
                visible: rm["visible"].as_bool().unwrap_or(true),
            });
        }
        for gm in man["groups"].as_array().into_iter().flatten() {
            doc.groups.push(LayerGroup {
                name: gm["name"].as_str().unwrap_or("Group").into(),
                mode: mode_of(&gm["mode"]).unwrap_or(MixMode::Normal),
                opacity: f(&gm["opacity"], 1.0),
                visible: gm["visible"].as_bool().unwrap_or(true),
            });
        }
        doc.profile = man["profile"].as_str().and_then(RgbSpace::from_name);
        doc.update_stencils();
        Ok(doc)
    }
}
