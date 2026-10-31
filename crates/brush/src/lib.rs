//! Stamp-based wet-media brush: turns pointer samples into dabs and injects water and
//! suspended pigment into a [`wet_fluid::WetLayer`]. Tuning constants are in
//! [`Calibration`].

pub mod bristle;
pub mod curve;
pub mod path;
pub mod ruler;
pub mod stroke;
pub mod tip;
pub mod tool;
pub mod volume;

use wet_preset::BrushPreset;
use wet_fluid::{WetLayer, grid::{ld, st}};

pub use stroke::{DabPos, Rng, Sample, Spacer};
pub use tip::Mask;
pub use bristle::{BristleParams, BristleSet};
pub use tool::{CellPaint, Liquify, Medium, MediumModel, Tool};

/// How paint and canvas interact (`paint_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintMode {
    /// Lays paint without picking up the canvas.
    Paint,
    /// Picks up wet paint into the brush and mixes it into what is laid (default).
    Mix,
    /// Paint plus stronger blending with the canvas.
    PaintBlend,
    /// No new paint: moves the canvas paint along (smudge-like).
    Blend,
    /// Removes paint.
    Erase,
}

impl PaintMode {
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "PAINT" => Self::Paint,
            "PAINT_MIX" => Self::Mix,
            "PAINT_BLEND" => Self::PaintBlend,
            "BLEND" => Self::Blend,
            "ERASE" => Self::Erase,
            _ => return None,
        })
    }

    /// The `paint_type` name [`Self::from_name`] reads.
    pub fn name(self) -> &'static str {
        match self {
            Self::Paint => "PAINT",
            Self::Mix => "PAINT_MIX",
            Self::PaintBlend => "PAINT_BLEND",
            Self::Blend => "BLEND",
            Self::Erase => "ERASE",
        }
    }
}

/// Dual brush (`dual_brush_*`, `DualBrushShape_0`): secondary stamps scattered around each
/// dab build their own stroke layer `D` (union, `D += (1 − D)·op·shape`), composited onto
/// the stroke mask with its weight: `E = M + w·D·(1 − M)`.
#[derive(Clone, Debug)]
pub struct DualBrush {
    pub shape: Mask,
    pub count: usize,
    /// Secondary stamp size relative to the dab, and its relative jitter.
    pub size: f32,
    pub size_jitter: f32,
    /// Offset of the stamps from the dab centre, in dab diameters (min and jitter).
    pub min_scatter: f32,
    pub scatter: f32,
    pub rotation_jitter: f32,
    pub opacity: f32,
    pub opacity_jitter: f32,
    pub weight: f32,
    /// `dual_brush_alternative_mode`: the dual stamps add paint instead of masking.
    pub alternative: bool,
}

impl DualBrush {
    fn from_preset(p: &BrushPreset, shape: Mask) -> Option<Self> {
        if !p.parameters.get("dual_brush_enabled").and_then(|v| v.as_bool()).unwrap_or(false) {
            return None;
        }
        let f = |k: &str, d: f32| p.param_f64(k).map_or(d, |v| v as f32);
        Some(Self {
            shape,
            count: f("dual_brush_splat_count", 1.0).max(1.0) as usize,
            size: f("dual_brush_splat_size", 1.0),
            size_jitter: f("dual_brush_splat_size_jitter", 0.0),
            min_scatter: f("dual_brush_min_scatter", 0.0),
            scatter: f("dual_brush_scattering_jitter", 0.0),
            rotation_jitter: f("dual_brush_rotation_jitter", 0.0),
            opacity: f("dual_brush_opacity", 1.0),
            opacity_jitter: f("dual_brush_opacity_jitter", 0.0),
            weight: f("dual_brush_layer_blend_mode_weight", 1.0),
            alternative: p.parameters.get("dual_brush_alternative_mode").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    }
}

/// How a dab chooses among several shape/grain textures (`texture_order`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureOrder {
    Sequential,
    Random,
    Pressure,
    /// Selected by pen tilt; without tilt input it follows pressure.
    Tilt,
}

/// How the dab is rotated (preset `brush_rotation_mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationMode {
    None,
    FollowTrajectory,
    PenTilt,
    PenRotation,
}

/// Preset-derived brush parameters. Ranges follow the preset JSON.
#[derive(Clone, Debug)]
pub struct BrushParams {
    /// Dab diameter in canvas pixels at full pressure.
    pub size: f32,
    /// Volume opacity 0–1 (`volume_params.opacity`, UI opacity / 100).
    pub opacity: f32,
    /// Volume water 0–1 (`volume_params.water`, UI water / 100).
    pub water: f32,
    /// Gap between dabs as a fraction of the current size.
    pub spacing: f32,
    pub spacing_jitter: f32,
    pub size_jitter: f32,
    pub opacity_jitter: f32,
    pub rotation: f32,
    pub rotation_jitter: f32,
    pub scatter_jitter: f32,
    /// Per-dab opacity (`splat_opacity · splat_opacity_multiplier`).
    pub splat_opacity: f32,
    pub pressure_size_influence: f32,
    pub pressure_opacity_influence: f32,
    pub rotation_mode: RotationMode,
    pub paint_mode: PaintMode,
    /// How the brush colour combines with the colour already on the canvas
    /// (`glaze_rgb_blend_mode`; effect markers use Burn, Color, Saturation, …).
    pub glaze_blend: wet_compose::MixMode,
    /// Watercolour glazing (`SET_GLAZE_MODE`): how strongly new paint covers the dry
    /// paint below it (semi-opaque by default, the state the calibration was done in).
    pub glaze_mode: GlazeMode,
    /// Paint water only, no pigment (`only_water`).
    pub only_water: bool,
    /// Dirty brush (`dirty_brush`): paint picked up during a stroke stays on the brush
    /// for the next strokes instead of being discarded at the next press.
    pub dirty: bool,
    /// Grain texture scale (`grain_scale`) and the brush size it was authored for.
    pub grain_scale: f32,
    pub grain_brush_size: f32,
    pub grain_follow_brush_size: bool,
    /// Grain moves with the brush (`grain_follow_distance`) instead of staying on the canvas.
    pub grain_follow_distance: bool,
    /// How far a following grain moves with the brush (`grain_tile_follow_distance`):
    /// 1 = fully in brush space (streaks), 0 = anchored to the canvas (0.09 paints
    /// speckle, 1 streaks).
    pub grain_follow_amount: f32,
    /// Grain rotates with the dab (`grain_follow_shape_rotation`), staying anchored to the
    /// canvas otherwise: striped grains then streak along the stroke.
    pub grain_follow_shape_rotation: bool,
    /// Each dab is swept from the previous dab centre (`splat_smudge`), turning a dotted
    /// tip into continuous lines (streaks, not dots). `smudge_density` is not modelled.
    pub splat_smudge: bool,
    /// Pen pressure mapping: `p_eff = min(p · pressure_multiplier, 1)`. `pressure_max` does
    /// not scale the response; it is kept for presets but unused.
    pub pressure_multiplier: f32,
    pub pressure_max: f32,
    /// How strongly the brush's paper texture (`PaperTexture_0`) gates deposition.
    pub paper_structure_influence: f32,
    /// Explicit pressure response curves (override the influence values when present).
    pub pressure_size_curve: Option<curve::Curve>,
    pub pressure_opacity_curve: Option<curve::Curve>,
}

impl Default for BrushParams {
    fn default() -> Self {
        Self {
            size: 30.0,
            opacity: 1.0,
            water: 1.0,
            spacing: 0.1,
            spacing_jitter: 0.0,
            size_jitter: 0.0,
            opacity_jitter: 0.0,
            rotation: 0.0,
            rotation_jitter: 0.0,
            scatter_jitter: 0.0,
            splat_opacity: 1.0,
            pressure_size_influence: 0.0,
            pressure_opacity_influence: 0.0,
            rotation_mode: RotationMode::None,
            paint_mode: PaintMode::Mix,
            glaze_blend: wet_compose::MixMode::Normal,
            glaze_mode: GlazeMode::SemiOpaque,
            only_water: false,
            dirty: false,
            grain_scale: 1.0,
            grain_brush_size: 0.0,
            grain_follow_brush_size: false,
            grain_follow_distance: false,
            grain_follow_amount: 1.0,
            splat_smudge: false,
            grain_follow_shape_rotation: false,
            pressure_multiplier: 1.0,
            pressure_max: 1.0,
            paper_structure_influence: 0.0,
            pressure_size_curve: None,
            pressure_opacity_curve: None,
        }
    }
}

impl BrushParams {
    pub fn from_preset(p: &BrushPreset) -> Self {
        let d = Self::default();
        let f = |k: &str, def: f32| p.param_f64(k).map_or(def, |v| v as f32);
        let b = |k: &str| p.parameters.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
        let rotation_mode = match p.parameters.get("brush_rotation_mode").and_then(|v| v.as_str()) {
            Some("FOLLOW_TRAJECTORY") => RotationMode::FollowTrajectory,
            Some("PEN_TILT") => RotationMode::PenTilt,
            Some("PEN_ROTATION") => RotationMode::PenRotation,
            _ => RotationMode::None,
        };
        Self {
            size: p.volume("size").map_or(d.size, |v| v as f32),
            opacity: p.volume("opacity").map_or(d.opacity, |v| v as f32),
            water: p.volume("water").map_or(d.water, |v| v as f32),
            spacing: f("splat_spacing", d.spacing),
            spacing_jitter: f("splat_spacing_jitter", 0.0),
            size_jitter: f("splat_size_jitter", 0.0),
            opacity_jitter: f("splat_opacity_jitter", 0.0),
            rotation: f("splat_rotation", 0.0),
            rotation_jitter: f("splat_rotation_jitter", 0.0),
            scatter_jitter: f("splat_scattering_jitter", 0.0),
            splat_opacity: f("splat_opacity", 1.0) * f("splat_opacity_multiplier", 1.0),
            pressure_size_influence: f("pressure_size_influence", 0.0),
            pressure_opacity_influence: f("pressure_opacity_influence", 0.0),
            rotation_mode,
            paint_mode: p
                .parameters_ex
                .as_ref()
                .and_then(|v| v.get("painting_params")?.get("paint_type")?.as_str())
                .and_then(PaintMode::from_name)
                .unwrap_or(PaintMode::Mix),
            glaze_blend: p
                .parameters
                .get("glaze_rgb_blend_mode")
                .and_then(|v| v.as_str())
                .and_then(wet_compose::MixMode::from_name)
                .unwrap_or(wet_compose::MixMode::Normal),
            glaze_mode: GlazeMode::SemiOpaque,
            only_water: false,
            dirty: false,
            grain_scale: f("grain_scale", 1.0),
            grain_brush_size: f("grain_brush_size", 0.0),
            grain_follow_brush_size: b("grain_follow_brush_size"),
            grain_follow_distance: b("grain_follow_distance"),
            grain_follow_amount: f("grain_tile_follow_distance", 1.0).clamp(0.0, 1.0),
            splat_smudge: b("splat_smudge"),
            grain_follow_shape_rotation: b("grain_follow_shape_rotation"),
            pressure_multiplier: f("pressure_multiplier", 1.0),
            pressure_max: f("pressure_max", 1.0),
            paper_structure_influence: f("paper_structure_influence", 0.0),
            pressure_size_curve: p.parameters.get("pressure_size_curve").and_then(curve::Curve::from_json),
            pressure_opacity_curve: p.parameters.get("pressure_opacity_curve").and_then(curve::Curve::from_json),
        }
    }

    /// Write these parameters back into a preset's JSON, the inverse of
    /// [`Self::from_preset`] (keys it does not read are left as they are; the splat opacity
    /// multiplier is folded into `splat_opacity`). Glaze mode, only-water and the dirty brush
    /// are session settings, not preset values, and are not written.
    pub fn write_preset(&self, p: &mut BrushPreset) {
        use serde_json::{Value, json};
        if !p.parameters.is_object() {
            p.parameters = json!({});
        }
        let m = p.parameters.as_object_mut().expect("object");
        let mut num = |k: &str, v: f32| {
            m.insert(k.into(), json!(v as f64));
        };
        num("splat_spacing", self.spacing);
        num("splat_spacing_jitter", self.spacing_jitter);
        num("splat_size_jitter", self.size_jitter);
        num("splat_opacity_jitter", self.opacity_jitter);
        num("splat_rotation", self.rotation);
        num("splat_rotation_jitter", self.rotation_jitter);
        num("splat_scattering_jitter", self.scatter_jitter);
        num("splat_opacity", self.splat_opacity);
        num("splat_opacity_multiplier", 1.0);
        num("pressure_size_influence", self.pressure_size_influence);
        num("pressure_opacity_influence", self.pressure_opacity_influence);
        num("grain_scale", self.grain_scale);
        num("grain_brush_size", self.grain_brush_size);
        num("grain_tile_follow_distance", self.grain_follow_amount);
        num("pressure_multiplier", self.pressure_multiplier);
        num("pressure_max", self.pressure_max);
        num("paper_structure_influence", self.paper_structure_influence);
        for (k, v) in [
            ("grain_follow_brush_size", self.grain_follow_brush_size),
            ("grain_follow_distance", self.grain_follow_distance),
            ("splat_smudge", self.splat_smudge),
            ("grain_follow_shape_rotation", self.grain_follow_shape_rotation),
        ] {
            m.insert(k.into(), Value::Bool(v));
        }
        let rot = match self.rotation_mode {
            RotationMode::None => "NONE",
            RotationMode::FollowTrajectory => "FOLLOW_TRAJECTORY",
            RotationMode::PenTilt => "PEN_TILT",
            RotationMode::PenRotation => "PEN_ROTATION",
        };
        m.insert("brush_rotation_mode".into(), json!(rot));
        m.insert("glaze_rgb_blend_mode".into(), json!(self.glaze_blend.name()));
        let ex = p.parameters_ex.get_or_insert_with(|| json!({}));
        if !ex.is_object() {
            *ex = json!({});
        }
        let ex = ex.as_object_mut().expect("object");
        let set = |ex: &mut serde_json::Map<String, Value>, group: &str, k: &str, v: Value| {
            let g = ex.entry(group).or_insert_with(|| json!({}));
            if !g.is_object() {
                *g = json!({});
            }
            g.as_object_mut().expect("object").insert(k.into(), v);
        };
        set(ex, "volume_params", "size", json!(self.size as f64));
        set(ex, "volume_params", "opacity", json!(self.opacity as f64));
        set(ex, "volume_params", "water", json!(self.water as f64));
        set(ex, "painting_params", "paint_type", json!(self.paint_mode.name()));
    }

    /// Pressure response: `1 - influence + influence·p` (explicit curves are applied separately).
    fn pressure_factor(influence: f32, p: f32) -> f32 {
        1.0 - influence + influence * p.clamp(0.0, 1.0)
    }

    /// Effective pressure `min(p · pressure_multiplier, 1)` fed to explicit pressure curves
    /// and to influence-based responses `1 − k + k·p_eff`.
    pub fn effective_pressure(&self, p: f32) -> f32 {
        (p * self.pressure_multiplier).clamp(0.0, 1.0)
    }

    pub fn size_at(&self, pressure: f32) -> f32 {
        let pe = self.effective_pressure(pressure);
        self.size
            * match &self.pressure_size_curve {
                Some(c) => c.eval(pe),
                None => Self::pressure_factor(self.pressure_size_influence, pe),
            }
    }

    /// Pressure response of opacity (curve or influence).
    pub fn opacity_factor(&self, pressure: f32) -> f32 {
        let pe = self.effective_pressure(pressure);
        match &self.pressure_opacity_curve {
            Some(c) => c.eval(pe),
            None => Self::pressure_factor(self.pressure_opacity_influence, pe),
        }
    }
}

/// Calibration of the injection.
///
/// Each stroke keeps its own coverage masks. A dab with shape value `s` and grain `g`
/// raises the paint mask by `Δm = (1 − m)·min(1, rate·splat_opacity·s·g)` and the water
/// mask by `(1 − m_w)·s`. A paint dab adds `A·Δm` pigment (`A` = the medium's amount ×
/// opacity) and, for wet media, `W·Δm_w` water (`W = water_target · water^water_gamma`),
/// and removes `pick·Δm` of the wet paint that was there before the stroke. Overlapping
/// dabs saturate within a stroke (flat top, linear in opacity) while separate strokes add.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    /// Eraser removal, blend and smudge strengths per unit of stroke mask.
    pub erase: f32,
    pub blend: f32,
    pub smudge_deposit: f32,
    pub smudge_pick: f32,
    /// Water follows the (grain-modulated) paint mask instead of the plain shape.
    pub water_follows_paint: bool,
    /// PAINT mode pigment relative to PAINT_MIX, and the share of picked-up paint the
    /// brush carries on.
    pub paint_only_scale: f32,
    pub carry: f32,
    /// Gain on released carried pigment, and the share of the reservoir released per unit
    /// stroke mask on clean paper.
    pub tint_k: f32,
    pub release: f32,
    /// Force `amount_per_splat · splat_opacity` targets for every medium (otherwise per
    /// medium, `MediumModel::splat_target`).
    pub splat_scales_target: bool,
    /// Pigment gain of bristle lines relative to the medium amount (round bristle
    /// lines reach coverage ≈ 1).
    pub bristle_gain: f32,
    /// Per-dab rate of the water mask (1 = saturate under the first dab; small values build
    /// a centre-heavy water profile from overlapping dabs).
    pub water_rate: f32,
    /// Contrast gain of the brush paper texture's cap (pencil comparison).
    pub tooth_gain: f32,
    /// Below `tooth_knee` effective pressure the brush paper texture's cap drops by up to
    /// `tooth_light`, so light strokes only catch the paper's peaks.
    pub tooth_light: f32,
    pub tooth_knee: f32,
    pub water_target: f32,
    pub water_gamma: f32,
    pub rate: f32,
    /// Shape-to-dab radius factor.
    pub radius_factor: f32,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            erase: 1.4,
            blend: 10.0,
            smudge_deposit: 0.55,
            smudge_pick: 3.0,
            water_follows_paint: false,
            tooth_gain: 3.5,
            tooth_light: 3.0,
            tooth_knee: 0.45,
            water_rate: 0.16,
            bristle_gain: 10.0,
            splat_scales_target: false,
            paint_only_scale: 0.75,
            carry: 0.2,
            tint_k: 1.0,
            release: 0.05,
            water_target: 4.7 * wet_fluid::params::WATER_SCALE,
            water_gamma: 3.3,
            rate: 2.2,
            radius_factor: 1.0,
        }
    }
}

/// The colour a cell shows over white paper (settled then suspended paint), 0–255.
fn visible_color(l: &WetLayer, c: usize) -> [f32; 3] {
    let mix = wet_fluid::mix::mix;
    let mc = [0, 1, 2].map(|k| ld(l.mid.rgb[k].cur(), c));
    let dc = [0, 1, 2].map(|k| ld(l.dens.rgb[k].cur(), c));
    let col = mix([255.0; 3], 1.0, mc, l.lut.eval(l.mid.amount.cur().data[c]), None);
    mix(col, 1.0, dc, l.lut.eval(l.dens.amount.cur().data[c]), None)
}

/// Watercolour glazing modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlazeMode {
    Transparent,
    SemiOpaque,
    Opaque,
}

impl GlazeMode {
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "TRANSPARENT" => Some(Self::Transparent),
            "SEMI-OPAQUE" | "SEMI_OPAQUE" => Some(Self::SemiOpaque),
            "OPAQUE" => Some(Self::Opaque),
            _ => None,
        }
    }

    /// (amount factor, cover): the deposit is scaled by the factor, and the dry paint
    /// below loses `cover × new coverage` of its own coverage. Fitted on a blue stroke over
    /// paper and over a dried red band: semi-opaque is the calibrated
    /// baseline, transparent lays 0.85× the pigment, opaque also hides half of the
    /// covered dry paint.
    fn effect(self) -> (f32, f32) {
        match self {
            Self::Transparent => (0.85, 0.0),
            Self::SemiOpaque => (1.0, 0.0),
            Self::Opaque => (1.0, 0.5),
        }
    }
}

/// Pigment gain of watercolour brushes with a canvas-anchored grain relative to the
/// streak-grain calibration reference (fitted over 1–7).
pub const ANCHORED_GRAIN_GAIN: f32 = 4.0;

/// Stroke length, in dab diameters, over which a multi-coloured brush's colours blend
/// into their average.
pub const MULTI_MIX_LENGTH: f32 = 8.0;

/// A loaded brush: parameters, tip images and the per-stroke state.
#[derive(Clone)]
pub struct Brush {
    pub params: BrushParams,
    /// Pen orientation of the current pointer sample: an explicit tip rotation in degrees
    /// and whether the input carries pen tilt.
    pub pen_rotation: Option<f32>,
    pub pen_tilt: bool,
    /// Share of new paint the current pointer sample lays (0–1): scales the
    /// deposit of every paint type except plain PAINT.
    pub paint_ratio: f32,
    pub calib: Calibration,
    pub tool: Tool,
    pub medium: MediumModel,
    /// Smudge carry strength (0–1) and blend strength.
    pub smudge: f32,
    pub shape: Mask,
    pub grain: Option<Mask>,
    /// Extra shapes/grains (`Shape_1..3`, `Texture_1..3`) and how a dab picks one
    /// (`texture_order`, `texture_<i>_start/end` in % of pressure).
    pub alt_shapes: Vec<Mask>,
    pub dual: Option<DualBrush>,
    pub alt_grains: Vec<Mask>,
    pub texture_order: TextureOrder,
    pub texture_ranges: [(f32, f32); 4],
    dab_count: usize,
    /// The preset's paper texture (`PaperTexture_0`), tiled on the canvas.
    pub paper_tex: Option<Mask>,
    /// Bristle groups (bristle presets paint with bristles instead of the shape).
    pub bristle: Option<BristleParams>,
    bristles: BristleSet,
    last: Option<Sample>,
    pub color: [f32; 3],
    spacer: Spacer,
    rng: Rng,
    seed: u64,
    /// Per-stroke paint mask, water mask and pigment deposited by this stroke.
    mask: Vec<f32>,
    mask_w: Vec<f32>,
    own: Vec<f32>,
    /// Relief (settled paint thickness) laid by this stroke.
    own_th: Vec<f32>,
    /// Per-stroke dual-brush mask `D`.
    dual_acc: Vec<f32>,
    /// Centre of the previous dab of this stroke (for `splat_smudge` sweeps).
    prev_dab: Option<(f32, f32)>,
    /// Multi-coloured brush: colours loaded side by side across the tip (two or more
    /// enable it). They blend into their average over `MULTI_MIX_LENGTH` dab diameters.
    pub multi: Vec<[f32; 3]>,
    /// Distance the dabs of this stroke have travelled (px).
    stroke_dist: f32,
    /// Paint carried by a smudge stroke.
    carry: Option<CellPaint>,
    /// Paint picked up by the brush (colour 0–255, amount), mixed into what it lays.
    reservoir: ([f32; 3], f32),
    /// Current dab (centre x, centre y, radius, direction) for the warping tools, and the
    /// paint before the current Liquify session (for Reconstruct).
    dab: (f32, f32, f32, f32),
    stroke_start: Vec<CellPaint>,
}

impl Brush {
    pub fn new(params: BrushParams, shape: Mask, grain: Option<Mask>) -> Self {
        Self {
            params,
            calib: Calibration::default(),
            tool: Tool::Paint(Medium::Watercolor),
            medium: Medium::Watercolor.model(),
            smudge: 0.1,
            shape,
            grain,
            alt_shapes: Vec::new(),
            dual: None,
            alt_grains: Vec::new(),
            texture_order: TextureOrder::Sequential,
            pen_rotation: None,
            pen_tilt: false,
            paint_ratio: 1.0,
            texture_ranges: [(0.0, 100.0); 4],
            dab_count: 0,
            paper_tex: None,
            bristle: None,
            bristles: BristleSet::default(),
            last: None,
            color: [0.0, 0.0, 0.0],
            spacer: Spacer::new(),
            rng: Rng::new(1),
            seed: 1,
            mask: Vec::new(),
            mask_w: Vec::new(),
            own: Vec::new(),
            own_th: Vec::new(),
            dual_acc: Vec::new(),
            prev_dab: None,
            multi: Vec::new(),
            stroke_dist: 0.0,
            carry: None,
            reservoir: ([0.0; 3], 0.0),
            dab: (0.0, 0.0, 0.0, 0.0),
            stroke_start: Vec::new(),
        }
    }

    /// Build from a preset: `Shape_0` is the dab, `Texture_0` the grain.
    pub fn from_preset(p: &BrushPreset) -> Self {
        let tex = |slot: &str| p.textures.get(slot).and_then(|b| Mask::from_png(b).ok());
        let shape = tex("Shape_0").unwrap_or_else(|| Mask::disk(128, 0.5));
        let mut b = Self::new(BrushParams::from_preset(p), shape, tex("Texture_0"));
        b.paper_tex = tex("PaperTexture_0");
        b.alt_shapes = (1..4).map_while(|i| tex(&format!("Shape_{i}"))).collect();
        b.dual = tex("DualBrushShape_0").and_then(|m| DualBrush::from_preset(p, m));
        b.alt_grains = (1..4).map_while(|i| tex(&format!("Texture_{i}"))).collect();
        b.texture_order = match p.parameters.get("texture_order").and_then(|v| v.as_str()) {
            Some("RANDOM") => TextureOrder::Random,
            Some("PRESSURE") => TextureOrder::Pressure,
            Some("TILT") => TextureOrder::Tilt,
            _ => TextureOrder::Sequential,
        };
        for i in 0..4 {
            let g = |k: String, d: f32| p.param_f64(&k).map_or(d, |v| v as f32);
            b.texture_ranges[i] = (g(format!("texture_{i}_start"), 0.0), g(format!("texture_{i}_end"), 100.0));
        }
        b.bristle = BristleParams::from_preset(p);
        b.set_tool(Tool::from_id(&p.tool));
        b
    }

    pub fn set_tool(&mut self, tool: Tool) {
        if !matches!(tool, Tool::Liquify(_)) {
            self.stroke_start.clear();
        }
        self.tool = tool;
        if let Tool::Paint(m) = tool {
            self.medium = m.model();
        }
    }

    /// Seed for the per-stroke jitter sequence (`SET_RANDOM_SEED` analogue).
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    pub fn press(&mut self, s: Sample) {
        self.rng = Rng::new(self.seed);
        self.spacer.press(s);
        for buf in [&mut self.mask, &mut self.mask_w, &mut self.own, &mut self.own_th, &mut self.dual_acc] {
            buf.iter_mut().for_each(|v| *v = 0.0);
        }
        self.carry = None;
        if !self.params.dirty {
            self.reservoir = ([0.0; 3], 0.0);
        }
        self.prev_dab = None;
        self.stroke_dist = 0.0;
        self.last = Some(s);
        if let Some(bp) = &self.bristle {
            self.bristles = BristleSet::new(bp, &mut self.rng);
        }
    }

    pub fn move_to(&mut self, layer: &mut WetLayer, s: Sample) {
        if self.bristle.is_some() {
            self.bristle_segment(layer, s);
            return;
        }
        let mut dabs = Vec::new();
        let (params, rng) = (&self.params, &mut self.rng);
        self.spacer.move_to(
            s,
            |p| {
                let base = params.spacing * params.size_at(p);
                base * (1.0 + params.spacing_jitter * rng.signed())
            },
            |d| dabs.push(d),
        );
        for d in dabs {
            self.stamp(layer, d);
        }
    }

    /// Paint along polylines ("stroke path"), one stroke per subpath, at a constant
    /// pressure.
    pub fn stroke_path(&mut self, layer: &mut WetLayer, subpaths: &[Vec<(f32, f32)>], pressure: f32) {
        for sp in subpaths.iter().filter(|sp| sp.len() >= 2) {
            let at = |&(x, y): &(f32, f32)| Sample { x, y, pressure };
            self.press(at(&sp[0]));
            for p in &sp[1..] {
                self.move_to(layer, at(p));
            }
            self.release(layer, at(sp.last().unwrap()));
        }
    }

    pub fn release(&mut self, layer: &mut WetLayer, s: Sample) {
        if self.bristle.is_some() {
            self.bristle_segment(layer, s);
            self.last = None;
            return;
        }
        let mut dabs = Vec::new();
        self.spacer.release(s, |d| dabs.push(d));
        for d in dabs {
            self.stamp(layer, d);
        }
    }

    /// The canvas rectangle (x0, y0, x1, y1, inclusive, within `w`×`h`) that
    /// [`Self::move_to`] or [`Self::release`] at `s` can read or change: the dabs lie on
    /// the segment from the previous pointer position, swept from the previous dab,
    /// widened by the largest dab, scatter and dual-brush reach. `None`: the tool reads
    /// the whole layer (Liquify, Clone).
    pub fn reach_bounds(&self, s: Sample, w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
        if matches!(self.tool, Tool::Liquify(_) | Tool::Clone { .. }) {
            return None;
        }
        let p = &self.params;
        let size = (0..=16).map(|i| p.size_at(i as f32 / 16.0)).fold(p.size_at(s.pressure), f32::max) * (1.0 + p.size_jitter.abs());
        let mut reach = 0.5 * size * self.calib.radius_factor + p.scatter_jitter.abs() * size;
        if let Some(db) = &self.dual {
            let rr = (0.25 * size * db.size * (1.0 + db.size_jitter.abs())).max(0.5);
            reach = reach.max(0.5 * size * self.calib.radius_factor + size * (db.min_scatter + db.scatter).abs() + rr);
        }
        if let Some(bp) = &self.bristle {
            reach = reach.max(size * (1.0 + bp.wave_jitter.abs()) + 2.0 * bp.edge_size.unwrap_or(0.0));
        }
        let reach = reach + 3.0;
        let points = [Some((s.x, s.y)), self.spacer.last().map(|a| (a.x, a.y)), self.last.map(|a| (a.x, a.y)), self.prev_dab];
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (x, y) in points.into_iter().flatten() {
            (x0, y0, x1, y1) = (x0.min(x - reach), y0.min(y - reach), x1.max(x + reach), y1.max(y + reach));
        }
        let (x0, y0) = (x0.floor().max(0.0) as usize, y0.floor().max(0.0) as usize);
        let (x1, y1) = ((x1.ceil().max(0.0) as usize).min(w - 1), (y1.ceil().max(0.0) as usize).min(h - 1));
        // off the canvas: nothing to read
        Some(if x0 <= x1 && y0 <= y1 { (x0, y0, x1, y1) } else { (0, 0, 0, 0) })
    }

    /// Drag the bristles from the previous pointer position to `s`.
    fn bristle_segment(&mut self, layer: &mut WetLayer, s: Sample) {
        let (Some(a), Some(bp)) = (self.last, self.bristle.clone()) else {
            self.last = Some(s);
            return;
        };
        self.last = Some(s);
        let p = &self.params;
        let pressure = 0.5 * (a.pressure + s.pressure);
        let size = p.size_at(pressure);
        let opacity = (p.opacity * self.medium.opacity_gain).min(1.0) * p.opacity_factor(pressure);
        let water_t = self.calib.water_target * p.water.max(0.0).powf(self.calib.water_gamma);
        let amount_t = self.medium.amount * self.calib.bristle_gain * opacity;
        let (w, h) = (layer.w, layer.h);
        let n = w * h;
        if self.mask.len() != n {
            for buf in [&mut self.mask, &mut self.mask_w, &mut self.own, &mut self.own_th, &mut self.dual_acc] {
                *buf = vec![0.0; n];
            }
        }
        let mut cells = Vec::new();
        let (mask, mask_w) = (&mut self.mask, &mut self.mask_w);
        self.bristles.segment(&bp, (a.x, a.y), (s.x, s.y), size, |cx, cy, r| {
            let (x0, x1) = ((cx - r - 1.0).floor().max(0.0) as usize, ((cx + r + 1.0).ceil() as usize).min(w - 1));
            let (y0, y1) = ((cy - r - 1.0).floor().max(0.0) as usize, ((cy + r + 1.0).ceil() as usize).min(h - 1));
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                    let cov = (r + 0.5 - d).clamp(0.0, 1.0);
                    if cov <= 0.0 {
                        continue;
                    }
                    let c = y * w + x;
                    let dm = (1.0 - mask[c]) * cov;
                    let dmw = (1.0 - mask_w[c]) * cov;
                    mask[c] += dm;
                    mask_w[c] += dmw;
                    if dm > 0.0 || dmw > 0.0 {
                        cells.push((c, dm, dmw));
                    }
                }
            }
        });
        let texture = vec![1.0; cells.len()];
        let angle = (s.y - a.y).atan2(s.x - a.x);
        self.apply(layer, &cells, &texture, amount_t, water_t, (angle, opacity), w);
    }

    /// Stamp one dab into the layer.
    pub fn stamp(&mut self, layer: &mut WetLayer, d: DabPos) {
        let p = &self.params;
        let size = p.size_at(d.pressure) * (1.0 + p.size_jitter * self.rng.signed());
        let r = 0.5 * size * self.calib.radius_factor;
        if r < 0.25 {
            return;
        }
        let pe_dab = p.effective_pressure(d.pressure);
        let opacity = (p.opacity * self.medium.opacity_gain).min(1.0)
            * p.opacity_factor(d.pressure)
            * (1.0 - p.opacity_jitter * self.rng.next_f32());
        let mut angle = p.rotation.to_radians() + p.rotation_jitter * std::f32::consts::PI * self.rng.signed();
        // pen-rotation and pen-tilt brushes follow the stroke when the input carries no pen
        // orientation (a mouse), so flat tips paint vertical strokes as wide as horizontal
        // ones. An explicit rotation sets the tip angle; pen tilt without rotation turns it
        // to 90° whatever the azimuth
        match p.rotation_mode {
            RotationMode::PenRotation | RotationMode::PenTilt => match self.pen_rotation {
                Some(r) => angle += r.to_radians(),
                None if self.pen_tilt => angle += std::f32::consts::FRAC_PI_2,
                None => angle += d.angle,
            },
            RotationMode::FollowTrajectory => angle += d.angle,
            RotationMode::None => {}
        }
        let scatter = p.scatter_jitter * size;
        let (cx, cy) = (
            d.x + scatter * self.rng.signed(),
            d.y + scatter * self.rng.signed(),
        );
        let water_t = self.calib.water_target * p.water.max(0.0).powf(self.calib.water_gamma);
        // splat opacity scales the target amount;
        // per-medium base amounts are per unit splat opacity
        let (amount_t, rate) = if self.medium.splat_target || self.calib.splat_scales_target {
            (self.medium.amount_per_splat * opacity * p.splat_opacity, self.calib.rate * 0.5)
        } else {
            (self.medium.amount * opacity, self.calib.rate * p.splat_opacity)
        };
        // watercolour brushes whose grain stays on the canvas (not streaking along the
        // stroke) and that do not scatter lay more pigment than the streak-grain
        // brush the target was calibrated on
        let solid = p.size_jitter == 0.0 && p.scatter_jitter < 0.1;
        let amount_t = if self.medium.wet && self.medium.splat_target && self.grain.is_some() && !p.grain_follow_distance && solid {
            amount_t * ANCHORED_GRAIN_GAIN
        } else {
            amount_t
        };
        self.dab = (cx, cy, r, d.angle);
        // the Liquify session snapshot: taken at the first liquify dab, kept until another
        // tool is chosen, so Reconstruct restores what was there before warping began
        if matches!(self.tool, Tool::Liquify(_)) && self.stroke_start.is_empty() {
            self.stroke_start = (0..layer.w * layer.h).map(|c| tool::read_cell(layer, c)).collect();
        }
        // Grain is anchored to the canvas; its pixel scale optionally follows brush size.
        let grain_px = if p.grain_follow_brush_size && p.grain_brush_size > 0.0 {
            p.grain_scale * size / p.grain_brush_size
        } else {
            p.grain_scale
        }
        .max(1e-3);
        // pick this dab's shape/grain (index 0 = Shape_0/Texture_0)
        let n_tex = 1 + self.alt_shapes.len().max(self.alt_grains.len());
        let tex_i = match self.texture_order {
            _ if n_tex == 1 => 0,
            TextureOrder::Sequential => self.dab_count % n_tex,
            TextureOrder::Random => ((self.rng.next_f32() * n_tex as f32) as usize).min(n_tex - 1),
            TextureOrder::Pressure | TextureOrder::Tilt => {
                let pp = 100.0 * p.effective_pressure(d.pressure);
                (0..n_tex)
                    .rev()
                    .find(|&i| pp >= self.texture_ranges[i].0 && pp <= self.texture_ranges[i].1.max(self.texture_ranges[i].0))
                    .unwrap_or(0)
            }
        };
        self.dab_count += 1;
        let shape = if tex_i == 0 { &self.shape } else { self.alt_shapes.get(tex_i - 1).unwrap_or(&self.shape) };
        let grain = if tex_i == 0 { self.grain.as_ref() } else { self.alt_grains.get(tex_i - 1).or(self.grain.as_ref()) };
        // dual brush stamps for this dab: (offset x, offset y, radius, cos, sin, opacity)
        let duals: Vec<(f32, f32, f32, f32, f32, f32)> = match &self.dual {
            Some(db) => (0..db.count)
                .map(|_| {
                    let dist = size * (db.min_scatter + db.scatter * self.rng.next_f32());
                    let a = std::f32::consts::TAU * self.rng.next_f32();
                    // stamp diameter ≈ ½·size·dual size
                    let rr = 0.25 * size * db.size * (1.0 + db.size_jitter * self.rng.signed());
                    let rot = db.rotation_jitter * std::f32::consts::PI * self.rng.signed();
                    let op = db.opacity * (1.0 - db.opacity_jitter * self.rng.next_f32());
                    (dist * a.cos(), dist * a.sin(), rr.max(0.5), rot.cos(), rot.sin(), op)
                })
                .collect(),
            None => Vec::new(),
        };
        let (sin, cos) = angle.sin_cos();
        let (w, h) = (layer.w as i64, layer.h as i64);
        let n = layer.w * layer.h;
        if self.mask.len() != n {
            for buf in [&mut self.mask, &mut self.mask_w, &mut self.own, &mut self.own_th, &mut self.dual_acc] {
                *buf = vec![0.0; n];
            }
        }
        // sweep positions relative to the centre (just the centre without splat_smudge)
        let sweep: Vec<(f32, f32)> = match self.prev_dab.filter(|_| p.splat_smudge) {
            Some((px, py)) => {
                let n = ((cx - px).hypot(cy - py).ceil() as usize).clamp(1, 64);
                (1..=n)
                    .map(|k| {
                        let t = 1.0 - k as f32 / n as f32;
                        (t * (px - cx), t * (py - cy))
                    })
                    .collect()
            }
            None => vec![(0.0, 0.0)],
        };
        if let Some((px, py)) = self.prev_dab {
            self.stroke_dist += (cx - px).hypot(cy - py);
        }
        self.prev_dab = Some((cx, cy));
        let (ox0, oy0, ox1, oy1) = sweep.iter().fold((0.0f32, 0.0f32, 0.0f32, 0.0f32), |a, &(ox, oy)| {
            (a.0.min(ox), a.1.min(oy), a.2.max(ox), a.3.max(oy))
        });
        // dual stamps can reach beyond the main dab
        let reach = duals.iter().map(|d| d.0.hypot(d.1) + d.2).fold(r, f32::max);
        let (x0, x1) = (((cx + ox0 - reach).floor() as i64).max(0), ((cx + ox1 + reach).ceil() as i64).min(w - 1));
        let (y0, y1) = (((cy + oy0 - reach).floor() as i64).max(0), ((cy + oy1 + reach).ceil() as i64).min(h - 1));
        let mut cells = Vec::new();
        let mut texture = Vec::new();
        // prefilter the tip for its minification, with a one-level sharpening bias (a full
        // prefilter looks slightly too soft; point sampling misses the dots of sparse
        // bristle tips)
        let shape_lod = 0.5 * shape.w.max(shape.h) as f32 / (2.0 * r);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (dx, dy) = (x as f32 - cx, y as f32 - cy);
                // rotate into dab space, then to normalised texture coordinates
                let (u, v) = (dx * cos + dy * sin, -dx * sin + dy * cos);
                // with splat_smudge the grain of oils travels with the dab and is
                // swept with it, drawing streaks along the stroke instead of canvas speckle
                let swept_grain = p.splat_smudge && !self.medium.wet && self.medium.thickness > 0.0 && !p.grain_follow_shape_rotation && p.scatter_jitter < 0.1;
                let s = sweep
                    .iter()
                    .map(|&(ox, oy)| {
                        let (ex, ey) = (dx - ox, dy - oy);
                        let (su, sv) = (ex * cos + ey * sin, -ex * sin + ey * cos);
                        let sh = shape.sample_lod(0.5 + su / (2.0 * r), 0.5 + sv / (2.0 * r), shape_lod);
                        match grain.filter(|_| swept_grain) {
                            Some(g) => sh * g.sample_tiled(su / grain_px + self.seed as f32 * 37.0, sv / grain_px),
                            None => sh,
                        }
                    })
                    .fold(0.0f32, f32::max);
                let c = (y * w + x) as usize;
                // dual-brush stamps add to the stroke's dual layer: (weighted value before, after)
                let dual_f = self.dual.as_ref().map(|db| {
                    let hit = duals
                        .iter()
                        .map(|&(ox, oy, rr, c2, s2, op)| {
                            let (qx, qy) = (dx - ox, dy - oy);
                            let (du, dv) = (qx * c2 + qy * s2, -qx * s2 + qy * c2);
                            op * db.shape.sample_lod_normalized(
                                0.5 + du / (2.0 * rr),
                                0.5 + dv / (2.0 * rr),
                                db.shape.w as f32 / (2.0 * rr),
                            )
                        })
                        .fold(0.0f32, f32::max);
                    let d0 = self.dual_acc[c];
                    let d1 = d0 + (1.0 - d0) * hit.min(1.0);
                    self.dual_acc[c] = d1;
                    (db.weight * d0, db.weight * d1)
                });
                let confined = self.dual.as_ref().is_some_and(|d| d.alternative);
                // the dual layer adds coverage (union); alternative-mode stamps only inside the
                // main dab, normal-mode stamps may reach beyond it
                if s <= 0.0 && (confined || dual_f.is_none_or(|(f0, f1)| f1 <= f0)) {
                    continue;
                }
                let (gx, gy) = if p.grain_follow_distance && p.grain_follow_amount >= 1.0 {
                    (u, v)
                } else if p.grain_follow_distance {
                    let t = p.grain_follow_amount;
                    (x as f32 - t * cx, y as f32 - t * cy)
                } else if p.grain_follow_shape_rotation {
                    let (xf, yf) = (x as f32, y as f32);
                    (xf * cos + yf * sin, -xf * sin + yf * cos)
                } else {
                    (x as f32, y as f32)
                };
                let g = if swept_grain {
                    1.0
                } else {
                    grain.map_or(1.0, |g| g.sample_tiled(gx / grain_px, gy / grain_px))
                };
                // the brush's paper tooth gates deposition (dry media catch the ridges)
                let tooth = self.paper_tex.as_ref().map_or(1.0, |t| {
                    let k = p.paper_structure_influence.clamp(0.0, 1.0);
                    let cap = (1.0 - k + k * self.calib.tooth_gain * t.sample_tiled(x as f32, y as f32)).min(1.0);
                    // light pressure only reaches the paper's peaks
                    let knee = self.calib.tooth_knee;
                    (cap - self.calib.tooth_light * ((knee - pe_dab) / knee).clamp(0.0, 1.0)).clamp(0.0, 1.0)
                });
                // Grain normally modulates the deposit rate (calibrated on a streaking
                // watercolour brush). A grain that rotates with the dab is a bristle-streak texture: for settled
                // media it caps the reachable coverage, leaving dry-brush streaks.
                let streaks = p.grain_follow_shape_rotation
                    && !self.medium.wet
                    && matches!(self.tool, Tool::Paint(_));
                let (g_rate, g_cap) = if streaks { (1.0, g) } else { (g, 1.0) };
                let r = (rate * s * g_rate).min(1.0);
                let m0 = self.mask[c];
                self.mask[c] += (1.0 - m0) * r;
                // the dual layer is composited onto the stroke mask with its blend weight
                // (NORMAL: over, i.e. union)
                let dm = match dual_f {
                    Some((f0, f1)) => {
                        let e = |m: f32, f: f32| 1.0 - (1.0 - m) * (1.0 - f);
                        (e(self.mask[c], f1) - e(m0, f0)).max(0.0)
                    }
                    None => self.mask[c] - m0,
                };
                let dmw = (1.0 - self.mask_w[c]) * (self.calib.water_rate * s).min(1.0);
                self.mask_w[c] += dmw;
                // the tooth caps how much paint a cell can take (valleys stay light)
                cells.push((c, dm * tooth * g_cap, dmw));
                texture.push(s.min(1.0));
            }
        }
        self.apply(layer, &cells, &texture, amount_t, water_t, (d.angle, opacity), w as usize);
    }

    /// The colour the brush lays at cell `c`: its colour, or for a multi-coloured brush
    /// the band of the tip the cell lies under (across the stroke direction), blended
    /// towards the average of the loaded colours as the stroke goes on.
    fn color_at(&self, c: usize, w: usize) -> [f32; 3] {
        let n = self.multi.len();
        if n < 2 {
            return self.color;
        }
        let (cx, cy, r, angle) = self.dab;
        let (x, y) = ((c % w) as f32 - cx, (c / w) as f32 - cy);
        let (sn, cs) = angle.sin_cos();
        let across = ((-x * sn + y * cs) / r.max(1e-3)).clamp(-1.0, 1.0);
        let band = (((across + 1.0) * 0.5 * n as f32) as usize).min(n - 1);
        let avg = [0, 1, 2].map(|k| self.multi.iter().map(|m| m[k]).sum::<f32>() / n as f32);
        let t = (self.stroke_dist / (MULTI_MIX_LENGTH * 2.0 * r.max(1e-3))).min(1.0);
        [0, 1, 2].map(|k| self.multi[band][k] + (avg[k] - self.multi[band][k]) * t)
    }

    /// Dab-weighted average of the paint under a dab.
    fn dab_average(layer: &WetLayer, cells: &[(usize, f32, f32)]) -> Option<CellPaint> {
        let mut avg = CellPaint::default();
        let mut wsum = 0.0;
        for &(c, dm, dmw) in cells {
            let wgt = dm.max(dmw * 0.25);
            if wgt <= 0.0 {
                continue;
            }
            wsum += wgt;
            avg = tool::lerp_paint(&avg, &tool::read_cell(layer, c), wgt / wsum);
        }
        (wsum > 0.0).then_some(avg)
    }

    /// Apply the tool to the cells of one dab (`(cell, Δm, Δm_w)`).
    fn apply(
        &mut self,
        layer: &mut WetLayer,
        cells: &[(usize, f32, f32)],
        texture: &[f32],
        amount_t: f32,
        water_t: f32,
        (angle, opacity): (f32, f32),
        w: usize,
    ) {
        let tool = match (self.tool, self.params.paint_mode) {
            (Tool::Paint(_), PaintMode::Blend) => Tool::Smudge,
            (Tool::Paint(_), PaintMode::Erase) => Tool::Eraser,
            (t, _) => t,
        };
        // tools that edit cells directly honour the stencil here (deposits do it themselves)
        let masked: Vec<(usize, f32, f32)>;
        let cells = if layer.maps.stencil.is_some() && !matches!(tool, Tool::Paint(_) | Tool::Water) {
            masked = cells
                .iter()
                .map(|&(c, dm, dmw)| {
                    let k = layer.maps.stencil_at(c);
                    (c, dm * k, dmw * k)
                })
                .collect();
            &masked[..]
        } else {
            cells
        };
        match tool {
            Tool::Paint(_) => {
                let m = self.medium;
                // PAINT lays ≈ 2/3 of PAINT_MIX's pigment and picks nothing up;
                // PAINT_BLEND picks up more and carries it further
                // concave: a small ratio lays more than its share (0.1 → 43 % of the full
                // coverage)
                let ratio = self.paint_ratio.clamp(0.0, 1.0).powf(0.75);
                let (scale, pick, carry) = match self.params.paint_mode {
                    PaintMode::Paint => (self.calib.paint_only_scale, 0.0, 0.0),
                    PaintMode::PaintBlend => (ratio, m.pick * 1.6, self.calib.carry * 2.0),
                    _ => (ratio, m.pick, self.calib.carry),
                };
                for (&(c, dm, dmw), &g) in cells.iter().zip(texture) {
                    let before = (layer.dens.amount.cur().data[c] - self.own[c]).max(0.0);
                    if m.wet && dm > 0.0 && pick > 0.0 {
                        // pick-up of the wet paint that was there before this stroke loads the brush
                        let taken = pick * dm * before;
                        if taken > 0.0 {
                            layer.dens.amount.cur_mut().data[c] -= taken;
                            let rgb = [0, 1, 2].map(|k| ld(layer.dens.rgb[k].cur(), c));
                            let (r_rgb, r_a) = self.reservoir;
                            let tot = r_a + taken * carry;
                            if tot > 0.0 {
                                self.reservoir = (
                                    [0, 1, 2].map(|k| (r_rgb[k] * r_a + rgb[k] * taken * carry) / tot),
                                    tot,
                                );
                            }
                        }
                    }
                    let (glaze_k, cover) = if m.wet { self.params.glaze_mode.effect() } else { (1.0, 0.0) };
                    let add = if self.params.only_water { 0.0 } else { amount_t * scale * dm * glaze_k };
                    if cover > 0.0 && add > 0.0 {
                        // the new paint hides part of the dry paint below
                        let mid = layer.mid.amount.cur().data[c];
                        if mid > 0.0 {
                            let lut = &layer.lut;
                            let hide = cover * lut.eval(add);
                            layer.mid.amount.cur_mut().data[c] =
                                wet_fluid::lut::OpacityLut::inverse(lut.eval(mid) * (1.0 - hide)).min(mid);
                        }
                    }
                    self.own[c] += add;
                    // the brush releases what it picked up as extra pigment, on paper that had
                    // no wet paint (where it is still picking up, the canvas paint dominates)
                    let (r_rgb, r_a) = self.reservoir;
                    if m.wet && r_a > 0.0 && dm > 0.0 && before < 1.0 {
                        let rel = r_a * (dm * self.calib.release).min(1.0);
                        self.reservoir.1 -= rel;
                        layer.deposit(c, 0.0, r_rgb, rel * self.calib.tint_k);
                    }
                    let own = self.color_at(c, w);
                    let color = if self.params.glaze_blend == wet_compose::MixMode::Normal {
                        own
                    } else {
                        // blend the brush colour with the paint at this cell, as effect
                        // markers do. Where there is no paint the plain colour is laid
                        // (separable compositing over a transparent backdrop), so Color,
                        // Dodge, Overlay and Luminosity markers paint the same tint on blank
                        // paper.
                        let base = visible_color(layer, c);
                        let a_b = {
                            let (am, ad) = (layer.mid.amount.cur().data[c], layer.dens.amount.cur().data[c]);
                            1.0 - (1.0 - layer.lut.eval(am)) * (1.0 - layer.lut.eval(ad))
                        };
                        let mixed = wet_compose::blend(
                            self.params.glaze_blend,
                            base.map(|v| v / 255.0),
                            own.map(|v| v / 255.0),
                        );
                        [0, 1, 2].map(|k| (1.0 - a_b) * own[k] + a_b * mixed[k] * 255.0)
                    };
                    if m.wet {
                        layer.deposit(c, water_t * if self.calib.water_follows_paint { dm } else { dmw }, color, add);
                    } else {
                        // fresh paint pushes the relief of the strokes below flat as it covers
                        // their colour, so their edges don't show through as ridges
                        let th = layer.mid.thickness.cur().data[c];
                        let under = (th - self.own_th[c]).max(0.0);
                        let th = th - under * layer.lut.eval(add * layer.maps.stencil_at(c));
                        layer.mid.thickness.cur_mut().data[c] = th;
                        // thickness follows the dab profile (brush space), so repeated dabs build
                        // relief ridges along the stroke even where the colour saturates
                        tool::deposit_settled(layer, c, color, add, m.thickness * (0.4 + 0.6 * g));
                        self.own_th[c] += layer.mid.thickness.cur().data[c] - th;
                    }
                }
            }
            Tool::Water => {
                for &(c, _, dmw) in cells {
                    layer.deposit(c, water_t * dmw, self.color, 0.0);
                }
            }
            Tool::Dry => {
                for &(c, dm, _) in cells {
                    let k = (opacity * dm * 4.0).min(1.0);
                    let wat = layer.water.cur_mut();
                    let v = ld(wat, c) * (1.0 - k);
                    st(wat, c, if v < 1e-3 { 0.0 } else { v });
                }
            }
            Tool::Eraser => {
                // the eraser takes away a share of the paint's coverage (not of its amount,
                // which would leave a veil of thickly laid media) and its relief with it
                for &(c, dm, _) in cells {
                    let k = (self.calib.erase * opacity * dm).min(1.0);
                    for p in [&mut layer.dens, &mut layer.mid] {
                        let a = p.amount.cur().data[c];
                        if a <= 0.0 {
                            continue;
                        }
                        let left = if k >= 1.0 {
                            0.0
                        } else {
                            wet_fluid::lut::OpacityLut::inverse(layer.lut.eval(a) * (1.0 - k)).min(a)
                        };
                        p.amount.cur_mut().data[c] = left;
                        p.thickness.cur_mut().data[c] *= left / a;
                    }
                }
            }
            Tool::Blend => {
                // every dab cell moves towards the dab-weighted average of the paint under
                // the dab (spreads paint into neighbouring paper, softens edges)
                let avg = Self::dab_average(layer, cells);
                if let Some(avg) = avg {
                    for &(c, dm, _) in cells {
                        let cur = tool::read_cell(layer, c);
                        let t = (self.calib.blend * opacity * dm).min(1.0);
                        tool::write_cell(layer, c, &tool::lerp_paint(&cur, &avg, t));
                    }
                }
            }
            Tool::Smudge => {
                // Conservative transfer through a single reservoir: deposit part of the
                // carried paint (weighted by the dab), then pick up part of what is under
                // the dab into the carry.
                let wsum: f32 = cells.iter().map(|&(_, dm, _)| dm).sum();
                if wsum <= 0.0 {
                    return;
                }
                let mut carry = self.carry.unwrap_or_default();
                let give = (self.calib.smudge_deposit * self.params.spacing).min(1.0);
                let deposit_a = carry.dens_a * give;
                let mut picked = CellPaint::default();
                for &(c, dm, _) in cells {
                    let mut cur = tool::read_cell(layer, c);
                    let add = deposit_a * dm / wsum;
                    if add > 0.0 {
                        let into = CellPaint { dens_rgb: carry.dens_rgb, dens_a: add, ..cur };
                        cur = CellPaint {
                            dens_rgb: tool::lerp_paint(&cur, &into, add / (cur.dens_a + add)).dens_rgb,
                            dens_a: cur.dens_a + add,
                            ..cur
                        };
                    }
                    self.own[c] += add;
                    // only paint that was there before this stroke is picked up
                    // a paint brush in BLEND mode smudges more gently than the smudge tool
                    let gentle = if matches!(self.tool, Tool::Paint(_)) { 0.5 } else { 1.0 };
                    let k = (self.calib.smudge_pick * gentle * opacity * dm).min(1.0);
                    let lift = (cur.dens_a - self.own[c]).max(0.0);
                    let take = CellPaint {
                        dens_rgb: cur.dens_rgb,
                        dens_a: lift * k,
                        mid_rgb: cur.mid_rgb,
                        mid_a: cur.mid_a * k,
                    };
                    cur.dens_a -= take.dens_a;
                    cur.mid_a -= take.mid_a;
                    tool::write_cell(layer, c, &cur);
                    // the carry holds everything as suspended paint
                    let t_a = take.dens_a + take.mid_a;
                    if t_a > 0.0 {
                        let rgb = [0, 1, 2].map(|i| {
                            (take.dens_rgb[i] * take.dens_a + take.mid_rgb[i] * take.mid_a) / t_a
                        });
                        let total = picked.dens_a + t_a;
                        picked.dens_rgb = [0, 1, 2]
                            .map(|i| (picked.dens_rgb[i] * picked.dens_a + rgb[i] * t_a) / total);
                        picked.dens_a = total;
                    }
                }
                carry.dens_a -= deposit_a;
                let total = carry.dens_a + picked.dens_a;
                if total > 0.0 {
                    carry.dens_rgb = [0, 1, 2].map(|i| {
                        (carry.dens_rgb[i] * carry.dens_a + picked.dens_rgb[i] * picked.dens_a)
                            / total
                    });
                }
                carry.dens_a = total;
                self.carry = Some(carry);
            }
            Tool::Blow => {
                let (sin, cos) = angle.sin_cos();
                for &(c, dm, _) in cells {
                    if layer.maps.open.data[c] == 0 {
                        continue;
                    }
                    let k = 0.4 * opacity * dm * 4.0;
                    let u = layer.u.cur_mut();
                    st(u, c, (ld(u, c) + k * cos).clamp(-0.4, 0.4));
                    let v = layer.v.cur_mut();
                    st(v, c, (ld(v, c) + k * sin).clamp(-0.4, 0.4));
                }
            }
            Tool::Liquify(mode) => {
                let (cx, cy, r, dir) = self.dab;
                let strength = opacity;
                // source position for each cell, sampled from the state before this dab
                let new: Vec<(usize, CellPaint)> = cells
                    .iter()
                    .filter_map(|&(c, _, _)| {
                        let (x, y) = ((c % w) as f32, (c / w) as f32);
                        let (dx, dy) = (x - cx, y - cy);
                        let rho = (dx * dx + dy * dy).sqrt() / r.max(1e-3);
                        if rho >= 1.0 {
                            return None;
                        }
                        let f = (1.0 - rho).powi(2) * strength;
                        let (sx, sy) = match mode {
                            Liquify::Push => {
                                let step = 0.6 * r * f;
                                (x - dir.cos() * step, y - dir.sin() * step)
                            }
                            Liquify::Twirl => {
                                let t = -0.35 * f;
                                let (sn, cs) = t.sin_cos();
                                (cx + dx * cs - dy * sn, cy + dx * sn + dy * cs)
                            }
                            Liquify::Pucker => (cx + dx * (1.0 + 0.25 * f), cy + dy * (1.0 + 0.25 * f)),
                            Liquify::Bloat => (cx + dx * (1.0 - 0.25 * f), cy + dy * (1.0 - 0.25 * f)),
                            Liquify::Reconstruct => {
                                let cur = tool::read_cell(layer, c);
                                let orig = self.stroke_start.get(c).copied().unwrap_or(cur);
                                return Some((c, tool::lerp_paint(&cur, &orig, (0.5 * f).min(1.0))));
                            }
                        };
                        Some((c, tool::sample_paint(layer, sx, sy)))
                    })
                    .collect();
                for (c, p) in new {
                    tool::write_cell(layer, c, &p);
                }
            }
            Tool::Clone { dx, dy } => {
                let src: Vec<Option<CellPaint>> = cells
                    .iter()
                    .map(|&(c, _, _)| {
                        let (x, y) = ((c % w) as f32 + dx, (c / w) as f32 + dy);
                        if x < 0.0 || y < 0.0 || x >= w as f32 || y >= layer.h as f32 {
                            None
                        } else {
                            Some(tool::read_cell(layer, y as usize * w + x as usize))
                        }
                    })
                    .collect();
                for (&(c, dm, _), sp) in cells.iter().zip(src) {
                    if let Some(sp) = sp {
                        let cur = tool::read_cell(layer, c);
                        let t = (opacity * dm * 4.0).min(1.0);
                        tool::write_cell(layer, c, &tool::lerp_paint(&cur, &sp, t));
                    }
                }
            }
        }
    }
}
