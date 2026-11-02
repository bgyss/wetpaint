//! Application state and the commands shared by menus, panels and keyboard shortcuts.

use std::path::{Path, PathBuf};

use web_time::Instant;

use wet_preset::BrushPreset;
use wet_brush::{
    Brush, BrushParams, Liquify, Mask, Medium, Sample, Tool,
    ruler::{Ruler, RulerStroke},
    volume::Volume,
};
use wet_doc::{Document, filters::Filter};
use winit::keyboard::{Key, NamedKey};

use crate::platform::{self, Pick, Target};
use crate::transform::Xform;

pub const PALETTE: [[f32; 3]; 12] = [
    [200.0, 40.0, 40.0], [230.0, 120.0, 30.0], [235.0, 200.0, 40.0], [60.0, 150.0, 60.0],
    [30.0, 110.0, 120.0], [40.0, 70.0, 190.0], [110.0, 50.0, 150.0], [190.0, 70.0, 140.0],
    [120.0, 80.0, 50.0], [60.0, 60.0, 60.0], [15.0, 15.0, 20.0], [250.0, 250.0, 245.0],
];

/// Simulation steps per second.
pub const STEPS_PER_SECOND: f32 = 25.0;

/// The tool's default brush: a soft round tip.
fn make_brush(tool: Tool) -> Brush {
    let mut b = Brush::new(BrushParams { spacing: 0.05, ..BrushParams::default() }, Mask::disk(128, 0.6), None);
    b.set_tool(tool);
    b.params.size = 30.0;
    b.params.water = 1.0;
    b.params.opacity = 1.0;
    b
}

/// Decode an image file (PNG, JPEG, TIFF, BMP) to straight RGBA8.
pub fn decode_rgba(bytes: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Some((img.into_raw(), w, h))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ClickMode {
    Paint,
    Pick,
    Fill,
    SelectEllipse,
    SelectRect,
    Lasso,
    Wand,
    /// Move / scale / rotate / distort the active layer with handles ([`Xform`]).
    Transform,
}

/// Everything the user can ask for from a menu, a panel button or a key.
#[derive(Clone, Debug, PartialEq)]
pub enum Cmd {
    Tool(Tool),
    CycleLiquify,
    Mode(ClickMode),
    NextColor,
    SizeBy(f32),
    WaterBy(f32),
    OpacityBy(f32),
    Volume(Volume),
    ToggleMulti,
    TogglePause,
    ToggleTilt,
    DryLayer,
    WetLayer,
    ClearLayer,
    TogglePigment,
    ToggleGranulation,
    NewLayer,
    NextLayer,
    DeleteLayer,
    DuplicateLayer,
    MergeDown,
    MoveLayer(isize),
    GroupWithBelow,
    LayerPigments,
    ToggleMaskingFluid,
    ToggleLock,
    ToggleClip,
    ToggleMetallic,
    Transform((f32, f32), f32, f32),
    ApplyTransform,
    CancelTransform,
    /// Brush Creator: restore the brush's preset parameters, or save them as a preset.
    ResetBrush,
    SavePreset,
    ToggleGpu,
    /// Undo (negative) or redo (positive) this many steps, as a History panel click does.
    History(isize),
    Undo,
    Redo,
    Filter(Filter),
    SelectAll,
    Deselect,
    RestoreSelection,
    InvertSelection,
    StencilFromSelection,
    CutSelectionToLayer,
    ToggleStencil,
    InvertStencil,
    Tracing,
    ToggleReference,
    CycleRuler,
    ToggleGrid,
    ToggleGreyscale,
    ToggleNatural,
    ToggleSoftShadows,
    NextVisualPreset,
    Fit,
    ToggleRecording,
    NewDocument(usize, usize),
    Open,
    Save,
    SaveAs,
    SaveIterative,
    ExportPng,
    ExportPsd,
    ExportHeightmap,
}

impl Cmd {
    /// Does the command change the document (so it needs saving)? Tool, brush, view and
    /// file commands do not.
    fn edits_document(&self) -> bool {
        !matches!(
            self,
            Cmd::Tool(_)
                | Cmd::CycleLiquify
                | Cmd::Mode(_)
                | Cmd::NextColor
                | Cmd::SizeBy(_)
                | Cmd::WaterBy(_)
                | Cmd::OpacityBy(_)
                | Cmd::Volume(_)
                | Cmd::ToggleMulti
                | Cmd::TogglePause
                | Cmd::TogglePigment
                | Cmd::ToggleNatural
                | Cmd::ToggleGreyscale
                | Cmd::CycleRuler
                | Cmd::Fit
                | Cmd::ToggleRecording
                | Cmd::ToggleGpu
                | Cmd::ApplyTransform
                | Cmd::CancelTransform
                | Cmd::ResetBrush
                | Cmd::SavePreset
                | Cmd::NewDocument(..)
                | Cmd::Open
                | Cmd::Save
                | Cmd::SaveAs
                | Cmd::SaveIterative
                | Cmd::ExportPng
                | Cmd::ExportPsd
                | Cmd::ExportHeightmap
        )
    }
}

pub struct App {
    pub doc: Document,
    /// Where the document was last opened from or saved to.
    pub path: Option<PathBuf>,
    /// View: screen points per canvas pixel and the canvas origin relative to the canvas
    /// panel (`zoom <= 0` fits the canvas on the next frame).
    pub zoom: f32,
    pub pan: (f32, f32),
    /// View the canvas in greyscale (View > Greyscale).
    pub greyscale: bool,
    /// Red–yellow–blue mixing (Edit > Natural Colors).
    pub natural: bool,
    /// Index of the last applied Visual Settings preset.
    pub visual_preset: usize,
    pub mode: ClickMode,
    pub recording: Option<(usize, Instant)>,
    pub select_start: Option<(f32, f32)>,
    /// Lasso path being drawn (canvas coordinates).
    pub lasso: Vec<(f32, f32)>,
    /// Pressure from a Force Touch trackpad (0–1) or touch force, when the platform reports
    /// one; `None` paints at full pressure.
    pub pressure: Option<f32>,
    pub volumes: wet_brush::volume::VolumePresets,
    /// Active ruler guide and its per-stroke state.
    pub ruler: Option<Ruler>,
    pub ruler_stroke: RulerStroke,
    pub brush: Brush,
    pub color: usize,
    pub painting: bool,
    /// The stroke in progress paints a layer the GPU holds ([`App::flush_stroke`]).
    pub gpu_stroke: bool,
    /// Brush steps not yet stamped: a stroke on the GPU stamps a frame's steps together.
    queued: Vec<Sample>,
    /// Canvas position under the pointer.
    pub cursor: (f32, f32),
    pub paused: bool,
    pub pigment: bool,
    pub last_step: Instant,
    pub debt: f32,
    /// Name of the library preset the brush came from (`None`: the tool's default).
    pub preset: Option<String>,
    /// The preset file the brush was built from (its tips and textures), kept for the
    /// Brush Creator's reset and save.
    pub source: Option<BrushPreset>,
    /// Last message for the status bar.
    pub status: String,
    /// Measured simulation rate (steps per second).
    pub sim_rate: f32,
    /// Interactive transform of the active layer, until applied or cancelled.
    pub xform: Option<Xform>,
    /// ⌘/Ctrl held: a dragged transform corner moves on its own (perspective).
    pub free_corner: bool,
    /// An agent's gesture is still pressed: its next stroke or drag goes on with it.
    pub held: bool,
    /// Wet layers step on the GPU (View > GPU Simulation).
    pub gpu: Option<crate::gpu::GpuSim>,
    /// The window's GPU device and renderer: the GPU simulation shares the device and the
    /// display composited on it is drawn from there (`None` without a window: the
    /// simulation opens its own device and the display is composited on the CPU).
    pub render_state: Option<egui_wgpu::RenderState>,
    /// Changes whenever a layer may have been edited (commands, strokes, opening), so the
    /// GPU display uploads CPU layers again.
    pub epoch: u64,
    /// The document has changed since it was last opened, saved or created.
    pub dirty: bool,
    /// File > Quit or ⌘/Ctrl+Q: the window closes once [`App::confirm_discard`] agrees.
    pub quit: bool,
    steps_this_second: (u32, Instant),
}

impl App {
    pub fn new(w: usize, h: usize) -> Self {
        let (mut brush, source) = (make_brush(Tool::Paint(Medium::Watercolor)), None);
        brush.color = PALETTE[0];
        Self {
            doc: Document::new(w, h),
            path: None,
            zoom: 0.0,
            pan: (0.0, 0.0),
            greyscale: false,
            natural: false,
            visual_preset: 0,
            mode: ClickMode::Paint,
            recording: None,
            select_start: None,
            lasso: Vec::new(),
            pressure: None,
            volumes: wet_brush::volume::VolumePresets::default(),
            ruler: None,
            ruler_stroke: RulerStroke::default(),
            brush,
            color: 0,
            painting: false,
            gpu_stroke: false,
            queued: Vec::new(),
            cursor: (0.0, 0.0),
            paused: false,
            pigment: false,
            last_step: Instant::now(),
            debt: 0.0,
            preset: None,
            source,
            status: String::new(),
            sim_rate: 0.0,
            xform: None,
            free_corner: false,
            held: false,
            gpu: None,
            render_state: None,
            epoch: 0,
            dirty: false,
            quit: false,
            steps_this_second: (0, Instant::now()),
        }
    }

    pub fn set_tool(&mut self, tool: Tool) {
        let (size, water, opacity, color) =
            (self.brush.params.size, self.brush.params.water, self.brush.params.opacity, self.brush.color);
        (self.brush, self.source) = (make_brush(tool), None);
        self.brush.params.size = size;
        self.brush.params.water = water;
        self.brush.params.opacity = opacity;
        self.brush.color = color;
        self.mode = ClickMode::Paint;
        self.preset = None;
    }

    /// Paint with a `.wetbrush` preset (its own size, spacing and textures), keeping the
    /// current tool and colour.
    pub fn load_preset(&mut self, path: &Path) {
        match std::fs::read(path) {
            Ok(b) => self.load_preset_bytes(&crate::library::preset_name(path).unwrap_or_default(), &b),
            Err(e) => self.status = format!("cannot read preset {}: {e}", path.display()),
        }
    }

    /// [`Self::load_preset`] from the file's contents; `name` is the file's preset name, if
    /// any.
    pub fn load_preset_bytes(&mut self, name: &str, bytes: &[u8]) {
        let tool = self.brush.tool;
        match BrushPreset::from_bytes(bytes) {
            Ok(p) => {
                let color = self.brush.color;
                self.brush = Brush::from_preset(&p);
                self.brush.set_tool(tool);
                self.brush.color = color;
                self.mode = ClickMode::Paint;
                let name = if name.is_empty() { p.name.clone() } else { name.to_owned() };
                self.status = format!("brush preset {name}");
                self.preset = Some(name);
                self.source = Some(p);
            }
            Err(e) => self.status = format!("cannot read preset {name}: {e}"),
        }
    }

    /// Open a dropped or command-line file: a `.wetdoc`, a brush preset, or an image as a
    /// new layer.
    pub fn open_path(&mut self, path: &Path) {
        match std::fs::read(path) {
            Ok(bytes) => {
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                self.open_bytes(&name, Some(path), &bytes);
            }
            Err(e) => self.status = format!("cannot open {}: {e}", path.display()),
        }
    }

    /// [`Self::open_path`] from a file's name, path (if it has one) and contents.
    pub fn open_bytes(&mut self, name: &str, path: Option<&Path>, bytes: &[u8]) {
        self.epoch += 1;
        self.release_gpu();
        let lower = name.to_lowercase();
        if let Some(preset) = crate::library::preset_name(Path::new(name)) {
            self.load_preset_bytes(&preset, bytes);
        } else if lower.ends_with(".wetdoc") {
            if !self.confirm_discard() {
                return;
            }
            match Document::load(bytes) {
                Ok(d) => {
                    self.doc = d;
                    self.zoom = 0.0;
                    self.status = format!("opened {name}");
                    self.path = path.map(Path::to_owned);
                    self.dirty = false;
                }
                Err(e) => self.status = format!("open failed: {e}"),
            }
        } else if let Some((rgba, w, h)) = decode_rgba(bytes) {
            self.doc.checkpoint_as("Import image");
            let stem = Path::new(name).file_stem().and_then(|n| n.to_str()).unwrap_or("image").to_owned();
            self.doc.import_image(&stem, &rgba, w, h);
            self.dirty = true;
            self.status = format!("imported {name} as a layer");
        } else {
            self.status = format!("cannot open {name}");
        }
    }

    /// Act on files the user picked (File > Open, tracing and reference images).
    pub fn handle_picked(&mut self) {
        for f in platform::take_picked() {
            match f.kind {
                Pick::Open => self.open_bytes(&f.name, f.path.as_deref(), &f.bytes),
                Pick::Tracing => {
                    if let Some((rgba, w, h)) = decode_rgba(&f.bytes) {
                        self.import_tracing(&rgba, w, h);
                    }
                }
                Pick::Reference => {
                    if let Some((rgba, w, h)) = decode_rgba(&f.bytes) {
                        self.import_reference(rgba, w, h);
                    }
                }
            }
        }
    }

    /// An image (straight RGBA8) as a locked tracing layer that strokes take their colour
    /// from, with a new layer above it to paint on.
    pub fn import_tracing(&mut self, rgba: &[u8], w: usize, h: usize) {
        let idx = self.doc.import_image("reference", rgba, w, h);
        self.doc.layers[idx].tracing = true;
        self.doc.layers[idx].locked = true;
        self.doc.add_layer("tracing");
        self.epoch += 1;
    }

    /// An image (straight RGBA8) for the reference panel.
    pub fn import_reference(&mut self, rgba: Vec<u8>, w: usize, h: usize) {
        self.doc.references.push(wet_doc::ReferenceImage { name: "reference".into(), w, h, rgba, visible: true });
    }

    pub fn set_color(&mut self, c: [f32; 3]) {
        self.brush.color = c;
    }

    /// Advance the simulation by wall-clock time.
    pub fn tick(&mut self) {
        self.handle_picked();
        let now = Instant::now();
        let dt = now.duration_since(self.last_step).as_secs_f32();
        self.last_step = now;
        if self.paused {
            self.sim_rate = 0.0;
            return;
        }
        self.debt = (self.debt + dt * STEPS_PER_SECOND).min(4.0);
        let steps = self.debt.floor() as usize;
        self.debt -= steps as f32;
        self.steps_this_second.0 += steps as u32;
        if steps > 0 {
            self.step_sim(steps);
        }
        let elapsed = self.steps_this_second.1.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.sim_rate = self.steps_this_second.0 as f32 / elapsed;
            self.steps_this_second = (0, Instant::now());
        }
    }

    /// Run `n` simulation steps, on the GPU when it is on.
    pub fn step_sim(&mut self, n: usize) {
        // the layer a stroke is painting steps on the CPU, unless the stroke runs on the GPU
        let skip = (self.painting && !self.gpu_stroke).then_some(self.doc.active);
        if let Some(g) = &mut self.gpu {
            match g.step(&mut self.doc, n, skip) {
                Ok(()) => return,
                Err(e) => {
                    g.release(&mut self.doc);
                    self.gpu = None;
                    self.status = format!("GPU simulation stopped: {e}");
                }
            }
        }
        for _ in 0..n {
            self.doc.step();
        }
    }

    /// Bring layers simulated on the GPU back to the CPU before editing or saving them.
    pub fn release_gpu(&mut self) {
        if let Some(g) = &mut self.gpu {
            g.release(&mut self.doc);
        }
    }

    /// Can the display be composited on the GPU? It needs the GPU simulation and a window;
    /// time-lapse recording reads the painting on the CPU.
    pub fn gpu_display_ok(&self) -> bool {
        self.gpu.is_some() && self.render_state.is_some() && self.recording.is_none()
    }

    /// Composite the display on the GPU when [`Self::gpu_display_ok`]: the texture to draw.
    pub fn gpu_display(&mut self, filter: wgpu::FilterMode) -> Option<egui::TextureId> {
        let ok = self.gpu_display_ok();
        let g = self.gpu.as_mut()?;
        if !ok {
            g.shown_on_cpu();
            return None;
        }
        let render = self.render_state.as_ref()?;
        let view = crate::gpu::View { greyscale: self.greyscale, pigment: self.pigment, epoch: self.epoch, painting: self.painting.then_some(self.doc.active) };
        Some(g.compose(&self.doc, render, filter, &view))
    }

    /// Refresh what the display reads of layers simulated on the GPU.
    pub fn show_gpu(&mut self) {
        if let Some(g) = &mut self.gpu {
            g.show(&mut self.doc);
        }
    }

    pub fn sample(&mut self) -> Sample {
        let (x, y) = match &self.ruler {
            Some(r) => self.ruler_stroke.constrain(r, self.cursor),
            None => self.cursor,
        };
        Sample { x, y, pressure: self.pressure.unwrap_or(1.0) }
    }

    /// Pointer pressed on the canvas at `self.cursor`.
    pub fn press(&mut self) {
        self.epoch += 1;
        // a stroke changes only the active layer, unless masking fluid recomputes every
        // layer's stencil
        let partial = self.mode == ClickMode::Paint && !self.doc.layers.iter().any(|l| l.masking_fluid);
        let active = self.doc.active;
        // a stroke on a layer the GPU holds keeps it there: each brush step brings back only
        // the cells it can reach (not for tools that read the whole layer)
        let (w, h) = (self.doc.w, self.doc.h);
        let at = Sample { x: self.cursor.0, y: self.cursor.1, pressure: 1.0 };
        let reach = self.brush.reach_bounds(at, w, h).is_some();
        let gpu_stroke = partial && reach && self.gpu.as_ref().is_some_and(|g| g.holds(active));
        match &mut self.gpu {
            Some(_) if gpu_stroke => {}
            Some(g) if partial => g.release_layer(&mut self.doc, active),
            _ => self.release_gpu(),
        }
        let (x, y) = (self.cursor.0, self.cursor.1);
        let inside = x >= 0.0 && y >= 0.0 && (x as usize) < self.doc.w && (y as usize) < self.doc.h;
        match self.mode {
            ClickMode::Transform => {
                let tol = 8.0 / self.zoom.max(0.05);
                if let Some(x) = &mut self.xform {
                    x.press(self.cursor, tol);
                }
            }
            ClickMode::SelectEllipse | ClickMode::SelectRect => self.select_start = Some(self.cursor),
            ClickMode::Lasso => {
                self.select_start = Some(self.cursor);
                self.lasso = vec![self.cursor];
            }
            ClickMode::Wand => {
                if inside {
                    self.dirty = true;
                    self.doc.select_color((x as usize, y as usize), 32, true, wet_doc::SelectOp::Replace);
                }
                self.mode = ClickMode::Paint;
            }
            ClickMode::Pick | ClickMode::Fill => {
                if inside {
                    let (x, y) = (x as usize, y as usize);
                    if self.mode == ClickMode::Pick {
                        let img = self.doc.render(None);
                        let p = &img[(y * self.doc.w + x) * 4..];
                        self.brush.color = [p[0] as f32, p[1] as f32, p[2] as f32];
                    } else {
                        self.doc.checkpoint_as("Fill");
                        self.dirty = true;
                        let opts = wet_doc::FillOptions { opacity: self.brush.params.opacity.min(1.0), ..Default::default() };
                        self.doc.fill((x, y), &wet_doc::Fill::Color(self.brush.color), opts);
                    }
                }
                self.mode = ClickMode::Paint;
            }
            ClickMode::Paint => {
                if self.doc.layers[self.doc.active].locked {
                    self.status = "layer is locked".into();
                    return;
                }
                self.ruler_stroke.press(self.cursor);
                let s = self.sample();
                if let Some(c) = self.doc.tracing_color(s.x as usize, s.y as usize) {
                    self.brush.color = c;
                }
                self.doc.update_stencils();
                let label = match self.brush.tool {
                    Tool::Paint(m) => format!("{m:?} stroke"),
                    Tool::Liquify(l) => format!("Liquify ({l:?})"),
                    Tool::Clone { .. } => "Clone stroke".into(),
                    t => format!("{t:?} stroke"),
                };
                match &mut self.gpu {
                    Some(g) if gpu_stroke => g.begin_stroke(&mut self.doc, active, &label),
                    _ => self.doc.checkpoint_as(&label),
                }
                self.dirty = true;
                self.brush.press(s);
                self.queued.clear();
                self.painting = true;
                self.gpu_stroke = gpu_stroke;
            }
        }
    }

    /// Stamp the brush steps queued since the last call (the last one ends the stroke
    /// when `end`). On a layer the GPU holds, the cells they can reach come back from the
    /// GPU once and go back once, however many steps there are.
    pub fn flush_stroke(&mut self, end: bool) {
        if self.queued.is_empty() {
            return;
        }
        let steps = std::mem::take(&mut self.queued);
        let i = self.doc.active;
        let (w, h) = (self.doc.w, self.doc.h);
        let on_gpu = self.gpu_stroke && self.gpu.as_ref().is_some_and(|g| g.holds(i));
        // every step's reach, from the state before the first: the dabs lie on the path
        // through the queued samples
        let rect = on_gpu
            .then(|| {
                steps.iter().map(|&s| self.brush.reach_bounds(s, w, h)).try_fold(None, |acc: Option<wet_fluid_gpu::Rect>, r| {
                    let r = r?;
                    Some(Some(acc.map_or(r, |a| (a.0.min(r.0), a.1.min(r.1), a.2.max(r.2), a.3.max(r.3)))))
                })
            })
            .flatten()
            .flatten();
        if let (Some(r), Some(g)) = (rect, &mut self.gpu) {
            g.read_region(&mut self.doc, i, r);
        }
        let layer = &mut self.doc.layers[i].wet;
        let last = steps.len() - 1;
        for (k, s) in steps.into_iter().enumerate() {
            if end && k == last {
                self.brush.release(layer, s);
            } else {
                self.brush.move_to(layer, s);
            }
        }
        if let (Some(r), Some(g)) = (rect, &mut self.gpu) {
            g.write_region(&self.doc, i, r);
        }
    }

    /// Pointer moved to `self.cursor` (pressed or not).
    pub fn moved(&mut self) {
        if self.painting {
            let s = self.sample();
            self.queued.push(s);
            if !self.gpu_stroke {
                self.flush_stroke(false);
            }
        }
        if self.mode == ClickMode::Lasso && self.select_start.is_some() {
            self.lasso.push(self.cursor);
        }
        if let Some(x) = &mut self.xform {
            x.drag_to(self.cursor, self.free_corner);
        }
    }

    pub fn release(&mut self) {
        self.epoch += 1;
        match self.mode {
            ClickMode::Transform => {
                if let Some(x) = &mut self.xform {
                    x.release();
                }
            }
            ClickMode::SelectEllipse | ClickMode::SelectRect => {
                if let Some(a) = self.select_start.take() {
                    let (w, h) = (self.doc.w, self.doc.h);
                    let tol = 6.0 / self.zoom.max(0.05);
                    let a = self.doc.guides.snap(a, w, h, tol);
                    let b = self.doc.guides.snap(self.cursor, w, h, tol);
                    self.doc.select_shape(a, b, self.mode == ClickMode::SelectEllipse);
                    self.dirty = true;
                    self.mode = ClickMode::Paint;
                }
            }
            ClickMode::Lasso => {
                if self.select_start.take().is_some() {
                    let pts = std::mem::take(&mut self.lasso);
                    self.doc.select_polygon(&pts, wet_doc::SelectOp::Replace);
                    self.mode = ClickMode::Paint;
                }
            }
            _ => {
                if self.painting {
                    let s = self.sample();
                    self.queued.push(s);
                    self.flush_stroke(true);
                    self.painting = false;
                }
            }
        }
    }

    /// Map a key pressed over the canvas to a command.
    pub fn key_cmd(key: &Key) -> Option<Cmd> {
        Some(match key {
            Key::Named(NamedKey::Space) => Cmd::TogglePause,
            Key::Named(NamedKey::Tab) => Cmd::NextLayer,
            Key::Named(NamedKey::Home) => Cmd::Fit,
            Key::Named(NamedKey::Enter) => Cmd::ApplyTransform,
            Key::Named(NamedKey::Escape) => Cmd::CancelTransform,
            Key::Named(NamedKey::Insert) => Cmd::ToggleReference,
            Key::Named(NamedKey::F1) => Cmd::CycleRuler,
            Key::Named(NamedKey::F2) => Cmd::Tracing,
            Key::Named(NamedKey::F3) => Cmd::ToggleMulti,
            Key::Named(NamedKey::F4) => Cmd::LayerPigments,
            Key::Named(NamedKey::F5) => Cmd::ExportPng,
            Key::Named(NamedKey::F6) => Cmd::GroupWithBelow,
            Key::Named(NamedKey::F7) => Cmd::StencilFromSelection,
            Key::Named(NamedKey::F8) => Cmd::CutSelectionToLayer,
            Key::Named(NamedKey::F9) => Cmd::ToggleGrid,
            Key::Named(NamedKey::F10) => Cmd::Volume(Volume::Size),
            Key::Named(NamedKey::F11) => Cmd::Volume(Volume::Opacity),
            Key::Named(NamedKey::F12) => Cmd::Volume(Volume::Water),
            Key::Named(NamedKey::ArrowLeft) => Cmd::Transform((-10.0, 0.0), 1.0, 0.0),
            Key::Named(NamedKey::ArrowRight) => Cmd::Transform((10.0, 0.0), 1.0, 0.0),
            Key::Named(NamedKey::ArrowUp) => Cmd::Transform((0.0, -10.0), 1.0, 0.0),
            Key::Named(NamedKey::ArrowDown) => Cmd::Transform((0.0, 10.0), 1.0, 0.0),
            Key::Character(c) => match c.as_str() {
                "1" => Cmd::Tool(Tool::Paint(Medium::Watercolor)),
                "2" => Cmd::Tool(Tool::Paint(Medium::Ink)),
                "3" => Cmd::Tool(Tool::Paint(Medium::Oil)),
                "4" => Cmd::Tool(Tool::Paint(Medium::Pastel)),
                "5" => Cmd::Tool(Tool::Paint(Medium::Pencil)),
                "6" => Cmd::Tool(Tool::Paint(Medium::Marker)),
                "7" => Cmd::Tool(Tool::Paint(Medium::Airbrush)),
                "8" => Cmd::Tool(Tool::Water),
                "9" => Cmd::Tool(Tool::Dry),
                "0" => Cmd::Tool(Tool::Eraser),
                "b" => Cmd::Tool(Tool::Blend),
                "m" => Cmd::Tool(Tool::Smudge),
                "o" => Cmd::Tool(Tool::Blow),
                "h" => Cmd::CycleLiquify,
                "c" => Cmd::NextColor,
                "C" => Cmd::ToggleGpu,
                "[" => Cmd::SizeBy(1.0 / 1.2),
                "]" => Cmd::SizeBy(1.2),
                "," => Cmd::WaterBy(-0.1),
                "." => Cmd::WaterBy(0.1),
                "-" => Cmd::OpacityBy(-0.1),
                "=" => Cmd::OpacityBy(0.1),
                "t" => Cmd::ToggleTilt,
                "d" => Cmd::DryLayer,
                "w" => Cmd::WetLayer,
                "x" => Cmd::ClearLayer,
                "p" => Cmd::TogglePigment,
                "g" => Cmd::ToggleGranulation,
                "n" => Cmd::NewLayer,
                "z" => Cmd::Undo,
                "y" => Cmd::Redo,
                "I" => Cmd::Filter(Filter::Invert),
                "U" => Cmd::Filter(Filter::Desaturate { luminosity: true }),
                "H" => Cmd::Filter(Filter::HueSaturation { hue: 30.0, saturation: 0.0, lightness: 0.0 }),
                "B" => Cmd::Filter(Filter::GaussianBlur { radius: 4.0 }),
                "S" => Cmd::Filter(Filter::Sharpen { radius: 2.0, strength: 0.5 }),
                "i" => Cmd::Mode(ClickMode::Pick),
                "e" => Cmd::Mode(ClickMode::SelectEllipse),
                "u" => Cmd::Mode(ClickMode::SelectRect),
                "'" => Cmd::Mode(ClickMode::Lasso),
                ";" => Cmd::Mode(ClickMode::Wand),
                "f" => Cmd::Mode(ClickMode::Fill),
                "q" => Cmd::Deselect,
                "Q" => Cmd::RestoreSelection,
                "/" => Cmd::InvertSelection,
                "N" => Cmd::DuplicateLayer,
                "M" => Cmd::MergeDown,
                "\\" => Cmd::ToggleStencil,
                "|" => Cmd::InvertStencil,
                "<" => Cmd::Transform((0.0, 0.0), 1.0, -5f32.to_radians()),
                ">" => Cmd::Transform((0.0, 0.0), 1.0, 5f32.to_radians()),
                "{" => Cmd::Transform((0.0, 0.0), 1.0 / 1.1, 0.0),
                "}" => Cmd::Transform((0.0, 0.0), 1.1, 0.0),
                "k" => Cmd::ToggleMaskingFluid,
                "a" => Cmd::ToggleLock,
                "j" => Cmd::ToggleClip,
                "r" => Cmd::ToggleRecording,
                "R" => Cmd::Mode(ClickMode::Transform),
                "G" => Cmd::ToggleGreyscale,
                "T" => Cmd::ToggleMetallic,
                "V" => Cmd::NextVisualPreset,
                "O" => Cmd::ToggleSoftShadows,
                "P" => Cmd::ToggleNatural,
                "E" => Cmd::ExportHeightmap,
                "D" => Cmd::SaveIterative,
                "s" => Cmd::Save,
                "v" => Cmd::ExportPsd,
                "l" => Cmd::Open,
                _ => return None,
            },
            _ => return None,
        })
    }

    pub fn report(&mut self, what: &str, r: std::io::Result<()>) {
        self.status = match r {
            Ok(()) => what.to_owned(),
            Err(e) => format!("{what} failed: {e}"),
        };
        eprintln!("{}", self.status);
    }

    pub fn save_to(&mut self, target: Target) {
        let r = platform::write(&target, |out| self.doc.save(out));
        if r.is_ok() {
            self.dirty = false;
        }
        self.report(&format!("saved {}", target.describe()), r);
        let Target::Path(path) = target else { return };
        self.path = Some(path);
    }

    /// May the current document be replaced or closed? Asks to save unsaved changes first:
    /// Save (then only if it was saved), Don't Save, or Cancel.
    pub fn confirm_discard(&mut self) -> bool {
        if !self.dirty || cfg!(test) {
            return true;
        }
        let name = self.path.as_deref().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("Untitled");
        match platform::ask_save_changes(name) {
            Some(true) => {
                self.run(Cmd::Save);
                !self.dirty
            }
            Some(false) => true,
            None => false,
        }
    }

    /// Start transforming the active layer.
    fn begin_transform(&mut self) {
        if self.doc.layers[self.doc.active].locked {
            self.status = "layer is locked".into();
            return;
        }
        self.xform = Some(Xform::new(self.doc.w, self.doc.h));
        self.mode = ClickMode::Transform;
        self.status = "transforming the layer".into();
    }

    /// Resample the active layer through the transform being edited and leave the tool.
    fn apply_transform(&mut self) {
        if let Some(x) = self.xform.take()
            && x.changed()
        {
            self.doc.checkpoint_as("Transform");
            self.doc.perspective_layer(self.doc.active, x.quad);
            self.dirty = true;
            self.status = "transformed the layer".into();
        }
        if self.mode == ClickMode::Transform {
            self.mode = ClickMode::Paint;
        }
    }

    pub fn run(&mut self, cmd: Cmd) {
        self.epoch += 1;
        // the CPU copy of every layer is complete before any command reads or edits it
        self.release_gpu();
        // any other command applies a pending transform first; undo abandons it
        if self.xform.is_some() {
            match cmd {
                Cmd::Transform(offset, scale, rotation) => {
                    if let Some(x) = &mut self.xform {
                        x.nudge(offset, scale, rotation);
                    }
                    return;
                }
                Cmd::Undo | Cmd::History(_) | Cmd::CancelTransform => {
                    self.xform = None;
                    self.mode = ClickMode::Paint;
                    self.status = "transform cancelled".into();
                    return;
                }
                Cmd::Mode(ClickMode::Transform) | Cmd::TogglePause | Cmd::Fit => {}
                _ => self.apply_transform(),
            }
        }
        if cmd.edits_document() {
            self.dirty = true;
        }
        let i = self.doc.active;
        match cmd {
            Cmd::Tool(t) => self.set_tool(t),
            Cmd::CycleLiquify => {
                let next = match self.brush.tool {
                    Tool::Liquify(Liquify::Push) => Liquify::Twirl,
                    Tool::Liquify(Liquify::Twirl) => Liquify::Pucker,
                    Tool::Liquify(Liquify::Pucker) => Liquify::Bloat,
                    Tool::Liquify(Liquify::Bloat) => Liquify::Reconstruct,
                    _ => Liquify::Push,
                };
                let (size, opacity) = (self.brush.params.size, self.brush.params.opacity);
                self.brush.set_tool(Tool::Liquify(next));
                self.brush.params.size = size;
                self.brush.params.opacity = opacity;
                self.mode = ClickMode::Paint;
            }
            Cmd::Mode(ClickMode::Transform) => {
                if self.xform.is_none() {
                    self.begin_transform();
                }
            }
            Cmd::Mode(m) => self.mode = m,
            Cmd::ApplyTransform | Cmd::CancelTransform => {}
            Cmd::ToggleGpu => {
                if self.gpu.take().is_some() {
                    self.status = "simulation on the CPU".into();
                } else {
                    let compute = |r: &egui_wgpu::RenderState| r.adapter.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS);
                    // on the web the window's device is the only one there is
                    if self.render_state.as_ref().map_or(cfg!(target_arch = "wasm32"), |r| !compute(r)) {
                        self.status = "this GPU has no compute shaders (WebGPU is needed)".into();
                        return;
                    }
                    let ctx = self.render_state.as_ref().map(|r| wet_fluid_gpu::GpuContext {
                        device: r.device.clone(),
                        queue: r.queue.clone(),
                        adapter_info: r.adapter.get_info(),
                    });
                    match crate::gpu::GpuSim::new(&self.doc, ctx) {
                        Ok(g) => {
                            let what = if g.simulate { "simulation" } else { "display" };
                            self.status = format!("{what} on the GPU ({})", g.adapter);
                            self.gpu = Some(g);
                        }
                        Err(e) => self.status = format!("no GPU simulation: {e}"),
                    }
                }
            }
            Cmd::History(n) => {
                for _ in 0..n.unsigned_abs() {
                    if !(if n < 0 { self.doc.undo() } else { self.doc.redo() }) {
                        break;
                    }
                }
            }
            Cmd::ResetBrush => {
                let tool = self.brush.tool;
                let (size, opacity, water, color) =
                    (self.brush.params.size, self.brush.params.opacity, self.brush.params.water, self.brush.color);
                self.brush = match &self.source {
                    Some(p) => Brush::from_preset(p),
                    None => make_brush(tool),
                };
                self.brush.set_tool(tool);
                (self.brush.params.size, self.brush.params.opacity, self.brush.params.water, self.brush.color) = (size, opacity, water, color);
                self.status = "brush parameters reset".into();
            }
            Cmd::SavePreset => {
                let name = self.preset.clone().unwrap_or_else(|| "My Brush".into());
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let mut dialog = rfd::FileDialog::new().set_file_name(format!("{name}.wetbrush")).add_filter("brush preset", &["wetbrush"]);
                    if let Some(dir) = crate::library::root().zip(crate::library::folder_for(self.brush.tool)).map(|(r, f)| r.join(f).join("My Brushes"))
                        && std::fs::create_dir_all(&dir).is_ok()
                    {
                        dialog = dialog.set_directory(dir);
                    }
                    if let Some(path) = dialog.save_file() {
                        let r = self.save_preset(&path);
                        self.report(&format!("saved preset {}", path.display()), r);
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    let target = Target::Download(format!("{name}.wetbrush"));
                    let r = platform::write(&target, |out| std::io::Write::write_all(out, &self.preset_bytes(&name)?));
                    self.report(&format!("saved preset {}", target.describe()), r);
                }
            }
            Cmd::NextColor => {
                self.color = (self.color + 1) % PALETTE.len();
                self.brush.color = PALETTE[self.color];
            }
            Cmd::SizeBy(k) => self.brush.params.size = (self.brush.params.size * k).clamp(2.0, 400.0),
            Cmd::WaterBy(d) => self.brush.params.water = (self.brush.params.water + d).clamp(0.0, 1.0),
            Cmd::OpacityBy(d) => self.brush.params.opacity = (self.brush.params.opacity + d).clamp(0.05, 1.0),
            Cmd::Volume(v) => {
                let k = self.volumes.next_index(v, &self.brush.params);
                self.volumes.apply(v, k, &mut self.brush.params);
            }
            Cmd::ToggleMulti => {
                // multi-coloured brush: the current colour plus the next two palette colours
                self.brush.multi = if self.brush.multi.is_empty() {
                    let mut m = vec![self.brush.color];
                    m.extend((1..3).map(|k| PALETTE[(self.color + k) % PALETTE.len()]));
                    m
                } else {
                    Vec::new()
                };
            }
            Cmd::TogglePause => self.paused = !self.paused,
            Cmd::ToggleTilt => {
                let l = self.doc.active_layer();
                if l.params.tilt_x == 0.0 && l.params.tilt_y == 0.0 {
                    l.params.tilt_y = 1.0;
                }
                l.params.tilt_active = !l.params.tilt_active;
                l.params.create_drops = l.params.tilt_active;
            }
            Cmd::DryLayer => {
                self.doc.checkpoint_as("Dry layer");
                self.doc.active_layer().dry_all();
            }
            Cmd::WetLayer => {
                self.doc.checkpoint_as("Wet layer");
                self.doc.active_layer().wet_all(wet_fluid::params::WATER_SCALE);
            }
            Cmd::ClearLayer => {
                self.doc.checkpoint_as("Clear layer");
                self.doc.active_layer().clear();
            }
            Cmd::TogglePigment => {
                self.pigment = !self.pigment;
                for l in &mut self.doc.layers {
                    l.wet.pigment = self.pigment.then(|| Box::new(wet_pigment::KmMixer::new()) as _);
                }
            }
            Cmd::ToggleGranulation => {
                let l = self.doc.active_layer();
                l.params.gran_enabled = !l.params.gran_enabled;
            }
            Cmd::NewLayer => {
                let n = self.doc.layers.len() + 1;
                self.doc.add_layer(&format!("Layer {n}"));
            }
            Cmd::NextLayer => self.doc.active = (self.doc.active + 1) % self.doc.layers.len(),
            Cmd::DeleteLayer => {
                if self.doc.layers.len() > 1 {
                    self.doc.checkpoint_as("Delete layer");
                    self.doc.remove_layer(i);
                }
            }
            Cmd::DuplicateLayer => {
                self.doc.duplicate_layer(i);
            }
            Cmd::MergeDown => {
                self.doc.merge_down(i);
            }
            Cmd::MoveLayer(d) => {
                let to = i as isize + d;
                if to >= 0 && (to as usize) < self.doc.layers.len() {
                    self.doc.move_layer(i, to as usize);
                }
            }
            Cmd::GroupWithBelow => {
                if i > 0 {
                    let name = format!("Group {}", self.doc.groups.len() + 1);
                    self.doc.group_layers(&name, i - 1, i);
                }
            }
            Cmd::LayerPigments => {
                let on = !self.doc.layers[i].pigments;
                self.doc.set_layer_pigments(i, on);
            }
            Cmd::ToggleMaskingFluid => self.doc.layers[i].masking_fluid = !self.doc.layers[i].masking_fluid,
            Cmd::ToggleLock => self.doc.layers[i].locked = !self.doc.layers[i].locked,
            Cmd::ToggleClip => self.doc.layers[i].clip = !self.doc.layers[i].clip,
            Cmd::ToggleMetallic => {
                self.doc.layers[i].metallic = !self.doc.layers[i].metallic;
            }
            Cmd::Transform(offset, scale, rotation) => {
                if !self.doc.layers[i].locked {
                    self.doc.checkpoint_as("Transform");
                    let centre = (self.doc.w as f32 / 2.0, self.doc.h as f32 / 2.0);
                    self.doc.transform_layer(i, offset, scale, rotation, centre);
                }
            }
            Cmd::Undo => {
                self.doc.undo();
            }
            Cmd::Redo => {
                self.doc.redo();
            }
            Cmd::Filter(f) => self.doc.apply_filter(i, &f),
            Cmd::SelectAll => self.doc.select_all(),
            Cmd::Deselect => self.doc.deselect(),
            Cmd::RestoreSelection => self.doc.restore_selection(),
            Cmd::InvertSelection => self.doc.invert_selection(),
            Cmd::StencilFromSelection => {
                if let Some(st) = self.doc.stencil_from_selection() {
                    self.doc.stencils.push(st);
                    self.doc.selection = None;
                    self.doc.update_stencils();
                }
            }
            Cmd::CutSelectionToLayer => {
                if self.doc.copy_selection(i, true).is_some() {
                    self.doc.selection = None;
                }
            }
            Cmd::ToggleStencil => {
                if let Some(s) = self.doc.stencils.first_mut() {
                    s.visible = !s.visible;
                } else {
                    let centre = (self.doc.w as f32 / 2.0, self.doc.h as f32 / 2.0);
                    let st = match std::fs::read("stencil.png").ok().and_then(|b| wet_brush::Mask::from_png(&b).ok()) {
                        Some(m) => {
                            let mask = m.data.iter().map(|&v| (v * 255.0).round() as u8).collect();
                            wet_doc::Stencil::new(mask, m.w, m.h, centre)
                        }
                        None => {
                            let n = 128;
                            let disc = (0..n * n)
                                .map(|i| {
                                    let (x, y) = ((i % n) as f32 - 63.5, (i / n) as f32 - 63.5);
                                    ((64.0 - (x * x + y * y).sqrt()).clamp(0.0, 1.0) * 255.0) as u8
                                })
                                .collect();
                            wet_doc::Stencil::new(disc, n, n, centre)
                        }
                    };
                    self.doc.stencils.push(st);
                }
                self.doc.update_stencils();
            }
            Cmd::InvertStencil => {
                if let Some(s) = self.doc.stencils.first_mut() {
                    s.invert = !s.invert;
                }
                self.doc.update_stencils();
            }
            Cmd::Tracing => {
                // tracing: import an image as a locked tracing layer (strokes take their colour
                // from it), or toggle tracing on the existing one
                if let Some(l) = self.doc.layers.iter_mut().find(|l| l.locked && l.name == "reference") {
                    l.tracing = !l.tracing;
                } else {
                    platform::pick(Pick::Tracing, &[("image", platform::IMAGES)]);
                    self.handle_picked();
                }
            }
            Cmd::ToggleReference => {
                if let Some(r) = self.doc.references.first_mut() {
                    r.visible = !r.visible;
                } else {
                    platform::pick(Pick::Reference, &[("image", platform::IMAGES)]);
                    self.handle_picked();
                }
            }
            Cmd::CycleRuler => {
                // off → horizontal → vertical → concentric ellipses → 1-point perspective,
                // all about the canvas centre
                let c = (self.doc.w as f32 / 2.0, self.doc.h as f32 / 2.0);
                self.ruler = match &self.ruler {
                    None => Some(Ruler::Line { a: c, b: (c.0 + 1.0, c.1), parallel: true }),
                    Some(Ruler::Line { a, b, .. }) if a.1 == b.1 => Some(Ruler::Line { a: c, b: (c.0, c.1 + 1.0), parallel: true }),
                    Some(Ruler::Line { .. }) => Some(Ruler::Ellipse { centre: c, radii: (c.0 * 0.5, c.1 * 0.5), rotation: 0.0, parallel: true }),
                    Some(Ruler::Ellipse { .. }) => Some(Ruler::Perspective { vanishing: vec![c] }),
                    Some(Ruler::Perspective { .. }) => None,
                };
            }
            Cmd::ToggleGrid => {
                // grid every 64 px with 4 subdivisions, snapping selections to it
                let g = &mut self.doc.guides;
                g.grid = match g.grid {
                    None => Some(wet_doc::guides::Grid::Spacing { spacing: 64.0, subdivisions: 4 }),
                    Some(_) => None,
                };
                g.snap_to_grid = g.grid.is_some();
            }
            Cmd::ToggleGreyscale => self.greyscale = !self.greyscale,
            Cmd::ToggleNatural => {
                self.natural = !self.natural;
                self.doc.set_natural_colors(self.natural);
            }
            Cmd::ToggleSoftShadows => self.doc.soft_shadows = if self.doc.soft_shadows > 0.0 { 0.0 } else { 6.0 },
            Cmd::NextVisualPreset => {
                let ps = wet_doc::visual::presets();
                self.visual_preset = (self.visual_preset + 1) % ps.len();
                let p = &ps[self.visual_preset];
                self.doc.apply_visual_preset(p);
                self.status = format!("watercolour behaviour: {}", p.name);
            }
            Cmd::Fit => self.zoom = 0.0,
            Cmd::ToggleRecording => {
                if cfg!(target_arch = "wasm32") {
                    self.status = "time-lapse recording needs the desktop app".into();
                    return;
                }
                self.recording = match self.recording {
                    Some(_) => None,
                    None => {
                        let _ = std::fs::create_dir_all("timelapse");
                        Some((0, Instant::now()))
                    }
                };
                self.status = if self.recording.is_some() { "recording time-lapse to timelapse/" } else { "time-lapse stopped" }.into();
            }
            Cmd::NewDocument(w, h) => {
                if !self.confirm_discard() {
                    return;
                }
                self.dirty = false;
                self.doc = Document::new(w.clamp(8, 8192), h.clamp(8, 8192));
                self.path = None;
                self.zoom = 0.0;
            }
            Cmd::Open => platform::pick(
                Pick::Open,
                &[("wetpaint document", &["wetdoc"]), ("image (as a layer)", platform::IMAGES), ("brush preset", &["wetbrush"])],
            ),
            Cmd::Save => match self.path.clone() {
                Some(p) => self.save_to(Target::Path(p)),
                None => self.run(Cmd::SaveAs),
            },
            Cmd::SaveAs => {
                if let Some(t) = platform::save_target("untitled.wetdoc", "wetdoc") {
                    self.save_to(t);
                }
            }
            Cmd::SaveIterative => match &self.path {
                Some(p) => self.save_to(Target::Path(wet_doc::iterative_path(p))),
                None => self.run(Cmd::SaveAs),
            },
            Cmd::ExportPng => {
                if let Some(t) = platform::save_target("painting.png", "png") {
                    let r = platform::write(&t, |out| self.doc.export_png(out));
                    self.report(&format!("exported {}", t.describe()), r);
                }
            }
            Cmd::ExportPsd => {
                if let Some(t) = platform::save_target("painting.psd", "psd") {
                    let r = platform::write(&t, |out| self.doc.export_psd(out));
                    self.report(&format!("exported {}", t.describe()), r);
                }
            }
            Cmd::ExportHeightmap => {
                if let Some(t) = platform::save_target("painting_height.exr", "exr") {
                    let r = platform::write(&t, |out| self.doc.export_heightmap_exr(out));
                    self.report(&format!("exported {}", t.describe()), r);
                }
            }
        }
    }

    /// Write a time-lapse frame every half second while recording.
    /// A sample stroke with the current brush on a small blank layer, left to flow for a
    /// moment: an S-curve with pressure rising and falling, over white. Straight RGBA8.
    pub fn stroke_preview(&self) -> (Vec<u8>, usize, usize) {
        let (w, h) = (360, 110);
        let mut layer = wet_fluid::WetLayer::new(w, h);
        let mut brush = self.brush.clone();
        brush.params.size = brush.params.size.min(h as f32 * 0.45);
        let n = 60;
        let at = |k: usize| {
            let t = k as f32 / n as f32;
            let x = 30.0 + t * (w as f32 - 60.0);
            let y = h as f32 / 2.0 + (t * std::f32::consts::TAU).sin() * h as f32 * 0.22;
            Sample { x, y, pressure: 0.15 + 0.85 * (t * std::f32::consts::PI).sin() }
        };
        brush.press(at(0));
        for k in 1..n {
            brush.move_to(&mut layer, at(k));
        }
        brush.release(&mut layer, at(n));
        for _ in 0..12 {
            layer.step();
        }
        let rgba = wet_fluid::render::render_layer(&layer)
            .iter()
            .flat_map(|p| {
                let a = p[3].clamp(0.0, 1.0);
                let c = |v: f32| ((v * a + (1.0 - a)) * 255.0).round().clamp(0.0, 255.0) as u8;
                [c(p[0]), c(p[1]), c(p[2]), 255]
            })
            .collect();
        (rgba, w, h)
    }

    /// Write the brush as a `.wetbrush`: its preset (tips, textures and the keys we do not
    /// model) with the edited parameters, and the preview stroke as thumbnail.
    pub fn save_preset(&self, path: &Path) -> std::io::Result<()> {
        let name = crate::library::preset_name(path).unwrap_or_default();
        std::fs::write(path, self.preset_bytes(&name)?)
    }

    /// The brush as a `.wetbrush` file's contents, named `name`.
    pub fn preset_bytes(&self, name: &str) -> std::io::Result<Vec<u8>> {
        let name = name.to_owned();
        let mut p = self.source.clone().unwrap_or_default();
        p.name = name;
        p.tool = self.brush.tool.id().into();
        self.brush.params.write_preset(&mut p);
        let (rgba, w, h) = self.stroke_preview();
        let mut thumb = Vec::new();
        image::RgbaImage::from_raw(w as u32, h as u32, rgba)
            .expect("preview size")
            .write_to(&mut std::io::Cursor::new(&mut thumb), image::ImageFormat::Png)
            .map_err(std::io::Error::other)?;
        p.thumbnail = Some(thumb);
        Ok(p.to_bytes())
    }

    pub fn record(&mut self, img: &[u8]) {
        let Some((n, t)) = self.recording else { return };
        if t.elapsed().as_secs_f32() < 0.5 {
            return;
        }
        let path = format!("timelapse/frame_{n:04}.png");
        if let Ok(f) = std::fs::File::create(&path) {
            let mut enc = png::Encoder::new(std::io::BufWriter::new(f), self.doc.w as u32, self.doc.h as u32);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let _ = enc.write_header().and_then(|mut wr| wr.write_image_data(img));
        }
        self.recording = Some((n + 1, Instant::now()));
    }
}
