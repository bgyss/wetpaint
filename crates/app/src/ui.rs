//! The window's panels: menu bar, tool palette, brush / colour / canvas / layer panels,
//! status bar and the canvas view.

use egui::{Align2, Color32, Pos2, Rect, RichText, Sense, Stroke, TextureHandle, TextureOptions, Vec2};
use wet_brush::{Medium, Tool, volume::Volume};
use wet_compose::MixMode;
use wet_doc::filters::Filter;

use std::{collections::HashMap, path::PathBuf};

use crate::{
    app::{App, ClickMode, Cmd, PALETTE},
    library,
};

/// UI-only state kept between frames.
pub struct Ui {
    canvas_tex: Option<TextureHandle>,
    /// The active layer alone, drawn through the transform being edited.
    layer_tex: Option<TextureHandle>,
    reference_tex: Option<(usize, TextureHandle)>,
    new_doc: Option<(usize, usize)>,
    show_shortcuts: bool,
    /// Canvas panel size last frame: the view is refitted while the panels settle on the
    /// first frames, and on window resizes while the view is still fitted.
    last_rect: Vec2,
    fitted_zoom: f32,
    /// Brush library: categories per tool folder and preset thumbnails, read on first use.
    library: HashMap<&'static str, Vec<library::Category>>,
    thumbs: HashMap<PathBuf, Option<TextureHandle>>,
    /// Brush Creator window: open, and its preview stroke with the parameters it shows.
    show_creator: bool,
    creator_tex: Option<(String, TextureHandle)>,
    /// Layer thumbnails in the Layers panel, refreshed one layer every few frames.
    layer_thumbs: Vec<Option<TextureHandle>>,
    frame: usize,
    /// Layer being renamed and the edited name.
    renaming: Option<(usize, String)>,
    /// Commands collected while drawing the UI, run after it (the app is borrowed while drawing).
    queue: Vec<Cmd>,
}

/// File > Quit's shortcut, as the platform writes it.
const QUIT_KEY: &str = if cfg!(target_os = "macos") { "⌘Q" } else { "Ctrl+Q" };
const ACCENT: Color32 = Color32::from_rgb(70, 130, 200);

const MEDIA: [(Medium, &str, &str); 8] = [
    (Medium::Watercolor, "Watercolour", "1"),
    (Medium::Ink, "Ink", "2"),
    (Medium::Oil, "Oil & Acrylic", "3"),
    (Medium::ExpressOil, "Express Oil", ""),
    (Medium::Pastel, "Pastel", "4"),
    (Medium::Pencil, "Pencil", "5"),
    (Medium::Marker, "Marker", "6"),
    (Medium::Airbrush, "Airbrush", "7"),
];

const UTILITIES: [(Tool, &str, &str); 6] = [
    (Tool::Water, "Water", "8"),
    (Tool::Dry, "Dry", "9"),
    (Tool::Eraser, "Eraser", "0"),
    (Tool::Blend, "Blend", "B"),
    (Tool::Smudge, "Smudge", "M"),
    (Tool::Blow, "Blow", "O"),
];

const SELECTIONS: [(ClickMode, &str, &str); 6] = [
    (ClickMode::SelectRect, "Rectangle", "U"),
    (ClickMode::SelectEllipse, "Ellipse", "E"),
    (ClickMode::Lasso, "Lasso", "'"),
    (ClickMode::Wand, "Magic wand", ";"),
    (ClickMode::Pick, "Pick colour", "I"),
    (ClickMode::Fill, "Fill", "F"),
];

fn rgb(c: [f32; 3]) -> Color32 {
    Color32::from_rgb(c[0].clamp(0.0, 255.0) as u8, c[1].clamp(0.0, 255.0) as u8, c[2].clamp(0.0, 255.0) as u8)
}

fn tool_name(t: Tool) -> String {
    match t {
        Tool::Paint(m) => MEDIA.iter().find(|x| x.0 == m).map_or("Paint", |x| x.1).to_owned(),
        Tool::Liquify(l) => format!("Liquify ({l:?})"),
        Tool::Clone { .. } => "Clone".into(),
        other => UTILITIES.iter().find(|x| x.0 == other).map_or_else(|| format!("{other:?}"), |x| x.1.to_owned()),
    }
}

impl Ui {
    pub fn new() -> Self {
        Self { canvas_tex: None, layer_tex: None, reference_tex: None, new_doc: None, show_shortcuts: false, last_rect: Vec2::ZERO, fitted_zoom: 0.0, library: HashMap::new(), thumbs: HashMap::new(), show_creator: false, creator_tex: None, layer_thumbs: Vec::new(), frame: 0, renaming: None, queue: Vec::new() }
    }

    pub fn style(ctx: &egui::Context) {
        ctx.options_mut(|o| o.theme_preference = egui::ThemePreference::Dark);
        ctx.global_style_mut(|s| {
            s.spacing.item_spacing = Vec2::new(6.0, 5.0);
            s.spacing.slider_width = 150.0;
            s.visuals.selection.bg_fill = ACCENT;
        });
    }

    /// Draw the whole window and run the commands it produced.
    pub fn show(&mut self, ui: &mut egui::Ui, app: &mut App) {
        if ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Q)) {
            app.quit = true;
        }
        self.menu_bar(ui, app);
        self.status_bar(ui, app);
        self.tool_palette(ui, app);
        self.side_panel(ui, app);
        self.canvas(ui, app);
        self.dialogs(ui.ctx(), app);
        for cmd in std::mem::take(&mut self.queue) {
            app.run(cmd);
        }
    }

    fn item(&mut self, ui: &mut egui::Ui, label: &str, key: &str, cmd: Cmd) {
        let b = egui::Button::new(label).shortcut_text(key);
        if ui.add(b).clicked() {
            self.queue.push(cmd);
            ui.close();
        }
    }

    fn check(&mut self, ui: &mut egui::Ui, label: &str, key: &str, on: bool, cmd: Cmd) {
        let b = egui::Button::new(label).shortcut_text(key).selected(on);
        if ui.add(b).clicked() {
            self.queue.push(cmd);
            ui.close();
        }
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui, app: &mut App) {
        egui::Panel::top("menu").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New…").clicked() {
                        self.new_doc = Some((app.doc.w, app.doc.h));
                        ui.close();
                    }
                    self.item(ui, "Open…", "L", Cmd::Open);
                    ui.separator();
                    self.item(ui, "Save", "S", Cmd::Save);
                    self.item(ui, "Save As…", "", Cmd::SaveAs);
                    self.item(ui, "Save Iteration", "Shift+D", Cmd::SaveIterative);
                    ui.separator();
                    self.item(ui, "Export PNG…", "F5", Cmd::ExportPng);
                    self.item(ui, "Export Layered PSD…", "V", Cmd::ExportPsd);
                    self.item(ui, "Export Heightmap (EXR)…", "Shift+E", Cmd::ExportHeightmap);
                    ui.separator();
                    self.check(ui, "Record Time-lapse", "R", app.recording.is_some(), Cmd::ToggleRecording);
                    ui.separator();
                    if ui.add(egui::Button::new("Quit").shortcut_text(QUIT_KEY)).clicked() {
                        app.quit = true;
                        ui.close();
                    }
                });
                ui.menu_button("Edit", |ui| {
                    self.item(ui, "Undo", "Z", Cmd::Undo);
                    self.item(ui, "Redo", "Y", Cmd::Redo);
                    ui.separator();
                    if ui.button("Brush Creator…").clicked() {
                        self.show_creator = true;
                        ui.close();
                    }
                    self.item(ui, "Save Brush Preset…", "", Cmd::SavePreset);
                    ui.separator();
                    self.check(ui, "Pigment Mixing", "P", app.pigment, Cmd::TogglePigment);
                    self.check(ui, "Natural Colours (RYB)", "Shift+P", app.natural, Cmd::ToggleNatural);
                    self.item(ui, "Next Visual Settings Preset", "Shift+V", Cmd::NextVisualPreset);
                });
                ui.menu_button("Layer", |ui| {
                    let l = &app.doc.layers[app.doc.active];
                    let (mf, lock, clip, pig, metal) = (l.masking_fluid, l.locked, l.clip, l.pigments, l.metallic);
                    self.item(ui, "New Layer", "N", Cmd::NewLayer);
                    self.item(ui, "Duplicate Layer", "Shift+N", Cmd::DuplicateLayer);
                    self.item(ui, "Merge Down", "Shift+M", Cmd::MergeDown);
                    self.item(ui, "Delete Layer", "", Cmd::DeleteLayer);
                    self.item(ui, "Group With Layer Below", "F6", Cmd::GroupWithBelow);
                    ui.separator();
                    self.check(ui, "Masking Fluid", "K", mf, Cmd::ToggleMaskingFluid);
                    self.check(ui, "Clip to Layer Below", "J", clip, Cmd::ToggleClip);
                    self.check(ui, "Lock", "A", lock, Cmd::ToggleLock);
                    self.check(ui, "Pigment Mixing on Layer", "F4", pig, Cmd::LayerPigments);
                    self.check(ui, "Metallic", "Shift+T", metal, Cmd::ToggleMetallic);
                    ui.separator();
                    self.item(ui, "Import Tracing Image…", "F2", Cmd::Tracing);
                    ui.separator();
                    if app.xform.is_some() {
                        self.item(ui, "Apply Transform", "Enter", Cmd::ApplyTransform);
                        self.item(ui, "Cancel Transform", "Esc", Cmd::CancelTransform);
                    } else {
                        self.item(ui, "Transform…", "Shift+R", Cmd::Mode(ClickMode::Transform));
                    }
                    self.item(ui, "Rotate −5°", "<", Cmd::Transform((0.0, 0.0), 1.0, -5f32.to_radians()));
                    self.item(ui, "Rotate +5°", ">", Cmd::Transform((0.0, 0.0), 1.0, 5f32.to_radians()));
                    self.item(ui, "Scale Down", "{", Cmd::Transform((0.0, 0.0), 1.0 / 1.1, 0.0));
                    self.item(ui, "Scale Up", "}", Cmd::Transform((0.0, 0.0), 1.1, 0.0));
                });
                ui.menu_button("Select", |ui| {
                    self.item(ui, "All", "", Cmd::SelectAll);
                    self.item(ui, "Deselect", "Q", Cmd::Deselect);
                    self.item(ui, "Reselect", "Shift+Q", Cmd::RestoreSelection);
                    self.item(ui, "Invert", "/", Cmd::InvertSelection);
                    ui.separator();
                    for (m, name, key) in SELECTIONS.iter().take(4) {
                        self.item(ui, name, key, Cmd::Mode(*m));
                    }
                    ui.separator();
                    self.item(ui, "Stencil From Selection", "F7", Cmd::StencilFromSelection);
                    self.item(ui, "Cut to New Layer", "F8", Cmd::CutSelectionToLayer);
                    self.item(ui, "Show / Place Stencil", "\\", Cmd::ToggleStencil);
                    self.item(ui, "Invert Stencil", "|", Cmd::InvertStencil);
                });
                ui.menu_button("Filter", |ui| {
                    self.item(ui, "Invert", "Shift+I", Cmd::Filter(Filter::Invert));
                    self.item(ui, "Desaturate", "Shift+U", Cmd::Filter(Filter::Desaturate { luminosity: true }));
                    self.item(ui, "Hue +30°", "Shift+H", Cmd::Filter(Filter::HueSaturation { hue: 30.0, saturation: 0.0, lightness: 0.0 }));
                    self.item(ui, "Gaussian Blur", "Shift+B", Cmd::Filter(Filter::GaussianBlur { radius: 4.0 }));
                    self.item(ui, "Sharpen", "Shift+S", Cmd::Filter(Filter::Sharpen { radius: 2.0, strength: 0.5 }));
                });
                ui.menu_button("View", |ui| {
                    self.item(ui, "Fit Canvas", "Home", Cmd::Fit);
                    self.check(ui, "Greyscale", "Shift+G", app.greyscale, Cmd::ToggleGreyscale);
                    self.check(ui, "Grid", "F9", app.doc.guides.grid.is_some(), Cmd::ToggleGrid);
                    self.item(ui, "Next Ruler", "F1", Cmd::CycleRuler);
                    self.check(ui, "GPU Simulation", "Shift+C", app.gpu.is_some(), Cmd::ToggleGpu);
                    self.check(ui, "Soft Shadows", "Shift+O", app.doc.soft_shadows > 0.0, Cmd::ToggleSoftShadows);
                    self.check(ui, "Reference Image", "Insert", app.doc.references.iter().any(|r| r.visible), Cmd::ToggleReference);
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("Keyboard Shortcuts").clicked() {
                        self.show_shortcuts = true;
                        ui.close();
                    }
                });
            });
        });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui, app: &mut App) {
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                let l = &app.doc.layers[app.doc.active];
                let mode = match app.mode {
                    ClickMode::Paint => tool_name(app.brush.tool),
                    ClickMode::Transform => "Transform".to_owned(),
                    m => SELECTIONS.iter().find(|x| x.0 == m).map_or("", |x| x.1).to_owned(),
                };
                ui.label(RichText::new(mode).strong());
                ui.separator();
                ui.label(format!("{} × {} px", app.doc.w, app.doc.h));
                ui.separator();
                ui.label(format!("{:.0}%", app.zoom * 100.0));
                ui.separator();
                ui.label(format!("{:.0}, {:.0}", app.cursor.0, app.cursor.1));
                ui.separator();
                ui.label(format!("layer \"{}\"", l.name));
                ui.separator();
                let on = if app.gpu.is_some() { " (GPU)" } else { "" };
                let sim = if app.paused { "simulation paused".to_owned() } else { format!("simulation {:.0} steps/s{on}", app.sim_rate) };
                ui.label(sim);
                if l.wet.params.tilt_active {
                    ui.separator();
                    ui.colored_label(ACCENT, "tilted");
                }
                if let Some(p) = app.pressure {
                    ui.separator();
                    ui.label(format!("pressure {:.2}", p));
                }
                if !app.status.is_empty() {
                    ui.separator();
                    ui.add(egui::Label::new(RichText::new(&app.status).weak()).truncate()).on_hover_text(&app.status);
                }
            });
        });
    }

    fn tool_palette(&mut self, ui: &mut egui::Ui, app: &mut App) {
        egui::Panel::left("tools").resizable(false).exact_size(132.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(4.0);
                ui.label(RichText::new("MEDIA").small().weak());
                for (m, name, key) in MEDIA {
                    let on = app.mode == ClickMode::Paint && app.brush.tool == Tool::Paint(m);
                    self.tool_button(ui, name, key, on, Cmd::Tool(Tool::Paint(m)));
                }
                ui.add_space(6.0);
                ui.label(RichText::new("TOOLS").small().weak());
                for (t, name, key) in UTILITIES {
                    let on = app.mode == ClickMode::Paint && app.brush.tool == t;
                    self.tool_button(ui, name, key, on, Cmd::Tool(t));
                }
                let liq = matches!(app.brush.tool, Tool::Liquify(_)) && app.mode == ClickMode::Paint;
                let label = if liq { tool_name(app.brush.tool) } else { "Liquify".to_owned() };
                self.tool_button(ui, &label, "H", liq, Cmd::CycleLiquify);
                self.tool_button(ui, "Transform", "Shift+R", app.mode == ClickMode::Transform, Cmd::Mode(ClickMode::Transform));
                ui.add_space(6.0);
                ui.label(RichText::new("SELECT").small().weak());
                for (m, name, key) in SELECTIONS {
                    self.tool_button(ui, name, key, app.mode == m, Cmd::Mode(m));
                }
            });
        });
    }

    fn tool_button(&mut self, ui: &mut egui::Ui, name: &str, key: &str, on: bool, cmd: Cmd) {
        let b = egui::Button::new(name).selected(on).min_size(Vec2::new(ui.available_width(), 22.0));
        let r = ui.add(b);
        let r = if key.is_empty() { r } else { r.on_hover_text(format!("{name} ({key})")) };
        if r.clicked() {
            self.queue.push(cmd);
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, app: &mut App) {
        egui::Panel::right("panels").resizable(true).default_size(270.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::CollapsingHeader::new(RichText::new("Brush").strong()).default_open(true).show(ui, |ui| self.brush_panel(ui, app));
                egui::CollapsingHeader::new(RichText::new("Brush Library").strong()).default_open(true).show(ui, |ui| self.library_panel(ui, app));
                egui::CollapsingHeader::new(RichText::new("Colour").strong()).default_open(true).show(ui, |ui| self.colour_panel(ui, app));
                egui::CollapsingHeader::new(RichText::new("Canvas").strong()).default_open(true).show(ui, |ui| self.canvas_panel(ui, app));
                egui::CollapsingHeader::new(RichText::new("Paper").strong()).default_open(false).show(ui, |ui| self.paper_panel(ui, app));
                egui::CollapsingHeader::new(RichText::new("Layers").strong()).default_open(true).show(ui, |ui| self.layers_panel(ui, app));
                egui::CollapsingHeader::new(RichText::new("History").strong()).default_open(false).show(ui, |ui| self.history_panel(ui, app));
            });
        });
    }

    fn brush_panel(&mut self, ui: &mut egui::Ui, app: &mut App) {
        let p = &mut app.brush.params;
        egui::Grid::new("brush").num_columns(2).show(ui, |ui| {
            ui.label("Size");
            ui.add(egui::Slider::new(&mut p.size, 2.0..=400.0).logarithmic(true).suffix(" px"));
            ui.end_row();
            ui.label("Opacity");
            let mut o = p.opacity * 100.0;
            if ui.add(egui::Slider::new(&mut o, 5.0..=100.0).suffix(" %")).changed() {
                p.opacity = o / 100.0;
            }
            ui.end_row();
            ui.label("Water");
            let mut w = p.water * 100.0;
            if ui.add(egui::Slider::new(&mut w, 0.0..=100.0).suffix(" %")).changed() {
                p.water = w / 100.0;
            }
            ui.end_row();
        });
        ui.horizontal_wrapped(|ui| {
            for (v, label) in [(Volume::Size, "Size preset"), (Volume::Opacity, "Opacity preset"), (Volume::Water, "Water preset")] {
                if ui.small_button(label).clicked() {
                    self.queue.push(Cmd::Volume(v));
                }
            }
        });
        if ui.button("Brush Creator…").on_hover_text("Edit every brush parameter with a preview stroke").clicked() {
            self.show_creator = true;
        }
        let multi = !app.brush.multi.is_empty();
        if ui.selectable_label(multi, "Multi-coloured brush").clicked() {
            self.queue.push(Cmd::ToggleMulti);
        }
        let rulers = match &app.ruler {
            None => "off",
            Some(wet_brush::ruler::Ruler::Line { a, b, .. }) if a.1 == b.1 => "horizontal",
            Some(wet_brush::ruler::Ruler::Line { .. }) => "vertical",
            Some(wet_brush::ruler::Ruler::Ellipse { .. }) => "ellipses",
            Some(wet_brush::ruler::Ruler::Perspective { .. }) => "perspective",
        };
        ui.horizontal(|ui| {
            ui.label("Ruler");
            if ui.button(rulers).on_hover_text("Cycle rulers (F1)").clicked() {
                self.queue.push(Cmd::CycleRuler);
            }
        });
    }

    fn library_panel(&mut self, ui: &mut egui::Ui, app: &mut App) {
        let Some(folder) = library::folder_for(app.brush.tool) else {
            ui.weak("This tool has no presets.");
            return;
        };
        let Some(res) = library::root() else {
            ui.weak("No brush library folder.");
            return;
        };
        let cats = self.library.entry(folder).or_insert_with(|| library::categories(&res, folder));
        if cats.is_empty() {
            ui.weak("No saved brushes for this tool yet. Brush Creator > Save Preset adds them here.");
            return;
        }
        let mut chosen = None;
        for cat in cats.iter() {
            let open = cat.presets.iter().any(|p| app.preset.as_deref() == Some(p.name.as_str()));
            egui::CollapsingHeader::new(format!("{} ({})", cat.name, cat.presets.len()))
                .id_salt((folder, &cat.name))
                .default_open(open)
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = Vec2::splat(4.0);
                        for p in &cat.presets {
                            let tex = self.thumbs.entry(p.path.clone()).or_insert_with(|| {
                                library::thumbnail(&p.path, 96).map(|(rgba, w, h)| {
                                    let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
                                    ui.ctx().load_texture(p.name.clone(), img, TextureOptions::LINEAR)
                                })
                            });
                            let selected = app.preset.as_deref() == Some(p.name.as_str());
                            let r = match tex {
                                Some(t) => ui.add(egui::Button::image((t.id(), Vec2::splat(44.0))).selected(selected)),
                                None => ui.add(egui::Button::new(&p.name).selected(selected)),
                            };
                            if r.on_hover_text(&p.name).clicked() {
                                chosen = Some(p.path.clone());
                            }
                        }
                    });
                });
        }
        if let Some(path) = chosen {
            app.load_preset(&path);
        }
    }

    fn paper_panel(&mut self, ui: &mut egui::Ui, app: &mut App) {
        use wet_paper::Surface;
        let mut paper = app.doc.paper;
        let mut apply = false;
        egui::Grid::new("paper").num_columns(2).show(ui, |ui| {
            ui.label("Surface");
            egui::ComboBox::from_id_salt("surface").selected_text(format!("{:?}", paper.spec.surface)).show_ui(ui, |ui| {
                for (s, name) in [(Surface::HotPressed, "Hot pressed"), (Surface::ColdPressed, "Cold pressed"), (Surface::Rough, "Rough")] {
                    apply |= ui.selectable_value(&mut paper.spec.surface, s, name).changed();
                }
            });
            ui.end_row();
            ui.label("Scale");
            let r = ui.add(egui::Slider::new(&mut paper.spec.scale, 0.25..=4.0).logarithmic(true));
            apply |= r.drag_stopped() || (r.changed() && !r.dragged());
            ui.end_row();
            ui.label("Relief");
            let r = ui.add(egui::Slider::new(&mut paper.spec.relief, 0.0..=3.0));
            apply |= r.drag_stopped() || (r.changed() && !r.dragged());
            ui.end_row();
            ui.label("Colour");
            let mut c = rgb(paper.color);
            if egui::color_picker::color_edit_button_srgba(ui, &mut c, egui::color_picker::Alpha::Opaque).changed() {
                paper.color = [c.r() as f32, c.g() as f32, c.b() as f32];
                apply = true;
            }
            ui.end_row();
        });
        apply |= ui.checkbox(&mut paper.deckled, "Deckled edges").changed();
        apply |= ui.checkbox(&mut paper.visible, "Show paper").changed();
        if apply {
            app.dirty = true;
            app.release_gpu();
            app.doc.set_paper(paper);
        }
    }

    fn colour_panel(&mut self, ui: &mut egui::Ui, app: &mut App) {
        let mut c = rgb(app.brush.color);
        ui.horizontal(|ui| {
            if egui::color_picker::color_edit_button_srgba(ui, &mut c, egui::color_picker::Alpha::Opaque).changed() {
                app.set_color([c.r() as f32, c.g() as f32, c.b() as f32]);
            }
            ui.label(format!("#{:02X}{:02X}{:02X}", c.r(), c.g(), c.b()));
        });
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::splat(3.0);
            for (k, p) in PALETTE.iter().enumerate() {
                let (rect, r) = ui.allocate_exact_size(Vec2::splat(20.0), Sense::click());
                let sel = rgb(*p) == rgb(app.brush.color);
                ui.painter().rect_filled(rect, 3.0, rgb(*p));
                if sel || r.hovered() {
                    ui.painter().rect_stroke(rect, 3.0, Stroke::new(2.0, Color32::WHITE), egui::StrokeKind::Outside);
                }
                if r.clicked() {
                    app.color = k;
                    app.set_color(*p);
                }
            }
        });
    }

    fn canvas_panel(&mut self, ui: &mut egui::Ui, app: &mut App) {
        ui.horizontal(|ui| {
            let label = if app.paused { "▶ Resume" } else { "⏸ Pause" };
            if ui.button(label).on_hover_text("Space").clicked() {
                self.queue.push(Cmd::TogglePause);
            }
            for (label, cmd, key) in [("Dry", Cmd::DryLayer, "D"), ("Wet", Cmd::WetLayer, "W"), ("Clear", Cmd::ClearLayer, "X")] {
                if ui.button(label).on_hover_text(key).clicked() {
                    self.queue.push(cmd);
                }
            }
        });
        let mut gpu = app.gpu.is_some();
        let tip = app.gpu.as_ref().map_or("Step wet layers on the GPU (Shift+C)".to_owned(), |g| format!("On {} (Shift+C)", g.adapter));
        if ui.checkbox(&mut gpu, "GPU simulation").on_hover_text(tip).changed() {
            self.queue.push(Cmd::ToggleGpu);
        }
        let params = &mut app.doc.layers[app.doc.active].wet.params;
        let mut tilt = params.tilt_active;
        if ui.checkbox(&mut tilt, "Tilt canvas").on_hover_text("T").changed() {
            self.queue.push(Cmd::ToggleTilt);
        }
        if params.tilt_active {
            // direction and strength of gravity, from the tilt vector
            let mut angle = params.tilt_x.atan2(params.tilt_y).to_degrees();
            let mut amount = (params.tilt_x * params.tilt_x + params.tilt_y * params.tilt_y).sqrt();
            let a = ui.add(egui::Slider::new(&mut angle, -180.0..=180.0).text("direction").suffix("°"));
            let s = ui.add(egui::Slider::new(&mut amount, 0.05..=1.0).text("strength"));
            if a.changed() || s.changed() {
                app.dirty = true;
                params.tilt_x = amount * angle.to_radians().sin();
                params.tilt_y = amount * angle.to_radians().cos();
            }
        }
        let mut gran = params.gran_enabled;
        if ui.checkbox(&mut gran, "Granulation").on_hover_text("G").changed() {
            self.queue.push(Cmd::ToggleGranulation);
        }
        let mut pig = app.pigment;
        if ui.checkbox(&mut pig, "Pigment mixing").on_hover_text("P").changed() {
            self.queue.push(Cmd::TogglePigment);
        }
        let mut nat = app.natural;
        if ui.checkbox(&mut nat, "Natural colours (RYB)").on_hover_text("Shift+P").changed() {
            self.queue.push(Cmd::ToggleNatural);
        }
        app.dirty |= ui.add(egui::Slider::new(&mut app.doc.impasto, 0.0..=3.0).text("impasto")).changed();
    }

    fn layers_panel(&mut self, ui: &mut egui::Ui, app: &mut App) {
        ui.horizontal(|ui| {
            for (label, cmd, tip) in [
                ("New", Cmd::NewLayer, "New layer (N)"),
                ("Copy", Cmd::DuplicateLayer, "Duplicate (Shift+N)"),
                ("Merge", Cmd::MergeDown, "Merge down (Shift+M)"),
                ("⬆", Cmd::MoveLayer(1), "Move up"),
                ("⬇", Cmd::MoveLayer(-1), "Move down"),
                ("🗑", Cmd::DeleteLayer, "Delete layer"),
            ] {
                if ui.button(label).on_hover_text(tip).clicked() {
                    self.queue.push(cmd);
                }
            }
        });
        let active = app.doc.active;
        let mut edited = false;
        let l = &mut app.doc.layers[active];
        egui::Grid::new("layer").num_columns(2).show(ui, |ui| {
            ui.label("Blend");
            egui::ComboBox::from_id_salt("blend").selected_text(format!("{:?}", l.mode)).show_ui(ui, |ui| {
                for m in MixMode::ALL {
                    edited |= ui.selectable_value(&mut l.mode, m, format!("{m:?}")).changed();
                }
            });
            ui.end_row();
            ui.label("Opacity");
            let mut o = l.opacity * 100.0;
            if ui.add(egui::Slider::new(&mut o, 0.0..=100.0).suffix(" %")).changed() {
                l.opacity = o / 100.0;
                edited = true;
            }
            ui.end_row();
        });
        ui.separator();
        self.refresh_layer_thumb(ui.ctx(), app);
        // top layer first
        for idx in (0..app.doc.layers.len()).rev() {
            let l = &mut app.doc.layers[idx];
            ui.horizontal(|ui| {
                if let Some(Some(t)) = self.layer_thumbs.get(idx) {
                    let size = t.size_vec2();
                    let (r, _) = ui.allocate_exact_size(Vec2::new(38.0, 26.0), Sense::hover());
                    let img = Rect::from_center_size(r.center(), size);
                    ui.painter().image(t.id(), img, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                    ui.painter().rect_stroke(img, 0.0, Stroke::new(1.0, Color32::from_gray(80)), egui::StrokeKind::Outside);
                }
                let eye = if l.visible { "👁" } else { "◌" };
                if ui.small_button(eye).on_hover_text("Visible").clicked() {
                    l.visible = !l.visible;
                    edited = true;
                }
                let lock = if l.locked { "🔒" } else { "🔓" };
                if ui.small_button(lock).on_hover_text("Lock (A)").clicked() {
                    l.locked = !l.locked;
                    edited = true;
                }
                let mut name = l.name.clone();
                if l.masking_fluid {
                    name += "  · masking fluid";
                }
                if l.clip {
                    name += "  · clipped";
                }
                if l.wet.params.tilt_active {
                    name += "  · tilted";
                }
                if let Some((k, text)) = &mut self.renaming
                    && *k == idx
                {
                    let r = ui.add(egui::TextEdit::singleline(text).desired_width(ui.available_width()));
                    r.request_focus();
                    if r.lost_focus() {
                        if !text.trim().is_empty() && !ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                            l.name = text.trim().to_owned();
                            edited = true;
                        }
                        self.renaming = None;
                    }
                    return;
                }
                let r = ui.add(egui::Button::selectable(idx == active, name).min_size(Vec2::new(ui.available_width(), 20.0)));
                if r.clicked() {
                    app.doc.active = idx;
                }
                if r.on_hover_text("Double-click to rename").double_clicked() {
                    self.renaming = Some((idx, l.name.clone()));
                }
            });
        }
        app.dirty |= edited;
    }

    /// Undo history: the edits so far (oldest first) and those that can be redone (dimmed).
    /// Clicking a row returns the painting to the state just after that edit.
    fn history_panel(&mut self, ui: &mut egui::Ui, app: &mut App) {
        let (done, undone) = app.doc.history();
        let n = done.len() as isize;
        let layer = |k: usize| app.doc.layers.get(k).map_or(String::new(), |l| format!("  · {}", l.name));
        if ui.add(egui::Button::selectable(done.is_empty(), "Start").min_size(Vec2::new(ui.available_width(), 18.0))).clicked() {
            self.queue.push(Cmd::History(-n));
        }
        for (k, (label, l)) in done.iter().enumerate() {
            let r = ui.add(egui::Button::selectable(k + 1 == done.len(), format!("{label}{}", layer(*l))).min_size(Vec2::new(ui.available_width(), 18.0)));
            if r.clicked() {
                self.queue.push(Cmd::History(k as isize + 1 - n));
            }
        }
        for (j, (label, l)) in undone.iter().enumerate() {
            let text = RichText::new(format!("{label}{}", layer(*l))).weak().italics();
            if ui.add(egui::Button::selectable(false, text).min_size(Vec2::new(ui.available_width(), 18.0))).clicked() {
                self.queue.push(Cmd::History(j as isize + 1));
            }
        }
        ui.weak(format!("{} of {} steps kept", done.len(), app.doc.undo_limit));
    }

    fn dialogs(&mut self, ctx: &egui::Context, app: &mut App) {
        if let Some((mut w, mut h)) = self.new_doc {
            let mut open = true;
            let mut create = false;
            egui::Window::new("New Canvas").collapsible(false).resizable(false).open(&mut open).show(ctx, |ui| {
                egui::Grid::new("newdoc").show(ui, |ui| {
                    ui.label("Width");
                    ui.add(egui::DragValue::new(&mut w).range(8..=8192).suffix(" px"));
                    ui.end_row();
                    ui.label("Height");
                    ui.add(egui::DragValue::new(&mut h).range(8..=8192).suffix(" px"));
                    ui.end_row();
                });
                ui.horizontal(|ui| {
                    for (label, pw, ph) in [("A4 @100 dpi", 827, 1169), ("1080p", 1920, 1080), ("Square", 1024, 1024)] {
                        if ui.small_button(label).clicked() {
                            (w, h) = (pw, ph);
                        }
                    }
                });
                create = ui.button("Create").clicked();
            });
            self.new_doc = (open && !create).then_some((w, h));
            if create {
                self.queue.push(Cmd::NewDocument(w, h));
            }
        }
        if self.show_creator {
            self.brush_creator(ctx, app);
        }
        if self.show_shortcuts {
            egui::Window::new("Keyboard Shortcuts").open(&mut self.show_shortcuts).default_width(420.0).show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| ui.label(crate::SHORTCUTS));
            });
        }
        for cmd in std::mem::take(&mut self.queue) {
            app.run(cmd);
        }
        if let Some(r) = app.doc.references.iter().find(|r| r.visible) {
            let id = r.rgba.as_ptr() as usize;
            if self.reference_tex.as_ref().is_none_or(|(k, _)| *k != id) {
                let img = egui::ColorImage::from_rgba_unmultiplied([r.w, r.h], &r.rgba);
                self.reference_tex = Some((id, ctx.load_texture("reference", img, TextureOptions::LINEAR)));
            }
            let (_, tex) = self.reference_tex.as_ref().expect("reference texture");
            let aspect = r.h as f32 / r.w.max(1) as f32;
            egui::Window::new("Reference").default_width(260.0).show(ctx, |ui| {
                let w = ui.available_width();
                ui.image((tex.id(), Vec2::new(w, w * aspect)));
            });
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui, app: &mut App) {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(Color32::from_gray(38))).show(ui, |ui| {
            let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
            let (dw, dh) = (app.doc.w as f32, app.doc.h as f32);
            if rect.size() != self.last_rect && (self.last_rect == Vec2::ZERO || app.zoom == self.fitted_zoom) {
                app.zoom = 0.0;
            }
            self.last_rect = rect.size();
            if app.zoom <= 0.0 {
                app.zoom = (rect.width() / dw).min(rect.height() / dh) * 0.96;
                self.fitted_zoom = app.zoom;
                app.pan = ((rect.width() - dw * app.zoom) / 2.0, (rect.height() - dh * app.zoom) / 2.0);
            }
            let to_canvas = |p: Pos2, app: &App| ((p.x - rect.min.x - app.pan.0) / app.zoom, (p.y - rect.min.y - app.pan.1) / app.zoom);

            // zoom around the pointer: scroll wheel or pinch
            if let Some(hover) = resp.hover_pos() {
                let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
                let k = pinch * 1.0015f32.powf(scroll);
                if k != 1.0 {
                    let before = to_canvas(hover, app);
                    app.zoom = (app.zoom * k).clamp(0.05, 32.0);
                    app.pan = (hover.x - rect.min.x - before.0 * app.zoom, hover.y - rect.min.y - before.1 * app.zoom);
                }
                app.cursor = to_canvas(hover, app);
            }
            // pan: middle drag or Shift + drag
            let shift = ui.input(|i| i.modifiers.shift);
            if resp.dragged_by(egui::PointerButton::Middle) || (shift && resp.dragged_by(egui::PointerButton::Primary)) {
                let d = resp.drag_delta();
                app.pan = (app.pan.0 + d.x, app.pan.1 + d.y);
            } else {
                // every pointer sample since the last frame, so fast strokes keep their shape
                if resp.drag_started_by(egui::PointerButton::Primary) || resp.clicked() {
                    if let Some(p) = ui.input(|i| i.pointer.press_origin()) {
                        app.cursor = to_canvas(p, app);
                    }
                    app.press();
                }
                app.free_corner = ui.input(|i| i.modifiers.command);
                let xdrag = app.xform.as_ref().is_some_and(|x| x.dragging().is_some());
                if app.painting || app.select_start.is_some() || xdrag {
                    let moves: Vec<Pos2> = ui.input(|i| {
                        i.events.iter().filter_map(|e| if let egui::Event::PointerMoved(p) = e { Some(*p) } else { None }).collect()
                    });
                    for p in moves {
                        app.cursor = to_canvas(p, app);
                        app.moved();
                    }
                    // a stroke on the GPU stamps the frame's steps together
                    app.flush_stroke(false);
                }
                if resp.drag_stopped() || (resp.clicked() && (app.painting || app.select_start.is_some() || xdrag)) {
                    app.release();
                }
            }

            // the painting, with view overlays baked in (greyscale, stencils, selection); a
            // layer being transformed is left out and drawn on its own through the transform
            let active = app.doc.active;
            let shown = app.doc.layers[active].visible;
            let transforming = app.xform.is_some();
            if transforming {
                app.doc.layers[active].visible = false;
            }
            app.show_gpu();
            let nearest = app.zoom >= 2.0;
            // composited on the GPU when the document allows, on the CPU otherwise
            let filter = if nearest { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear };
            let tex_id = match app.gpu_display(filter) {
                Some(id) => id,
                None => {
                    let mut img = app.doc.render(app.pigment.then(wet_pigment::KmMixer::new).as_ref().map(|p| p as &dyn wet_compose::PigmentMix));
                    app.doc.layers[active].visible = shown;
                    app.record(&img);
                    decorate(app, &mut img);
                    let image = color_image(app.doc.w, app.doc.h, &img);
                    let opts = if nearest { TextureOptions::NEAREST } else { TextureOptions::LINEAR };
                    match &mut self.canvas_tex {
                        Some(t) => t.set(image, opts),
                        None => self.canvas_tex = Some(ui.ctx().load_texture("canvas", image, opts)),
                    }
                    self.canvas_tex.as_ref().expect("canvas texture").id()
                }
            };
            app.doc.layers[active].visible = shown;
            let canvas_rect = Rect::from_min_size(rect.min + Vec2::new(app.pan.0, app.pan.1), Vec2::new(dw, dh) * app.zoom);
            let painter = ui.painter_at(rect);
            painter.rect_filled(canvas_rect.translate(Vec2::new(3.0, 4.0)), 0.0, Color32::from_black_alpha(90));
            painter.image(tex_id, canvas_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);

            let to_screen = |x: f32, y: f32| rect.min + Vec2::new(app.pan.0 + x * app.zoom, app.pan.1 + y * app.zoom);
            if transforming {
                self.transform_overlay(ui, app, &painter, canvas_rect, shown, &to_screen);
            }
            if app.doc.guides.grid.is_some() {
                for (step, alpha) in [(16.0, 30u8), (64.0, 70u8)] {
                    let s = Stroke::new(1.0, Color32::from_black_alpha(alpha));
                    let mut x = 0.0;
                    while x <= dw {
                        painter.line_segment([to_screen(x, 0.0), to_screen(x, dh)], s);
                        x += step;
                    }
                    let mut y = 0.0;
                    while y <= dh {
                        painter.line_segment([to_screen(0.0, y), to_screen(dw, y)], s);
                        y += step;
                    }
                }
            }
            // marching ants around the selection
            if let Some(m) = app.doc.selection.as_deref() {
                let t = ui.input(|i| i.time) as f32;
                for (a, b) in selection_edges(m, app.doc.w, app.doc.h) {
                    let (p, q) = (to_screen(a.0, a.1), to_screen(b.0, b.1));
                    painter.line_segment([p, q], Stroke::new(1.0, Color32::WHITE));
                    let phase = (t * 12.0 + (a.0 + a.1) * app.zoom).rem_euclid(8.0);
                    painter.extend(egui::Shape::dashed_line_with_offset(&[p, q], Stroke::new(1.0, Color32::BLACK), &[4.0], &[4.0], phase));
                }
            }
            // selection being dragged
            let dash = Stroke::new(1.5, ACCENT);
            if let Some(a) = app.select_start {
                let b = app.cursor;
                match app.mode {
                    ClickMode::SelectRect => {
                        painter.rect_stroke(Rect::from_two_pos(to_screen(a.0, a.1), to_screen(b.0, b.1)), 0.0, dash, egui::StrokeKind::Middle);
                    }
                    ClickMode::SelectEllipse => {
                        let r = Rect::from_two_pos(to_screen(a.0, a.1), to_screen(b.0, b.1));
                        let pts: Vec<Pos2> = (0..=64)
                            .map(|k| {
                                let t = k as f32 / 64.0 * std::f32::consts::TAU;
                                r.center() + Vec2::new(r.width() / 2.0 * t.cos(), r.height() / 2.0 * t.sin())
                            })
                            .collect();
                        painter.add(egui::Shape::line(pts, dash));
                    }
                    ClickMode::Lasso => {
                        painter.add(egui::Shape::line(app.lasso.iter().map(|p| to_screen(p.0, p.1)).collect(), dash));
                    }
                    _ => {}
                }
            }
            // brush outline under the pointer
            if let Some(hover) = resp.hover_pos() {
                if let Some(x) = app.xform.as_ref().filter(|_| app.mode == ClickMode::Transform && !shift) {
                    use crate::transform::Handle;
                    let h = x.dragging().unwrap_or_else(|| x.handle_at(app.cursor, 8.0 / app.zoom));
                    ui.ctx().set_cursor_icon(match h {
                        Handle::Move => egui::CursorIcon::Move,
                        Handle::Corner(0 | 2) => egui::CursorIcon::ResizeNwSe,
                        Handle::Corner(_) => egui::CursorIcon::ResizeNeSw,
                        Handle::Rotate if x.dragging().is_some() => egui::CursorIcon::Grabbing,
                        Handle::Rotate => egui::CursorIcon::Grab,
                    });
                } else if app.mode == ClickMode::Paint && !shift {
                    let r = (app.brush.params.size * 0.5 * app.zoom).max(2.0);
                    painter.circle_stroke(hover, r, Stroke::new(1.0, Color32::from_white_alpha(170)));
                    painter.circle_stroke(hover, r + 1.0, Stroke::new(1.0, Color32::from_black_alpha(120)));
                    ui.ctx().set_cursor_icon(egui::CursorIcon::None);
                } else if app.mode != ClickMode::Paint {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
                } else {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            }
            if app.layers_locked_hint() {
                painter.text(rect.left_top() + Vec2::new(10.0, 10.0), Align2::LEFT_TOP, "🔒 layer locked", egui::FontId::proportional(13.0), Color32::from_white_alpha(160));
            }
        });
    }
}

impl Ui {
    /// Re-render one layer's thumbnail every few frames, cycling through the layers, so
    /// thumbnails follow the paint without rendering every layer twice per frame.
    fn refresh_layer_thumb(&mut self, ctx: &egui::Context, app: &App) {
        let n = app.doc.layers.len();
        self.layer_thumbs.resize(n, None);
        self.frame += 1;
        let missing = self.layer_thumbs.iter().position(Option::is_none);
        let idx = match missing {
            Some(i) => i,
            None if self.frame.is_multiple_of(6) => (self.frame / 6) % n,
            None => return,
        };
        let (w, h) = (app.doc.w, app.doc.h);
        let k = (36.0 / w as f32).min(24.0 / h as f32);
        let (tw, th) = (((w as f32 * k).round() as usize).max(1), ((h as f32 * k).round() as usize).max(1));
        let px = wet_fluid::render::render_layer(&app.doc.layers[idx].wet);
        let paper = app.doc.paper.color.map(|c| c / 255.0);
        let mut rgba = Vec::with_capacity(tw * th * 4);
        for y in 0..th {
            for x in 0..tw {
                // average a few samples of the cell for a steadier small image
                let mut acc = [0.0f32; 3];
                for (sx, sy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                    let cx = (((x as f32 + sx) / k) as usize).min(w - 1);
                    let cy = (((y as f32 + sy) / k) as usize).min(h - 1);
                    let p = px[cy * w + cx];
                    for c in 0..3 {
                        acc[c] += (p[c] * p[3] + paper[c] * (1.0 - p[3])) / 4.0;
                    }
                }
                rgba.extend(acc.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
                rgba.push(255);
            }
        }
        let img = egui::ColorImage::from_rgba_unmultiplied([tw, th], &rgba);
        match &mut self.layer_thumbs[idx] {
            Some(t) => t.set(img, TextureOptions::LINEAR),
            slot => *slot = Some(ctx.load_texture(format!("layer thumb {idx}"), img, TextureOptions::LINEAR)),
        }
    }

    /// Brush Creator: a preview stroke and every brush parameter, grouped by stamp, grain
    /// and dynamics, with reset and save as a preset.
    fn brush_creator(&mut self, ctx: &egui::Context, app: &mut App) {
        // the preview is repainted when the parameters, tool or colour change
        let key = format!("{:?}{:?}{:?}{:?}", app.brush.params, app.brush.tool, app.brush.color, app.source.as_ref().map(|p| &p.name));
        if self.creator_tex.as_ref().is_none_or(|(k, _)| *k != key) {
            let (rgba, w, h) = app.stroke_preview();
            let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
            self.creator_tex = Some((key, ctx.load_texture("brush preview", img, TextureOptions::LINEAR)));
        }
        let mut open = true;
        egui::Window::new("Brush Creator").open(&mut open).default_width(380.0).show(ctx, |ui| {
            let (_, tex) = self.creator_tex.as_ref().expect("preview");
            ui.label(RichText::new(format!("{} · {}", tool_name(app.brush.tool), app.preset.as_deref().or(app.source.as_ref().map(|p| p.name.as_str())).unwrap_or("built-in round"))).strong());
            ui.image((tex.id(), Vec2::new(360.0, 110.0)));
            let p = &mut app.brush.params;
            egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                let section = |ui: &mut egui::Ui, name: &str, body: &mut dyn FnMut(&mut egui::Ui)| {
                    egui::CollapsingHeader::new(RichText::new(name).strong()).default_open(true).show(ui, |ui| {
                        egui::Grid::new(name).num_columns(2).show(ui, |ui| body(ui));
                    });
                };
                let pct = |ui: &mut egui::Ui, label: &str, v: &mut f32| {
                    ui.label(label);
                    let mut x = *v * 100.0;
                    if ui.add(egui::Slider::new(&mut x, 0.0..=100.0).suffix(" %")).changed() {
                        *v = x / 100.0;
                    }
                    ui.end_row();
                };
                section(ui, "Stroke", &mut |ui| {
                    ui.label("Spacing");
                    let mut x = p.spacing * 100.0;
                    if ui.add(egui::Slider::new(&mut x, 1.0..=200.0).logarithmic(true).suffix(" % of size")).changed() {
                        p.spacing = x / 100.0;
                    }
                    ui.end_row();
                    pct(ui, "Spacing jitter", &mut p.spacing_jitter);
                    pct(ui, "Scatter", &mut p.scatter_jitter);
                    ui.label("Smudge between dabs");
                    ui.checkbox(&mut p.splat_smudge, "");
                    ui.end_row();
                });
                section(ui, "Shape", &mut |ui| {
                    pct(ui, "Size jitter", &mut p.size_jitter);
                    ui.label("Rotation");
                    ui.add(egui::Slider::new(&mut p.rotation, 0.0..=360.0).suffix("°"));
                    ui.end_row();
                    pct(ui, "Rotation jitter", &mut p.rotation_jitter);
                    ui.label("Rotation follows");
                    egui::ComboBox::from_id_salt("rotation mode").selected_text(format!("{:?}", p.rotation_mode)).show_ui(ui, |ui| {
                        use wet_brush::RotationMode as R;
                        for m in [R::None, R::FollowTrajectory, R::PenTilt, R::PenRotation] {
                            ui.selectable_value(&mut p.rotation_mode, m, format!("{m:?}"));
                        }
                    });
                    ui.end_row();
                });
                section(ui, "Opacity", &mut |ui| {
                    pct(ui, "Dab opacity", &mut p.splat_opacity);
                    pct(ui, "Opacity jitter", &mut p.opacity_jitter);
                });
                section(ui, "Pressure", &mut |ui| {
                    pct(ui, "Size", &mut p.pressure_size_influence);
                    pct(ui, "Opacity", &mut p.pressure_opacity_influence);
                    ui.label("Multiplier");
                    ui.add(egui::Slider::new(&mut p.pressure_multiplier, 0.1..=4.0).logarithmic(true));
                    ui.end_row();
                });
                section(ui, "Paint", &mut |ui| {
                    ui.label("Mode");
                    egui::ComboBox::from_id_salt("paint mode").selected_text(format!("{:?}", p.paint_mode)).show_ui(ui, |ui| {
                        use wet_brush::PaintMode as M;
                        for m in [M::Paint, M::Mix, M::PaintBlend, M::Blend, M::Erase] {
                            ui.selectable_value(&mut p.paint_mode, m, format!("{m:?}"));
                        }
                    });
                    ui.end_row();
                    ui.label("Colour blend");
                    egui::ComboBox::from_id_salt("glaze blend").selected_text(format!("{:?}", p.glaze_blend)).show_ui(ui, |ui| {
                        for m in MixMode::ALL {
                            ui.selectable_value(&mut p.glaze_blend, m, format!("{m:?}"));
                        }
                    });
                    ui.end_row();
                    ui.label("Glazing");
                    egui::ComboBox::from_id_salt("glaze mode").selected_text(format!("{:?}", p.glaze_mode)).show_ui(ui, |ui| {
                        use wet_brush::GlazeMode as G;
                        for m in [G::Transparent, G::SemiOpaque, G::Opaque] {
                            ui.selectable_value(&mut p.glaze_mode, m, format!("{m:?}"));
                        }
                    });
                    ui.end_row();
                    ui.label("Water only");
                    ui.checkbox(&mut p.only_water, "");
                    ui.end_row();
                    ui.label("Dirty brush");
                    ui.checkbox(&mut p.dirty, "");
                    ui.end_row();
                });
                section(ui, "Grain", &mut |ui| {
                    ui.label("Scale");
                    ui.add(egui::Slider::new(&mut p.grain_scale, 0.1..=8.0).logarithmic(true));
                    ui.end_row();
                    ui.label("Scales with size");
                    ui.checkbox(&mut p.grain_follow_brush_size, "");
                    ui.end_row();
                    ui.label("Moves with brush");
                    ui.checkbox(&mut p.grain_follow_distance, "");
                    ui.end_row();
                    pct(ui, "Follow amount", &mut p.grain_follow_amount);
                    ui.label("Rotates with dab");
                    ui.checkbox(&mut p.grain_follow_shape_rotation, "");
                    ui.end_row();
                    pct(ui, "Paper structure", &mut p.paper_structure_influence);
                });
            });
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Reset").on_hover_text("Restore the preset's parameters").clicked() {
                    self.queue.push(Cmd::ResetBrush);
                }
                if ui.button("Save Preset…").on_hover_text("Write a .wetbrush preset with these parameters").clicked() {
                    self.queue.push(Cmd::SavePreset);
                }
            });
        });
        self.show_creator &= open;
    }

    /// The active layer drawn through the transform quad, its outline and corner handles,
    /// and an Apply / Cancel bar.
    fn transform_overlay(&mut self, ui: &egui::Ui, app: &App, painter: &egui::Painter, canvas_rect: Rect, shown: bool, to_screen: &dyn Fn(f32, f32) -> Pos2) {
        let Some(x) = &app.xform else { return };
        let rect = painter.clip_rect();
        let (w, h) = (app.doc.w, app.doc.h);
        // pixel centres sit half a pixel into each screen pixel square
        let scr = |p: (f32, f32)| to_screen(p.0 + 0.5, p.1 + 0.5);
        if shown && let Some(hm) = wet_doc::homography([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], x.quad) {
            let l = &app.doc.layers[app.doc.active];
            let rgba: Vec<u8> = wet_fluid::render::render_layer(&l.wet)
                .iter()
                .enumerate()
                .flat_map(|(c, p)| {
                    let m = l.mask.as_ref().map_or(1.0, |m| m[c] as f32 / 255.0);
                    let a = p[3] * l.opacity * m;
                    [p[0], p[1], p[2], a].map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
                })
                .collect();
            let image = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
            match &mut self.layer_tex {
                Some(t) => t.set(image, TextureOptions::LINEAR),
                None => self.layer_tex = Some(ui.ctx().load_texture("transform layer", image, TextureOptions::LINEAR)),
            }
            // a grid of small quads follows a perspective map closely
            const N: u32 = 16;
            let mut mesh = egui::Mesh::with_texture(self.layer_tex.as_ref().expect("layer texture").id());
            for j in 0..=N {
                for i in 0..=N {
                    let (u, v) = (i as f32 / N as f32, j as f32 / N as f32);
                    let d = hm[6] * u + hm[7] * v + 1.0;
                    let p = ((hm[0] * u + hm[1] * v + hm[2]) / d, (hm[3] * u + hm[4] * v + hm[5]) / d);
                    let uv = Pos2::new((0.5 + u * (w - 1) as f32) / w as f32, (0.5 + v * (h - 1) as f32) / h as f32);
                    mesh.vertices.push(egui::epaint::Vertex { pos: scr(p), uv, color: Color32::WHITE });
                }
            }
            for j in 0..N {
                for i in 0..N {
                    let k = j * (N + 1) + i;
                    mesh.add_triangle(k, k + 1, k + N + 2);
                    mesh.add_triangle(k, k + N + 2, k + N + 1);
                }
            }
            // the resample keeps the canvas size, so what falls outside it is cut off
            painter.with_clip_rect(canvas_rect.intersect(rect)).add(mesh);
        }
        let pts: Vec<Pos2> = x.quad.iter().map(|&p| scr(p)).collect();
        painter.add(egui::Shape::closed_line(pts.clone(), Stroke::new(1.0, Color32::from_black_alpha(160))));
        painter.add(egui::Shape::closed_line(pts.clone(), Stroke::new(1.0, ACCENT)));
        for p in &pts {
            let r = Rect::from_center_size(*p, Vec2::splat(9.0));
            painter.rect_filled(r, 1.0, Color32::WHITE);
            painter.rect_stroke(r, 1.0, Stroke::new(1.0, ACCENT), egui::StrokeKind::Middle);
        }
        let c = x.centre();
        painter.circle_stroke(scr(c), 4.0, Stroke::new(1.0, ACCENT));
        egui::Area::new(egui::Id::new("transform bar")).fixed_pos(rect.center_top() + Vec2::new(-150.0, 8.0)).show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Transform").strong());
                    ui.weak("⌘-drag a corner to distort");
                    if ui.button("Apply").on_hover_text("Enter").clicked() {
                        self.queue.push(Cmd::ApplyTransform);
                    }
                    if ui.button("Cancel").on_hover_text("Esc").clicked() {
                        self.queue.push(Cmd::CancelTransform);
                    }
                });
            });
        });
    }
}

impl App {
    fn layers_locked_hint(&self) -> bool {
        self.doc.layers[self.doc.active].locked
    }
}

/// The selection's outline as canvas-space segments along pixel edges (mask ≥ 128 is
/// selected), with runs along a row or column merged into one segment.
fn selection_edges(m: &[u8], w: usize, h: usize) -> Vec<((f32, f32), (f32, f32))> {
    let on = |x: isize, y: isize| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && m[y as usize * w + x as usize] >= 128;
    let mut out = Vec::new();
    // horizontal edges between rows y-1 and y, vertical edges between columns x-1 and x
    for y in 0..=h as isize {
        let mut start = None;
        for x in 0..=w as isize {
            let edge = x < w as isize && on(x, y - 1) != on(x, y);
            match (edge, start) {
                (true, None) => start = Some(x),
                (false, Some(s)) => {
                    out.push(((s as f32, y as f32), (x as f32, y as f32)));
                    start = None;
                }
                _ => {}
            }
        }
    }
    for x in 0..=w as isize {
        let mut start = None;
        for y in 0..=h as isize {
            let edge = y < h as isize && on(x - 1, y) != on(x, y);
            match (edge, start) {
                (true, None) => start = Some(y),
                (false, Some(s)) => {
                    out.push(((x as f32, s as f32), (x as f32, y as f32)));
                    start = None;
                }
                _ => {}
            }
        }
    }
    out
}

/// Straight RGBA8 to an egui image, converting rows on all cores.
fn color_image(w: usize, h: usize, rgba: &[u8]) -> egui::ColorImage {
    use rayon::prelude::*;
    let pixels = rgba.par_chunks_exact(4).map(|p| Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3])).collect();
    egui::ColorImage::new([w, h], pixels)
}

/// Greyscale view, stencil film and the dimmed area outside a selection.
fn decorate(app: &App, img: &mut [u8]) {
    let w = app.doc.w;
    let stencils: Vec<&wet_doc::Stencil> = app.doc.stencils.iter().filter(|s| s.visible).collect();
    let sel = app.doc.selection.as_deref();
    if !app.greyscale && stencils.is_empty() && sel.is_none() {
        return;
    }
    use rayon::prelude::*;
    img.par_chunks_exact_mut(4).enumerate().for_each(|(i, px)| {
        if app.greyscale {
            let y = (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32).round() as u8;
            px[..3].fill(y);
        }
        if !stencils.is_empty() {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let k = stencils.iter().map(|s| s.coverage(x, y)).fold(0.0f32, f32::max);
            for (v, t) in px[..3].iter_mut().zip([150.0, 90.0, 200.0]) {
                *v = (*v as f32 * (1.0 - 0.35 * k) + t * 0.35 * k) as u8;
            }
        }
        if let Some(m) = sel {
            let k = 1.0 - 0.35 * (1.0 - m[i] as f32 / 255.0);
            for v in &mut px[..3] {
                *v = (*v as f32 * k) as u8;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One frame of the whole UI with the given input events.
    fn frame(ctx: &egui::Context, ui: &mut Ui, app: &mut App, events: Vec<egui::Event>) {
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 800.0))),
            events,
            ..Default::default()
        };
        let _ = ctx.run_ui(input, |root| ui.show(root, app));
    }

    fn button(pos: Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() }
    }

    /// Pixels that differ from the blank canvas.
    fn painted(app: &App, blank: &[u8]) -> usize {
        app.doc.render(None).chunks_exact(4).zip(blank.chunks_exact(4)).filter(|(a, b)| a != b).count()
    }

    #[test]
    fn dragging_on_the_canvas_paints() {
        let ctx = egui::Context::default();
        let (mut ui, mut app) = (Ui::new(), App::new(300, 200));
        app.paused = true;
        frame(&ctx, &mut ui, &mut app, vec![]);
        frame(&ctx, &mut ui, &mut app, vec![]);
        assert!(app.zoom > 0.0, "the view is fitted");
        // the canvas centre on screen
        let (x0, y0) = (ui.last_rect.x / 2.0, ui.last_rect.y / 2.0);
        let c = Pos2::new(132.0 + x0, 30.0 + y0);
        let blank = app.doc.render(None);
        frame(&ctx, &mut ui, &mut app, vec![egui::Event::PointerMoved(c), button(c, true)]);
        for k in 1..=10 {
            let p = c + Vec2::new(6.0 * k as f32, 0.0);
            frame(&ctx, &mut ui, &mut app, vec![egui::Event::PointerMoved(p)]);
        }
        let end = c + Vec2::new(60.0, 0.0);
        frame(&ctx, &mut ui, &mut app, vec![button(end, false)]);
        assert!(!app.painting, "the stroke ended");
        assert!(painted(&app, &blank) > 100, "paint was laid");
    }

    /// Coverage-weighted centre of the active layer's paint.
    fn paint_centre(app: &App) -> (f32, f32) {
        let px = wet_fluid::render::render_layer(&app.doc.layers[app.doc.active].wet);
        let (mut sx, mut sy, mut sa) = (0.0, 0.0, 0.0);
        for (c, p) in px.iter().enumerate() {
            sx += (c % app.doc.w) as f32 * p[3];
            sy += (c / app.doc.w) as f32 * p[3];
            sa += p[3];
        }
        (sx / sa, sy / sa)
    }

    #[test]
    fn dragging_a_transform_moves_the_layer() {
        let ctx = egui::Context::default();
        let (mut ui, mut app) = (Ui::new(), App::new(300, 200));
        app.paused = true;
        frame(&ctx, &mut ui, &mut app, vec![]);
        frame(&ctx, &mut ui, &mut app, vec![]);
        let c = Pos2::new(132.0 + ui.last_rect.x / 2.0, 30.0 + ui.last_rect.y / 2.0);
        let stroke = |ui: &mut Ui, app: &mut App, from: Pos2, d: Vec2| {
            frame(&ctx, ui, app, vec![egui::Event::PointerMoved(from), button(from, true)]);
            for k in 1..=10 {
                frame(&ctx, ui, app, vec![egui::Event::PointerMoved(from + d * (k as f32 / 10.0))]);
            }
            frame(&ctx, ui, app, vec![button(from + d, false)]);
        };
        stroke(&mut ui, &mut app, c - Vec2::new(30.0, 0.0), Vec2::new(60.0, 0.0));
        let before = paint_centre(&app);

        app.run(Cmd::Mode(ClickMode::Transform));
        assert!(app.xform.is_some() && app.mode == ClickMode::Transform);
        frame(&ctx, &mut ui, &mut app, vec![]);
        stroke(&mut ui, &mut app, c, Vec2::new(0.0, 40.0));
        let quad = app.xform.as_ref().expect("still transforming").quad;
        let dy = 40.0 / app.zoom;
        assert!((quad[0].1 - dy).abs() < 0.5, "the quad followed the drag: {quad:?}, expected dy {dy}");
        // painting is untouched until the transform is applied
        assert!((paint_centre(&app).1 - before.1).abs() < 0.01);

        app.run(Cmd::ApplyTransform);
        assert!(app.xform.is_none() && app.mode == ClickMode::Paint);
        let after = paint_centre(&app);
        assert!((after.1 - before.1 - dy).abs() < 1.0, "moved by {:?}, expected dy {dy}", (after.0 - before.0, after.1 - before.1));
        assert!((after.0 - before.0).abs() < 1.0);

        // undo restores the layer; a cancelled transform leaves it alone
        app.run(Cmd::Undo);
        assert!((paint_centre(&app).1 - before.1).abs() < 0.01);
        app.run(Cmd::Mode(ClickMode::Transform));
        app.run(Cmd::Transform((25.0, 0.0), 1.0, 0.0));
        app.run(Cmd::CancelTransform);
        assert!((paint_centre(&app).0 - before.0).abs() < 0.01);
    }

    /// A horizontal stroke across the middle of the canvas, without the UI.
    fn stroke(app: &mut App) {
        let y = app.doc.h as f32 / 2.0;
        app.cursor = (20.0, y);
        app.press();
        for k in 1..=20 {
            app.cursor = (20.0 + k as f32 * (app.doc.w as f32 - 40.0) / 20.0, y);
            app.moved();
        }
        app.release();
    }

    fn mean_rgb(img: &[u8]) -> [f32; 3] {
        let n = (img.len() / 4) as f32;
        let mut m = [0.0; 3];
        for p in img.chunks_exact(4) {
            for c in 0..3 {
                m[c] += p[c] as f32 / n;
            }
        }
        m
    }

    /// The GPU path runs the same simulation (multi-step runs agree in aggregate, as in
    /// the backend's parity test) and hands layers back for undo. Skipped without a GPU.
    #[test]
    fn gpu_simulation_matches_the_cpu() {
        let (mut cpu, mut gpu) = (App::new(160, 120), App::new(160, 120));
        let blank = cpu.doc.render(None);
        gpu.run(Cmd::ToggleGpu);
        if gpu.gpu.is_none() {
            eprintln!("no GPU adapter: {}", gpu.status);
            return;
        }
        stroke(&mut cpu);
        stroke(&mut gpu);
        cpu.step_sim(40);
        gpu.step_sim(40);
        gpu.gpu.as_mut().unwrap().show_now(&mut gpu.doc);
        assert!(gpu.gpu.as_ref().unwrap().holds_any(), "the wet layer runs on the GPU");
        let (a, b) = (mean_rgb(&cpu.doc.render(None)), mean_rgb(&gpu.doc.render(None)));
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1.5), "cpu {a:?} vs gpu {b:?}");
        assert!(painted(&gpu, &blank) > 500);

        gpu.run(Cmd::Undo);
        assert!(!gpu.gpu.as_ref().unwrap().holds_any(), "commands bring layers back first");
        assert_eq!(painted(&gpu, &blank), 0, "undo removes the stroke");
        // a tilted layer with drips stays on the GPU too
        cpu.run(Cmd::Undo);
        for app in [&mut cpu, &mut gpu] {
            stroke(app);
            app.run(Cmd::ToggleTilt);
            app.step_sim(60);
        }
        gpu.gpu.as_mut().unwrap().show_now(&mut gpu.doc);
        assert!(gpu.gpu.as_ref().unwrap().holds_any(), "the tilted layer runs on the GPU");
        let (a, b) = (mean_rgb(&cpu.doc.render(None)), mean_rgb(&gpu.doc.render(None)));
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1.5), "tilted: cpu {a:?} vs gpu {b:?}");
        gpu.run(Cmd::ToggleGpu);
        assert!(gpu.gpu.is_none());
    }

    /// The GPU display compositor draws what the CPU renderer does: layers held on the GPU
    /// and on the CPU, blend modes, opacity, hidden and masking-fluid layers, paper relief,
    /// impasto and the greyscale view.
    #[test]
    fn gpu_display_matches_the_cpu_render() {
        let mut app = App::new(160, 120);
        app.run(Cmd::ToggleGpu);
        if app.gpu.is_none() {
            return;
        }
        app.doc.impasto = 2.0;
        let blank = app.doc.render(None);
        stroke(&mut app);
        let line = |app: &mut App, y0: f32, y1: f32| {
            app.cursor = (20.0, y0);
            app.press();
            for k in 1..=20 {
                let t = k as f32 / 20.0;
                app.cursor = (20.0 + 120.0 * t, y0 + (y1 - y0) * t);
                app.moved();
            }
            app.release();
        };
        for (mode, opacity, medium, color, ys) in [
            (MixMode::Multiply, 0.6, Medium::Oil, 3, (20.0, 100.0)),
            (MixMode::Hue, 1.0, Medium::Watercolor, 5, (100.0, 20.0)),
            (MixMode::Screen, 0.8, Medium::Ink, 1, (30.0, 30.0)),
            (MixMode::Normal, 1.0, Medium::Oil, 7, (90.0, 90.0)),
        ] {
            app.run(Cmd::NewLayer);
            let l = app.doc.active;
            app.doc.layers[l].mode = mode;
            app.doc.layers[l].opacity = opacity;
            app.run(Cmd::Tool(Tool::Paint(medium)));
            app.color = color;
            app.set_color(crate::app::PALETTE[color]);
            line(&mut app, ys.0, ys.1);
        }
        // a hidden layer and a masking-fluid one are left out of the colour
        app.doc.layers[3].visible = false;
        app.doc.layers[4].masking_fluid = true;
        app.step_sim(25);
        let active = app.doc.active;
        let g = app.gpu.as_mut().unwrap();
        // one wet layer back on the CPU, so both sources are composited
        g.release_layer(&mut app.doc, 1);
        g.show_now(&mut app.doc);
        assert!(g.holds_any(), "some layers run on the GPU");
        // features added one at a time, each compared with the CPU render
        type Step = fn(&mut App);
        let steps: [(&str, Step); 10] = [
            ("plain", |_| {}),
            ("greyscale view", |a| a.greyscale = true),
            ("masks", |a| {
                a.greyscale = false;
                let (w, h) = (a.doc.w, a.doc.h);
                a.doc.layers[2].mask = Some((0..w * h).map(|i| ((i % w) * 255 / w) as u8).collect());
            }),
            ("clipping", |a| a.doc.layers[3].clip = true),
            ("a group", |a| {
                a.doc.active = 2;
                a.run(Cmd::GroupWithBelow);
                a.doc.groups[0].mode = MixMode::Multiply;
                a.doc.groups[0].opacity = 0.7;
            }),
            ("pigment mixing", |a| {
                a.pigment = true;
                a.doc.layers[1].mode = MixMode::Pigment;
                a.doc.active = 0;
                a.run(Cmd::LayerPigments);
            }),
            ("gloss and soft shadows", |a| {
                a.doc.gloss = 0.8;
                a.doc.soft_shadows = 6.0;
                a.doc.shadow_altitude = 5.0;
            }),
            ("metallic", |a| {
                a.doc.active = 1;
                a.run(Cmd::ToggleMetallic);
            }),
            ("stencil and selection", |a| {
                a.run(Cmd::ToggleStencil);
                a.doc.select_shape((30.0, 20.0), (120.0, 90.0), true);
            }),
            ("deckle", |a| {
                let mut p = a.doc.paper;
                p.deckled = true;
                a.doc.set_paper(p);
            }),
        ];
        let mut before: Vec<u8> = Vec::new();
        for (name, step) in steps {
            step(&mut app);
            // commands bring layers back to the CPU: a few steps return the wet ones
            app.step_sim(2);
            app.epoch += 1;
            let view = crate::gpu::View { greyscale: app.greyscale, pigment: app.pigment, epoch: app.epoch, painting: Some(active) };
            let g = app.gpu.as_mut().unwrap();
            g.show_now(&mut app.doc);
            assert!(g.holds_any(), "{name}: some layers run on the GPU");
            g.composite(&app.doc, &view);
            let gpu = g.compositor().unwrap().read_rgba();
            // the view the window draws it through
            let _ = g.compositor().unwrap().srgb_view();
            let km = wet_pigment::KmMixer::new();
            let mut cpu = app.doc.render(app.pigment.then_some(&km as &dyn wet_compose::PigmentMix));
            decorate(&app, &mut cpu);
            // premultiplied, as egui's Color32 is
            for p in cpu.chunks_exact_mut(4) {
                let c = Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]);
                p.copy_from_slice(&c.to_array());
            }
            let diff: Vec<u8> = cpu.iter().zip(&gpu).map(|(a, b)| a.abs_diff(*b)).collect();
            let worst = *diff.iter().max().unwrap();
            let off = diff.iter().filter(|&&d| d > 1).count();
            let moved = before.iter().zip(&cpu).filter(|(a, b)| a.abs_diff(**b) > 3).count();
            eprintln!("{name}: worst {worst}, {off} channels off by more than 1 ({moved} changed by the step)");
            assert!(before.is_empty() || moved > 200, "{name}: the step shows");
            before = cpu.clone();
            assert!(worst <= 3 && off < cpu.len() / 500, "{name}: GPU display differs from the CPU render: worst {worst}, {off} channels");
        }
        app.greyscale = false;
        assert!(painted(&app, &blank) > 5000, "the comparison covers paint");
    }

    /// A stroke on a layer the GPU holds, which brings back only the cells under the brush,
    /// leaves the layer exactly as the same stroke on a full CPU copy does, for tools that
    /// pick up, settle, smudge, blend, erase, wet and blow; undo restores the layer as it was.
    #[test]
    fn gpu_strokes_match_cpu_strokes() {
        let mut app = App::new(160, 120);
        app.run(Cmd::ToggleGpu);
        if app.gpu.is_none() {
            return;
        }
        stroke(&mut app);
        app.step_sim(6);
        let i = app.doc.active;
        let state = |l: &wet_fluid::WetLayer| {
            let h = |p: &wet_fluid::grid::Plane<half::f16>| p.data.iter().map(|v| v.to_f32()).collect::<Vec<f32>>();
            let mut v = vec![h(l.water.cur()), h(l.u.cur()), h(l.v.cur())];
            for k in 0..3 {
                v.push(h(l.dens.rgb[k].cur()));
                v.push(h(l.mid.rgb[k].cur()));
            }
            for p in [l.dens.amount.cur(), l.dens.thickness.cur(), l.mid.amount.cur(), l.mid.thickness.cur()] {
                v.push(p.data.clone());
            }
            v.push(l.maps.open.data.iter().map(|&b| b as f32).collect());
            v
        };
        let tools = [
            Tool::Paint(Medium::Watercolor),
            Tool::Paint(Medium::Oil),
            Tool::Smudge,
            Tool::Blend,
            Tool::Eraser,
            Tool::Water,
            Tool::Blow,
        ];
        for tool in tools {
            app.run(Cmd::Tool(tool));
            app.brush.params.size = 24.0;
            app.step_sim(2);
            let g = app.gpu.as_mut().unwrap();
            assert!(g.holds(i), "{tool:?}: the layer is on the GPU");
            let mut reference = wet_fluid::WetLayer::new(app.doc.w, app.doc.h);
            g.download_copy(i, &mut reference);
            let before = state(&reference);
            let mut brush = app.brush.clone();
            let path: Vec<(f32, f32)> = (0..=16).map(|k| (25.0 + 7.0 * k as f32, 40.0 + 30.0 * (k as f32 * 0.5).sin())).collect();
            let at = |p: (f32, f32)| wet_brush::Sample { x: p.0, y: p.1, pressure: 1.0 };
            app.cursor = path[0];
            app.press();
            assert!(app.gpu_stroke && app.gpu.as_ref().unwrap().holds(i), "{tool:?}: the stroke runs on the GPU");
            brush.press(at(path[0]));
            for &p in &path[1..] {
                app.cursor = p;
                app.moved();
                brush.move_to(&mut reference, at(p));
            }
            app.release();
            brush.release(&mut reference, at(*path.last().unwrap()));
            let g = app.gpu.as_mut().unwrap();
            let mut painted = wet_fluid::WetLayer::new(app.doc.w, app.doc.h);
            g.download_copy(i, &mut painted);
            let (a, b) = (state(&painted), state(&reference));
            let changed = before.iter().zip(&b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count()).sum::<usize>();
            assert!(changed > 50, "{tool:?}: the stroke changed the layer ({changed} values)");
            for (k, (x, y)) in a.iter().zip(&b).enumerate() {
                let off = x.iter().zip(y).filter(|(p, q)| p != q).count();
                assert_eq!(off, 0, "{tool:?}: plane {k} differs from the CPU stroke in {off} cells");
            }
            // undo, with its snapshot copied on the GPU
            app.run(Cmd::Undo);
            let undone = state(&app.doc.layers[i].wet);
            for (k, (x, y)) in undone.iter().zip(&before).enumerate() {
                assert!(x == y, "{tool:?}: undo leaves plane {k} different from before the stroke");
            }
            assert_eq!(app.doc.history().1.first().map(|s| s.0), Some(match tool {
                Tool::Paint(m) => format!("{m:?} stroke"),
                t => format!("{t:?} stroke"),
            }.as_str()), "the undone step is the stroke");
        }
    }

    /// Frame cost of the simulation on a large wet canvas: one second of steps plus a
    /// display refresh, CPU against GPU. `cargo test -p wetpaint --release -- --ignored
    /// --nocapture sim_speed`
    #[test]
    #[ignore]
    fn sim_speed() {
        for (gpu_on, gpu_display) in [(false, false), (true, false), (true, true)] {
            let mut app = App::new(1600, 1200);
            app.brush.params.size = 300.0;
            for _ in 0..4 {
                stroke(&mut app);
                app.cursor = (800.0, 300.0);
            }
            if gpu_on {
                app.run(Cmd::ToggleGpu);
            }
            app.step_sim(5);
            app.show_gpu();
            let (mut ts, mut tw, mut tr) = (0.0, 0.0, 0.0);
            let t = std::time::Instant::now();
            for _ in 0..25 {
                let t0 = std::time::Instant::now();
                app.step_sim(1);
                let t1 = std::time::Instant::now();
                app.show_gpu();
                let t2 = std::time::Instant::now();
                if gpu_display {
                    app.epoch += 1;
                    let g = app.gpu.as_mut().unwrap();
                    g.composite(&app.doc, &crate::gpu::View { greyscale: false, pigment: false, epoch: app.epoch, painting: None });
                    g.wait();
                } else {
                    let mut img = app.doc.render(None);
                    decorate(&app, &mut img);
                    let _ = color_image(app.doc.w, app.doc.h, &img);
                }
                let t3 = std::time::Instant::now();
                ts += (t1 - t0).as_secs_f32();
                tw += (t2 - t1).as_secs_f32();
                tr += (t3 - t2).as_secs_f32();
            }
            let dt = t.elapsed().as_secs_f32();
            println!("  per frame: step {:.1} ms, readback {:.1} ms, render + image {:.1} ms", ts * 40.0, tw * 40.0, tr * 40.0);
            let name = match (gpu_on, gpu_display) {
                (false, _) => "CPU",
                (true, false) => "GPU simulation, CPU display",
                (true, true) => "GPU simulation and display",
            };
            println!("{name}: 25 frames of one step + display in {dt:.2}s ({:.1} frames/s)", 25.0 / dt);
            if gpu_on && !gpu_display {
                // a stroke starting: what the press costs before paint can be laid, with
                // three wet layers on the GPU
                for _ in 0..2 {
                    app.run(Cmd::NewLayer);
                    stroke(&mut app);
                }
                app.step_sim(2);
                app.gpu.as_mut().unwrap().show_now(&mut app.doc);
                // a stroke on a layer the GPU holds: the press, then brush steps that move
                // only the cells under the brush
                app.brush.params.size = 60.0;
                app.cursor = (100.0, 100.0);
                let t = std::time::Instant::now();
                app.press();
                println!("  stroke press, three wet layers on the GPU: {:.1} ms (on the GPU: {})", t.elapsed().as_secs_f32() * 1000.0, app.gpu_stroke);
                let t = std::time::Instant::now();
                for k in 1..=20 {
                    app.cursor = (100.0 + 10.0 * k as f32, 100.0 + 4.0 * k as f32);
                    app.moved();
                    app.flush_stroke(false);
                }
                println!("  one brush step a frame (60 px, 10 px moves): {:.2} ms each", t.elapsed().as_secs_f32() * 1000.0 / 20.0);
                let t = std::time::Instant::now();
                for f in 0..10 {
                    for k in 1..=4 {
                        app.cursor = (300.0 + 40.0 * f as f32 + 10.0 * k as f32, 180.0 + 4.0 * k as f32);
                        app.moved();
                    }
                    app.flush_stroke(false);
                }
                println!("  four brush steps a frame: {:.2} ms a frame", t.elapsed().as_secs_f32() * 1000.0 / 10.0);
                let t = std::time::Instant::now();
                app.gpu.as_mut().unwrap().flush_snapshot(&mut app.doc, true);
                println!("  undo snapshot, waited for here (in the app it converts on a worker thread): {:.1} ms", t.elapsed().as_secs_f32() * 1000.0);
                app.release();
            }
        }
    }

    #[test]
    fn history_jumps_between_labelled_steps() {
        let mut app = App::new(64, 48);
        app.run(Cmd::DryLayer);
        app.run(Cmd::WetLayer);
        app.run(Cmd::Filter(Filter::Invert));
        let labels = |app: &App| {
            let (d, u) = app.doc.history();
            (d.iter().map(|x| x.0.to_owned()).collect::<Vec<_>>(), u.iter().map(|x| x.0.to_owned()).collect::<Vec<_>>())
        };
        assert_eq!(labels(&app).0, ["Dry layer", "Wet layer", "Invert"]);
        app.run(Cmd::History(-2));
        assert_eq!(labels(&app), (vec!["Dry layer".to_owned()], vec!["Wet layer".to_owned(), "Invert".to_owned()]));
        app.run(Cmd::History(2));
        assert_eq!(labels(&app).0.len(), 3);
        app.run(Cmd::History(-10));
        assert!(labels(&app).0.is_empty(), "jumping past the start stops there");
    }

    #[test]
    fn edits_mark_the_document_unsaved_and_quit_is_requested() {
        let ctx = egui::Context::default();
        let (mut ui, mut app) = (Ui::new(), App::new(300, 200));
        app.paused = true;
        frame(&ctx, &mut ui, &mut app, vec![]);
        for cmd in [Cmd::Fit, Cmd::ToggleGreyscale, Cmd::SizeBy(1.2), Cmd::TogglePause] {
            app.run(cmd);
        }
        assert!(!app.dirty, "view and brush commands leave the document saved");
        app.run(Cmd::NewLayer);
        assert!(app.dirty, "a new layer is an unsaved change");
        app.run(Cmd::NewDocument(64, 48));
        assert!(!app.dirty, "a new document starts saved");
        // a stroke on the canvas
        frame(&ctx, &mut ui, &mut app, vec![]);
        let c = Pos2::new(132.0 + ui.last_rect.x / 2.0, 30.0 + ui.last_rect.y / 2.0);
        frame(&ctx, &mut ui, &mut app, vec![egui::Event::PointerMoved(c), button(c, true)]);
        frame(&ctx, &mut ui, &mut app, vec![button(c, false)]);
        assert!(app.dirty, "painting is an unsaved change");
        let q = egui::Event::Key { key: egui::Key::Q, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::COMMAND };
        frame(&ctx, &mut ui, &mut app, vec![q]);
        assert!(app.quit, "⌘/Ctrl+Q asks to quit");
    }

    #[test]
    fn selection_outline_follows_pixel_edges() {
        // a 3×2 block in a 6×4 mask: four merged edges of lengths 3, 3, 2, 2
        let (w, h) = (6, 4);
        let mut m = vec![0u8; w * h];
        for y in 1..3 {
            for x in 2..5 {
                m[y * w + x] = 255;
            }
        }
        let mut e = selection_edges(&m, w, h);
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(e, vec![((2.0, 1.0), (2.0, 3.0)), ((2.0, 1.0), (5.0, 1.0)), ((2.0, 3.0), (5.0, 3.0)), ((5.0, 1.0), (5.0, 3.0))]);
        assert!(selection_edges(&vec![255; w * h], w, h).len() == 4, "a full selection outlines the canvas");
    }

    #[test]
    fn brush_creator_previews_and_saves_presets() {
        let ctx = egui::Context::default();
        let (mut ui, mut app) = (Ui::new(), App::new(64, 48));
        let (rgba, w, h) = app.stroke_preview();
        assert_eq!(rgba.len(), w * h * 4);
        let marked = rgba.chunks_exact(4).filter(|p| p[0] < 230 || p[1] < 230 || p[2] < 230).count();
        assert!(marked > 500, "the preview stroke shows ({marked} px)");
        ui.show_creator = true;
        frame(&ctx, &mut ui, &mut app, vec![]);
        assert!(ui.creator_tex.is_some());

        app.brush.params.spacing = 0.33;
        app.brush.params.scatter_jitter = 0.4;
        let dir = std::env::temp_dir().join(format!("wetpaint-preset-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cat = dir.join("watercolor").join("Mine");
        std::fs::create_dir_all(&cat).unwrap();
        let path = cat.join("Edited.wetbrush");
        app.save_preset(&path).unwrap();
        app.run(Cmd::ResetBrush);
        assert!((app.brush.params.spacing - 0.33).abs() > 1e-4, "reset restored the preset's spacing");
        app.load_preset(&path);
        assert_eq!(app.preset.as_deref(), Some("Edited"), "{}", app.status);
        assert!((app.brush.params.spacing - 0.33).abs() < 1e-4);
        assert!((app.brush.params.scatter_jitter - 0.4).abs() < 1e-4);
        assert!(library::thumbnail(&path, 48).is_some(), "the thumbnail is a readable PNG");
        let cats = library::categories(&dir, "watercolor");
        assert_eq!(cats[0].presets[0].name, "Edited", "the library lists it");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn panels_and_menus_draw_with_every_tool() {
        let ctx = egui::Context::default();
        let (mut ui, mut app) = (Ui::new(), App::new(64, 48));
        for (m, _, _) in MEDIA {
            app.run(Cmd::Tool(Tool::Paint(m)));
            frame(&ctx, &mut ui, &mut app, vec![]);
        }
        for cmd in [Cmd::DryLayer, Cmd::NewLayer, Cmd::ToggleTilt, Cmd::ToggleGrid, Cmd::CycleLiquify, Cmd::Mode(ClickMode::Lasso), Cmd::SelectAll, Cmd::Mode(ClickMode::Transform)] {
            app.run(cmd);
            frame(&ctx, &mut ui, &mut app, vec![]);
        }
        assert_eq!(app.doc.layers.len(), 2);
    }
}
