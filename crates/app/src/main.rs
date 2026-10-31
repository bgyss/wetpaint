//! `wetpaint`: an interactive wet-media painting app on the open engine.
//!
//! ```text
//! cargo run -p wetpaint --release [-- WIDTH HEIGHT] [FILE.wetdoc | IMAGE | PRESET]
//! ```
//! A menu bar, a tool palette (left), brush / colour / canvas / layer panels (right), a
//! status bar and the canvas: paint with the mouse, pen or Force Touch trackpad (pressure),
//! scroll or pinch to zoom at the pointer, middle-drag or Shift+drag to pan. Every command
//! also has a key over the canvas; Help > Keyboard Shortcuts lists them ([`SHORTCUTS`]).
//!
//! Each tool starts with a soft round brush; Brush Creator saves `.wetbrush` presets into the
//! brush library (see [`library`]).

mod app;
mod gpu;
mod library;
mod transform;
mod ui;
mod window;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use app::App;

pub const SHORTCUTS: &str = "\
Tools: 1 watercolour, 2 ink, 3 oil, 4 pastel, 5 pencil, 6 marker, 7 airbrush, 8 water,
  9 dry, 0 eraser, B blend, M smudge, O blow, H liquify (again: push → twirl → pucker →
  bloat → reconstruct).
Brush: C next colour, [ / ] size, , / . water, - / = opacity, F3 multi-coloured brush,
  F10 / F11 / F12 size / opacity / water presets.
Canvas: Space pause/resume, T tilt, D dry, W wet, X clear layer, P pigment mixing,
  G granulation.
Layers: N new, Tab next, K masking fluid, J clip to layer below, A lock, F4 pigment mixing
  on the layer, F6 group with the layer below, Shift+N duplicate, Shift+M merge down,
  Shift+T metallic.
File: Z undo, Y redo, S save, L open, V export PSD, F5 export PNG, Shift+D save iteration,
  Shift+E export heightmap, R time-lapse recording, ⌘/Ctrl+Q quit (asks to save changes).
Rulers: F1 cycles horizontal → vertical → ellipses → 1-point perspective → off.
Tracing: F2 imports an image as a locked tracing layer, or toggles tracing.
Reference: Insert opens a reference image panel, or shows/hides it.
Grid: F9 (64 px, 4 subdivisions) with snapping of rectangle/ellipse selections.
Selection: U rectangle, E ellipse, ' lasso, ; magic wand, / invert, Q deselect,
  Shift+Q reselect, F7 stencil from selection, F8 cut to a new layer.
Stencil: \\ places stencil.png (or a disc) or shows/hides it, | inverts it.
Click tools: I pick colour, F fill.
Transform the active layer: arrows move 10 px, < / > rotate 5°, { / } scale. Shift+R
  transforms with handles: drag inside to move, a corner to scale (⌘/Ctrl: distort that
  corner), outside to rotate; the keys above nudge it, Enter applies, Esc cancels.
Filters: Shift+I invert, Shift+U desaturate, Shift+H hue +30°, Shift+B blur, Shift+S sharpen.
View: Home fit, Shift+C GPU simulation, Shift+G greyscale, Shift+O soft shadows, Shift+P natural (RYB) colours,
  Shift+V next watercolour behaviour preset.
Navigation: scroll or pinch to zoom at the pointer, middle-drag or Shift+drag to pan.";

struct Shell {
    app: App,
    ui: ui::Ui,
    win: Option<window::EguiWindow>,
}

impl Shell {
    fn window(&self) -> Option<&Window> {
        self.win.as_ref().map(|w| &*w.window)
    }

    /// Close the window and exit, unless unsaved changes are kept by Cancel.
    fn close(&mut self, el: &ActiveEventLoop) {
        self.app.quit = false;
        if !self.app.confirm_discard() {
            return;
        }
        // taken out so a redraw still queued after exit finds nothing to paint with
        if let Some(mut win) = self.win.take() {
            win.destroy();
        }
        el.exit();
    }

    fn redraw(&mut self) {
        let Some(win) = &mut self.win else { return };
        let (app, ui) = (&mut self.app, &mut self.ui);
        win.frame(|root| ui.show(root, app));
    }
}

impl ApplicationHandler for Shell {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.win.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("wetpaint")
            .with_inner_size(winit::dpi::LogicalSize::new(1400.0, 900.0))
            .with_min_inner_size(winit::dpi::LogicalSize::new(800.0, 500.0));
        let win = window::EguiWindow::new(el, attrs);
        ui::Ui::style(&win.ctx);
        self.app.render_state = win.render_state();
        self.win = Some(win);
        el.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(win) = &mut self.win else { return };
        let response = win.on_window_event(&event);
        let wants_keys = win.ctx.egui_wants_keyboard_input();
        match event {
            WindowEvent::CloseRequested => {
                self.close(el);
                return;
            }
            WindowEvent::RedrawRequested => {
                self.redraw();
                if self.app.quit {
                    self.close(el);
                    return;
                }
            }
            // trackpad force (macOS Force Touch: stage 1 = click, pressure 0–1 within it);
            // egui does not forward it, so it is read here
            WindowEvent::TouchpadPressure { pressure, stage, .. } => {
                self.app.pressure = (stage > 0).then_some(pressure.clamp(0.05, 1.0));
                if self.app.painting {
                    self.app.moved();
                }
            }
            WindowEvent::DroppedFile(path) => self.app.open_path(&path),
            // touch screens and pens reported as touches carry a normalised force
            WindowEvent::Touch(t) => {
                if let Some(f) = t.force {
                    self.app.pressure = Some(f.normalized().clamp(0.05, 1.0) as f32);
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed && !response.consumed && !wants_keys => {
                if let Some(cmd) = App::key_cmd(&event.logical_key) {
                    self.app.run(cmd);
                }
            }
            _ => {}
        }
        if response.repaint
            && let Some(w) = self.window()
        {
            w.request_redraw();
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        self.app.tick();
        if let Some(w) = self.window() {
            w.request_redraw();
        }
    }
}

fn main() {
    // `wetpaint [WIDTH HEIGHT] [FILE]`: a canvas size and/or a document, image or preset
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a: Vec<usize> = args.iter().filter_map(|s| s.parse().ok()).collect();
    let (w, h) = (a.first().copied().unwrap_or(1200), a.get(1).copied().unwrap_or(800));
    let mut builder = EventLoop::builder();
    // the default macOS menu's Quit ends the process without asking about unsaved changes;
    // ⌘Q is handled as File > Quit instead
    #[cfg(target_os = "macos")]
    winit::platform::macos::EventLoopBuilderExtMacOS::with_default_menu(&mut builder, false);
    let el = builder.build().expect("event loop");
    let mut shell = Shell { app: App::new(w, h), ui: ui::Ui::new(), win: None };
    if let Some(file) = args.iter().find(|s| s.parse::<usize>().is_err()) {
        shell.app.open_path(std::path::Path::new(file));
    }
    el.run_app(&mut shell).expect("run");
}
