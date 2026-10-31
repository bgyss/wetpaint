//! wetpaint in the browser: the app and its interface on eframe's web runner, drawn with
//! WebGPU (WebGL when the browser has no WebGPU). With WebGPU the display is composited on
//! the GPU; the simulation runs on the CPU.
//!
//! The page needs a `<canvas id="wetpaint">`; `wasm-bindgen --target web` makes the module
//! that starts it.

#![cfg(target_arch = "wasm32")]

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use wetpaint::app::{App, Cmd};
use wetpaint::ui::Ui;
use winit::keyboard::{Key, NamedKey};

/// Canvas size of a new document: large enough to paint on, small enough for the CPU
/// simulation to keep up in one browser thread.
const CANVAS: (usize, usize) = (1024, 720);

struct WebApp {
    app: App,
    ui: Ui,
}

impl WebApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Ui::style(&cc.egui_ctx);
        let mut app = App::new(CANVAS.0, CANVAS.1);
        app.render_state = cc.wgpu_render_state.clone();
        // composite on the GPU when it can run compute shaders (WebGPU)
        app.run(Cmd::ToggleGpu);
        Self { app, ui: Ui::new() }
    }

    /// Keyboard shortcuts over the canvas, as the desktop app reads them from the window.
    fn keys(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let events = ctx.input(|i| i.events.clone());
        for e in events {
            let key = match e {
                egui::Event::Text(t) if t != " " => Key::Character(t.into()),
                egui::Event::Key { key, pressed: true, modifiers, .. } if !modifiers.command => match named(key) {
                    Some(k) => Key::Named(k),
                    None => continue,
                },
                _ => continue,
            };
            if let Some(cmd) = App::key_cmd(&key) {
                self.app.run(cmd);
            }
        }
    }
}

fn named(key: egui::Key) -> Option<NamedKey> {
    use egui::Key as K;
    Some(match key {
        K::Space => NamedKey::Space,
        K::Tab => NamedKey::Tab,
        K::Home => NamedKey::Home,
        K::Enter => NamedKey::Enter,
        K::Escape => NamedKey::Escape,
        K::Insert => NamedKey::Insert,
        K::ArrowLeft => NamedKey::ArrowLeft,
        K::ArrowRight => NamedKey::ArrowRight,
        K::ArrowUp => NamedKey::ArrowUp,
        K::ArrowDown => NamedKey::ArrowDown,
        K::F1 => NamedKey::F1,
        K::F2 => NamedKey::F2,
        K::F3 => NamedKey::F3,
        K::F4 => NamedKey::F4,
        K::F6 => NamedKey::F6,
        K::F7 => NamedKey::F7,
        K::F8 => NamedKey::F8,
        K::F9 => NamedKey::F9,
        K::F10 => NamedKey::F10,
        K::F11 => NamedKey::F11,
        K::F12 => NamedKey::F12,
        _ => return None,
    })
}

impl eframe::App for WebApp {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root.ctx().clone();
        self.keys(&ctx);
        self.app.tick();
        self.ui.show(root, &mut self.app);
        // a browser tab cannot quit itself
        self.app.quit = false;
        ctx.request_repaint();
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let mut options = eframe::WebOptions::default();
    if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        setup.device_descriptor = Arc::new(wet_fluid_gpu::GpuContext::device_descriptor);
    }
    wasm_bindgen_futures::spawn_local(async move {
        let document = web_sys::window().and_then(|w| w.document()).expect("a document");
        let canvas: web_sys::HtmlCanvasElement = document
            .get_element_by_id("wetpaint")
            .expect("a <canvas id=\"wetpaint\">")
            .dyn_into()
            .expect("a canvas");
        let started = eframe::WebRunner::new().start(canvas, options, Box::new(|cc| Ok(Box::new(WebApp::new(cc))))).await;
        if let Err(e) = started {
            show_error(&document, &format!("wetpaint could not start: {e:?}"));
        }
    });
}

fn show_error(document: &web_sys::Document, msg: &str) {
    if let Some(el) = document.get_element_by_id("status-text").or_else(|| document.get_element_by_id("status")) {
        el.set_text_content(Some(msg));
    }
}
