#![cfg_attr(target_family = "wasm", no_main)]
//! Spike: the wetpaint app with a gpui interface. It paints with the mouse or a
//! force-sensitive trackpad, runs the simulation by wall-clock time, and shows the
//! canvas as an image gpui uploads each frame. The panel changes settings through the
//! agent API, so the interface holds no painting state of its own.
//!
//! The side panel reports what each frame costs: simulation, compositing on the CPU,
//! conversion to the BGRA image gpui takes, and the frame interval.
//!
//! `wetpaint-gpui [W H] [--demo]`: `--demo` paints scripted strokes for a while through
//! the same press / move / release calls the mouse uses, prints the frame timings, and quits.
//! `--bench` does the demo's per-frame work without a window. The same code builds for
//! the browser (`web/build.sh`), on gpui's web platform.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use web_time::Instant;

use gpui::{
    App, Bounds, Context, Corners, MouseButton, MouseDownEvent, MouseMoveEvent, MousePressureEvent, MouseUpEvent,
    Pixels, RenderImage, SharedString, Window, WindowBounds, WindowOptions, canvas, div, point, prelude::*, px, rgb,
    size,
};
use gpui_platform::application;
use serde_json::{Value, json};
use wetpaint::{agent, app::App as Paint};

thread_local! {
    static DOC_SIZE: Cell<(usize, usize)> = const { Cell::new((1024, 768)) };
}

fn doc() -> (usize, usize) {
    DOC_SIZE.with(Cell::get)
}

/// Seconds the demo paints for before it reports and quits.
const DEMO_SECONDS: f32 = 12.0;
const PANEL: f32 = 232.0;
const TOOLS: [&str; 8] = ["watercolor", "ink", "oil", "pastel", "pencil", "marker", "water", "eraser"];
const PALETTE: [u32; 12] =
    [0xc82828, 0xe6781e, 0xebc828, 0x3c963c, 0x1e6e78, 0x2846be, 0x6e3296, 0xbe468c, 0x785032, 0x3c3c3c, 0x0f0f14, 0xfafaf5];

/// Running averages of what a frame costs, in milliseconds.
#[derive(Default)]
struct Stats {
    tick: f32,
    composite: f32,
    convert: f32,
    interval: f32,
    last_frame: Option<Instant>,
    intervals: Vec<f32>,
}

impl Stats {
    fn add(v: &mut f32, ms: f32) {
        *v = if *v == 0.0 { ms } else { *v * 0.9 + ms * 0.1 };
    }
}

struct Wetpaint {
    paint: Paint,
    /// The canvas element's bounds and the image's place in them, from the last paint.
    canvas: Rc<Cell<Option<Bounds<Pixels>>>>,
    image: Option<Arc<RenderImage>>,
    /// Images painted in earlier frames, to take out of gpui's atlas.
    stale: Rc<RefCell<Vec<Arc<RenderImage>>>>,
    /// Force-touch pressure, when the trackpad reports it.
    pressure: Option<f32>,
    stats: Stats,
    status: SharedString,
    /// The scripted demo's start, while it runs.
    demo: Option<Instant>,
}

impl Wetpaint {
    fn new() -> Self {
        let (w, h) = doc();
        let paint = agent::headless(w, h);
        Self {
            paint,
            canvas: Rc::new(Cell::new(None)),
            image: None,
            stale: Rc::new(RefCell::new(Vec::new())),
            pressure: None,
            stats: Stats::default(),
            status: "".into(),
            demo: None,
        }
    }

    /// One agent request; the reply's status or error goes to the status line.
    fn call(&mut self, req: Value) -> Value {
        let r = agent::handle(&mut self.paint, &req);
        if let Some(e) = r.get("error").and_then(Value::as_str) {
            self.status = e.to_owned().into();
        } else if let Some(s) = r.get("status").and_then(Value::as_str) {
            self.status = s.to_owned().into();
        }
        r
    }

    fn get(&mut self, key: &str) -> Value {
        self.call(json!({ "op": "get", "key": key }))["value"].clone()
    }

    /// Window position → canvas pixel, through where the image was last drawn.
    fn to_canvas(&self, p: gpui::Point<Pixels>) -> Option<(f32, f32)> {
        let b = fit(self.canvas.get()?);
        let k = doc().0 as f32 / f32::from(b.size.width);
        Some((f32::from(p.x - b.origin.x) * k, f32::from(p.y - b.origin.y) * k))
    }

    fn pointer(&mut self, p: gpui::Point<Pixels>) -> bool {
        let Some(c) = self.to_canvas(p) else { return false };
        self.paint.cursor = c;
        self.paint.pressure = self.pressure;
        true
    }

    /// Advance the simulation and turn the document into a new image for gpui.
    fn frame(&mut self) {
        let now = Instant::now();
        if let Some(t) = self.stats.last_frame.replace(now) {
            let ms = (now - t).as_secs_f32() * 1e3;
            Stats::add(&mut self.stats.interval, ms);
            self.stats.intervals.push(ms);
        }
        if self.demo.is_some() && self.stats.intervals.len() % 30 == 0 {
            eprintln!("frame {} at {:.1} s", self.stats.intervals.len(), self.demo.map_or(0.0, |s| s.elapsed().as_secs_f32()));
        }
        if let Some(start) = self.demo {
            self.demo_step(start.elapsed().as_secs_f32());
        }
        self.paint.tick();
        let t1 = Instant::now();
        let km = self.paint.pigment.then(wet_pigment::KmMixer::new);
        let mut rgba = self.paint.doc.render(km.as_ref().map(|p| p as &dyn wet_compose::PigmentMix));
        let t2 = Instant::now();
        // gpui images are BGRA
        for px in rgba.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        let (w, h) = doc();
        let buf = image::RgbaImage::from_raw(w as u32, h as u32, rgba).expect("canvas size");
        if let Some(old) = self.image.replace(Arc::new(RenderImage::new([image::Frame::new(buf)]))) {
            self.stale.borrow_mut().push(old);
        }
        let t3 = Instant::now();
        Stats::add(&mut self.stats.tick, (t1 - now).as_secs_f32() * 1e3);
        Stats::add(&mut self.stats.composite, (t2 - t1).as_secs_f32() * 1e3);
        Stats::add(&mut self.stats.convert, (t3 - t2).as_secs_f32() * 1e3);
    }

    /// The demo at `t` seconds: a new wavy stroke each 1.5 s, with the next tool and colour.
    fn demo_step(&mut self, t: f32) {
        const STROKE: f32 = 1.5;
        let (w, h) = doc();
        let (w, h) = (w as f32, h as f32);
        let k = (t / STROKE) as usize;
        let u = (t % STROKE) / STROKE;
        if u < 0.85 {
            let y = h * (0.15 + 0.7 * ((k * 3 % 7) as f32 / 6.0));
            self.paint.cursor = (w * (0.08 + 0.84 * u / 0.85), y + h * 0.06 * (u * 12.0).sin());
            self.paint.pressure = Some(0.5 + 0.5 * (u * 3.14).sin());
            if self.paint.painting {
                self.paint.moved();
            } else {
                let tool = ["watercolor", "oil", "watercolor", "pastel", "ink", "watercolor"][k % 6];
                let colour = format!("#{:06x}", PALETTE[k * 5 % 9]);
                self.call(json!({ "op": "set", "values": { "brush.tool": tool, "brush.color": colour, "brush.size": 48 } }));
                self.paint.press();
            }
        } else if self.paint.painting {
            self.paint.release();
        }
    }

    /// The demo's frame timings, on stdout.
    fn report(&self) {
        let mut v = self.stats.intervals.clone();
        v.sort_by(f32::total_cmp);
        let q = |p: f32| v.get(((v.len() as f32 - 1.0) * p) as usize).copied().unwrap_or(0.0);
        let (w, h) = doc();
        println!(
            "canvas {w}x{h}: {} frames; interval p50 {:.1} ms, p95 {:.1} ms, max {:.1} ms; sim {:.1} ms, composite {:.1} ms, to BGRA {:.1} ms (averages)",
            v.len(),
            q(0.5),
            q(0.95),
            q(1.0),
            self.stats.tick,
            self.stats.composite,
            self.stats.convert,
        );
    }
}

/// The largest rectangle with the canvas's aspect ratio, centred in `b`.
fn fit(b: Bounds<Pixels>) -> Bounds<Pixels> {
    let (bw, bh) = (f32::from(b.size.width), f32::from(b.size.height));
    let (dw, dh) = doc();
    let k = (bw / dw as f32).min(bh / dh as f32);
    let (w, h) = (dw as f32 * k, dh as f32 * k);
    Bounds::new(point(b.origin.x + px((bw - w) / 2.0), b.origin.y + px((bh - h) / 2.0)), size(px(w), px(h)))
}

fn label(text: impl Into<SharedString>) -> impl IntoElement {
    div().text_xs().text_color(rgb(0x8a8f9c)).mt_3().child(text.into())
}

fn button(id: impl Into<SharedString>, text: impl Into<SharedString>, on: bool) -> gpui::Stateful<gpui::Div> {
    div()
        .id(gpui::ElementId::Name(id.into()))
        .px_2()
        .py_1()
        .rounded_md()
        .text_sm()
        .cursor_pointer()
        .bg(if on { rgb(0x2c5aa8) } else { rgb(0x2a2d36) })
        .text_color(rgb(0xe8eaf0))
        .hover(|s| s.bg(rgb(0x3a3f4c)))
        .child(text.into())
}

impl Render for Wetpaint {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.frame();
        window.request_animation_frame();
        if self.demo.is_some_and(|s| s.elapsed().as_secs_f32() > DEMO_SECONDS) {
            self.report();
            cx.quit();
        }

        let tool = self.get("brush.tool").as_str().unwrap_or("").to_owned();
        let colour = self.get("brush.color").as_str().unwrap_or("").to_owned();
        let size_v = self.get("brush.size").as_f64().unwrap_or(0.0);
        let tilt = self.get("layer.sim.tilt_active") == true;
        let pigment = self.paint.pigment;
        let paused = self.paint.paused;
        let s = &self.stats;
        let stats = format!(
            "sim {:.1} ms · {:.0} steps/s\ncomposite {:.1} ms\nto BGRA {:.1} ms\nframe {:.1} ms ({:.0} fps)\npressure {}",
            s.tick,
            self.paint.sim_rate,
            s.composite,
            s.convert,
            s.interval,
            if s.interval > 0.0 { 1e3 / s.interval } else { 0.0 },
            self.pressure.map_or("— (mouse)".to_owned(), |p| format!("{p:.2} (force touch)")),
        );

        let tools = div().flex().flex_wrap().gap_1().children(TOOLS.iter().map(|&t| {
            button(format!("tool-{t}"), t, t == tool).on_click(cx.listener(move |this, _, _, _| {
                this.call(json!({ "op": "set", "key": "brush.tool", "value": t }));
            }))
        }));
        let palette = div().flex().flex_wrap().gap_1().children(PALETTE.iter().map(|&c| {
            let hex = format!("#{c:06x}");
            let on = hex == colour;
            div()
                .id(gpui::ElementId::Name(format!("colour-{c:06x}").into()))
                .size(px(26.0))
                .rounded_md()
                .bg(rgb(c))
                .border_2()
                .border_color(if on { rgb(0xffffff) } else { rgb(0x1b1d24) })
                .cursor_pointer()
                .on_click(cx.listener(move |this, _, _, _| {
                    this.call(json!({ "op": "set", "key": "brush.color", "value": hex }));
                }))
        }));
        let cmd = |id: &str, text: &str, on: bool, name: &'static str| {
            button(id.to_owned(), text.to_owned(), on).on_click(cx.listener(move |this, _, _, _| {
                this.call(json!({ "op": "command", "name": name }));
            }))
        };
        let sizes = div()
            .flex()
            .items_center()
            .gap_2()
            .child(button("size-down", "−", false).on_click(cx.listener(|this, _, _, _| {
                this.call(json!({ "op": "command", "name": "size_by", "args": { "factor": 0.8 } }));
            })))
            .child(div().text_sm().text_color(rgb(0xe8eaf0)).child(format!("{size_v:.0} px")))
            .child(button("size-up", "+", false).on_click(cx.listener(|this, _, _, _| {
                this.call(json!({ "op": "command", "name": "size_by", "args": { "factor": 1.25 } }));
            })));

        let panel = div()
            .flex()
            .flex_col()
            .w(px(PANEL))
            .h_full()
            .p_3()
            .gap_1()
            .bg(rgb(0x1b1d24))
            .child(div().text_lg().text_color(rgb(0xffffff)).child("wetpaint · gpui"))
            .child(label("Tool"))
            .child(tools)
            .child(label("Colour"))
            .child(palette)
            .child(label("Size"))
            .child(sizes)
            .child(label("Canvas"))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_1()
                    .child(cmd("tilt", "tilt", tilt, "toggle_tilt"))
                    .child(cmd("pigment", "pigment", pigment, "toggle_pigment"))
                    .child(cmd("pause", "pause", paused, "toggle_pause"))
                    .child(cmd("dry", "dry", false, "dry_layer"))
                    .child(cmd("clear", "clear", false, "clear_layer"))
                    .child(cmd("undo", "undo", false, "undo")),
            )
            .child(label("Frame"))
            .child(div().text_xs().text_color(rgb(0xc8ccd6)).child(stats))
            .child(div().flex_1())
            .child(div().text_xs().text_color(rgb(0x8a8f9c)).child(self.status.clone()));

        let image = self.image.clone();
        let stale = self.stale.clone();
        let canvas_bounds = self.canvas.clone();
        let surface = canvas(
            move |bounds, _, _| canvas_bounds.set(Some(bounds)),
            move |bounds, (), window, _| {
                for old in stale.borrow_mut().drain(..) {
                    let _ = window.drop_image(old);
                }
                if let Some(image) = image {
                    let b = fit(bounds);
                    let _ = window.paint_image(b, b, Corners::default(), image, 0, false);
                }
            },
        )
        .size_full();

        let area = div()
            .flex_1()
            .h_full()
            .p_4()
            .bg(rgb(0x2a2d36))
            .child(surface)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, ev: &MouseDownEvent, _, _| {
                    if this.pointer(ev.position) {
                        this.paint.press();
                    }
                }),
            )
            .on_mouse_move(cx.listener(|this, ev: &MouseMoveEvent, _, _| {
                if this.pointer(ev.position) {
                    this.paint.moved();
                }
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, ev: &MouseUpEvent, _, _| {
                    this.pointer(ev.position);
                    this.paint.release();
                    this.pressure = None;
                }),
            )
            .on_mouse_pressure(cx.listener(|this, ev: &MousePressureEvent, _, _| {
                this.pressure = Some(ev.pressure.clamp(0.0, 1.0));
            }));

        let root = div().flex().size_full().bg(rgb(0x1b1d24));
        // the browser has only the fonts the page loads
        #[cfg(target_family = "wasm")]
        let root = root.font_family("IBM Plex Sans");
        root.child(panel).child(area)
    }
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let demo = args.iter().any(|a| a == "--demo");
    let nums: Vec<usize> = args.iter().filter_map(|a| a.parse().ok()).collect();
    if let [w, h, ..] = nums[..] {
        DOC_SIZE.with(|d| d.set((w, h)));
    }
    if args.iter().any(|a| a == "--bench") {
        // the per-frame work without a window: the demo's strokes at 60 frames a second
        let mut app = Wetpaint::new();
        app.demo = Some(Instant::now());
        for k in 0..(DEMO_SECONDS * 60.0) as usize {
            let t = Instant::now();
            app.demo_step(k as f32 / 60.0);
            app.demo = None;
            app.frame();
            app.demo = Some(t);
        }
        app.report();
        return;
    }
    run(demo);
}

/// The browser has no system fonts gpui can lay out text with.
#[cfg(target_family = "wasm")]
fn load_fonts(cx: &App) {
    let fonts = [
        include_bytes!("../assets/fonts/IBMPlexSans-Regular.ttf").as_slice(),
        include_bytes!("../assets/fonts/IBMPlexSans-SemiBold.ttf").as_slice(),
    ];
    if let Err(e) = cx.text_system().add_fonts(fonts.map(std::borrow::Cow::Borrowed).into()) {
        web_sys::console::error_1(&format!("fonts: {e:#}").into());
    }
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    gpui_platform::web_init();
    run(false);
}

fn run(demo: bool) {
    application().run(move |cx: &mut App| {
        #[cfg(target_family = "wasm")]
        load_fonts(cx);
        let bounds = Bounds::centered(None, size(px(PANEL + 1024.0 + 32.0), px(800.0)), cx);
        cx.open_window(
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), focus: true, ..Default::default() },
            |_, cx| {
                cx.new(|cx| {
                    // redraw continuously: the paint keeps moving after the pointer stops
                    cx.spawn(async move |this, cx| {
                        while this.update(cx, |_, cx| cx.notify()).is_ok() {
                            cx.background_executor().timer(Duration::from_millis(8)).await;
                        }
                    })
                    .detach();
                    let mut app = Wetpaint::new();
                    app.demo = demo.then(Instant::now);
                    app
                })
            },
        )
        .expect("open a window");
        cx.on_window_closed(|cx, _| cx.quit()).detach();
        cx.activate(true);
    });
}
