//! `wetpaint`: an interactive wet-media painting app on the open engine.
//!
//! ```text
//! cargo run -p wetpaint --release [-- WIDTH HEIGHT] [FILE.wetdoc | IMAGE | PRESET]
//! wetpaint --listen [ADDR] …     the window, also driven by agents over local TCP
//! wetpaint --agent [W H] [FILE]  no window: JSON requests on stdin, replies on stdout
//! wetpaint --mcp [W H] [FILE]    no window: an MCP server on stdio
//! wetpaint --mcp --connect ADDR  an MCP server driving a window started with --listen
//! ```
//! A menu bar, a tool palette (left), brush / colour / canvas / layer panels (right), a
//! status bar and the canvas: paint with the mouse, pen or Force Touch trackpad (pressure),
//! scroll or pinch to zoom at the pointer, middle-drag or Shift+drag to pan. Every command
//! also has a key over the canvas; Help > Keyboard Shortcuts lists them
//! ([`wetpaint::SHORTCUTS`]).
//!
//! Each tool starts with a soft round brush; Brush Creator saves `.wetbrush` presets into the
//! brush library (see [`wetpaint::library`]).
//!
//! Agents reach every command, setting and canvas gesture through [`wetpaint::agent`]
//! (`{"op": "describe"}` lists them). `--listen` takes requests on 127.0.0.1:7878 by
//! default, one JSON object per line.

mod window;

use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use wetpaint::{agent, app::App, serve, ui};

struct Shell {
    app: App,
    ui: ui::Ui,
    win: Option<window::EguiWindow>,
    /// Agent requests from `--listen`, answered between frames.
    agent: Option<std::sync::mpsc::Receiver<serve::Request>>,
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

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        if let Some(rx) = &self.agent {
            while let Ok((line, reply)) = rx.try_recv() {
                let _ = reply.send(agent::handle_line(&mut self.app, &line));
            }
            if self.app.quit {
                self.close(el);
                return;
            }
        }
        self.app.tick();
        if let Some(w) = self.window() {
            w.request_redraw();
        }
    }
}

/// Command-line options: the flags, and the positional canvas size and file.
struct Args {
    agent: bool,
    mcp: bool,
    listen: Option<String>,
    connect: Option<String>,
    size: (usize, usize),
    file: Option<String>,
}

fn args() -> Args {
    let mut a = Args { agent: false, mcp: false, listen: None, connect: None, size: (1200, 800), file: None };
    let mut nums = Vec::new();
    let mut it = std::env::args().skip(1).peekable();
    while let Some(s) = it.next() {
        // a flag's value: `--flag=VALUE`, or the next argument unless it is a flag
        let (flag, inline) = s.split_once('=').map_or((s.as_str(), None), |(f, v)| (f, Some(v.to_owned())));
        let mut value = |default: Option<&str>| inline.clone().or_else(|| it.next_if(|n| !n.starts_with("--") && n.contains(':'))).or(default.map(str::to_owned));
        match flag {
            "--agent" => a.agent = true,
            "--mcp" => a.mcp = true,
            "--listen" => a.listen = value(Some(serve::DEFAULT_ADDR)),
            "--connect" => a.connect = value(Some(serve::DEFAULT_ADDR)),
            "--help" | "-h" => {
                eprintln!("usage: wetpaint [--listen [ADDR] | --agent | --mcp [--connect ADDR]] [WIDTH HEIGHT] [FILE]");
                std::process::exit(0);
            }
            f if f.starts_with("--") => {
                eprintln!("unknown option {f} (try --help)");
                std::process::exit(2);
            }
            _ => match s.parse::<usize>() {
                Ok(n) => nums.push(n),
                Err(_) => a.file = Some(s),
            },
        }
    }
    if let [w, h, ..] = nums[..] {
        a.size = (w, h);
    }
    a
}

/// `--agent` and `--mcp`: no window; requests on stdin, replies on stdout.
fn headless(a: &Args) -> std::io::Result<()> {
    if let Some(addr) = &a.connect {
        let mut call = serve::connect(addr).map_err(|e| std::io::Error::other(format!("cannot reach wetpaint at {addr}: {e}")))?;
        return wetpaint::mcp::serve(&mut call, std::io::stdin().lock(), std::io::stdout().lock());
    }
    let mut app = agent::headless(a.size.0, a.size.1);
    if let Some(file) = &a.file {
        app.open_path(std::path::Path::new(file));
    }
    if a.mcp {
        let mut call = |r: &serde_json::Value| agent::handle(&mut app, r);
        wetpaint::mcp::serve(&mut call, std::io::stdin().lock(), std::io::stdout().lock())
    } else {
        serve::stdio(&mut app)
    }
}

fn main() {
    // `wetpaint [WIDTH HEIGHT] [FILE]`: a canvas size and/or a document, image or preset
    let args = args();
    if args.agent || args.mcp {
        if let Err(e) = headless(&args) {
            eprintln!("wetpaint: {e}");
            std::process::exit(1);
        }
        return;
    }
    let (w, h) = args.size;
    let agent = args.listen.as_deref().map(|addr| match serve::listen(addr) {
        Ok((at, rx)) => {
            if !at.ip().is_loopback() {
                eprintln!("wetpaint: warning: {at} accepts agent requests from other machines");
            }
            eprintln!("wetpaint: agents can connect to {at}");
            rx
        }
        Err(e) => {
            eprintln!("wetpaint: cannot listen on {addr}: {e}");
            std::process::exit(1);
        }
    });
    let mut builder = EventLoop::builder();
    // the default macOS menu's Quit ends the process without asking about unsaved changes;
    // ⌘Q is handled as File > Quit instead
    #[cfg(target_os = "macos")]
    winit::platform::macos::EventLoopBuilderExtMacOS::with_default_menu(&mut builder, false);
    let el = builder.build().expect("event loop");
    let mut shell = Shell { app: App::new(w, h), ui: ui::Ui::new(), win: None, agent };
    if let Some(file) = &args.file {
        shell.app.open_path(std::path::Path::new(file));
    }
    el.run_app(&mut shell).expect("run");
}
