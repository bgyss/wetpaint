# Spike: wetpaint on gpui

The wetpaint app with a [gpui](https://gpui.rs) interface instead of egui: a tool and
colour panel, the canvas, and per-frame timings. Painting goes through the same
`App::press` / `moved` / `release` calls as the egui app, and the panel changes settings
through the agent API (`wetpaint::agent::handle`), so the interface holds no painting
state of its own. It builds for macOS and for the browser on gpui's web platform.

gpui comes from Zed's repository at a pinned revision (the crates.io release is older and
has no web platform), with Rust 1.98.1 (`rust-toolchain.toml`) and one patch,
`patches/external-texture.patch` (below). `./vendor.sh` fetches and patches it into
`../gpui-vendor/zed`. This directory is its own Cargo workspace, so gpui's dependency
tree stays out of the main build.

```
./vendor.sh                             # once: gpui at the pinned revision, patched
cargo run --release                     # paint with the mouse (force touch: pressure)
cargo run --release -- 1600 1200        # canvas size
cargo run --release -- --demo           # scripted strokes for 12 s, then frame timings on stdout
cargo run --release -- 2048 1536 --bench   # the per-frame work without a window
WASM_BINDGEN=/path/to/wasm-bindgen-0.2.130 web/build.sh && python3 -m http.server -d web/dist 8766
```

`runtime_shaders` compiles gpui's Metal shaders at startup, because building them ahead
of time needs Xcode's Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`).

## What it needed from the engine

- One wgpu. gpui pins a git revision of wgpu 30; the engine used crates.io 30.0.1. Two
  copies don't link for the browser (both export the WebGPU bindings), so this workspace
  patches crates.io `wgpu` to gpui's revision, and `wet-fluid-gpu` now asks for `"30"`.
- No egui-wgpu, which does not build against that revision. The app crate's new `window`
  feature (on by default) holds the egui interface and the window's GPU device; without
  it `wetpaint` is the document, commands and agent ops. `decorate` (the view overlays)
  moved from `ui.rs` to `app.rs`, and `App::window_gpu` replaced direct `render_state`
  reads.

## The external-texture patch

`patches/external-texture.patch` (170 lines against Zed `c7bea34`) lets an application
draw a texture it renders itself, on gpui's wgpu renderer:

- `Window::external_gpu()` hands out the renderer's device: for wgpu,
  `(wgpu::Device, wgpu::Queue, wgpu::AdapterInfo)` behind `Box<dyn Any>`, so gpui's core
  stays free of wgpu. Implemented for the web platform; other renderers return `None`.
- `Window::paint_external_texture(bounds, ExternalTexture)` adds the texture to the scene
  as a surface (the primitive macOS uses for video frames). `ExternalTexture` wraps an
  `Rc<dyn Any>`: a `wgpu::TextureView`, RGBA with straight alpha, sampled as stored.
- The wgpu renderer draws surfaces with a new `fs_surface_rgba` fragment shader on the
  video surfaces' vertex shader and bind group layout, which it already had and never used.

The spike asks for the device on its first frame, gives it to the engine
(`App::external_gpu`), and turns the GPU display on. Each frame the engine composites
the document into its display texture on that device (`App::gpu_texture`) and gpui draws
it in the same frame. The **gpu** button in the panel switches between the two paths.

Verified in Chrome on WebGPU (M1 Max, 1024 × 768). With the paint dry:

| display | composite | frame |
|---|---|---|
| GPU texture (patched) | 14.5 ms | 25.0 ms (40 fps) |
| CPU image | 27.5 ms, then gpui's upload | 32.5 ms (31 fps) |

Those figures are with the simulation on the CPU, as it was in the browser then: each
layer was rendered there and uploaded before the GPU composited, and while three wet
strokes spread the single-threaded simulation (about 160 ms a step) held either path to
6 fps.

The engine now simulates on the GPU in the browser too (its readbacks no longer block:
see `crates/app/src/gpu.rs`), and with the texture drawn in place nothing of a wet
layer crosses to the CPU each frame. Same page, same three wet strokes:

| | simulation | composite | frame |
|---|---|---|---|
| GPU simulation, GPU texture | 0.8 ms | 1.3 ms | 8.9 ms (113 fps) |
| while painting over wet paint | 7.3 ms | 1.3 ms | 17.1 ms (59 fps) |
| CPU simulation, GPU texture (before) | 161 ms | 11 ms | 173 ms (6 fps) |

gpui's device was requested with wgpu's downlevel limits, and they were enough for the
compositor at this size. Larger canvases need larger storage buffers, so an application
should be able to raise the limits when gpui creates the device.

Not done: the Linux platform (the same renderer; `external_gpu` needs wiring into
`gpui_linux`'s windows, untested here), and macOS, where gpui draws with Metal.

## Findings (Zed `c7bea34`, November 2026)

**Without the patch, the canvas goes through the CPU.** gpui draws images from its texture atlas
(`Window::paint_image` with a BGRA `RenderImage`), so each frame the document is
composited on the CPU, converted to BGRA and uploaded. There is no way to draw a wgpu
texture the app owns:

- macOS: gpui renders with Metal, not wgpu. Its `surface()` element takes a CoreVideo
  buffer, but only 4:2:0 YCbCr video frames, which halve the colour resolution.
- Linux and the browser: gpui renders with wgpu, the same crate as the engine now, but its
  device is private to the renderer and the scene has no primitive for an external texture.

So the GPU simulation and the GPU display compositor (`wet-fluid-gpu`) can't feed the
gpui canvas without a copy back to the CPU each frame. The egui app avoids that: the
display is composited on the window's own device and drawn from there.

Per-frame cost on the CPU path (M1 Max, release; CPU simulation, the `--bench` strokes):

| canvas | simulation | composite | to BGRA |
|---|---|---|---|
| 1024 × 768 | 2.4–2.9 ms | 5.8–6.0 ms | 0.3 ms |
| 1600 × 1200 | 17.2 ms | 11.3 ms | 0.7 ms |
| 2048 × 1536 | 115.6 ms | 18.5 ms | 1.1 ms |

In Chrome (WebGPU, single-threaded wasm, 1024 × 768): about 20 fps once the paint
settles (simulation 19 ms, composite 26 ms); about 5 fps while wet watercolour spreads.
The browser build has no threads unless gpui's `multithreaded` feature is on (nightly
Rust and a cross-origin-isolated page), so rayon runs on one core there.

Not yet measured: the upload and frame interval of the windowed macOS app. gpui does not
draw windows macOS reports as hidden, and the run here had no visible window; run
`--demo` with the window in front.

**Pen input.** gpui reports force-touch trackpad pressure (`MousePressureEvent`), not
tablet pressure, tilt or rotation, on any platform. In the browser it reads pointer
events but drops `pressure` and `tiltX`/`tiltY`. Pen input would need patches to the
macOS, Windows, Linux and web platform layers.

**The interface code.** The panel, buttons and canvas take about 150 lines, comparable to
the egui equivalents, with layout and text noticeably better. gpui needs the view to ask
for frames: `request_animation_frame` from inside `render` did not keep a frame loop
going here, so a timer re-renders the view every 8 ms.

**The browser.** gpui's web platform (`gpui_web`) is in Zed's tree. The spike runs in
Chrome on WebGPU: panel, canvas, painting with the pointer. Script-dispatched pointer
events report `offsetX` as half of `clientX` in Chrome on this display, which only matters
for testing with synthetic events.

## What would make it the main interface

1. The external-texture patch above, upstream, with the Linux windows wired and a way
   to raise the device limits.
2. On macOS, a BGRA `CVPixelBuffer` path in the Metal renderer's surfaces, with the
   compositor rendering into an IOSurface-backed texture through wgpu's Metal interop, or
   wgpu as gpui's macOS renderer.
3. Pen pressure, tilt and rotation in gpui's input events.
4. Until then, on macOS, the CPU path: fine at 1024 × 768, short of the egui app's GPU
   display at larger sizes.
