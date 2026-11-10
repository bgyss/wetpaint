# wetpaint: an open natural-media painting engine

A cross-platform Rust engine and app for physically based wet and dry media painting:
watercolour, ink, oil, pastel, pencil, marker and airbrush. Paint flows, spreads into wet
paper, granulates, dries with darkened edges and runs down a tilted canvas. Pigments mix
subtractively with a Kubelka–Munk model.

## Crates

| crate | what it does |
|---|---|
| `wet-fluid` | Grid-based wet-media simulation: velocity diffusion, gravity and tilt, projection, conservative transport of water and pigment, evaporation, deposition, re-wetting, edge darkening, granulation, drips. The CPU reference. |
| `wet-fluid-gpu` | wgpu/WGSL port of the simulation kernels (tested against the CPU reference), and the GPU display compositor. |
| `wet-brush` | Brush engine: tip and grain textures, spacing, jitter and pressure curves, dual brushes, bristle brushes, per-medium paint models, glazing, multi-coloured brushes, rulers, and the water, dry, eraser, blend, smudge, blow, clone and liquify tools. |
| `wet-preset` | `.wetbrush` brush preset files. |
| `wet-paper` | Procedural papers (height, absorbency, display shading), deckled edges and granulation textures. |
| `wet-compose` | Layer compositing with 27 colour blend modes. |
| `wet-pigment` | Kubelka–Munk pigment mixer and RYB colour mixing. |
| `wet-doc` | Documents: wet layers with groups, clipping, masks, masking fluid and per-layer pigments; selections, stencils, fill, tracing, transforms, filters; undo/redo; the open `.wetdoc` format, and layered PSD and PNG export with ICC profiles. |
| `wet-color` | RGB working spaces (sRGB, Display P3, Adobe RGB, Rec. 2020), conversions and ICC profiles. |
| `wetpaint` | The painting app: egui on winit + wgpu, with the simulation and display on the GPU when available. |

## Build and run

```
cargo run -p wetpaint --release            # Help > Keyboard Shortcuts lists the keys
cargo run -p wetpaint --release -- 1600 1200 painting.wetdoc
cargo test --workspace --release
cargo run -p wet-fluid-gpu --release --example bench -- 512 1024 2048
```

View > GPU Simulation (Shift+C) runs the simulation on the GPU. Brush Creator saves
`.wetbrush` presets into the brush library: `$WETPAINT_BRUSHES`, or `wetpaint/Brushes` in
the platform's application data folder (`~/Library/Application Support` on macOS).

## In the browser

`crates/web` runs the same app in a browser with eframe: WebGPU when the browser has it
(the simulation and the display run on the GPU), WebGL otherwise (on the CPU). A browser
cannot wait for the GPU, so there a stroke's cells and the layers a command needs come
back a frame or two after they are asked for, and the input waits for them.

```
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129   # must match Cargo.lock
web/build.sh                                       # → web/dist (wasm-opt is used if installed)
python3 -m http.server -d web/dist 8099            # open http://127.0.0.1:8099/
```

`web/dist/artifact.html` is the same page laid out for hosts that serve one HTML file with
the module beside it in 1 MiB gzipped parts.

## Agent control

Everything the app does can also be driven by an agent or a script: every menu, panel and
keyboard command, every slider, checkbox and picker, strokes and other canvas gestures,
the simulation clock, rendering, and files. Requests are JSON objects with an `"op"`;
`{"op": "describe"}` lists every op, command (by name), setting (by key, with its current
value) and enumerated value. See `crates/app/src/agent.rs`.

```
wetpaint --mcp [W H] [FILE]        # MCP server on stdio, no window
wetpaint --agent [W H] [FILE]      # JSON lines on stdin → replies on stdout, no window
wetpaint --listen [ADDR]           # the window, also taking JSON lines on 127.0.0.1:7878
wetpaint --mcp --connect [ADDR]    # MCP server that drives a --listen window
```

For example, with Claude Code: `claude mcp add wetpaint -- /path/to/wetpaint --mcp`. The
`render` tool returns the painting as an image the model can see.

```json
{"op": "set", "values": {"brush.tool": "oil", "brush.color": "#c04020", "brush.size": 40}}
{"op": "stroke", "points": [[40, 100, 0.3], [200, 110, 1.0], [280, 100, 0.5]], "steps_per_point": 2}
{"op": "step", "n": 50}
{"op": "command", "name": "filter", "args": {"filter": "hue_saturation", "hue": 30}}
{"op": "render", "path": "painting.png"}
```

Without a window, the simulation advances only by `step` (25 steps a second), so a script
gives the same pixels every time. Commands that open a dialog take a `path` instead, or
`data` (base64) plus a `filename`. Without a path, exports come back as base64. Replacing
or closing a document with unsaved changes needs `"discard": true`. In the browser, the
page exposes the same API as `await window.wetpaint.call({op: "state"})`.

`--listen` accepts requests from any process on this machine. Use it only on a machine
you trust.

## Licence

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.
