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

## Licence

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.
