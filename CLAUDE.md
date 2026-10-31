# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

`wetpaint`: an open (MIT OR Apache-2.0) Rust natural-media painting engine and app. This
repository is public. It must stay free of third-party proprietary material: no other
painting app's code, decompiled output, assets (papers, brush textures, presets), file
formats' private details, or names of their internals. Engine behaviour is described in
the engine's own terms.

## Commands

```
cargo test --workspace --release
cargo run -p wetpaint --release [-- WIDTH HEIGHT] [FILE]
cargo test -p wet-fluid-gpu --release -- --nocapture   # GPU vs CPU parity (skips if no adapter)
cargo run -p wet-fluid-gpu --release --example bench -- 512 1024 2048
cargo run -p wet-fluid --release --example stroke -- <out_dir> [steps]
```

`.cargo/config.toml` forces `/usr/bin/clang` as the linker on aarch64 macOS (the nix-profile
`gcc` breaks linking on the maintainer's machine).

## Structure

Crates under `crates/`: `fluid` (CPU reference simulation, one module per kernel under
`src/kernels/`, step driver `WetLayer::step` in `src/layer.rs`), `fluid-gpu` (WGSL port in
`src/fluid.wgsl`, display compositor in `src/display.wgsl`), `brush`, `preset`
(`.wetbrush`), `paper`, `compose`, `pigment`, `color`, `doc`, `app` (`wetpaint`).

- Any change to a CPU kernel must be mirrored in `crates/fluid-gpu/src/fluid.wgsl` (the
  parity test's single-step check catches mismatches; multi-step runs diverge cell by
  cell by design and are compared in aggregate).
- `src/display.wgsl` mirrors the `wet-compose` blend modes and the `Document::render`
  shading (`wetpaint` test `gpu_display_matches_the_cpu_render`). Features it does not
  cover fall back to the CPU path (`App::gpu_display_ok`).
- Brush stamping stays on the CPU for exact parity; GPU-held layers copy only the stroke's
  reach rectangle (`Brush::reach_bounds`) back and forth.
