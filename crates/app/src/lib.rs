//! The wetpaint app: document state and commands ([`app`]), the egui interface ([`ui`]),
//! GPU simulation and display ([`gpu`]), and file access ([`platform`]). The desktop
//! binary and the web build drive it. Agents drive it too: [`agent`] answers JSON
//! requests for every command, setting and canvas gesture, over stdio, TCP ([`serve`]),
//! MCP ([`mcp`]) or, on the web, from JavaScript.

pub mod agent;
pub mod app;
pub mod gpu;
pub mod library;
pub mod mcp;
pub mod platform;
#[cfg(not(target_arch = "wasm32"))]
pub mod serve;
pub mod transform;
#[cfg(feature = "window")]
pub mod ui;

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
