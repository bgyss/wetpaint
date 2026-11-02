//! Agent control: everything the app does, as JSON requests that an agent or a script sends.
//!
//! A request is an object with an `"op"`. The reply is `{"ok": true, ...}` or
//! `{"ok": false, "error": "..."}`, plus the status-bar message when the request changed it.
//! `{"op": "describe"}` lists every op, command, setting and enumerated value. One
//! [`handle`] serves every way in:
//! - `wetpaint --agent`: JSON lines on stdin and stdout, without a window;
//! - `wetpaint --mcp`: a Model Context Protocol server on stdio ([`crate::mcp`]);
//! - `wetpaint --listen [ADDR]`: the window, also driven by JSON lines over local TCP;
//! - the web build: `window.wetpaint.call(request)`.
//!
//! Commands are the menu, panel and keyboard [`Cmd`]s by name. Settings are the panels'
//! sliders, checkboxes and pickers by key (`brush.size`, `layer.opacity`, `paper.relief`,
//! …). Gestures (`stroke`, `drag`, `click`) do what the pointer does on the canvas. File
//! ops take paths or base64 data instead of dialogs.
//!
//! Coverage is checked by the compiler: [`cmd_name`] matches every [`Cmd`], and the
//! settings destructure the brush, layer, group, paper and simulation structs field by
//! field, so a new command or field does not build until the agent can reach it.

use std::fmt::Debug;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde_json::{Map, Value, json};

use wet_brush::{
    BrushParams, GlazeMode, Liquify, Medium, PaintMode, RotationMode, Tool,
    curve::Curve,
    ruler::Ruler,
    volume::Volume,
};
use wet_compose::MixMode;
use wet_doc::{
    Fill, FillOptions, LayerGroup, PaintLayer, PaperSettings, SelectOp,
    filters::{Channel, Filter},
    guides::Grid,
};
use wet_fluid::params::FluidParams;
use wet_paper::{PaperSpec, Surface};

use crate::app::{App, ClickMode, Cmd, PALETTE, decode_rgba};
use crate::platform::{self, Target};

type R<T> = Result<T, String>;

/// Answer one request.
pub fn handle(app: &mut App, req: &Value) -> Value {
    let before = app.status.clone();
    match dispatch(app, req) {
        Ok(v) => {
            let mut m = match v {
                Value::Object(m) => m,
                Value::Null => Map::new(),
                v => Map::from_iter([("value".to_owned(), v)]),
            };
            m.insert("ok".into(), true.into());
            if app.status != before && !app.status.is_empty() {
                m.insert("status".into(), app.status.clone().into());
            }
            Value::Object(m)
        }
        Err(e) => json!({ "ok": false, "error": e }),
    }
}

/// [`handle`] for one line of JSON; the reply is one line.
pub fn handle_line(app: &mut App, line: &str) -> String {
    let reply = match serde_json::from_str::<Value>(line) {
        Ok(req) => handle(app, &req),
        Err(e) => json!({ "ok": false, "error": format!("not JSON: {e}") }),
    };
    reply.to_string()
}

/// An app for agents without a window: the simulation advances only by `step`, and the
/// view is 1:1 (the transform handles and snapping use screen distances).
pub fn headless(w: usize, h: usize) -> App {
    let mut app = App::new(w, h);
    app.zoom = 1.0;
    app
}

/// An op: name, what it does, and its arguments (name, JSON type, meaning).
pub struct Op {
    pub name: &'static str,
    pub doc: &'static str,
    pub args: &'static [(&'static str, &'static str, &'static str)],
}

const LAYER_ARG: (&str, &str, &str) = ("layer", "any", "layer index (0 = bottom) or name; the active layer by default");
const SOURCE_ARGS: [(&str, &str, &str); 3] = [
    ("path", "string", "file to read (desktop)"),
    ("data", "string", "the file's contents, base64"),
    ("filename", "string", "file name with extension, needed with data"),
];
const OUT_ARGS: [(&str, &str, &str); 2] = [
    ("path", "string", "file to write (desktop); without it the reply carries the bytes as base64 `data`"),
    ("download", "boolean", "web: hand the file to the browser as a download"),
];

pub const OPS: &[Op] = &[
    Op { name: "describe", doc: "List every op, command, setting (with its current value) and enumerated value.", args: &[] },
    Op { name: "state", doc: "The document and app at a glance: size, layers, groups, tool, mode, colour, history, file.", args: &[] },
    Op {
        name: "command",
        doc: "Run a menu / panel / keyboard command by name (see describe.commands). Commands that open a dialog take a path or data instead.",
        args: &[("name", "string", "command name"), ("args", "object", "the command's arguments, if any (they may also be given at the top level)")],
    },
    Op { name: "key", doc: "Press a keyboard shortcut over the canvas (Help > Keyboard Shortcuts), e.g. \"x\", \"Shift+N\", \"F1\", \"Space\".", args: &[("key", "string", "the key")] },
    Op {
        name: "get",
        doc: "Read settings: one key, or all of them without a key.",
        args: &[("key", "string", "setting key, e.g. brush.size"), LAYER_ARG, ("group", "integer", "group index for group.* keys")],
    },
    Op {
        name: "set",
        doc: "Change settings: key and value, or several at once in values. Toggles run their command.",
        args: &[("key", "string", "setting key"), ("value", "any", "new value"), ("values", "object", "{key: value, ...}"), LAYER_ARG, ("group", "integer", "group index for group.* keys")],
    },
    Op {
        name: "stroke",
        doc: "Paint a stroke with the current brush on the active (or given) layer, as a pointer drag does.",
        args: &[
            ("points", "array", "[[x, y], [x, y, pressure], {\"x\", \"y\", \"pressure\"}, ...] in canvas pixels; pressure 0–1"),
            ("steps_per_point", "integer", "simulation steps between points (0: the stroke is instantaneous)"),
            ("hold", "boolean", "keep the pointer down: the next stroke goes on with the same gesture (render and step in between to watch it)"),
            LAYER_ARG,
        ],
    },
    Op {
        name: "drag",
        doc: "A pointer drag in the current or given mode: a selection rectangle / ellipse / lasso, transform handles, or paint.",
        args: &[
            ("points", "array", "as for stroke"),
            ("mode", "string", "click mode to use (see enums.mode)"),
            ("free_corner", "boolean", "transform: a dragged corner moves on its own (perspective), as with ⌘/Ctrl held"),
            ("steps_per_point", "integer", "simulation steps between points"),
            ("hold", "boolean", "keep the pointer down: the next drag goes on with the same gesture"),
        ],
    },
    Op {
        name: "click",
        doc: "A click on the canvas in the current or given mode: pick a colour, fill, magic wand, or a dab of paint.",
        args: &[("x", "number", "canvas x"), ("y", "number", "canvas y"), ("mode", "string", "click mode (see enums.mode)"), ("pressure", "number", "0–1")],
    },
    Op {
        name: "select",
        doc: "Change the selection: all, none, invert, restore, rect, ellipse, polygon, color (magic wand) or visible (a layer's paint).",
        args: &[
            ("shape", "string", "all | none | invert | restore | rect | ellipse | polygon | color | visible"),
            ("from", "array", "rect / ellipse corner [x, y]"),
            ("to", "array", "rect / ellipse opposite corner [x, y]"),
            ("points", "array", "polygon [[x, y], ...]"),
            ("at", "array", "color: the pixel [x, y] to match"),
            ("tolerance", "integer", "color: 0–255 (32)"),
            ("contiguous", "boolean", "color: connected pixels only (true)"),
            ("combine", "string", "with the current selection: replace | add | subtract | intersect (replace)"),
            LAYER_ARG,
        ],
    },
    Op {
        name: "fill",
        doc: "Fill the area around a pixel of the active layer with a colour or a gradient (Fill tool, with all its options).",
        args: &[
            ("at", "array", "[x, y]"),
            ("color", "any", "\"#rrggbb\" or [r, g, b] 0–255; the brush colour by default"),
            ("gradient", "object", "{\"from\": [x, y], \"to\": [x, y], \"colors\": [c0, c1], \"radial\": false}"),
            ("tolerance", "integer", "0–255 (32)"),
            ("opacity", "number", "0–1 (the brush opacity)"),
            ("contiguous", "boolean", "true"),
            ("wet", "boolean", "the fill is wet paint (false)"),
        ],
    },
    Op {
        name: "transform",
        doc: "Move, scale and rotate the active layer about the canvas centre, or map its corners to four points.",
        args: &[
            ("dx", "number", "pixels"),
            ("dy", "number", "pixels"),
            ("scale", "number", "factor (1)"),
            ("rotation", "number", "degrees"),
            ("corners", "array", "perspective: where the canvas corners go, [[x, y] × 4] clockwise from top-left"),
        ],
    },
    Op { name: "resize", doc: "Resize the canvas (crop or extend) or resample the image.", args: &[("width", "integer", "px"), ("height", "integer", "px"), ("resample", "boolean", "scale the painting instead of cropping / extending"), ("offset", "array", "canvas: where the old top-left goes, [x, y]")] },
    Op { name: "step", doc: "Advance the wet simulation by n steps (25 steps = one second).", args: &[("n", "integer", "steps (1)")] },
    Op {
        name: "render",
        doc: "The painting (or one layer) as a PNG.",
        args: &[
            LAYER_ARG,
            ("view", "boolean", "include the view overlays: greyscale, stencils, the dimmed area outside a selection"),
            ("max_size", "integer", "downscale so the longer side is at most this"),
            OUT_ARGS[0],
        ],
    },
    Op { name: "pixel", doc: "The displayed colour [r, g, b, a] at a canvas pixel.", args: &[("x", "integer", ""), ("y", "integer", "")] },
    Op { name: "new", doc: "A new blank document.", args: &[("width", "integer", "px"), ("height", "integer", "px"), ("discard", "boolean", "allow losing unsaved changes")] },
    Op {
        name: "open",
        doc: "Open a .wetdoc, a .wetbrush preset, or an image as a new layer.",
        args: &[SOURCE_ARGS[0], SOURCE_ARGS[1], SOURCE_ARGS[2], ("discard", "boolean", "allow losing unsaved changes")],
    },
    Op {
        name: "import",
        doc: "Import an image as a layer, a tracing layer, a reference image, or a stencil (PNG).",
        args: &[SOURCE_ARGS[0], SOURCE_ARGS[1], SOURCE_ARGS[2], ("as", "string", "layer | tracing | reference | stencil")],
    },
    Op { name: "save", doc: "Save the document (.wetdoc) to its file or to path.", args: &[("path", "string", "where to save (desktop)")] },
    Op {
        name: "export",
        doc: "Write the painting as png, psd (layered), exr (heightmap), wetdoc, or the brush as wetbrush.",
        args: &[("format", "string", "png | psd | exr | wetdoc | wetbrush"), ("name", "string", "wetbrush: preset name"), OUT_ARGS[0], OUT_ARGS[1]],
    },
    Op { name: "presets", doc: "The brush library's presets for a tool.", args: &[("tool", "string", "tool name (the current tool)")] },
    Op {
        name: "load_preset",
        doc: "Paint with a brush preset: by library name, path or data.",
        args: &[("name", "string", "library preset name"), SOURCE_ARGS[0], SOURCE_ARGS[1], SOURCE_ARGS[2]],
    },
    Op {
        name: "save_preset",
        doc: "Save the brush (Brush Creator parameters) as a .wetbrush preset.",
        args: &[("name", "string", "preset name"), ("library", "boolean", "save into the brush library's My Brushes"), OUT_ARGS[0], OUT_ARGS[1]],
    },
    Op { name: "history", doc: "The undo history: steps done and steps that can be redone.", args: &[] },
    Op { name: "quit", doc: "Close the app (desktop) or end the session.", args: &[("discard", "boolean", "allow losing unsaved changes")] },
];

/// Command names, their arguments and what they do.
pub const COMMANDS: &[(&str, &str, &str)] = &[
    ("tool", "tool, dx, dy (clone)", "choose a tool (see enums.tool)"),
    ("cycle_liquify", "", "liquify: push → twirl → pucker → bloat → reconstruct"),
    ("mode", "mode", "click mode (see enums.mode)"),
    ("next_color", "", "next palette colour"),
    ("size_by", "factor", "multiply the brush size"),
    ("water_by", "delta", "add to the brush water (0–1)"),
    ("opacity_by", "delta", "add to the brush opacity (0–1)"),
    ("volume", "which", "next size / opacity / water / pressure / length preset"),
    ("toggle_multi", "", "multi-coloured brush on / off"),
    ("toggle_pause", "", "pause / resume the simulation"),
    ("toggle_tilt", "", "tilt the active layer (paint runs)"),
    ("dry_layer", "", "dry the active layer"),
    ("wet_layer", "", "wet the active layer"),
    ("clear_layer", "", "clear the active layer"),
    ("toggle_pigment", "", "pigment (Kubelka–Munk) mixing on all layers"),
    ("toggle_granulation", "", "granulation on the active layer"),
    ("new_layer", "", "add a layer above the active one"),
    ("next_layer", "", "make the next layer active"),
    ("delete_layer", "", "delete the active layer"),
    ("duplicate_layer", "", "duplicate the active layer"),
    ("merge_down", "", "merge the active layer into the one below"),
    ("move_layer", "by", "move the active layer up (+) or down (−)"),
    ("group_with_below", "", "group the active layer with the one below"),
    ("layer_pigments", "", "pigment mixing on the active layer"),
    ("toggle_masking_fluid", "", "the active layer is masking fluid"),
    ("toggle_lock", "", "lock / unlock the active layer"),
    ("toggle_clip", "", "clip the active layer to the one below"),
    ("toggle_metallic", "", "metallic paint on the active layer"),
    ("transform", "dx, dy, scale, rotation (degrees)", "move / scale / rotate the active layer (nudges a transform in progress)"),
    ("apply_transform", "", "apply the transform in progress"),
    ("cancel_transform", "", "cancel the transform in progress"),
    ("reset_brush", "", "restore the brush's preset parameters"),
    ("save_preset", "path or library, name", "save the brush as a preset"),
    ("toggle_gpu", "", "simulate on the GPU / CPU"),
    ("history", "steps", "undo (−) or redo (+) this many steps"),
    ("undo", "", "undo"),
    ("redo", "", "redo"),
    ("filter", "filter, its parameters", "filter the active layer (see enums.filter)"),
    ("select_all", "", "select everything"),
    ("deselect", "", "drop the selection"),
    ("restore_selection", "", "bring back the last selection"),
    ("invert_selection", "", "invert the selection"),
    ("stencil_from_selection", "", "turn the selection into a stencil"),
    ("cut_selection_to_layer", "", "cut the selection to a new layer"),
    ("toggle_stencil", "", "show / hide the stencil (places a disc, or ./stencil.png, if none)"),
    ("invert_stencil", "", "invert the stencil"),
    ("tracing", "path or data (first time)", "import a tracing image, or toggle tracing"),
    ("toggle_reference", "path or data (first time)", "import a reference image, or show / hide it"),
    ("cycle_ruler", "", "ruler: off → horizontal → vertical → ellipses → perspective"),
    ("toggle_grid", "", "64 px grid with snapping"),
    ("toggle_greyscale", "", "greyscale view"),
    ("toggle_natural", "", "natural (RYB) colour mixing"),
    ("toggle_soft_shadows", "", "soft shadows on the relief"),
    ("next_visual_preset", "", "next watercolour behaviour preset"),
    ("fit", "", "fit the canvas in the window"),
    ("toggle_recording", "", "time-lapse recording (desktop window)"),
    ("new_document", "width, height, discard", "a new blank document"),
    ("open", "path or data + name, discard", "open a document, preset or image"),
    ("save", "path", "save the document"),
    ("save_as", "path", "save the document to a new file"),
    ("save_iterative", "", "save as the next numbered file"),
    ("export_png", "path", "export a PNG"),
    ("export_psd", "path", "export a layered PSD"),
    ("export_heightmap", "path", "export the heightmap (EXR)"),
];

/// The command's name in [`COMMANDS`].
pub fn cmd_name(c: &Cmd) -> &'static str {
    match c {
        Cmd::Tool(_) => "tool",
        Cmd::CycleLiquify => "cycle_liquify",
        Cmd::Mode(_) => "mode",
        Cmd::NextColor => "next_color",
        Cmd::SizeBy(_) => "size_by",
        Cmd::WaterBy(_) => "water_by",
        Cmd::OpacityBy(_) => "opacity_by",
        Cmd::Volume(_) => "volume",
        Cmd::ToggleMulti => "toggle_multi",
        Cmd::TogglePause => "toggle_pause",
        Cmd::ToggleTilt => "toggle_tilt",
        Cmd::DryLayer => "dry_layer",
        Cmd::WetLayer => "wet_layer",
        Cmd::ClearLayer => "clear_layer",
        Cmd::TogglePigment => "toggle_pigment",
        Cmd::ToggleGranulation => "toggle_granulation",
        Cmd::NewLayer => "new_layer",
        Cmd::NextLayer => "next_layer",
        Cmd::DeleteLayer => "delete_layer",
        Cmd::DuplicateLayer => "duplicate_layer",
        Cmd::MergeDown => "merge_down",
        Cmd::MoveLayer(_) => "move_layer",
        Cmd::GroupWithBelow => "group_with_below",
        Cmd::LayerPigments => "layer_pigments",
        Cmd::ToggleMaskingFluid => "toggle_masking_fluid",
        Cmd::ToggleLock => "toggle_lock",
        Cmd::ToggleClip => "toggle_clip",
        Cmd::ToggleMetallic => "toggle_metallic",
        Cmd::Transform(..) => "transform",
        Cmd::ApplyTransform => "apply_transform",
        Cmd::CancelTransform => "cancel_transform",
        Cmd::ResetBrush => "reset_brush",
        Cmd::SavePreset => "save_preset",
        Cmd::ToggleGpu => "toggle_gpu",
        Cmd::History(_) => "history",
        Cmd::Undo => "undo",
        Cmd::Redo => "redo",
        Cmd::Filter(_) => "filter",
        Cmd::SelectAll => "select_all",
        Cmd::Deselect => "deselect",
        Cmd::RestoreSelection => "restore_selection",
        Cmd::InvertSelection => "invert_selection",
        Cmd::StencilFromSelection => "stencil_from_selection",
        Cmd::CutSelectionToLayer => "cut_selection_to_layer",
        Cmd::ToggleStencil => "toggle_stencil",
        Cmd::InvertStencil => "invert_stencil",
        Cmd::Tracing => "tracing",
        Cmd::ToggleReference => "toggle_reference",
        Cmd::CycleRuler => "cycle_ruler",
        Cmd::ToggleGrid => "toggle_grid",
        Cmd::ToggleGreyscale => "toggle_greyscale",
        Cmd::ToggleNatural => "toggle_natural",
        Cmd::ToggleSoftShadows => "toggle_soft_shadows",
        Cmd::NextVisualPreset => "next_visual_preset",
        Cmd::Fit => "fit",
        Cmd::ToggleRecording => "toggle_recording",
        Cmd::NewDocument(..) => "new_document",
        Cmd::Open => "open",
        Cmd::Save => "save",
        Cmd::SaveAs => "save_as",
        Cmd::SaveIterative => "save_iterative",
        Cmd::ExportPng => "export_png",
        Cmd::ExportPsd => "export_psd",
        Cmd::ExportHeightmap => "export_heightmap",
    }
}

/// The command `name` with its arguments from `a`.
pub fn parse_cmd(name: &str, a: &Value) -> R<Cmd> {
    Ok(match name {
        "tool" => Cmd::Tool(parse_tool(string(a, "tool")?, a)?),
        "cycle_liquify" => Cmd::CycleLiquify,
        "mode" => Cmd::Mode(parse_enum(&MODES, string(a, "mode")?, "mode")?),
        "next_color" => Cmd::NextColor,
        "size_by" => Cmd::SizeBy(num(a, "factor")?),
        "water_by" => Cmd::WaterBy(num(a, "delta")?),
        "opacity_by" => Cmd::OpacityBy(num(a, "delta")?),
        "volume" => Cmd::Volume(parse_enum(&VOLUMES, string(a, "which")?, "volume")?),
        "toggle_multi" => Cmd::ToggleMulti,
        "toggle_pause" => Cmd::TogglePause,
        "toggle_tilt" => Cmd::ToggleTilt,
        "dry_layer" => Cmd::DryLayer,
        "wet_layer" => Cmd::WetLayer,
        "clear_layer" => Cmd::ClearLayer,
        "toggle_pigment" => Cmd::TogglePigment,
        "toggle_granulation" => Cmd::ToggleGranulation,
        "new_layer" => Cmd::NewLayer,
        "next_layer" => Cmd::NextLayer,
        "delete_layer" => Cmd::DeleteLayer,
        "duplicate_layer" => Cmd::DuplicateLayer,
        "merge_down" => Cmd::MergeDown,
        "move_layer" => Cmd::MoveLayer(int(a, "by")? as isize),
        "group_with_below" => Cmd::GroupWithBelow,
        "layer_pigments" => Cmd::LayerPigments,
        "toggle_masking_fluid" => Cmd::ToggleMaskingFluid,
        "toggle_lock" => Cmd::ToggleLock,
        "toggle_clip" => Cmd::ToggleClip,
        "toggle_metallic" => Cmd::ToggleMetallic,
        "transform" => Cmd::Transform(
            (num_or(a, "dx", 0.0)?, num_or(a, "dy", 0.0)?),
            num_or(a, "scale", 1.0)?,
            num_or(a, "rotation", 0.0)?.to_radians(),
        ),
        "apply_transform" => Cmd::ApplyTransform,
        "cancel_transform" => Cmd::CancelTransform,
        "reset_brush" => Cmd::ResetBrush,
        "save_preset" => Cmd::SavePreset,
        "toggle_gpu" => Cmd::ToggleGpu,
        "history" => Cmd::History(int(a, "steps")? as isize),
        "undo" => Cmd::Undo,
        "redo" => Cmd::Redo,
        "filter" => Cmd::Filter(parse_filter(a)?),
        "select_all" => Cmd::SelectAll,
        "deselect" => Cmd::Deselect,
        "restore_selection" => Cmd::RestoreSelection,
        "invert_selection" => Cmd::InvertSelection,
        "stencil_from_selection" => Cmd::StencilFromSelection,
        "cut_selection_to_layer" => Cmd::CutSelectionToLayer,
        "toggle_stencil" => Cmd::ToggleStencil,
        "invert_stencil" => Cmd::InvertStencil,
        "tracing" => Cmd::Tracing,
        "toggle_reference" => Cmd::ToggleReference,
        "cycle_ruler" => Cmd::CycleRuler,
        "toggle_grid" => Cmd::ToggleGrid,
        "toggle_greyscale" => Cmd::ToggleGreyscale,
        "toggle_natural" => Cmd::ToggleNatural,
        "toggle_soft_shadows" => Cmd::ToggleSoftShadows,
        "next_visual_preset" => Cmd::NextVisualPreset,
        "fit" => Cmd::Fit,
        "toggle_recording" => Cmd::ToggleRecording,
        "new_document" => Cmd::NewDocument(uint(a, "width")?, uint(a, "height")?),
        "open" => Cmd::Open,
        "save" => Cmd::Save,
        "save_as" => Cmd::SaveAs,
        "save_iterative" => Cmd::SaveIterative,
        "export_png" => Cmd::ExportPng,
        "export_psd" => Cmd::ExportPsd,
        "export_heightmap" => Cmd::ExportHeightmap,
        _ => return Err(format!("unknown command \"{name}\" (describe lists them)")),
    })
}

// ---- enumerations, named by their variants in snake case ----

const MEDIA: [Medium; 8] = [
    Medium::Watercolor, Medium::Ink, Medium::Oil, Medium::ExpressOil,
    Medium::Pastel, Medium::Pencil, Medium::Marker, Medium::Airbrush,
];
const LIQUIFY: [Liquify; 5] = [Liquify::Push, Liquify::Twirl, Liquify::Pucker, Liquify::Bloat, Liquify::Reconstruct];
const MODES: [ClickMode; 8] = [
    ClickMode::Paint, ClickMode::Pick, ClickMode::Fill, ClickMode::SelectEllipse,
    ClickMode::SelectRect, ClickMode::Lasso, ClickMode::Wand, ClickMode::Transform,
];
const VOLUMES: [Volume; 5] = [Volume::Size, Volume::Opacity, Volume::Water, Volume::Pressure, Volume::Length];
const PAINT_MODES: [PaintMode; 5] = [PaintMode::Paint, PaintMode::Mix, PaintMode::PaintBlend, PaintMode::Blend, PaintMode::Erase];
const GLAZE_MODES: [GlazeMode; 3] = [GlazeMode::Transparent, GlazeMode::SemiOpaque, GlazeMode::Opaque];
const ROTATION_MODES: [RotationMode; 4] = [RotationMode::None, RotationMode::FollowTrajectory, RotationMode::PenTilt, RotationMode::PenRotation];
const SURFACES: [Surface; 3] = [Surface::HotPressed, Surface::ColdPressed, Surface::Rough];
const SELECT_OPS: [SelectOp; 4] = [SelectOp::Replace, SelectOp::Add, SelectOp::Subtract, SelectOp::Intersect];
const CHANNELS: [Channel; 4] = [Channel::Rgb, Channel::Red, Channel::Green, Channel::Blue];
const RULERS: [&str; 5] = ["off", "horizontal", "vertical", "ellipses", "perspective"];
const FILTERS: [&str; 16] = [
    "brightness_contrast", "levels", "curves", "hue_saturation", "color_balance", "color_filter",
    "colorize", "desaturate", "invert", "gradient_map", "gaussian_blur", "lens_blur", "sharpen",
    "subtle_sharpen", "white_to_alpha", "black_to_alpha",
];
const UTILITY_TOOLS: [&str; 7] = ["water", "dry", "eraser", "blend", "smudge", "blow", "clone"];

/// `CamelCase` → `camel_case`.
fn snake(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.extend(c.to_lowercase());
    }
    out
}

fn name_of<T: Debug>(v: &T) -> String {
    snake(&format!("{v:?}"))
}

fn names<T: Debug>(all: &[T]) -> Vec<String> {
    all.iter().map(name_of).collect()
}

fn parse_enum<T: Debug + Copy>(all: &[T], s: &str, what: &str) -> R<T> {
    all.iter().copied().find(|v| name_of(v) == s).ok_or_else(|| format!("unknown {what} \"{s}\"; one of: {}", names(all).join(", ")))
}

fn tool_names() -> Vec<String> {
    let mut v: Vec<String> = MEDIA.iter().map(|m| m.id().to_owned()).collect();
    v.extend(UTILITY_TOOLS.iter().map(|s| s.to_string()));
    v.extend(LIQUIFY.iter().map(|l| format!("liquify_{}", name_of(l))));
    v
}

fn tool_name(t: Tool) -> String {
    match t {
        Tool::Liquify(l) => format!("liquify_{}", name_of(&l)),
        t => t.id().to_owned(),
    }
}

fn parse_tool(s: &str, a: &Value) -> R<Tool> {
    if let Some(m) = MEDIA.iter().find(|m| m.id() == s) {
        return Ok(Tool::Paint(*m));
    }
    if let Some(l) = s.strip_prefix("liquify_") {
        return Ok(Tool::Liquify(parse_enum(&LIQUIFY, l, "liquify mode")?));
    }
    Ok(match s {
        "water" => Tool::Water,
        "dry" => Tool::Dry,
        "eraser" => Tool::Eraser,
        "blend" => Tool::Blend,
        "smudge" => Tool::Smudge,
        "blow" => Tool::Blow,
        "clone" => Tool::Clone { dx: num_or(a, "dx", 0.0)?, dy: num_or(a, "dy", 0.0)? },
        "liquify" => Tool::Liquify(Liquify::Push),
        _ => return Err(format!("unknown tool \"{s}\"; one of: {}", tool_names().join(", "))),
    })
}

fn parse_filter(a: &Value) -> R<Filter> {
    let name = string(a, "filter")?;
    let channel = |a: &Value| arg(a, "channel").map_or(Ok(Channel::Rgb), |v| parse_enum(&CHANNELS, str_of(v)?, "channel"));
    let triple = |k: &str| arg(a, k).map_or(Ok([0.0; 3]), |v| {
        let v = floats(v)?;
        <[f32; 3]>::try_from(v.as_slice()).map_err(|_| format!("{k} needs 3 numbers"))
    });
    Ok(match name {
        "brightness_contrast" => Filter::BrightnessContrast { brightness: num_or(a, "brightness", 0.0)?, contrast: num_or(a, "contrast", 0.0)? },
        "levels" => Filter::Levels {
            channel: channel(a)?,
            in_black: num_or(a, "in_black", 0.0)?,
            in_white: num_or(a, "in_white", 255.0)?,
            gamma: num_or(a, "gamma", 1.0)?,
            out_black: num_or(a, "out_black", 0.0)?,
            out_white: num_or(a, "out_white", 255.0)?,
        },
        "curves" => Filter::Curves { channel: channel(a)?, points: points2(need(a, "points")?)? },
        "hue_saturation" => Filter::HueSaturation { hue: num_or(a, "hue", 0.0)?, saturation: num_or(a, "saturation", 0.0)?, lightness: num_or(a, "lightness", 0.0)? },
        "color_balance" => Filter::ColorBalance {
            shadows: triple("shadows")?,
            midtones: triple("midtones")?,
            highlights: triple("highlights")?,
            preserve_luminosity: bool_or(a, "preserve_luminosity", true)?,
        },
        "color_filter" => Filter::ColorFilter { color: color(need(a, "color")?)?, strength: num_or(a, "strength", 0.5)? },
        "colorize" => Filter::Colorize { hue: num(a, "hue")?, saturation: num_or(a, "saturation", 0.5)?, strength: num_or(a, "strength", 1.0)? },
        "desaturate" => Filter::Desaturate { luminosity: bool_or(a, "luminosity", true)? },
        "invert" => Filter::Invert,
        "gradient_map" => Filter::GradientMap {
            stops: need(a, "stops")?
                .as_array()
                .ok_or("stops: [[position, colour], ...]")?
                .iter()
                .map(|s| Ok((f32_of(s.get(0).ok_or("a stop needs a position")?)?, color(s.get(1).ok_or("a stop needs a colour")?)?)))
                .collect::<R<_>>()?,
        },
        "gaussian_blur" => Filter::GaussianBlur { radius: num_or(a, "radius", 4.0)? },
        "lens_blur" => Filter::LensBlur { size: num_or(a, "size", 10.0)?, threshold: num_or(a, "threshold", 255.0)?, brightness: num_or(a, "brightness", 0.0)? },
        "sharpen" => Filter::Sharpen { radius: num_or(a, "radius", 2.0)?, strength: num_or(a, "strength", 0.5)? },
        "subtle_sharpen" => Filter::SubtleSharpen,
        "white_to_alpha" => Filter::WhiteToAlpha,
        "black_to_alpha" => Filter::BlackToAlpha,
        _ => return Err(format!("unknown filter \"{name}\"; one of: {}", FILTERS.join(", "))),
    })
}

// ---- argument helpers ----

fn arg<'a>(a: &'a Value, k: &str) -> Option<&'a Value> {
    a.get(k).filter(|v| !v.is_null())
}

fn need<'a>(a: &'a Value, k: &str) -> R<&'a Value> {
    arg(a, k).ok_or_else(|| format!("missing \"{k}\""))
}

fn f32_of(v: &Value) -> R<f32> {
    v.as_f64().map(|x| x as f32).filter(|x| x.is_finite()).ok_or_else(|| format!("expected a number, got {v}"))
}

fn str_of(v: &Value) -> R<&str> {
    v.as_str().ok_or_else(|| format!("expected a string, got {v}"))
}

fn bool_of(v: &Value) -> R<bool> {
    v.as_bool().ok_or_else(|| format!("expected true or false, got {v}"))
}

fn num(a: &Value, k: &str) -> R<f32> {
    f32_of(need(a, k)?).map_err(|e| format!("{k}: {e}"))
}

fn num_or(a: &Value, k: &str, d: f32) -> R<f32> {
    arg(a, k).map_or(Ok(d), |v| f32_of(v).map_err(|e| format!("{k}: {e}")))
}

fn int(a: &Value, k: &str) -> R<i64> {
    need(a, k)?.as_i64().ok_or_else(|| format!("{k}: expected an integer"))
}

fn uint(a: &Value, k: &str) -> R<usize> {
    need(a, k)?.as_u64().map(|v| v as usize).ok_or_else(|| format!("{k}: expected a whole number"))
}

fn string<'a>(a: &'a Value, k: &str) -> R<&'a str> {
    str_of(need(a, k)?).map_err(|e| format!("{k}: {e}"))
}

fn bool_or(a: &Value, k: &str, d: bool) -> R<bool> {
    arg(a, k).map_or(Ok(d), |v| bool_of(v).map_err(|e| format!("{k}: {e}")))
}

fn floats(v: &Value) -> R<Vec<f32>> {
    v.as_array().ok_or_else(|| format!("expected an array of numbers, got {v}"))?.iter().map(f32_of).collect()
}

/// `[x, y]` or `{"x", "y"}`.
fn point(v: &Value) -> R<(f32, f32)> {
    if let Some(o) = v.as_object() {
        return Ok((f32_of(o.get("x").ok_or("a point needs x")?)?, f32_of(o.get("y").ok_or("a point needs y")?)?));
    }
    match floats(v)?.as_slice() {
        [x, y, ..] => Ok((*x, *y)),
        _ => Err(format!("a point is [x, y], got {v}")),
    }
}

fn points2(v: &Value) -> R<Vec<(f32, f32)>> {
    v.as_array().ok_or("expected an array of points")?.iter().map(point).collect()
}

/// `"#rrggbb"` or `[r, g, b]` (0–255).
fn color(v: &Value) -> R<[f32; 3]> {
    if let Some(s) = v.as_str() {
        let h = s.trim_start_matches('#');
        if h.len() != 6 {
            return Err(format!("a colour is \"#rrggbb\" or [r, g, b], got {s}"));
        }
        let c = |i: usize| u8::from_str_radix(&h[i..i + 2], 16).map(f32::from).map_err(|_| format!("bad colour {s}"));
        return Ok([c(0)?, c(2)?, c(4)?]);
    }
    match floats(v)?.as_slice() {
        [r, g, b] => Ok([r.clamp(0.0, 255.0), g.clamp(0.0, 255.0), b.clamp(0.0, 255.0)]),
        _ => Err(format!("a colour is \"#rrggbb\" or [r, g, b], got {v}")),
    }
}

fn hex(c: [f32; 3]) -> Value {
    let b = |v: f32| v.round().clamp(0.0, 255.0) as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2])).into()
}

fn curve_json(c: &Option<Curve>) -> Value {
    c.as_ref().map_or(Value::Null, |c| json!({ "points": c.points.iter().map(|p| [p.0, p.1]).collect::<Vec<_>>(), "output": [c.output.0, c.output.1] }))
}

fn curve_of(v: &Value) -> R<Option<Curve>> {
    if v.is_null() {
        return Ok(None);
    }
    let points = points2(need(v, "points")?)?;
    let output = arg(v, "output").map_or(Ok((0.0, 1.0)), point)?;
    Ok(Some(Curve { points, output }))
}

/// The layer `a` names (index or name), or the active layer.
fn layer_index(app: &App, a: &Value) -> R<usize> {
    let n = app.doc.layers.len();
    match arg(a, "layer") {
        None => Ok(app.doc.active),
        Some(Value::String(name)) => app.doc.layers.iter().position(|l| &l.name == name).ok_or_else(|| format!("no layer named \"{name}\"")),
        Some(v) => v.as_u64().map(|i| i as usize).filter(|&i| i < n).ok_or_else(|| format!("layer: an index below {n} or a name, got {v}")),
    }
}

/// A pointer position and its pressure, if given.
type GesturePoint = ((f32, f32), Option<f32>);

/// A gesture's points: `[x, y]`, `[x, y, pressure]` or `{"x", "y", "pressure"}`.
fn gesture_points(v: &Value) -> R<Vec<GesturePoint>> {
    let pts = v.as_array().ok_or("points: an array of [x, y] or [x, y, pressure]")?;
    if pts.is_empty() {
        return Err("points is empty".into());
    }
    pts.iter()
        .map(|p| {
            let xy = point(p)?;
            let pr = match p {
                Value::Object(o) => o.get("pressure").map(f32_of).transpose()?,
                _ => p.get(2).map(f32_of).transpose()?,
            };
            Ok((xy, pr.map(|v| v.clamp(0.0, 1.0))))
        })
        .collect()
}

/// A file's name, path (if read from disk) and contents, from `path` or `data` + `name`.
fn source(a: &Value) -> R<(String, Option<PathBuf>, Vec<u8>)> {
    if let Some(p) = arg(a, "path") {
        let path = PathBuf::from(str_of(p)?);
        let bytes = std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        return Ok((name, Some(path), bytes));
    }
    let data = string(a, "data").map_err(|_| "pass a path, or data (base64) with a filename".to_owned())?;
    let bytes = B64.decode(data.trim()).map_err(|e| format!("data is not base64: {e}"))?;
    // `filename`, or `name` where that is not the command's name
    let name = string(a, "filename").or_else(|_| string(a, "name")).map_err(|_| "data needs a filename with the file's extension".to_owned())?.to_owned();
    Ok((name, None, bytes))
}

/// Write `bytes` to `path`, hand them to the browser, or return them as base64.
fn deliver(a: &Value, bytes: Vec<u8>, default_name: &str) -> R<Value> {
    let n = bytes.len();
    if let Some(p) = arg(a, "path") {
        let p = str_of(p)?;
        std::fs::write(p, &bytes).map_err(|e| format!("cannot write {p}: {e}"))?;
        return Ok(json!({ "path": p, "bytes": n }));
    }
    if bool_or(a, "download", false)? {
        platform::write(&Target::Download(default_name.to_owned()), |o| std::io::Write::write_all(o, &bytes)).map_err(|e| e.to_string())?;
        return Ok(json!({ "downloaded": default_name, "bytes": n }));
    }
    Ok(json!({ "data": B64.encode(&bytes), "bytes": n }))
}

fn png(rgba: Vec<u8>, w: usize, h: usize, max: Option<usize>) -> R<(Vec<u8>, usize, usize)> {
    let mut img = image::RgbaImage::from_raw(w as u32, h as u32, rgba).ok_or("image size")?;
    if let Some(m) = max.filter(|&m| m > 0 && w.max(h) > m) {
        let k = m as f32 / w.max(h) as f32;
        let (nw, nh) = (((w as f32 * k).round() as u32).max(1), ((h as f32 * k).round() as u32).max(1));
        img = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok((out, w, h))
}

fn check_discard(app: &App, a: &Value) -> R<()> {
    if app.dirty && !bool_or(a, "discard", false)? {
        return Err("the document has unsaved changes: save it first, or pass \"discard\": true".into());
    }
    Ok(())
}

/// The status message as an error when it reports one.
fn status_result(app: &App, v: Value) -> R<Value> {
    let s = &app.status;
    if s.starts_with("cannot") || s.contains("failed") || s.starts_with("no ") {
        Err(s.clone())
    } else {
        Ok(v)
    }
}

// ---- ops ----

fn dispatch(app: &mut App, a: &Value) -> R<Value> {
    let op = a.get("op").and_then(Value::as_str).ok_or("a request needs an \"op\"; {\"op\": \"describe\"} lists them")?;
    match op {
        "describe" => Ok(describe(app)),
        "state" => Ok(state(app)),
        "command" => command(app, a),
        "key" => key(app, a),
        "get" => get(app, a),
        "set" => set(app, a),
        "stroke" => {
            let i = layer_index(app, a)?;
            app.doc.active = i;
            if app.mode != ClickMode::Paint {
                app.run(Cmd::Mode(ClickMode::Paint));
            }
            if app.doc.layers[i].locked {
                return Err(format!("layer \"{}\" is locked", app.doc.layers[i].name));
            }
            gesture(app, a)
        }
        "drag" => {
            if let Some(m) = arg(a, "mode") {
                app.run(Cmd::Mode(parse_enum(&MODES, str_of(m)?, "mode")?));
            }
            gesture(app, a)
        }
        "click" => {
            if let Some(m) = arg(a, "mode") {
                app.run(Cmd::Mode(parse_enum(&MODES, str_of(m)?, "mode")?));
            }
            let picking = app.mode == ClickMode::Pick;
            let mut p = json!([num(a, "x")?, num(a, "y")?]);
            if let Some(pr) = arg(a, "pressure") {
                p.as_array_mut().expect("point").push(pr.clone());
            }
            let mut r = gesture(app, &json!({ "points": [p] }))?;
            if picking {
                r["color"] = hex(app.brush.color);
            }
            Ok(r)
        }
        "select" => select(app, a),
        "fill" => fill(app, a),
        "transform" => transform(app, a),
        "resize" => resize(app, a),
        "step" => {
            let n = arg(a, "n").map_or(Ok(1), |_| uint(a, "n"))?.min(100_000);
            app.step_sim(n);
            app.epoch += 1;
            Ok(json!({ "steps": n }))
        }
        "render" => render(app, a),
        "pixel" => {
            let (x, y) = (uint(a, "x")?, uint(a, "y")?);
            if x >= app.doc.w || y >= app.doc.h {
                return Err(format!("({x}, {y}) is outside the {} × {} canvas", app.doc.w, app.doc.h));
            }
            app.release_gpu();
            let img = composite(app);
            let k = (y * app.doc.w + x) * 4;
            Ok(json!({ "rgba": &img[k..k + 4], "hex": hex([img[k] as f32, img[k + 1] as f32, img[k + 2] as f32]) }))
        }
        "new" => {
            check_discard(app, a)?;
            app.dirty = false;
            app.run(Cmd::NewDocument(uint(a, "width")?, uint(a, "height")?));
            Ok(json!({ "width": app.doc.w, "height": app.doc.h }))
        }
        "open" => open(app, a),
        "import" => import(app, a),
        "save" => save(app, a),
        "export" => export(app, a),
        "presets" => presets(app, a),
        "load_preset" => load_preset(app, a),
        "save_preset" => save_preset(app, a),
        "history" => {
            let (done, undone) = app.doc.history();
            let steps = |v: Vec<wet_doc::HistoryStep<'_>>| v.iter().map(|(label, l)| json!({ "label": label, "layer": l })).collect::<Vec<_>>();
            Ok(json!({ "done": steps(done), "undone": steps(undone), "limit": app.doc.undo_limit }))
        }
        "quit" => {
            check_discard(app, a)?;
            app.dirty = false;
            app.quit = true;
            Ok(Value::Null)
        }
        _ => Err(format!("unknown op \"{op}\"; one of: {}", OPS.iter().map(|o| o.name).collect::<Vec<_>>().join(", "))),
    }
}

/// Run a command, routing those that would open a dialog to the file ops.
fn command(app: &mut App, a: &Value) -> R<Value> {
    let name = string(a, "name")?;
    // arguments may be nested in "args" or given beside "name"
    let args = match arg(a, "args") {
        Some(Value::Object(m)) => {
            let mut all = a.as_object().cloned().unwrap_or_default();
            all.extend(m.clone());
            Value::Object(all)
        }
        _ => a.clone(),
    };
    let a = &args;
    let has_source = arg(a, "path").is_some() || arg(a, "data").is_some();
    match name {
        "open" => return open(app, a),
        "save" | "save_as" if arg(a, "path").is_some() => return save(app, a),
        "save" if app.path.is_some() => return save(app, a),
        "save" | "save_as" => return Err("give a path to save to (or use export with format wetdoc for the bytes)".into()),
        "save_iterative" if app.path.is_none() => return Err("save the document to a path first".into()),
        "export_png" | "export_psd" | "export_heightmap" => {
            let format = match name {
                "export_png" => "png",
                "export_psd" => "psd",
                _ => "exr",
            };
            let mut e = a.clone();
            e["format"] = format.into();
            return export(app, &e);
        }
        "save_preset" => return save_preset(app, a),
        "new_document" => check_discard(app, a).map(|_| app.dirty = false)?,
        "tracing" if !app.doc.layers.iter().any(|l| l.locked && l.name == "reference") => {
            if !has_source {
                return Err("tracing: give the image as path or data the first time".into());
            }
            let mut i = a.clone();
            i["as"] = "tracing".into();
            return import(app, &i);
        }
        "toggle_reference" if app.doc.references.is_empty() => {
            if !has_source {
                return Err("toggle_reference: give the image as path or data the first time".into());
            }
            let mut i = a.clone();
            i["as"] = "reference".into();
            return import(app, &i);
        }
        _ => {}
    }
    let cmd = parse_cmd(name, a)?;
    app.status.clear();
    app.run(cmd);
    status_result(app, Value::Null)
}

/// A shortcut key as `App::key_cmd` reads it: a character, `Shift+<letter>`, or a key name.
fn key(app: &mut App, a: &Value) -> R<Value> {
    use winit::keyboard::{Key, NamedKey};
    let s = string(a, "key")?;
    let k = match s.strip_prefix("Shift+") {
        Some(c) if c.chars().count() == 1 => Key::Character(c.to_uppercase().into()),
        _ if s.chars().count() == 1 => Key::Character(s.into()),
        _ => Key::Named(match s {
            "Space" => NamedKey::Space,
            "Tab" => NamedKey::Tab,
            "Home" => NamedKey::Home,
            "Enter" => NamedKey::Enter,
            "Escape" | "Esc" => NamedKey::Escape,
            "Insert" => NamedKey::Insert,
            "ArrowLeft" | "Left" => NamedKey::ArrowLeft,
            "ArrowRight" | "Right" => NamedKey::ArrowRight,
            "ArrowUp" | "Up" => NamedKey::ArrowUp,
            "ArrowDown" | "Down" => NamedKey::ArrowDown,
            "F1" => NamedKey::F1,
            "F2" => NamedKey::F2,
            "F3" => NamedKey::F3,
            "F4" => NamedKey::F4,
            "F5" => NamedKey::F5,
            "F6" => NamedKey::F6,
            "F7" => NamedKey::F7,
            "F8" => NamedKey::F8,
            "F9" => NamedKey::F9,
            "F10" => NamedKey::F10,
            "F11" => NamedKey::F11,
            "F12" => NamedKey::F12,
            _ => return Err(format!("unknown key \"{s}\"")),
        }),
    };
    let cmd = App::key_cmd(&k).ok_or_else(|| format!("\"{s}\" is not a shortcut"))?;
    let name = cmd_name(&cmd);
    // keys that would open a dialog go through the command op, which takes paths instead
    let mut r = if ROUTED.contains(&name) {
        command(app, &json!({ "name": name, "args": a }))?
    } else {
        app.status.clear();
        app.run(cmd);
        status_result(app, Value::Null)?
    };
    if r.is_null() {
        r = json!({});
    }
    r["command"] = name.into();
    Ok(r)
}

/// Commands that open a dialog or may discard the document: the command op handles them.
const ROUTED: [&str; 11] = [
    "open", "save", "save_as", "save_iterative", "export_png", "export_psd", "export_heightmap",
    "save_preset", "new_document", "tracing", "toggle_reference",
];

fn gesture(app: &mut App, a: &Value) -> R<Value> {
    let pts = gesture_points(need(a, "points")?)?;
    let steps = arg(a, "steps_per_point").map_or(Ok(0), |_| uint(a, "steps_per_point"))?.min(1000);
    let saved = app.pressure;
    app.free_corner = bool_or(a, "free_corner", false)?;
    let to = |app: &mut App, ((x, y), p): GesturePoint| {
        app.cursor = (x, y);
        app.pressure = p.or(saved);
    };
    let hold = bool_or(a, "hold", false)?;
    // a held gesture goes on from where it was: its first point is a move, not a press
    let rest = if app.held {
        &pts[..]
    } else {
        to(app, pts[0]);
        app.press();
        &pts[1..]
    };
    for &p in rest {
        if steps > 0 {
            app.flush_stroke(false);
            app.step_sim(steps);
        }
        to(app, p);
        app.moved();
    }
    app.held = hold;
    if hold {
        // the paint so far shows in renders and flows with `step` until the gesture goes on
        app.flush_stroke(false);
    } else {
        app.release();
        app.free_corner = false;
    }
    app.pressure = saved;
    app.epoch += 1;
    Ok(json!({ "points": pts.len(), "layer": app.doc.active, "held": hold }))
}

fn select(app: &mut App, a: &Value) -> R<Value> {
    app.release_gpu();
    app.epoch += 1;
    let op = arg(a, "combine").map_or(Ok(SelectOp::Replace), |v| parse_enum(&SELECT_OPS, str_of(v)?, "combine"))?;
    let shape = string(a, "shape")?;
    match shape {
        "all" => app.doc.select_all(),
        "none" => app.doc.deselect(),
        "invert" => app.doc.invert_selection(),
        "restore" => app.doc.restore_selection(),
        "rect" | "ellipse" => {
            let (p, q) = (point(need(a, "from")?)?, point(need(a, "to")?)?);
            if op == SelectOp::Replace {
                app.doc.select_shape(p, q, shape == "ellipse");
            } else {
                let (c, r) = (((p.0 + q.0) / 2.0, (p.1 + q.1) / 2.0), ((q.0 - p.0).abs() / 2.0, (q.1 - p.1).abs() / 2.0));
                let poly: Vec<(f32, f32)> = if shape == "rect" {
                    vec![p, (q.0, p.1), q, (p.0, q.1)]
                } else {
                    (0..256).map(|k| (k as f32 / 256.0 * std::f32::consts::TAU).sin_cos()).map(|(s, co)| (c.0 + r.0 * co, c.1 + r.1 * s)).collect()
                };
                app.doc.select_polygon(&poly, op);
            }
        }
        "polygon" => app.doc.select_polygon(&points2(need(a, "points")?)?, op),
        "color" => {
            let (x, y) = point(need(a, "at")?)?;
            if x < 0.0 || y < 0.0 || x as usize >= app.doc.w || y as usize >= app.doc.h {
                return Err("at is outside the canvas".into());
            }
            let tol = arg(a, "tolerance").map_or(Ok(32), |_| uint(a, "tolerance"))?.min(255) as u8;
            app.doc.select_color((x as usize, y as usize), tol, bool_or(a, "contiguous", true)?, op);
        }
        "visible" => {
            let i = layer_index(app, a)?;
            app.doc.select_visible(i, op);
        }
        _ => return Err(format!("unknown shape \"{shape}\"; one of: all, none, invert, restore, rect, ellipse, polygon, color, visible")),
    }
    app.dirty = true;
    Ok(json!({ "selected": selection_box(app) }))
}

/// The selection's bounding box `[x0, y0, x1, y1]` (exclusive), or `null`.
fn selection_box(app: &App) -> Value {
    let Some(m) = &app.doc.selection else { return Value::Null };
    let w = app.doc.w;
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, _) in m.iter().enumerate().filter(|(_, v)| **v > 0) {
        let (x, y) = (i % w, i / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
    }
    if x0 == usize::MAX { json!([0, 0, 0, 0]) } else { json!([x0, y0, x1, y1]) }
}

fn fill(app: &mut App, a: &Value) -> R<Value> {
    let (x, y) = point(need(a, "at")?)?;
    if x < 0.0 || y < 0.0 || x as usize >= app.doc.w || y as usize >= app.doc.h {
        return Err("at is outside the canvas".into());
    }
    let i = app.doc.active;
    if app.doc.layers[i].locked {
        return Err(format!("layer \"{}\" is locked", app.doc.layers[i].name));
    }
    let fill = match arg(a, "gradient") {
        Some(g) => {
            let cs = need(g, "colors")?.as_array().ok_or("gradient.colors: [c0, c1]")?;
            if cs.len() != 2 {
                return Err("gradient.colors needs two colours".into());
            }
            Fill::Gradient { from: point(need(g, "from")?)?, to: point(need(g, "to")?)?, colors: [color(&cs[0])?, color(&cs[1])?], radial: bool_or(g, "radial", false)? }
        }
        None => Fill::Color(arg(a, "color").map_or(Ok(app.brush.color), color)?),
    };
    let opts = FillOptions {
        tolerance: arg(a, "tolerance").map_or(Ok(32), |_| uint(a, "tolerance"))?.min(255) as u8,
        opacity: num_or(a, "opacity", app.brush.params.opacity.min(1.0))?.clamp(0.0, 1.0),
        contiguous: bool_or(a, "contiguous", true)?,
        wet: bool_or(a, "wet", false)?,
    };
    app.release_gpu();
    app.epoch += 1;
    app.doc.checkpoint_as("Fill");
    app.doc.fill((x as usize, y as usize), &fill, opts);
    app.dirty = true;
    Ok(json!({ "layer": i }))
}

fn transform(app: &mut App, a: &Value) -> R<Value> {
    let i = app.doc.active;
    if app.doc.layers[i].locked {
        return Err(format!("layer \"{}\" is locked", app.doc.layers[i].name));
    }
    if let Some(c) = arg(a, "corners") {
        let c = points2(c)?;
        let corners: [(f32, f32); 4] = c.try_into().map_err(|_| "corners needs four points")?;
        app.release_gpu();
        app.epoch += 1;
        app.doc.checkpoint_as("Transform");
        app.doc.perspective_layer(i, corners);
        app.dirty = true;
        return Ok(json!({ "layer": i }));
    }
    app.run(parse_cmd("transform", a)?);
    Ok(json!({ "layer": i }))
}

fn resize(app: &mut App, a: &Value) -> R<Value> {
    let (w, h) = (uint(a, "width")?, uint(a, "height")?);
    if !(8..=8192).contains(&w) || !(8..=8192).contains(&h) {
        return Err("width and height must be 8–8192".into());
    }
    // the GPU simulation holds buffers of the old size
    let gpu = app.gpu.is_some();
    if gpu {
        app.run(Cmd::ToggleGpu);
    }
    app.release_gpu();
    if bool_or(a, "resample", false)? {
        app.doc.resize_image(w, h);
    } else {
        let (x, y) = arg(a, "offset").map_or(Ok((0.0, 0.0)), point)?;
        app.doc.resize_canvas(w, h, (x as i64, y as i64));
    }
    if gpu {
        app.run(Cmd::ToggleGpu);
    }
    app.epoch += 1;
    app.dirty = true;
    app.zoom = if app.render_state.is_some() { 0.0 } else { app.zoom };
    Ok(json!({ "width": app.doc.w, "height": app.doc.h }))
}

/// The painting as displayed (pigment mixing included), straight RGBA8.
fn composite(app: &App) -> Vec<u8> {
    let km = app.pigment.then(wet_pigment::KmMixer::new);
    app.doc.render(km.as_ref().map(|p| p as &dyn wet_compose::PigmentMix))
}

fn render(app: &mut App, a: &Value) -> R<Value> {
    app.release_gpu();
    let (w, h) = (app.doc.w, app.doc.h);
    let rgba = if arg(a, "layer").is_some() {
        let i = layer_index(app, a)?;
        wet_fluid::render::render_layer(&app.doc.layers[i].wet)
            .iter()
            .flat_map(|p| {
                let c = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
                [c(p[0]), c(p[1]), c(p[2]), c(p[3])]
            })
            .collect()
    } else {
        let mut img = composite(app);
        if bool_or(a, "view", false)? {
            crate::ui::decorate(app, &mut img);
        }
        img
    };
    let max = arg(a, "max_size").map(|_| uint(a, "max_size")).transpose()?;
    let (bytes, w, h) = png(rgba, w, h, max)?;
    let mut r = deliver(a, bytes, "painting.png")?;
    if let Some(d) = r.as_object_mut().and_then(|m| m.remove("data")) {
        r["png"] = d;
    }
    r["width"] = w.into();
    r["height"] = h.into();
    Ok(r)
}

fn open(app: &mut App, a: &Value) -> R<Value> {
    let (name, path, bytes) = source(a)?;
    if name.to_lowercase().ends_with(".wetdoc") {
        check_discard(app, a)?;
        app.dirty = false;
    }
    app.status.clear();
    app.open_bytes(&name, path.as_deref(), &bytes);
    status_result(app, json!({ "width": app.doc.w, "height": app.doc.h, "layers": app.doc.layers.len() }))
}

fn import(app: &mut App, a: &Value) -> R<Value> {
    let (name, _, bytes) = source(a)?;
    let what = arg(a, "as").map_or(Ok("layer"), str_of)?;
    app.release_gpu();
    app.epoch += 1;
    if what == "stencil" {
        let m = wet_brush::Mask::from_png(&bytes).map_err(|e| format!("a stencil is a PNG: {e}"))?;
        let mask = m.data.iter().map(|&v| (v * 255.0).round() as u8).collect();
        let centre = (app.doc.w as f32 / 2.0, app.doc.h as f32 / 2.0);
        app.doc.stencils.push(wet_doc::Stencil::new(mask, m.w, m.h, centre));
        app.doc.update_stencils();
        app.dirty = true;
        return Ok(json!({ "stencils": app.doc.stencils.len() }));
    }
    let (rgba, w, h) = decode_rgba(&bytes).ok_or_else(|| format!("{name} is not an image wetpaint reads"))?;
    match what {
        "layer" => {
            app.doc.checkpoint_as("Import image");
            let stem = Path::new(&name).file_stem().and_then(|n| n.to_str()).unwrap_or("image").to_owned();
            app.doc.import_image(&stem, &rgba, w, h);
        }
        "tracing" => app.import_tracing(&rgba, w, h),
        "reference" => app.import_reference(rgba, w, h),
        _ => return Err(format!("import as layer, tracing, reference or stencil, not \"{what}\"")),
    }
    app.dirty = true;
    Ok(json!({ "layers": app.doc.layers.len(), "active": app.doc.active }))
}

fn save(app: &mut App, a: &Value) -> R<Value> {
    let path = match arg(a, "path") {
        Some(p) => PathBuf::from(str_of(p)?),
        None => app.path.clone().ok_or("the document has no file yet: give a path")?,
    };
    app.release_gpu();
    app.status.clear();
    app.save_to(Target::Path(path.clone()));
    status_result(app, json!({ "path": path }))
}

fn export(app: &mut App, a: &Value) -> R<Value> {
    let format = string(a, "format")?;
    app.release_gpu();
    let mut buf = Cursor::new(Vec::new());
    let name = arg(a, "name").map_or(Ok(app.preset.clone().unwrap_or_else(|| "My Brush".into())), |v| str_of(v).map(str::to_owned))?;
    let file = match format {
        "png" => app.doc.export_png(&mut buf).map(|_| "painting.png".to_owned()),
        "psd" => app.doc.export_psd(&mut buf).map(|_| "painting.psd".to_owned()),
        "exr" | "heightmap" => app.doc.export_heightmap_exr(&mut buf).map(|_| "painting_height.exr".to_owned()),
        "wetdoc" => app.doc.save(&mut buf).map(|_| "painting.wetdoc".to_owned()),
        "wetbrush" => app.preset_bytes(&name).map(|b| {
            buf = Cursor::new(b);
            format!("{name}.wetbrush")
        }),
        _ => return Err(format!("unknown format \"{format}\"; one of: png, psd, exr, wetdoc, wetbrush")),
    }
    .map_err(|e| format!("export failed: {e}"))?;
    deliver(a, buf.into_inner(), &file)
}

fn tool_arg(app: &App, a: &Value) -> R<Tool> {
    arg(a, "tool").map_or(Ok(app.brush.tool), |t| parse_tool(str_of(t)?, a))
}

fn presets(app: &App, a: &Value) -> R<Value> {
    let tool = tool_arg(app, a)?;
    let Some(folder) = crate::library::folder_for(tool) else { return Ok(json!({ "presets": [], "note": "this tool has no presets" })) };
    let root = crate::library::root().ok_or("no brush library folder")?;
    let list: Vec<Value> = crate::library::categories(&root, folder)
        .iter()
        .flat_map(|c| c.presets.iter().map(move |p| json!({ "category": c.name, "name": p.name, "path": p.path })))
        .collect();
    Ok(json!({ "tool": tool_name(tool), "library": root.join(folder), "presets": list }))
}

fn load_preset(app: &mut App, a: &Value) -> R<Value> {
    app.status.clear();
    if arg(a, "path").is_some() || arg(a, "data").is_some() {
        let (name, _, bytes) = source(a)?;
        let name = crate::library::preset_name(Path::new(&name)).unwrap_or(name);
        app.load_preset_bytes(&name, &bytes);
    } else {
        let want = string(a, "name")?;
        let found = presets(app, a)?["presets"].as_array().and_then(|l| l.iter().find(|p| p["name"] == want).and_then(|p| p["path"].as_str().map(PathBuf::from)));
        app.load_preset(&found.ok_or_else(|| format!("no preset \"{want}\" for this tool in the library"))?);
    }
    status_result(app, json!({ "preset": app.preset }))
}

fn save_preset(app: &mut App, a: &Value) -> R<Value> {
    let name = arg(a, "name").map_or(Ok(app.preset.clone().unwrap_or_else(|| "My Brush".into())), |v| str_of(v).map(str::to_owned))?;
    if bool_or(a, "library", false)? {
        let folder = crate::library::folder_for(app.brush.tool).ok_or("this tool has no presets")?;
        let dir = crate::library::root().ok_or("no brush library folder")?.join(folder).join("My Brushes");
        std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let path = dir.join(format!("{name}.wetbrush"));
        app.save_preset(&path).map_err(|e| format!("cannot save {}: {e}", path.display()))?;
        return Ok(json!({ "path": path }));
    }
    let bytes = app.preset_bytes(&name).map_err(|e| e.to_string())?;
    deliver(a, bytes, &format!("{name}.wetbrush"))
}

// ---- settings ----

/// A setting: its key, value and whether it can be set.
type Setting = (String, Value, bool);

fn brush_fields(p: &BrushParams) -> Vec<(&'static str, Value)> {
    // destructured field by field: a new parameter must be listed here to build
    let BrushParams {
        size, opacity, water, spacing, spacing_jitter, size_jitter, opacity_jitter, rotation, rotation_jitter,
        scatter_jitter, splat_opacity, pressure_size_influence, pressure_opacity_influence, rotation_mode,
        paint_mode, glaze_blend, glaze_mode, only_water, dirty, grain_scale, grain_brush_size,
        grain_follow_brush_size, grain_follow_distance, grain_follow_amount, grain_follow_shape_rotation,
        splat_smudge, pressure_multiplier, pressure_max, paper_structure_influence, pressure_size_curve,
        pressure_opacity_curve,
    } = p;
    vec![
        ("size", json!(size)),
        ("opacity", json!(opacity)),
        ("water", json!(water)),
        ("spacing", json!(spacing)),
        ("spacing_jitter", json!(spacing_jitter)),
        ("size_jitter", json!(size_jitter)),
        ("opacity_jitter", json!(opacity_jitter)),
        ("rotation", json!(rotation)),
        ("rotation_jitter", json!(rotation_jitter)),
        ("scatter_jitter", json!(scatter_jitter)),
        ("splat_opacity", json!(splat_opacity)),
        ("pressure_size_influence", json!(pressure_size_influence)),
        ("pressure_opacity_influence", json!(pressure_opacity_influence)),
        ("rotation_mode", name_of(rotation_mode).into()),
        ("paint_mode", name_of(paint_mode).into()),
        ("glaze_blend", name_of(glaze_blend).into()),
        ("glaze_mode", name_of(glaze_mode).into()),
        ("only_water", json!(only_water)),
        ("dirty", json!(dirty)),
        ("grain_scale", json!(grain_scale)),
        ("grain_brush_size", json!(grain_brush_size)),
        ("grain_follow_brush_size", json!(grain_follow_brush_size)),
        ("grain_follow_distance", json!(grain_follow_distance)),
        ("grain_follow_amount", json!(grain_follow_amount)),
        ("grain_follow_shape_rotation", json!(grain_follow_shape_rotation)),
        ("splat_smudge", json!(splat_smudge)),
        ("pressure_multiplier", json!(pressure_multiplier)),
        ("pressure_max", json!(pressure_max)),
        ("paper_structure_influence", json!(paper_structure_influence)),
        ("pressure_size_curve", curve_json(pressure_size_curve)),
        ("pressure_opacity_curve", curve_json(pressure_opacity_curve)),
    ]
}

fn set_brush_field(p: &mut BrushParams, k: &str, v: &Value) -> R<()> {
    let f = || f32_of(v);
    let unit = || f32_of(v).map(|x| x.clamp(0.0, 1.0));
    match k {
        "size" => p.size = f()?.clamp(1.0, 1000.0),
        "opacity" => p.opacity = f()?.clamp(0.05, 1.0),
        "water" => p.water = unit()?,
        "spacing" => p.spacing = f()?.clamp(0.01, 2.0),
        "spacing_jitter" => p.spacing_jitter = unit()?,
        "size_jitter" => p.size_jitter = unit()?,
        "opacity_jitter" => p.opacity_jitter = unit()?,
        "rotation" => p.rotation = f()?,
        "rotation_jitter" => p.rotation_jitter = unit()?,
        "scatter_jitter" => p.scatter_jitter = unit()?,
        "splat_opacity" => p.splat_opacity = unit()?,
        "pressure_size_influence" => p.pressure_size_influence = unit()?,
        "pressure_opacity_influence" => p.pressure_opacity_influence = unit()?,
        "rotation_mode" => p.rotation_mode = parse_enum(&ROTATION_MODES, str_of(v)?, "rotation mode")?,
        "paint_mode" => p.paint_mode = parse_enum(&PAINT_MODES, str_of(v)?, "paint mode")?,
        "glaze_blend" => p.glaze_blend = parse_enum(&MixMode::ALL, str_of(v)?, "blend mode")?,
        "glaze_mode" => p.glaze_mode = parse_enum(&GLAZE_MODES, str_of(v)?, "glaze mode")?,
        "only_water" => p.only_water = bool_of(v)?,
        "dirty" => p.dirty = bool_of(v)?,
        "grain_scale" => p.grain_scale = f()?.clamp(0.01, 100.0),
        "grain_brush_size" => p.grain_brush_size = f()?.max(0.0),
        "grain_follow_brush_size" => p.grain_follow_brush_size = bool_of(v)?,
        "grain_follow_distance" => p.grain_follow_distance = bool_of(v)?,
        "grain_follow_amount" => p.grain_follow_amount = unit()?,
        "grain_follow_shape_rotation" => p.grain_follow_shape_rotation = bool_of(v)?,
        "splat_smudge" => p.splat_smudge = bool_of(v)?,
        "pressure_multiplier" => p.pressure_multiplier = f()?.clamp(0.01, 100.0),
        "pressure_max" => p.pressure_max = f()?,
        "paper_structure_influence" => p.paper_structure_influence = unit()?,
        "pressure_size_curve" => p.pressure_size_curve = curve_of(v)?,
        "pressure_opacity_curve" => p.pressure_opacity_curve = curve_of(v)?,
        _ => return Err(format!("unknown setting brush.{k}")),
    }
    Ok(())
}

fn layer_fields(l: &PaintLayer) -> Vec<(&'static str, Value, bool)> {
    let PaintLayer { name, wet: _, mode, opacity, visible, locked, clip, mask, masking_fluid, group, tracing, pigments, metallic } = l;
    vec![
        ("name", json!(name), true),
        ("mode", name_of(mode).into(), true),
        ("opacity", json!(opacity), true),
        ("visible", json!(visible), true),
        ("locked", json!(locked), true),
        ("clip", json!(clip), true),
        ("mask", json!(mask.is_some()), true),
        ("masking_fluid", json!(masking_fluid), true),
        ("group", json!(group), false),
        ("tracing", json!(tracing), true),
        ("pigments", json!(pigments), true),
        ("metallic", json!(metallic), true),
    ]
}

fn set_layer_field(app: &mut App, i: usize, k: &str, v: &Value) -> R<()> {
    let l = &mut app.doc.layers[i];
    match k {
        "name" => l.name = str_of(v)?.to_owned(),
        "mode" => l.mode = parse_enum(&MixMode::ALL, str_of(v)?, "blend mode")?,
        "opacity" => l.opacity = f32_of(v)?.clamp(0.0, 1.0),
        "visible" => l.visible = bool_of(v)?,
        "locked" => l.locked = bool_of(v)?,
        "clip" => l.clip = bool_of(v)?,
        "mask" => match (bool_of(v)?, l.mask.is_some()) {
            (true, false) => app.doc.add_layer_mask(i),
            (false, true) => l.mask = None,
            _ => {}
        },
        "masking_fluid" => {
            l.masking_fluid = bool_of(v)?;
            app.doc.update_stencils();
        }
        "tracing" => l.tracing = bool_of(v)?,
        "pigments" => app.doc.set_layer_pigments(i, bool_of(v)?),
        "metallic" => l.metallic = bool_of(v)?,
        "group" => return Err("layer.group is read-only: use the group_with_below command".into()),
        _ => return Err(format!("unknown setting layer.{k}")),
    }
    Ok(())
}

fn sim_fields(p: &FluidParams) -> Vec<(&'static str, Value)> {
    let FluidParams {
        diffuse_uv, max_diffuse_uv, max_limit_uv, mass_limit, plaster_flow_limit, paper_struct, paint_struct,
        project_cycles, tilt_active, tilt_x, tilt_y, dryme_edge, absorbency, border_dry, paper_influence, rewet,
        gran_enabled, gran_strength, gran_contrast, dry_everything, create_drops, drops_size, drops_length,
        placeholders: _,
    } = p;
    vec![
        ("diffuse_uv", json!(diffuse_uv)),
        ("max_diffuse_uv", json!(max_diffuse_uv)),
        ("max_limit_uv", json!(max_limit_uv)),
        ("mass_limit", json!(mass_limit)),
        ("plaster_flow_limit", json!(plaster_flow_limit)),
        ("paper_struct", json!(paper_struct)),
        ("paint_struct", json!(paint_struct)),
        ("project_cycles", json!(project_cycles)),
        ("tilt_active", json!(tilt_active)),
        ("tilt_x", json!(tilt_x)),
        ("tilt_y", json!(tilt_y)),
        ("dryme_edge", json!(dryme_edge)),
        ("absorbency", json!(absorbency)),
        ("border_dry", json!(border_dry)),
        ("paper_influence", json!(paper_influence)),
        ("rewet", json!(rewet)),
        ("gran_enabled", json!(gran_enabled)),
        ("gran_strength", json!(gran_strength)),
        ("gran_contrast", json!(gran_contrast)),
        ("dry_everything", json!(dry_everything)),
        ("create_drops", json!(create_drops)),
        ("drops_size", json!(drops_size)),
        ("drops_length", json!(drops_length)),
    ]
}

fn set_sim_field(p: &mut FluidParams, k: &str, v: &Value) -> R<()> {
    let f = || f32_of(v);
    match k {
        "diffuse_uv" => p.diffuse_uv = f()?,
        "max_diffuse_uv" => p.max_diffuse_uv = f()?,
        "max_limit_uv" => p.max_limit_uv = f()?,
        "mass_limit" => p.mass_limit = f()?,
        "plaster_flow_limit" => p.plaster_flow_limit = f()?,
        "paper_struct" => p.paper_struct = f()?,
        "paint_struct" => p.paint_struct = f()?,
        "project_cycles" => p.project_cycles = v.as_u64().ok_or("project_cycles: a whole number")?.min(1000) as u32,
        "tilt_active" => p.tilt_active = bool_of(v)?,
        "tilt_x" => p.tilt_x = f()?.clamp(-1.0, 1.0),
        "tilt_y" => p.tilt_y = f()?.clamp(-1.0, 1.0),
        "dryme_edge" => p.dryme_edge = f()?,
        "absorbency" => p.absorbency = f()?,
        "border_dry" => p.border_dry = f()?,
        "paper_influence" => p.paper_influence = f()?,
        "rewet" => p.rewet = f()?,
        "gran_enabled" => p.gran_enabled = bool_of(v)?,
        "gran_strength" => p.gran_strength = f()?,
        "gran_contrast" => p.gran_contrast = f()?,
        "dry_everything" => p.dry_everything = bool_of(v)?,
        "create_drops" => p.create_drops = bool_of(v)?,
        "drops_size" => p.drops_size = f()?,
        "drops_length" => p.drops_length = f()?,
        _ => return Err(format!("unknown setting layer.sim.{k}")),
    }
    Ok(())
}

fn paper_fields(p: &PaperSettings) -> Vec<(&'static str, Value)> {
    let PaperSettings { spec, color, visible, shading, tone, deckled } = p;
    let PaperSpec { surface, seed, scale, relief, absorbency, absorbency_variation } = spec;
    vec![
        ("surface", name_of(surface).into()),
        ("seed", json!(seed)),
        ("scale", json!(scale)),
        ("relief", json!(relief)),
        ("absorbency", json!(absorbency)),
        ("absorbency_variation", json!(absorbency_variation)),
        ("color", hex(*color)),
        ("visible", json!(visible)),
        ("shading", json!(shading)),
        ("tone", json!(tone)),
        ("deckled", json!(deckled)),
    ]
}

fn set_paper_field(p: &mut PaperSettings, k: &str, v: &Value) -> R<()> {
    let f = || f32_of(v);
    match k {
        "surface" => p.spec.surface = parse_enum(&SURFACES, str_of(v)?, "surface")?,
        "seed" => p.spec.seed = v.as_u64().ok_or("seed: a whole number")?,
        "scale" => p.spec.scale = f()?.clamp(0.05, 20.0),
        "relief" => p.spec.relief = f()?.clamp(0.0, 10.0),
        "absorbency" => p.spec.absorbency = f()?.clamp(0.0, 1.0),
        "absorbency_variation" => p.spec.absorbency_variation = f()?.clamp(0.0, 1.0),
        "color" => p.color = color(v)?,
        "visible" => p.visible = bool_of(v)?,
        "shading" => p.shading = f()?,
        "tone" => p.tone = f()?,
        "deckled" => p.deckled = bool_of(v)?,
        _ => return Err(format!("unknown setting paper.{k}")),
    }
    Ok(())
}

fn group_fields(g: &LayerGroup) -> Vec<(&'static str, Value)> {
    let LayerGroup { name, mode, opacity, visible } = g;
    vec![("name", json!(name)), ("mode", name_of(mode).into()), ("opacity", json!(opacity)), ("visible", json!(visible))]
}

fn set_group_field(g: &mut LayerGroup, k: &str, v: &Value) -> R<()> {
    match k {
        "name" => g.name = str_of(v)?.to_owned(),
        "mode" => g.mode = parse_enum(&MixMode::ALL, str_of(v)?, "blend mode")?,
        "opacity" => g.opacity = f32_of(v)?.clamp(0.0, 1.0),
        "visible" => g.visible = bool_of(v)?,
        _ => return Err(format!("unknown setting group.{k}")),
    }
    Ok(())
}

fn ruler_name(r: &Option<Ruler>) -> &'static str {
    match r {
        None => "off",
        Some(Ruler::Line { a, b, .. }) if a.1 == b.1 => "horizontal",
        Some(Ruler::Line { .. }) => "vertical",
        Some(Ruler::Ellipse { .. }) => "ellipses",
        Some(Ruler::Perspective { .. }) => "perspective",
    }
}

fn grid_json(g: &Option<Grid>) -> Value {
    match g {
        None => Value::Null,
        Some(Grid::Spacing { spacing, subdivisions }) => json!({ "spacing": spacing, "subdivisions": subdivisions }),
        Some(Grid::Evenly { columns, rows }) => json!({ "columns": columns, "rows": rows }),
    }
}

fn grid_of(v: &Value) -> R<Option<Grid>> {
    if v.is_null() || v == &json!(false) {
        return Ok(None);
    }
    if v == &json!(true) {
        return Ok(Some(Grid::Spacing { spacing: 64.0, subdivisions: 4 }));
    }
    let u = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as u32);
    if let (Some(columns), Some(rows)) = (u("columns"), u("rows")) {
        return Ok(Some(Grid::Evenly { columns: columns.max(1), rows: rows.max(1) }));
    }
    let spacing = f32_of(v.get("spacing").ok_or("grid: {\"spacing\", \"subdivisions\"}, {\"columns\", \"rows\"}, true or null")?)?.max(2.0);
    Ok(Some(Grid::Spacing { spacing, subdivisions: u("subdivisions").unwrap_or(1).max(1) }))
}

/// Every setting for layer `li` and group `gi`.
fn settings(app: &App, li: usize, gi: Option<usize>) -> Vec<Setting> {
    let mut s: Vec<Setting> = Vec::new();
    let mut put = |k: String, v: Value, w: bool| s.push((k, v, w));
    let b = &app.brush;
    put("brush.tool".into(), tool_name(b.tool).into(), true);
    if let Tool::Clone { dx, dy } = b.tool {
        put("brush.clone_offset".into(), json!([dx, dy]), true);
    }
    put("brush.color".into(), hex(b.color), true);
    put("brush.color_index".into(), json!(app.color), true);
    put("brush.multi".into(), Value::Array(b.multi.iter().map(|c| hex(*c)).collect()), true);
    put("brush.preset".into(), json!(app.preset), false);
    put("brush.pressure".into(), json!(app.pressure), true);
    for (k, v) in brush_fields(&b.params) {
        put(format!("brush.{k}"), v, true);
    }
    let l = &app.doc.layers[li];
    for (k, v, w) in layer_fields(l) {
        put(format!("layer.{k}"), v, w);
    }
    for (k, v) in sim_fields(&l.wet.params) {
        put(format!("layer.sim.{k}"), v, true);
    }
    if let Some(g) = gi.and_then(|g| app.doc.groups.get(g)) {
        for (k, v) in group_fields(g) {
            put(format!("group.{k}"), v, true);
        }
    }
    for (k, v) in paper_fields(&app.doc.paper) {
        put(format!("paper.{k}"), v, true);
    }
    let d = &app.doc;
    put("doc.width".into(), json!(d.w), false);
    put("doc.height".into(), json!(d.h), false);
    put("doc.layers".into(), json!(d.layers.len()), false);
    put("doc.active_layer".into(), json!(d.active), true);
    put("doc.impasto".into(), json!(d.impasto), true);
    put("doc.gloss".into(), json!(d.gloss), true);
    put("doc.soft_shadows".into(), json!(d.soft_shadows), true);
    put("doc.shadow_altitude".into(), json!(d.shadow_altitude), true);
    put("doc.metallic_reflectivity".into(), json!(d.metallic.reflectivity), true);
    put("doc.metallic_scale".into(), json!(d.metallic.scale), true);
    put("doc.metallic_strength".into(), json!(d.metallic.strength), true);
    put("doc.undo_limit".into(), json!(d.undo_limit), true);
    put("doc.grid".into(), grid_json(&d.guides.grid), true);
    put("doc.snap_to_grid".into(), json!(d.guides.snap_to_grid), true);
    put("doc.guides_vertical".into(), json!(d.guides.vertical), true);
    put("doc.guides_horizontal".into(), json!(d.guides.horizontal), true);
    put("doc.snap_to_guides".into(), json!(d.guides.snap_to_guides), true);
    put("doc.selection".into(), selection_box(app), false);
    put("doc.stencil_visible".into(), json!(d.stencils.first().is_some_and(|s| s.visible)), true);
    put("doc.stencil_invert".into(), json!(d.stencils.first().is_some_and(|s| s.invert)), true);
    put("doc.reference_visible".into(), json!(d.references.iter().any(|r| r.visible)), true);
    put("doc.path".into(), json!(app.path), false);
    put("doc.dirty".into(), json!(app.dirty), false);
    put("sim.paused".into(), json!(app.paused), true);
    put("sim.gpu".into(), json!(app.gpu.is_some()), true);
    put("sim.gpu_adapter".into(), json!(app.gpu.as_ref().map(|g| g.adapter.clone())), false);
    put("sim.rate".into(), json!(app.sim_rate), false);
    put("sim.pigment".into(), json!(app.pigment), true);
    put("sim.natural".into(), json!(app.natural), true);
    put("sim.visual_preset".into(), json!(app.visual_preset), true);
    put("view.zoom".into(), json!(app.zoom), true);
    put("view.pan".into(), json!([app.pan.0, app.pan.1]), true);
    put("view.greyscale".into(), json!(app.greyscale), true);
    put("view.mode".into(), name_of(&app.mode).into(), true);
    put("view.ruler".into(), ruler_name(&app.ruler).into(), true);
    put("view.transforming".into(), json!(app.xform.is_some()), false);
    put("app.recording".into(), json!(app.recording.is_some()), true);
    put("app.status".into(), json!(app.status), false);
    s
}

fn group_arg(app: &App, a: &Value, li: usize) -> R<Option<usize>> {
    match arg(a, "group") {
        Some(v) => {
            let g = v.as_u64().map(|g| g as usize).filter(|&g| g < app.doc.groups.len()).ok_or_else(|| format!("no group {v}"))?;
            Ok(Some(g))
        }
        None => Ok(app.doc.layers[li].group),
    }
}

fn get(app: &mut App, a: &Value) -> R<Value> {
    let li = layer_index(app, a)?;
    let gi = group_arg(app, a, li)?;
    let all = settings(app, li, gi);
    match arg(a, "key") {
        Some(k) => {
            let k = str_of(k)?;
            all.into_iter().find(|s| s.0 == k).map(|s| json!({ "key": k, "value": s.1 })).ok_or_else(|| format!("unknown setting \"{k}\" (describe lists them)"))
        }
        None => Ok(json!({ "layer": li, "group": gi, "settings": Map::from_iter(all.into_iter().map(|s| (s.0, s.1))) })),
    }
}

fn set(app: &mut App, a: &Value) -> R<Value> {
    let li = layer_index(app, a)?;
    let gi = group_arg(app, a, li)?;
    let pairs: Vec<(String, Value)> = match (arg(a, "values"), arg(a, "key")) {
        (Some(Value::Object(m)), _) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        (None, Some(k)) => vec![(str_of(k)?.to_owned(), a.get("value").cloned().unwrap_or(Value::Null))],
        _ => return Err("set needs key and value, or values".into()),
    };
    app.release_gpu();
    app.epoch += 1;
    for (k, v) in &pairs {
        set_one(app, k, v, li, gi).map_err(|e| format!("{k}: {e}"))?;
    }
    let after = settings(app, li, gi);
    Ok(json!({ "values": Map::from_iter(pairs.iter().filter_map(|(k, _)| after.iter().find(|s| &s.0 == k).map(|s| (k.clone(), s.1.clone())))) }))
}

fn set_one(app: &mut App, key: &str, v: &Value, li: usize, gi: Option<usize>) -> R<()> {
    // a toggle runs its command when the value changes
    let toggle = |app: &mut App, on: bool, want: &Value, cmd: Cmd| -> R<()> {
        if bool_of(want)? != on {
            app.run(cmd);
        }
        Ok(())
    };
    if let Some(k) = key.strip_prefix("brush.") {
        return match k {
            "tool" => {
                app.run(Cmd::Tool(parse_tool(str_of(v)?, &Value::Null)?));
                Ok(())
            }
            "clone_offset" => {
                let (dx, dy) = point(v)?;
                app.brush.set_tool(Tool::Clone { dx, dy });
                Ok(())
            }
            "color" => {
                app.set_color(color(v)?);
                Ok(())
            }
            "color_index" => {
                let k = v.as_u64().map(|k| k as usize).filter(|&k| k < PALETTE.len()).ok_or(format!("a palette index below {}", PALETTE.len()))?;
                app.color = k;
                app.set_color(PALETTE[k]);
                Ok(())
            }
            "multi" => {
                app.brush.multi = v.as_array().ok_or("a list of colours")?.iter().map(color).collect::<R<_>>()?;
                Ok(())
            }
            "pressure" => {
                app.pressure = if v.is_null() { None } else { Some(f32_of(v)?.clamp(0.05, 1.0)) };
                Ok(())
            }
            "preset" => Err("read-only: use load_preset".into()),
            _ => set_brush_field(&mut app.brush.params, k, v),
        };
    }
    if let Some(k) = key.strip_prefix("layer.sim.") {
        app.dirty = true;
        return set_sim_field(&mut app.doc.layers[li].wet.params, k, v);
    }
    if let Some(k) = key.strip_prefix("layer.") {
        app.dirty = true;
        return set_layer_field(app, li, k, v);
    }
    if let Some(k) = key.strip_prefix("group.") {
        let g = gi.ok_or("the layer is in no group: pass group")?;
        app.dirty = true;
        return set_group_field(&mut app.doc.groups[g], k, v);
    }
    if let Some(k) = key.strip_prefix("paper.") {
        let mut p = app.doc.paper;
        set_paper_field(&mut p, k, v)?;
        app.dirty = true;
        app.doc.set_paper(p);
        return Ok(());
    }
    let f = || f32_of(v);
    match key {
        "doc.active_layer" => {
            let n = app.doc.layers.len();
            app.doc.active = v.as_u64().map(|i| i as usize).filter(|&i| i < n).ok_or(format!("a layer index below {n}"))?;
            return Ok(());
        }
        "doc.impasto" => app.doc.impasto = f()?.clamp(0.0, 3.0),
        "doc.gloss" => app.doc.gloss = f()?.max(0.0),
        "doc.soft_shadows" => app.doc.soft_shadows = f()?.clamp(0.0, 10.0),
        "doc.shadow_altitude" => app.doc.shadow_altitude = f()?.clamp(0.0, 10.0),
        "doc.metallic_reflectivity" => app.doc.metallic.reflectivity = f()?.clamp(0.0, 10.0),
        "doc.metallic_scale" => app.doc.metallic.scale = f()?.clamp(0.0, 10.0),
        "doc.metallic_strength" => app.doc.metallic.strength = f()?.clamp(0.0, 10.0),
        "doc.undo_limit" => app.doc.undo_limit = v.as_u64().ok_or("a whole number")?.clamp(1, 1000) as usize,
        "doc.grid" => {
            app.doc.guides.grid = grid_of(v)?;
            app.doc.guides.snap_to_grid = app.doc.guides.grid.is_some();
        }
        "doc.snap_to_grid" => app.doc.guides.snap_to_grid = bool_of(v)?,
        "doc.guides_vertical" => app.doc.guides.vertical = floats(v)?,
        "doc.guides_horizontal" => app.doc.guides.horizontal = floats(v)?,
        "doc.snap_to_guides" => app.doc.guides.snap_to_guides = bool_of(v)?,
        "doc.stencil_visible" => {
            let on = app.doc.stencils.first().is_some_and(|s| s.visible);
            toggle(app, on, v, Cmd::ToggleStencil)?;
        }
        "doc.stencil_invert" => {
            if app.doc.stencils.is_empty() {
                return if bool_of(v)? { Err("no stencil yet: show one first".into()) } else { Ok(()) };
            }
            let on = app.doc.stencils[0].invert;
            toggle(app, on, v, Cmd::InvertStencil)?;
        }
        "doc.reference_visible" => {
            let r = app.doc.references.first_mut().ok_or("no reference image: import one")?;
            r.visible = bool_of(v)?;
            return Ok(());
        }
        "sim.paused" => {
            app.paused = bool_of(v)?;
            return Ok(());
        }
        "sim.gpu" => {
            let on = app.gpu.is_some();
            toggle(app, on, v, Cmd::ToggleGpu)?;
            if bool_of(v)? && app.gpu.is_none() {
                return Err(app.status.clone());
            }
            return Ok(());
        }
        "sim.pigment" => return toggle(app, app.pigment, v, Cmd::TogglePigment),
        "sim.natural" => return toggle(app, app.natural, v, Cmd::ToggleNatural),
        "sim.visual_preset" => {
            let ps = wet_doc::visual::presets();
            let k = v.as_u64().map(|k| k as usize).filter(|&k| k < ps.len()).ok_or(format!("a preset index below {}", ps.len()))?;
            app.visual_preset = k;
            app.doc.apply_visual_preset(&ps[k]);
            app.status = format!("watercolour behaviour: {}", ps[k].name);
        }
        "view.zoom" => {
            app.zoom = f()?.clamp(0.0, 64.0);
            return Ok(());
        }
        "view.pan" => {
            app.pan = point(v)?;
            return Ok(());
        }
        "view.greyscale" => {
            app.greyscale = bool_of(v)?;
            return Ok(());
        }
        "view.mode" => {
            app.run(Cmd::Mode(parse_enum(&MODES, str_of(v)?, "mode")?));
            return Ok(());
        }
        "view.ruler" => {
            let want = str_of(v)?;
            if !RULERS.contains(&want) {
                return Err(format!("one of: {}", RULERS.join(", ")));
            }
            for _ in 0..RULERS.len() {
                if ruler_name(&app.ruler) == want {
                    break;
                }
                app.run(Cmd::CycleRuler);
            }
            return Ok(());
        }
        "app.recording" => return toggle(app, app.recording.is_some(), v, Cmd::ToggleRecording),
        _ => {
            return Err(match settings(app, li, gi).iter().find(|s| s.0 == key) {
                Some(_) => "read-only".into(),
                None => "unknown setting (describe lists them)".into(),
            });
        }
    }
    app.dirty = true;
    Ok(())
}

// ---- overview ----

fn describe(app: &App) -> Value {
    let li = app.doc.active;
    let gi = app.doc.layers[li].group;
    let ops: Vec<Value> = OPS
        .iter()
        .map(|o| json!({ "op": o.name, "doc": o.doc, "args": Map::from_iter(o.args.iter().map(|(n, t, d)| (n.to_string(), json!(format!("{t}: {d}"))))) }))
        .collect();
    let commands: Vec<Value> = COMMANDS.iter().map(|(n, a, d)| json!({ "name": n, "args": a, "doc": d })).collect();
    let settings: Vec<Value> = settings(app, li, gi).into_iter().map(|(k, v, w)| json!({ "key": k, "value": v, "writable": w })).collect();
    json!({
        "about": "wetpaint: a wet-media painting app (watercolour, ink, oil, pastel, pencil, marker, airbrush). Canvas coordinates are pixels from the top-left. Colours are \"#rrggbb\" or [r, g, b] 0–255. The wet simulation advances by `step` (25 steps a second); the windowed app also steps it in real time unless sim.paused.",
        "ops": ops,
        "commands": commands,
        "settings": settings,
        "enums": {
            "tool": tool_names(),
            "mode": names(&MODES),
            "volume": names(&VOLUMES),
            "blend_mode": names(&MixMode::ALL),
            "paint_mode": names(&PAINT_MODES),
            "glaze_mode": names(&GLAZE_MODES),
            "rotation_mode": names(&ROTATION_MODES),
            "surface": names(&SURFACES),
            "select_op": names(&SELECT_OPS),
            "channel": names(&CHANNELS),
            "ruler": RULERS,
            "filter": FILTERS,
            "visual_preset": wet_doc::visual::presets().iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            "palette": PALETTE.iter().map(|c| hex(*c)).collect::<Vec<_>>(),
        },
        "shortcuts": crate::SHORTCUTS,
    })
}

fn state(app: &App) -> Value {
    let d = &app.doc;
    let layers: Vec<Value> = d
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let mut m = Map::from_iter(layer_fields(l).into_iter().map(|(k, v, _)| (k.to_owned(), v)));
            m.insert("index".into(), i.into());
            m.insert("tilted".into(), l.wet.params.tilt_active.into());
            Value::Object(m)
        })
        .collect();
    let groups: Vec<Value> = d.groups.iter().map(|g| Value::Object(Map::from_iter(group_fields(g).into_iter().map(|(k, v)| (k.to_owned(), v))))).collect();
    let (done, undone) = d.history();
    json!({
        "width": d.w,
        "height": d.h,
        "active_layer": d.active,
        "layers": layers,
        "groups": groups,
        "tool": tool_name(app.brush.tool),
        "preset": app.preset,
        "mode": name_of(&app.mode),
        "color": hex(app.brush.color),
        "size": app.brush.params.size,
        "opacity": app.brush.params.opacity,
        "water": app.brush.params.water,
        "selection": selection_box(app),
        "stencils": d.stencils.len(),
        "references": d.references.len(),
        "transforming": app.xform.is_some(),
        "ruler": ruler_name(&app.ruler),
        "paused": app.paused,
        "gpu": app.gpu.is_some(),
        "undo_steps": done.len(),
        "redo_steps": undone.len(),
        "path": app.path,
        "dirty": app.dirty,
        "status": app.status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(app: &mut App, req: Value) -> Value {
        let r = handle(app, &req);
        assert_eq!(r["ok"], true, "{req} → {r}");
        r
    }

    fn painted(app: &App) -> usize {
        let blank = Document::new(app.doc.w, app.doc.h).render(None);
        app.doc.render(None).chunks_exact(4).zip(blank.chunks_exact(4)).filter(|(a, b)| a != b).count()
    }

    use wet_doc::Document;

    #[test]
    fn a_held_stroke_in_pieces_paints_as_one() {
        let pts: Vec<Value> = (0..16).map(|k| json!([10 + k * 8, 40 + (k % 3) * 4, 0.9])).collect();
        let paint = |pieces: &[std::ops::Range<usize>], hold: bool| {
            let mut app = headless(160, 80);
            call(&mut app, json!({ "op": "set", "values": { "brush.tool": "oil", "brush.size": 20 } }));
            for (k, r) in pieces.iter().enumerate() {
                let held = hold && k + 1 < pieces.len();
                let r = call(&mut app, json!({ "op": "stroke", "points": pts[r.clone()], "hold": held }));
                assert_eq!(r["held"], held);
                assert_eq!(app.held, held);
            }
            app.doc.render(None)
        };
        let whole = paint(&[0..16], false);
        assert!(whole != paint(&[0..8, 7..16], false), "two strokes are not one");
        assert!(whole == paint(&[0..5, 5..11, 11..16], true), "held pieces go on with the same gesture");
    }

    #[test]
    fn every_command_parses_and_names_itself() {
        let args = json!({
            "tool": "oil", "mode": "fill", "factor": 1.2, "delta": 0.1, "which": "size", "by": 1, "steps": -1,
            "filter": "gaussian_blur", "width": 64, "height": 48, "dx": 1, "scale": 1, "rotation": 5,
        });
        for (name, _, _) in COMMANDS {
            let c = parse_cmd(name, &args).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(cmd_name(&c), *name);
        }
        // every key the app maps to a command can be pressed
        let mut app = headless(64, 48);
        for k in ["1", "Shift+N", "F9", "Space", "ArrowLeft", "x", "[", "Shift+G"] {
            call(&mut app, json!({ "op": "key", "key": k }));
        }
    }

    #[test]
    fn every_filter_and_enum_parses() {
        for f in FILTERS {
            let a = json!({ "filter": f, "points": [[0, 0], [1, 1]], "color": "#ff8800", "hue": 30, "stops": [[0, "#000000"], [1, [255, 255, 255]]] });
            parse_filter(&a).unwrap_or_else(|e| panic!("{f}: {e}"));
        }
        for t in tool_names() {
            parse_tool(&t, &Value::Null).unwrap_or_else(|e| panic!("{t}: {e}"));
        }
        assert_eq!(name_of(&Medium::ExpressOil), "express_oil");
        assert_eq!(parse_enum(&MixMode::ALL, "linear_burn", "mode").unwrap(), MixMode::LinearBurn);
    }

    /// Every writable setting takes back the value it reports.
    #[test]
    fn every_setting_round_trips() {
        let mut app = headless(64, 48);
        call(&mut app, json!({ "op": "command", "name": "new_layer" }));
        call(&mut app, json!({ "op": "command", "name": "group_with_below" }));
        call(&mut app, json!({ "op": "set", "key": "brush.pressure_size_curve", "value": { "points": [[0, 0], [1, 1]] } }));
        let all = settings(&app, app.doc.active, Some(0));
        assert!(all.len() > 100, "{} settings", all.len());
        for (k, v, writable) in all {
            if !writable || ["sim.gpu", "app.recording", "doc.reference_visible"].contains(&k.as_str()) {
                continue;
            }
            let r = handle(&mut app, &json!({ "op": "set", "key": k, "value": v, "group": 0 }));
            assert_eq!(r["ok"], true, "{k} = {v}: {r}");
            assert_eq!(r["values"][&k], v, "{k}");
        }
        let r = handle(&mut app, &json!({ "op": "set", "key": "doc.width", "value": 5 }));
        assert_eq!(r["error"], "doc.width: read-only");
    }

    /// A session as an agent would run it: paint, look, select, fill, undo, save, reopen.
    #[test]
    fn an_agent_paints_and_saves() {
        let mut app = headless(160, 120);
        let d = call(&mut app, json!({ "op": "describe" }));
        assert!(d["ops"].as_array().unwrap().len() >= 20);
        call(&mut app, json!({ "op": "set", "values": { "brush.tool": "watercolor", "brush.size": 24, "brush.color": "#2050c0" } }));
        let pts: Vec<Value> = (0..20).map(|k| json!([20 + k * 6, 60, 0.8])).collect();
        call(&mut app, json!({ "op": "stroke", "points": pts, "steps_per_point": 1 }));
        call(&mut app, json!({ "op": "step", "n": 10 }));
        assert!(painted(&app) > 500, "the stroke painted");
        let px = call(&mut app, json!({ "op": "pixel", "x": 80, "y": 60 }));
        let rgba = px["rgba"].as_array().unwrap();
        assert!(rgba[2].as_u64() > rgba[0].as_u64(), "blue paint: {px}");
        let r = call(&mut app, json!({ "op": "render", "max_size": 80 }));
        assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(80), Some(60)));
        let png = B64.decode(r["png"].as_str().unwrap()).unwrap();
        assert_eq!(&png[1..4], b"PNG");

        // layers, selection, fill and pick
        call(&mut app, json!({ "op": "command", "name": "new_layer" }));
        assert_eq!(app.doc.layers.len(), 2);
        call(&mut app, json!({ "op": "set", "key": "layer.opacity", "value": 0.5 }));
        assert_eq!(app.doc.layers[1].opacity, 0.5);
        let s = call(&mut app, json!({ "op": "select", "shape": "rect", "from": [10, 10], "to": [50, 40] }));
        // the rectangle's far edge pixels are partly covered
        assert_eq!(s["selected"], json!([10, 10, 51, 41]));
        call(&mut app, json!({ "op": "fill", "at": [20, 20], "color": "#c02020" }));
        call(&mut app, json!({ "op": "select", "shape": "none" }));
        let c = call(&mut app, json!({ "op": "click", "x": 20, "y": 20, "mode": "pick" }));
        assert_ne!(c["color"], "#2050c0", "picked the fill: {c}");
        call(&mut app, json!({ "op": "drag", "mode": "select_ellipse", "points": [[60, 30], [100, 70]] }));
        assert!(app.doc.selection.is_some());
        call(&mut app, json!({ "op": "command", "name": "filter", "args": { "filter": "hue_saturation", "hue": 90 } }));

        // undo history
        let h = call(&mut app, json!({ "op": "history" }));
        let n = h["done"].as_array().unwrap().len();
        call(&mut app, json!({ "op": "command", "name": "undo" }));
        assert_eq!(app.doc.history().0.len(), n - 1);

        // dialogs become paths and data
        let err = handle(&mut app, &json!({ "op": "command", "name": "save_as" }));
        assert_eq!(err["ok"], false);
        let dir = std::env::temp_dir().join(format!("wetpaint-agent-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.wetdoc");
        call(&mut app, json!({ "op": "save", "path": file }));
        assert!(!app.dirty);
        let png = call(&mut app, json!({ "op": "command", "name": "export_png" }));
        assert!(png["bytes"].as_u64().unwrap() > 100);
        call(&mut app, json!({ "op": "new", "width": 64, "height": 64 }));
        assert_eq!(app.doc.w, 64);
        call(&mut app, json!({ "op": "stroke", "points": [[10, 10], [50, 50]] }));
        let refused = handle(&mut app, &json!({ "op": "open", "path": file }));
        assert_eq!(refused["ok"], false, "unsaved changes are kept: {refused}");
        call(&mut app, json!({ "op": "open", "path": file, "discard": true }));
        assert_eq!((app.doc.w, app.doc.layers.len()), (160, 2));
        let exr = call(&mut app, json!({ "op": "export", "format": "exr", "path": dir.join("h.exr") }));
        assert!(exr["bytes"].as_u64().unwrap() > 0);
        let brush = call(&mut app, json!({ "op": "export", "format": "wetbrush", "name": "Test" }));
        call(&mut app, json!({ "op": "load_preset", "data": brush["data"], "filename": "Test.wetbrush" }));
        assert_eq!(app.preset.as_deref(), Some("Test"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transforms_and_tracing_through_the_agent() {
        let mut app = headless(120, 90);
        call(&mut app, json!({ "op": "stroke", "points": [[20, 45], [60, 45]] }));
        let before = app.doc.history().0.len();
        call(&mut app, json!({ "op": "transform", "dx": 10, "rotation": 15 }));
        call(&mut app, json!({ "op": "transform", "corners": [[0, 0], [120, 10], [110, 90], [0, 80]] }));
        assert_eq!(app.doc.history().0.len(), before + 2);
        // handles: begin, drag the inside, apply
        call(&mut app, json!({ "op": "command", "name": "mode", "mode": "transform" }));
        assert!(app.xform.is_some());
        call(&mut app, json!({ "op": "drag", "points": [[60, 45], [70, 50]] }));
        call(&mut app, json!({ "op": "command", "name": "apply_transform" }));
        assert!(app.xform.is_none());
        // a tracing layer from image data
        let (bytes, _, _) = png(vec![200; 16 * 16 * 4], 16, 16, None).unwrap();
        let err = handle(&mut app, &json!({ "op": "command", "name": "tracing" }));
        assert_eq!(err["ok"], false);
        call(&mut app, json!({ "op": "command", "name": "tracing", "args": { "filename": "t.png", "data": B64.encode(&bytes) } }));
        assert!(app.doc.layers.iter().any(|l| l.tracing));
    }
}
