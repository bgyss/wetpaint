//! Brush presets: a brush's tool, its parameters and its textures, stored as a `.wetbrush`
//! file (JSON; textures are base64 PNG files).
//!
//! ```json
//! { "wetbrush": 1, "name": "Wash", "tool": "watercolor",
//!   "parameters": { "splat_spacing": 0.05, … },
//!   "painting": { "volume_params": { "size": 30, … } },
//!   "textures": { "Shape_0": "<base64 PNG>", … },
//!   "thumbnail": "<base64 PNG>" }
//! ```

use std::collections::BTreeMap;

use base64::Engine as _;
use serde_json::{Map, Value, json};

/// File name extension of a preset.
pub const EXTENSION: &str = "wetbrush";

#[derive(Debug)]
pub enum Error {
    Json(serde_json::Error),
    /// Not a `.wetbrush` document (no `wetbrush` version key, or a newer version).
    Format(String),
    Base64(String, base64::DecodeError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "invalid JSON: {e}"),
            Self::Format(s) => write!(f, "not a brush preset: {s}"),
            Self::Base64(k, e) => write!(f, "invalid texture {k}: {e}"),
        }
    }
}

impl std::error::Error for Error {}

/// A brush preset.
#[derive(Debug, Clone, Default)]
pub struct BrushPreset {
    /// Tool identifier (`watercolor`, `oil`, `eraser`, …; see `wet_brush::Tool::id`).
    pub tool: String,
    pub name: String,
    /// Stamp, grain and dynamics parameters.
    pub parameters: Value,
    /// Painting parameters (`volume_params`, `painting_params`), if any.
    pub parameters_ex: Option<Value>,
    /// Texture slot (`Shape_0`, `Texture_0`, `PaperTexture_0`, `DualBrushShape_0`, …) to
    /// greyscale PNG file bytes.
    pub textures: BTreeMap<String, Vec<u8>>,
    /// A preview image (PNG file bytes), if any.
    pub thumbnail: Option<Vec<u8>>,
}

const VERSION: u64 = 1;

fn b64_str(v: &Value) -> &str {
    v.as_str().unwrap_or_default()
}

impl BrushPreset {
    pub fn from_bytes(data: &[u8]) -> Result<Self, Error> {
        let v: Value = serde_json::from_slice(data).map_err(Error::Json)?;
        match v.get("wetbrush").and_then(Value::as_u64) {
            Some(n) if n <= VERSION => {}
            Some(n) => return Err(Error::Format(format!("version {n}"))),
            None => return Err(Error::Format("no version".into())),
        }
        let b64 = |k: &str, s: &str| base64::engine::general_purpose::STANDARD.decode(s).map_err(|e| Error::Base64(k.to_owned(), e));
        let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or_default().to_owned();
        let mut textures = BTreeMap::new();
        if let Some(t) = v.get("textures").and_then(Value::as_object) {
            for (k, v) in t {
                textures.insert(k.clone(), b64(k, b64_str(v))?);
            }
        }
        Ok(Self {
            tool: s("tool"),
            name: s("name"),
            parameters: v.get("parameters").cloned().unwrap_or_else(|| json!({})),
            parameters_ex: v.get("painting").cloned(),
            textures,
            thumbnail: v.get("thumbnail").map(|t| b64("thumbnail", b64_str(t))).transpose()?,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut m = Map::new();
        m.insert("wetbrush".into(), json!(VERSION));
        m.insert("name".into(), json!(self.name));
        m.insert("tool".into(), json!(self.tool));
        m.insert("parameters".into(), self.parameters.clone());
        if let Some(ex) = &self.parameters_ex {
            m.insert("painting".into(), ex.clone());
        }
        if !self.textures.is_empty() {
            let t: Map<String, Value> = self
                .textures
                .iter()
                .map(|(k, b)| (k.clone(), json!(base64::engine::general_purpose::STANDARD.encode(b))))
                .collect();
            m.insert("textures".into(), Value::Object(t));
        }
        if let Some(t) = &self.thumbnail {
            m.insert("thumbnail".into(), json!(base64::engine::general_purpose::STANDARD.encode(t)));
        }
        serde_json::to_vec_pretty(&Value::Object(m)).expect("JSON")
    }

    /// Looks up a numeric parameter.
    pub fn param_f64(&self, key: &str) -> Option<f64> {
        self.parameters.get(key)?.as_f64()
    }

    /// Looks up a `volume_params` value (`size`, `water`, `opacity`, `pressure`, `length`).
    pub fn volume(&self, key: &str) -> Option<f64> {
        self.parameters_ex.as_ref()?.get("volume_params")?.get(key)?.as_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut p = BrushPreset { tool: "oil".into(), name: "Flat".into(), parameters: json!({ "splat_spacing": 0.1 }), ..Default::default() };
        p.parameters_ex = Some(json!({ "volume_params": { "size": 30.0 } }));
        p.textures.insert("Shape_0".into(), vec![1, 2, 3, 250]);
        p.thumbnail = Some(vec![9, 8]);
        let q = BrushPreset::from_bytes(&p.to_bytes()).unwrap();
        assert_eq!((q.tool.as_str(), q.name.as_str()), ("oil", "Flat"));
        assert_eq!(q.param_f64("splat_spacing"), Some(0.1));
        assert_eq!(q.volume("size"), Some(30.0));
        assert_eq!(q.textures["Shape_0"], vec![1, 2, 3, 250]);
        assert_eq!(q.thumbnail, Some(vec![9, 8]));
        assert!(BrushPreset::from_bytes(b"{\"wetbrush\": 9}").is_err());
        assert!(BrushPreset::from_bytes(b"{}").is_err());
    }
}
