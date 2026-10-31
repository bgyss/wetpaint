//! Brush library: the user's `.wetbrush` presets, grouped by tool and category
//! (`<brushes>/<tool id>/<category>/<name>.wetbrush`). The folder is `$WETPAINT_BRUSHES`, or
//! `wetpaint/Brushes` in the platform's application data folder.

use std::path::{Path, PathBuf};

use wet_brush::Tool;
use wet_preset::BrushPreset;

const SUFFIX: &str = ".wetbrush";

/// The brush library folder (it need not exist yet).
pub fn root() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("WETPAINT_BRUSHES") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let data = if cfg!(target_os = "macos") {
        home?.join("Library").join("Application Support")
    } else if cfg!(windows) {
        PathBuf::from(std::env::var_os("APPDATA")?)
    } else {
        std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).or_else(|| Some(home?.join(".local").join("share")))?
    };
    Some(data.join("wetpaint").join("Brushes"))
}

/// The library folder for a tool's presets, if the tool takes presets.
pub fn folder_for(tool: Tool) -> Option<&'static str> {
    match tool {
        Tool::Blow | Tool::Liquify(_) => None,
        t => Some(t.id()),
    }
}

pub struct Preset {
    pub name: String,
    pub path: PathBuf,
}

pub struct Category {
    pub name: String,
    pub presets: Vec<Preset>,
}

fn sorted_dir(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).collect();
    v.sort();
    v
}

/// Categories of `<root>/<folder>`, each with its presets, sorted by name.
pub fn categories(root: &Path, folder: &str) -> Vec<Category> {
    sorted_dir(&root.join(folder))
        .into_iter()
        .filter(|p| p.is_dir())
        .map(|dir| Category {
            name: dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            presets: sorted_dir(&dir)
                .into_iter()
                .filter_map(|path| {
                    let name = preset_name(&path)?;
                    Some(Preset { name, path })
                })
                .collect(),
        })
        .filter(|c| !c.presets.is_empty())
        .collect()
}

/// The preset name of a `.wetbrush` path.
pub fn preset_name(path: &Path) -> Option<String> {
    Some(path.file_name()?.to_str()?.strip_suffix(SUFFIX)?.to_owned())
}

pub fn is_preset(path: &Path) -> bool {
    preset_name(path).is_some()
}

/// A small RGBA thumbnail of a preset file (its preview image), `size` pixels on the long side.
pub fn thumbnail(path: &Path, size: u32) -> Option<(Vec<u8>, usize, usize)> {
    let preset = BrushPreset::from_bytes(&std::fs::read(path).ok()?).ok()?;
    let img = image::load_from_memory(&preset.thumbnail?).ok()?.thumbnail(size, size).to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Some((img.into_raw(), w, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wet_brush::Medium;

    #[test]
    fn reads_a_library_folder() {
        let root = std::env::temp_dir().join(format!("wetpaint-lib-{}", std::process::id()));
        let cat = root.join("ink").join("Pen");
        std::fs::create_dir_all(&cat).unwrap();
        std::fs::write(cat.join("Fine.wetbrush"), b"").unwrap();
        std::fs::write(cat.join("Bold.wetbrush"), b"").unwrap();
        std::fs::write(cat.join("notes.txt"), b"").unwrap();
        std::fs::create_dir_all(root.join("ink").join("Empty")).unwrap();
        let cats = categories(&root, folder_for(Tool::Paint(Medium::Ink)).unwrap());
        assert_eq!(cats.len(), 1);
        assert_eq!(cats[0].name, "Pen");
        let names: Vec<&str> = cats[0].presets.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["Bold", "Fine"]);
        assert!(is_preset(&cats[0].presets[0].path));
        let _ = std::fs::remove_dir_all(&root);
    }
}
