//! Files and dialogs: native dialogs and the file system on the desktop; downloads and the
//! browser's file picker on the web. Picked files arrive through [`take_picked`], at once
//! on the desktop and a few frames later on the web.

use std::cell::RefCell;
use std::io;
use std::path::PathBuf;

/// What a picked file is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// File > Open: a document, an image (as a layer) or a brush preset.
    Open,
    /// An image for a tracing layer.
    Tracing,
    /// A reference image.
    Reference,
}

/// A picked file: its name, its path (desktop only) and contents.
pub struct Picked {
    pub kind: Pick,
    pub name: String,
    pub path: Option<PathBuf>,
    pub bytes: Vec<u8>,
}

thread_local! {
    static PICKED: RefCell<Vec<Picked>> = const { RefCell::new(Vec::new()) };
}

/// Files picked since the last call.
pub fn take_picked() -> Vec<Picked> {
    PICKED.with_borrow_mut(std::mem::take)
}

pub const IMAGES: &[&str] = &["png", "jpg", "jpeg", "tif", "tiff", "bmp"];

/// Ask for a file to open, with `(description, extensions)` filters.
pub fn pick(kind: Pick, filters: &[(&str, &[&str])]) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut d = rfd::FileDialog::new();
        for (name, ext) in filters {
            d = d.add_filter(*name, ext);
        }
        if let Some(path) = d.pick_file()
            && let Ok(bytes) = std::fs::read(&path)
        {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            PICKED.with_borrow_mut(|v| v.push(Picked { kind, name, path: Some(path), bytes }));
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let mut d = rfd::AsyncFileDialog::new();
        for (name, ext) in filters {
            d = d.add_filter(*name, ext);
        }
        wasm_bindgen_futures::spawn_local(async move {
            if let Some(f) = d.pick_file().await {
                let (name, bytes) = (f.file_name(), f.read().await);
                PICKED.with_borrow_mut(|v| v.push(Picked { kind, name, path: None, bytes }));
            }
        });
    }
}

/// Where a saved file goes: a path the user chose, or (on the web) a download.
pub enum Target {
    Path(PathBuf),
    Download(String),
}

impl Target {
    pub fn describe(&self) -> String {
        match self {
            Self::Path(p) => p.display().to_string(),
            Self::Download(n) => n.clone(),
        }
    }
}

/// Ask where to save a file called `name` by default.
pub fn save_target(name: &str, ext: &str) -> Option<Target> {
    #[cfg(not(target_arch = "wasm32"))]
    return rfd::FileDialog::new().set_file_name(name).add_filter(ext, &[ext]).save_file().map(Target::Path);
    #[cfg(target_arch = "wasm32")]
    {
        let _ = ext;
        Some(Target::Download(name.to_owned()))
    }
}

/// Write a file through `write`.
pub fn write(target: &Target, write: impl FnOnce(&mut io::Cursor<Vec<u8>>) -> io::Result<()>) -> io::Result<()> {
    let mut buf = io::Cursor::new(Vec::new());
    write(&mut buf)?;
    match target {
        Target::Path(p) => std::fs::write(p, buf.into_inner()),
        Target::Download(name) => download(name, &buf.into_inner()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn download(_name: &str, _bytes: &[u8]) -> io::Result<()> {
    Err(io::Error::other("downloads are only available on the web"))
}

/// Hand `bytes` to the browser as a download called `name`.
#[cfg(target_arch = "wasm32")]
fn download(name: &str, bytes: &[u8]) -> io::Result<()> {
    use wasm_bindgen::JsCast;
    let err = |e: wasm_bindgen::JsValue| io::Error::other(format!("{e:?}"));
    let parts = js_sys::Array::of1(&js_sys::Uint8Array::from(bytes));
    let blob = web_sys::Blob::new_with_u8_array_sequence(&parts).map_err(err)?;
    let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(err)?;
    let doc = web_sys::window().and_then(|w| w.document()).ok_or_else(|| io::Error::other("no document"))?;
    let a: web_sys::HtmlAnchorElement = doc.create_element("a").map_err(err)?.unchecked_into();
    a.set_href(&url);
    a.set_download(name);
    a.click();
    web_sys::Url::revoke_object_url(&url).map_err(err)
}

/// Ask whether unsaved changes may be lost: `Some(true)` save first, `Some(false)` discard,
/// `None` cancel.
pub fn ask_save_changes(name: &str) -> Option<bool> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let answer = rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Unsaved changes")
            .set_description(format!("Save the changes to “{name}” first?"))
            .set_buttons(rfd::MessageButtons::YesNoCancelCustom("Save".into(), "Don't Save".into(), "Cancel".into()))
            .show();
        match answer {
            rfd::MessageDialogResult::Custom(b) if b == "Save" => Some(true),
            rfd::MessageDialogResult::Custom(b) if b == "Don't Save" => Some(false),
            rfd::MessageDialogResult::Yes => Some(true),
            rfd::MessageDialogResult::No => Some(false),
            _ => None,
        }
    }
    // in the browser there is no dialog to ask with (pages embedded in a frame may not
    // show one): the changes are discarded
    #[cfg(target_arch = "wasm32")]
    {
        let _ = name;
        Some(false)
    }
}
