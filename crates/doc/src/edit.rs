//! Select- and Layer-menu editing operations: Expand/Contract, Feather, Rough Feather,
//! Restore Last Selection, Open/Save Selection, Duplicate Layer, Merge Layers, Merge
//! Visible, Merge Visible to New Layer, Crop Canvas.

use half::f16;
use wet_fluid::mix::mix;

use crate::{Document, LayerPlanes};

impl Document {
    /// Forget the selection (it can be restored with [`Document::restore_selection`]).
    pub fn deselect(&mut self) {
        if let Some(s) = self.selection.take() {
            self.last_selection = Some(s);
        }
    }

    /// Bring back the selection from before the last deselect or selection change.
    pub fn restore_selection(&mut self) {
        if let Some(s) = self.last_selection.take() {
            self.last_selection = self.selection.replace(s);
        }
    }

    /// Select the open (paintable) area of stencil `idx` (Create Selection from Stencil).
    pub fn select_from_stencil(&mut self, idx: usize, op: crate::SelectOp) -> bool {
        let Some(st) = self.stencils.get(idx) else { return false };
        let w = self.w;
        let mask = (0..self.w * self.h)
            .map(|c| (255.0 * (1.0 - st.coverage((c % w) as f32, (c / w) as f32))).round().clamp(0.0, 255.0) as u8)
            .collect();
        self.apply_selection(mask, op);
        true
    }

    /// Grow (`px > 0`) or shrink (`px < 0`) the selection by `px` pixels (disc-shaped
    /// dilation / erosion of the soft mask).
    pub fn expand_selection(&mut self, px: i32) {
        let Some(sel) = self.selection.clone() else { return };
        let (w, h, r) = (self.w as i64, self.h as i64, px.unsigned_abs() as i64);
        if r == 0 {
            return;
        }
        let grow = px > 0;
        let offsets: Vec<(i64, i64)> =
            (-r..=r).flat_map(|dy| (-r..=r).map(move |dx| (dx, dy))).filter(|(dx, dy)| dx * dx + dy * dy <= r * r).collect();
        let out = (0..w * h)
            .map(|c| {
                let (x, y) = (c % w, c / w);
                let vals = offsets.iter().map(|&(dx, dy)| {
                    let (sx, sy) = (x + dx, y + dy);
                    // outside the canvas counts as unselected when shrinking
                    if sx < 0 || sy < 0 || sx >= w || sy >= h { 0 } else { sel[(sy * w + sx) as usize] }
                });
                if grow { vals.max().unwrap_or(0) } else { vals.min().unwrap_or(0) }
            })
            .collect();
        self.last_selection = Some(sel);
        self.selection = Some(out);
    }

    /// Soften the selection edge with a Gaussian of `radius` pixels.
    pub fn feather_selection(&mut self, radius: f32) {
        let Some(sel) = self.selection.clone() else { return };
        let f: Vec<f32> = sel.iter().map(|&v| v as f32).collect();
        let b = crate::filters::gaussian(&f, self.w, self.h, radius);
        self.last_selection = Some(sel);
        self.selection = Some(b.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect());
    }

    /// Feather with a ragged, torn edge: the softened edge is thresholded against value
    /// noise of `roughness` 0…1 (Rough Feather).
    pub fn rough_feather_selection(&mut self, radius: f32, roughness: f32, seed: u64) {
        let Some(sel) = self.selection.clone() else { return };
        let f: Vec<f32> = sel.iter().map(|&v| v as f32 / 255.0).collect();
        let b = crate::filters::gaussian(&f, self.w, self.h, radius);
        let w = self.w;
        let cell = (radius * 0.5).max(1.0);
        let out = b
            .iter()
            .enumerate()
            .map(|(c, &v)| {
                let n = value_noise((c % w) as f32 / cell, (c / w) as f32 / cell, seed) - 0.5;
                let t = (v + roughness * n).clamp(0.0, 1.0);
                // steepen so the edge is ragged rather than blurry
                (255.0 * (((t - 0.5) * 3.0) + 0.5).clamp(0.0, 1.0)).round() as u8
            })
            .collect();
        self.last_selection = Some(sel);
        self.selection = Some(out);
    }

    /// The selection as a grayscale PNG (Save Selection As...).
    pub fn selection_png(&self) -> Option<Vec<u8>> {
        let sel = self.selection.as_ref()?;
        let mut out = Vec::new();
        let mut enc = png::Encoder::new(&mut out, self.w as u32, self.h as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().ok()?;
        wr.write_image_data(sel).ok()?;
        wr.finish().ok()?;
        Some(out)
    }

    /// Load a selection from a grayscale (or colour, by luminance) PNG of the canvas size
    /// (Open Selection...). Returns false when the image does not fit.
    pub fn load_selection_png(&mut self, data: &[u8]) -> bool {
        let Some((w, h, new)) = decode_gray(data) else { return false };
        if w != self.w || h != self.h {
            return false;
        }
        if let Some(s) = self.selection.replace(new) {
            self.last_selection = Some(s);
        }
        true
    }

    /// Duplicate layer `idx` above itself; returns the new layer's index.
    pub fn duplicate_layer(&mut self, idx: usize) -> Option<usize> {
        let src = self.layers.get_mut(idx)?;
        let planes = LayerPlanes::capture(&mut src.wet);
        let (name, mode, opacity, visible, clip, mask, pigments, group) =
            (format!("{} copy", src.name), src.mode, src.opacity, src.visible, src.clip, src.mask.clone(), src.pigments, src.group);
        let params = src.wet.params.clone();
        self.active = idx;
        let new = self.add_layer(&name);
        let l = &mut self.layers[new];
        planes.restore(&mut l.wet);
        l.wet.params = params;
        (l.mode, l.opacity, l.visible, l.clip, l.mask, l.group) = (mode, opacity, visible, clip, mask, group);
        if pigments {
            self.set_layer_pigments(new, true);
        }
        Some(new)
    }

    /// Merge layer `idx` into the layer below it (Merge Layers): the upper layer's
    /// deposited and suspended paint land on top of the lower layer's (the opacity-LUT
    /// `mix` rule, as when paint is laid over paint), water and wetness take the wetter of
    /// the two. Layer blend modes and opacity are not baked in. Not undoable.
    pub fn merge_down(&mut self, idx: usize) -> bool {
        if idx == 0 || idx >= self.layers.len() {
            return false;
        }
        let upper = self.layers.remove(idx);
        let lower = &mut self.layers[idx - 1];
        merge_into(&mut lower.wet, &upper.wet, upper.visible, upper.opacity);
        self.active = idx - 1;
        self.undo.clear();
        self.redo.clear();
        true
    }

    /// Merge all visible layers into the lowest visible one (Merge Visible).
    pub fn merge_visible(&mut self) -> Option<usize> {
        let vis: Vec<usize> = (0..self.layers.len()).filter(|&i| self.layers[i].visible && !self.layers[i].masking_fluid).collect();
        let &base = vis.first()?;
        for &i in vis.iter().skip(1).rev() {
            let upper = self.layers.remove(i);
            merge_into(&mut self.layers[base].wet, &upper.wet, true, upper.opacity);
        }
        self.active = base;
        self.undo.clear();
        self.redo.clear();
        Some(base)
    }

    /// Merge copies of all visible layers into a new layer on top, keeping the originals
    /// (Merge Visible to New Layer).
    pub fn merge_visible_to_new(&mut self) -> Option<usize> {
        let vis: Vec<usize> = (0..self.layers.len()).filter(|&i| self.layers[i].visible && !self.layers[i].masking_fluid).collect();
        if vis.is_empty() {
            return None;
        }
        self.active = self.layers.len() - 1;
        let new = self.add_layer("Merged");
        for i in vis {
            let (upper, base) = if i < new {
                let (a, b) = self.layers.split_at_mut(new);
                (&a[i], &mut b[0])
            } else {
                continue;
            };
            merge_into(&mut base.wet, &upper.wet, true, upper.opacity);
        }
        Some(new)
    }

    /// Impasto depth: the paint thickness of the visible layers, summed per pixel
    /// (deposited and suspended paint; the paper relief is not included).
    pub fn heightmap(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.w * self.h];
        for l in self.layers.iter().filter(|l| l.visible && !l.masking_fluid) {
            for (c, o) in out.iter_mut().enumerate() {
                *o += l.wet.mid.thickness.cur().data[c] + l.wet.dens.thickness.cur().data[c];
            }
        }
        out
    }

    /// Write [`Document::heightmap`] as a 16-bit (half float) EXR displacement map
    /// of the impasto paint's physical depth.
    pub fn export_heightmap_exr(&self, out: impl std::io::Write + std::io::Seek) -> std::io::Result<()> {
        use exr::prelude::*;
        let hm = self.heightmap();
        let w = self.w;
        let channels = SpecificChannels::rgba(|Vec2(x, y)| {
            let v = f16::from_f32(hm[y * w + x]);
            (v, v, v, f16::ONE)
        });
        Image::from_channels((self.w, self.h), channels).write().to_buffered(out).map_err(std::io::Error::other)
    }

    /// Crop the canvas to the bounding box of the selection (Crop Canvas).
    pub fn crop_to_selection(&mut self) -> bool {
        let Some(sel) = self.selection.as_ref() else { return false };
        let w = self.w;
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        for (c, &v) in sel.iter().enumerate() {
            if v > 0 {
                let (x, y) = (c % w, c / w);
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            }
        }
        if x0 > x1 {
            return false;
        }
        self.resize_canvas(x1 - x0 + 1, y1 - y0 + 1, (-(x0 as i64), -(y0 as i64)));
        true
    }
}

/// The next file name for Iterative Save: `name_1.ext` the first time, then the number
/// after the highest `name_N.ext` already on disk (a `name_N` input continues its series).
pub fn iterative_path(path: &std::path::Path) -> std::path::PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("artwork");
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("wetdoc");
    let base = match stem.rsplit_once('_') {
        Some((b, n)) if !b.is_empty() && n.parse::<u32>().is_ok() => b,
        _ => stem,
    };
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
    let highest = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let rest = name.strip_prefix(&format!("{base}_"))?.strip_suffix(&format!(".{ext}"))?;
            rest.parse::<u32>().ok()
        })
        .max()
        .unwrap_or(0);
    dir.join(format!("{base}_{}.{ext}", highest + 1))
}

/// Smooth value noise in 0…1 on a unit lattice.
fn value_noise(x: f32, y: f32, seed: u64) -> f32 {
    let hash = |i: i64, j: i64| {
        let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed;
        h ^= h >> 31;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 29;
        (h >> 40) as f32 / (1u64 << 24) as f32
    };
    let (x0, y0) = (x.floor(), y.floor());
    let (tx, ty) = (x - x0, y - y0);
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (i, j) = (x0 as i64, y0 as i64);
    let top = hash(i, j) + (hash(i + 1, j) - hash(i, j)) * s(tx);
    let bot = hash(i, j + 1) + (hash(i + 1, j + 1) - hash(i, j + 1)) * s(tx);
    top + (bot - top) * s(ty)
}

/// Decode an 8-bit PNG to one grey byte per pixel (luminance for colour images).
pub(crate) fn decode_gray(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(data));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut r = dec.read_info().ok()?;
    let mut buf = vec![0; r.output_buffer_size()?];
    let info = r.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width as usize, info.height as usize);
    let ch = info.color_type.samples();
    let px = |i: usize| -> u8 {
        let p = &buf[i * ch..i * ch + ch];
        if ch >= 3 { (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).round() as u8 } else { p[0] }
    };
    Some((w, h, (0..w * h).map(px).collect()))
}

/// Lay `upper`'s paint over `lower`'s, cell by cell (see [`Document::merge_down`]).
fn merge_into(lower: &mut wet_fluid::WetLayer, upper: &wet_fluid::WetLayer, visible: bool, opacity: f32) {
    if !visible {
        return;
    }
    let lut = lower.lut.clone();
    let k = opacity.clamp(0.0, 1.0);
    let n = lower.w * lower.h;
    let pig = lower.pigment.as_deref();
    for c in 0..n {
        for (lo, up) in [(&mut lower.mid, &upper.mid), (&mut lower.dens, &upper.dens)] {
            let a_up = up.amount.cur().data[c] * k;
            if a_up <= 0.0 {
                continue;
            }
            let a_lo = lo.amount.cur().data[c];
            let c_lo = [0, 1, 2].map(|i| lo.rgb[i].cur().data[c].to_f32());
            let c_up = [0, 1, 2].map(|i| up.rgb[i].cur().data[c].to_f32());
            let new = if a_lo <= 0.01 { c_up } else { mix(c_lo, lut.eval(a_lo), c_up, lut.eval(a_up), pig) };
            for i in 0..3 {
                lo.rgb[i].cur_mut().data[c] = f16::from_f32(new[i]);
            }
            lo.amount.cur_mut().data[c] = a_lo + a_up;
            let t = lo.thickness.cur().data[c] + up.thickness.cur().data[c] * k;
            lo.thickness.cur_mut().data[c] = t;
        }
        let (wl, wu) = (lower.water.cur().data[c].to_f32(), upper.water.cur().data[c].to_f32());
        if wu > wl {
            lower.water.cur_mut().data[c] = f16::from_f32(wu);
        }
        lower.maps.open.data[c] = lower.maps.open.data[c].max(upper.maps.open.data[c]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> Document {
        Document::new(40, 40)
    }

    fn paint(d: &mut Document, idx: usize, x0: usize, x1: usize, rgb: [f32; 3]) {
        let l = &mut d.layers[idx].wet;
        for y in 10..30 {
            for x in x0..x1 {
                let c = y * 40 + x;
                l.mid.amount.cur_mut().data[c] = 500.0;
                for k in 0..3 {
                    l.mid.rgb[k].cur_mut().data[c] = f16::from_f32(rgb[k]);
                }
            }
        }
    }

    #[test]
    fn expand_contract_feather_restore() {
        let mut d = doc();
        d.select_shape((10.0, 10.0), (30.0, 30.0), false);
        let count = |d: &Document| d.selection.as_ref().unwrap().iter().filter(|&&v| v > 127).count();
        let base = count(&d);
        d.expand_selection(3);
        assert!(count(&d) > base);
        d.expand_selection(-6);
        assert!(count(&d) < base);
        d.feather_selection(4.0);
        let s = d.selection.as_ref().unwrap();
        assert!(s.iter().any(|&v| v > 10 && v < 245), "feathered edge has partial values");
        d.deselect();
        assert!(d.selection.is_none());
        d.restore_selection();
        assert!(d.selection.is_some());
        let png = d.selection_png().unwrap();
        let mut d2 = doc();
        assert!(d2.load_selection_png(&png));
        assert_eq!(d2.selection, d.selection);
    }

    #[test]
    fn selection_from_stencil_selects_the_open_area() {
        let mut d = doc();
        // a stencil blocking everything but a central square
        let mut m = vec![255u8; 20 * 20];
        for y in 5..15 {
            for x in 5..15 {
                m[y * 20 + x] = 0;
            }
        }
        d.stencils.push(crate::Stencil::new(m, 20, 20, (20.0, 20.0)));
        assert!(d.select_from_stencil(0, crate::SelectOp::Replace));
        let s = d.selection.as_ref().unwrap();
        assert!(s[20 * 40 + 20] > 200);
        assert!(s[12 * 40 + 12] < 50);
    }

    #[test]
    fn duplicate_and_merge() {
        let mut d = doc();
        paint(&mut d, 0, 5, 20, [200.0, 40.0, 40.0]);
        let dup = d.duplicate_layer(0).unwrap();
        assert_eq!(d.layers.len(), 2);
        assert_eq!(d.layers[dup].wet.mid.amount.cur().data[15 * 40 + 10], 500.0);
        // paint blue on the copy over the right half, then merge down
        paint(&mut d, dup, 15, 35, [40.0, 40.0, 200.0]);
        assert!(d.merge_down(dup));
        assert_eq!(d.layers.len(), 1);
        let l = &d.layers[0].wet;
        let c = 15 * 40 + 17; // both layers: blue landed on red
        let col = [0, 1, 2].map(|k| l.mid.rgb[k].cur().data[c].to_f32());
        assert!(col[2] > col[0], "{col:?}");
        assert!(l.mid.amount.cur().data[c] > 900.0);
        assert_eq!(l.mid.amount.cur().data[15 * 40 + 30], 500.0);
    }

    #[test]
    fn heightmap_and_iterative_path() {
        let mut d = doc();
        d.layers[0].wet.mid.thickness.cur_mut().data[5] = 2.5;
        assert_eq!(d.heightmap()[5], 2.5);
        let dir = std::env::temp_dir().join(format!("wetdoc_iter_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("art.wetdoc");
        assert_eq!(iterative_path(&p), dir.join("art_1.wetdoc"));
        std::fs::write(dir.join("art_1.wetdoc"), b"x").unwrap();
        std::fs::write(dir.join("art_4.wetdoc"), b"x").unwrap();
        assert_eq!(iterative_path(&p), dir.join("art_5.wetdoc"));
        assert_eq!(iterative_path(&dir.join("art_4.wetdoc")), dir.join("art_5.wetdoc"));
        let mut exr = std::io::Cursor::new(Vec::new());
        d.export_heightmap_exr(&mut exr).unwrap();
        assert!(exr.get_ref().starts_with(&[0x76, 0x2f, 0x31, 0x01]), "an OpenEXR file");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn merge_visible_to_new_keeps_originals_and_crop() {
        let mut d = doc();
        paint(&mut d, 0, 5, 20, [200.0, 40.0, 40.0]);
        d.add_layer("Layer 2");
        paint(&mut d, 1, 20, 35, [40.0, 40.0, 200.0]);
        let m = d.merge_visible_to_new().unwrap();
        assert_eq!(d.layers.len(), 3);
        assert_eq!(d.layers[m].wet.mid.amount.cur().data[15 * 40 + 10], 500.0);
        assert_eq!(d.layers[m].wet.mid.amount.cur().data[15 * 40 + 25], 500.0);
        d.select_shape((5.0, 5.0), (25.0, 20.0), false);
        assert!(d.crop_to_selection());
        assert!(d.w <= 21 && d.h <= 16, "{}x{}", d.w, d.h);
        assert_eq!(d.merge_visible(), Some(0));
        assert_eq!(d.layers.len(), 1);
    }
}
