//! Minimal layered Photoshop (PSD) export, written from Adobe's public file-format
//! specification: 8-bit RGB, one raster layer per document layer (raw, uncompressed
//! channels with transparency), layer groups as folder/divider records (`lsct`), and the
//! flattened composite as the merged image.

use std::io::Write;

use wet_compose::MixMode;

/// Photoshop blend-mode keys for our mix modes (Normal where there is no equivalent).
fn blend_key(m: MixMode) -> &'static [u8; 4] {
    use MixMode::*;
    match m {
        Additive => b"lddg",
        Subtractive => b"fsub",
        LinearBurn => b"lbrn",
        Darken => b"dark",
        Lighten => b"lite",
        Multiply => b"mul ",
        Screen => b"scrn",
        ColorDodge => b"div ",
        ColorBurn => b"idiv",
        Overlay => b"over",
        HardLight => b"hLit",
        SoftLight => b"sLit",
        VividLight => b"vLit",
        LinearLight => b"lLit",
        PinLight => b"pLit",
        HardMix => b"hMix",
        Difference => b"diff",
        Exclusion => b"smud",
        Hue => b"hue ",
        Saturation => b"sat ",
        Color => b"colr",
        Luminosity => b"lum ",
        _ => b"norm",
    }
}

/// What a layer record is: pixels, or the top/bottom marker of a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Pixels,
    /// The group itself (name, mode, opacity), above its members.
    GroupStart,
    /// The hidden marker below a group's members.
    GroupEnd,
}

/// One exported layer record: straight RGBA8 pixels (empty for group markers) and its
/// properties.
pub struct PsdLayer<'a> {
    pub name: &'a str,
    pub rgba: Vec<u8>,
    pub opacity: f32,
    pub mode: MixMode,
    pub visible: bool,
    pub clip: bool,
    pub section: Section,
}

/// Write a PSD with the given layer records (bottom first; a group is its `GroupEnd`
/// marker, its members, then its `GroupStart`) and flattened RGBA8 composite.
pub fn write_psd(
    mut out: impl Write,
    w: usize,
    h: usize,
    layers: &[PsdLayer],
    composite: &[u8],
    icc: Option<&[u8]>,
) -> std::io::Result<()> {
    let be16 = |v: u16| v.to_be_bytes();
    let be32 = |v: u32| v.to_be_bytes();
    let n = w * h;
    // header
    out.write_all(b"8BPS")?;
    out.write_all(&be16(1))?; // version
    out.write_all(&[0; 6])?;
    out.write_all(&be16(3))?; // channels in merged image (RGB)
    out.write_all(&be32(h as u32))?;
    out.write_all(&be32(w as u32))?;
    out.write_all(&be16(8))?; // depth
    out.write_all(&be16(3))?; // RGB colour mode
    out.write_all(&be32(0))?; // colour mode data
    // image resources: the ICC profile (resource 1039) when the document is tagged
    match icc {
        Some(icc) => {
            let pad = icc.len() % 2;
            out.write_all(&be32((4 + 2 + 2 + 4 + icc.len() + pad) as u32))?;
            out.write_all(b"8BIM")?;
            out.write_all(&be16(1039))?;
            out.write_all(&[0, 0])?; // empty Pascal name, padded to even
            out.write_all(&be32(icc.len() as u32))?;
            out.write_all(icc)?;
            out.write_all(&vec![0; pad])?;
        }
        None => out.write_all(&be32(0))?,
    }

    // layer info
    let mut info: Vec<u8> = Vec::new();
    info.extend(&(layers.len() as i16).to_be_bytes());
    for l in layers {
        let pixels = l.section == Section::Pixels;
        // channel image data size per channel: 2 (compression) + n bytes (none for markers)
        let chan_len = (2 + if pixels { n } else { 0 }) as u32;
        let (bottom, right) = if pixels { (h as u32, w as u32) } else { (0, 0) };
        info.extend(be32(0)); // top
        info.extend(be32(0)); // left
        info.extend(be32(bottom));
        info.extend(be32(right));
        info.extend(be16(4)); // channels: A, R, G, B
        for id in [-1i16, 0, 1, 2] {
            info.extend(id.to_be_bytes());
            info.extend(be32(chan_len));
        }
        info.extend(b"8BIM");
        info.extend(blend_key(l.mode));
        info.push((l.opacity.clamp(0.0, 1.0) * 255.0).round() as u8);
        info.push(l.clip as u8); // clipping: 1 = non-base (clipped to the layer below)
        info.push(if l.visible { 0 } else { 0x02 }); // flags: bit 1 = hidden
        info.push(0); // filler
        // extra data: mask (none), blending ranges (none), Pascal name padded to 4
        let name = l.name.as_bytes();
        let name = &name[..name.len().min(255)];
        let mut pascal = vec![name.len() as u8];
        pascal.extend(name);
        while pascal.len() % 4 != 0 {
            pascal.push(0);
        }
        // section divider setting: 1 = open folder, 3 = bounding divider
        let mut lsct = Vec::new();
        if !pixels {
            lsct.extend(b"8BIMlsct");
            lsct.extend(be32(12));
            lsct.extend(be32(if l.section == Section::GroupStart { 1 } else { 3 }));
            lsct.extend(b"8BIM");
            lsct.extend(blend_key(l.mode));
        }
        let extra_len = 4 + 4 + pascal.len() + lsct.len();
        info.extend(be32(extra_len as u32));
        info.extend(be32(0));
        info.extend(be32(0));
        info.extend(&pascal);
        info.extend(&lsct);
    }
    for l in layers {
        for k in [3usize, 0, 1, 2] {
            info.extend(be16(0)); // raw
            if l.section == Section::Pixels {
                info.extend(l.rgba.chunks_exact(4).map(|p| p[k]));
            }
        }
    }
    if info.len() % 2 != 0 {
        info.push(0);
    }
    out.write_all(&be32((info.len() + 4 + 4) as u32))?; // layer & mask section length
    out.write_all(&be32(info.len() as u32))?; // layer info length
    out.write_all(&info)?;
    out.write_all(&be32(0))?; // global layer mask info

    // merged image: raw, planar R, G, B
    out.write_all(&be16(0))?;
    for k in 0..3 {
        let plane: Vec<u8> = composite.chunks_exact(4).map(|p| p[k]).collect();
        out.write_all(&plane)?;
    }
    Ok(())
}

/// A layer read from a PSD: straight RGBA8 over its bounds, placed at `(left, top)`.
pub struct ReadLayer {
    pub name: String,
    pub left: i32,
    pub top: i32,
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
    pub opacity: f32,
    pub mode: MixMode,
    pub visible: bool,
    pub clip: bool,
}

/// A decoded PSD: canvas size, pixel layers (bottom first; group markers dropped) and
/// the flattened image (RGBA8).
pub struct ReadPsd {
    pub w: usize,
    pub h: usize,
    pub layers: Vec<ReadLayer>,
    pub composite: Vec<u8>,
}

fn mode_from_key(k: &[u8]) -> MixMode {
    use MixMode::*;
    match k {
        b"lddg" => Additive,
        b"fsub" => Subtractive,
        b"lbrn" => LinearBurn,
        b"dark" => Darken,
        b"lite" => Lighten,
        b"mul " => Multiply,
        b"scrn" => Screen,
        b"div " => ColorDodge,
        b"idiv" => ColorBurn,
        b"over" => Overlay,
        b"hLit" => HardLight,
        b"sLit" => SoftLight,
        b"vLit" => VividLight,
        b"lLit" => LinearLight,
        b"pLit" => PinLight,
        b"hMix" => HardMix,
        b"diff" => Difference,
        b"smud" => Exclusion,
        b"hue " => Hue,
        b"sat " => Saturation,
        b"colr" => Color,
        b"lum " => Luminosity,
        _ => Normal,
    }
}

/// Decode a PackBits (Photoshop RLE) row into `out`.
fn unpack_bits(src: &[u8], out: &mut Vec<u8>, len: usize) -> Option<()> {
    let mut i = 0;
    let end = out.len() + len;
    while out.len() < end {
        let n = *src.get(i)? as i8;
        i += 1;
        if n >= 0 {
            let k = n as usize + 1;
            out.extend_from_slice(src.get(i..i + k)?);
            i += k;
        } else if n != -128 {
            let v = *src.get(i)?;
            i += 1;
            out.extend(std::iter::repeat_n(v, (1 - n as isize) as usize));
        }
    }
    out.truncate(end);
    Some(())
}

/// Read one channel's image data (compression word + data) of `w × h` pixels at `pos`;
/// returns the plane and the position after it.
fn read_channel(d: &[u8], pos: usize, w: usize, h: usize, total_len: Option<usize>) -> Option<(Vec<u8>, usize)> {
    let comp = u16::from_be_bytes(d.get(pos..pos + 2)?.try_into().ok()?);
    let mut p = pos + 2;
    let mut plane = Vec::with_capacity(w * h);
    match comp {
        0 => {
            plane.extend_from_slice(d.get(p..p + w * h)?);
            p += w * h;
        }
        1 => {
            let counts: Vec<usize> = (0..h)
                .map(|r| u16::from_be_bytes([d[p + 2 * r], d[p + 2 * r + 1]]) as usize)
                .collect();
            p += 2 * h;
            for c in counts {
                unpack_bits(d.get(p..p + c)?, &mut plane, w)?;
                p += c;
            }
        }
        _ => return None, // zip compression is not supported
    }
    if let Some(t) = total_len {
        p = pos + t;
    }
    Some((plane, p))
}

/// Read an 8-bit RGB PSD (raw or RLE channels). `None` for other formats.
pub fn read_psd(d: &[u8]) -> Option<ReadPsd> {
    let be16 = |p: usize| -> Option<u16> { Some(u16::from_be_bytes(d.get(p..p + 2)?.try_into().ok()?)) };
    let be32 = |p: usize| -> Option<u32> { Some(u32::from_be_bytes(d.get(p..p + 4)?.try_into().ok()?)) };
    if d.get(..4)? != b"8BPS" || be16(4)? != 1 || be16(22)? != 8 || be16(24)? != 3 {
        return None;
    }
    let n_chan = be16(12)? as usize;
    let (h, w) = (be32(14)? as usize, be32(18)? as usize);
    let mut p = 26;
    p += 4 + be32(p)? as usize; // colour mode data
    p += 4 + be32(p)? as usize; // image resources
    let lm_len = be32(p)? as usize;
    let lm_end = p + 4 + lm_len;
    let mut layers = Vec::new();
    if lm_len > 0 {
        let li_len = be32(p + 4)? as usize;
        let mut q = p + 8;
        if li_len > 0 {
            let count = (i16::from_be_bytes(d.get(q..q + 2)?.try_into().ok()?)).unsigned_abs() as usize;
            q += 2;
            struct Rec {
                name: String,
                top: i32,
                left: i32,
                w: usize,
                h: usize,
                chans: Vec<(i16, usize)>,
                opacity: u8,
                clip: bool,
                hidden: bool,
                mode: MixMode,
                marker: bool,
            }
            let mut recs = Vec::with_capacity(count);
            for _ in 0..count {
                let rd = |o: usize| -> Option<i32> { Some(be32(q + o)? as i32) };
                let (top, left, bottom, right) = (rd(0)?, rd(4)?, rd(8)?, rd(12)?);
                let nc = be16(q + 16)? as usize;
                q += 18;
                let mut chans = Vec::new();
                for _ in 0..nc {
                    chans.push((i16::from_be_bytes(d.get(q..q + 2)?.try_into().ok()?), be32(q + 2)? as usize));
                    q += 6;
                }
                let mode = mode_from_key(d.get(q + 4..q + 8)?);
                let (opacity, clip, flags) = (d[q + 8], d[q + 9] != 0, d[q + 10]);
                q += 12;
                let extra = be32(q)? as usize;
                let ex_end = q + 4 + extra;
                let mut e = q + 4;
                e += 4 + be32(e)? as usize; // layer mask data
                e += 4 + be32(e)? as usize; // blending ranges
                let nlen = *d.get(e)? as usize;
                let name = String::from_utf8_lossy(d.get(e + 1..e + 1 + nlen)?).into_owned();
                e += (1 + nlen).div_ceil(4) * 4;
                // additional info: section dividers mark group starts/ends
                let mut marker = false;
                while e + 12 <= ex_end {
                    let key = d.get(e + 4..e + 8)?;
                    let len = be32(e + 8)? as usize;
                    if key == b"lsct" && be32(e + 12)? != 0 {
                        marker = true;
                    }
                    e += 12 + len + len % 2;
                }
                q = ex_end;
                recs.push(Rec {
                    name,
                    top,
                    left,
                    w: (right - left).max(0) as usize,
                    h: (bottom - top).max(0) as usize,
                    chans,
                    opacity,
                    clip,
                    hidden: flags & 0x02 != 0,
                    mode,
                    marker,
                });
            }
            for r in recs {
                let mut planes: Vec<(i16, Vec<u8>)> = Vec::new();
                for &(id, len) in &r.chans {
                    let (plane, next) = read_channel(d, q, r.w, r.h, Some(len))?;
                    planes.push((id, plane));
                    q = next;
                }
                if r.marker || r.w == 0 || r.h == 0 {
                    continue;
                }
                let get = |id: i16| planes.iter().find(|(i, _)| *i == id).map(|(_, p)| p);
                let n = r.w * r.h;
                let rgba = (0..n)
                    .flat_map(|i| {
                        let ch = |id: i16, d: u8| get(id).map_or(d, |p| p[i]);
                        [ch(0, 0), ch(1, 0), ch(2, 0), ch(-1, 255)]
                    })
                    .collect();
                layers.push(ReadLayer {
                    name: r.name,
                    left: r.left,
                    top: r.top,
                    w: r.w,
                    h: r.h,
                    rgba,
                    opacity: r.opacity as f32 / 255.0,
                    mode: r.mode,
                    visible: !r.hidden,
                    clip: r.clip,
                });
            }
        }
    }
    // merged image: one compression word for all channels, then planar data
    let comp = be16(lm_end)?;
    let mut planes: Vec<Vec<u8>> = Vec::new();
    let mut m = lm_end + 2;
    if comp == 0 {
        for _ in 0..n_chan.min(4) {
            planes.push(d.get(m..m + w * h)?.to_vec());
            m += w * h;
        }
    } else if comp == 1 {
        let counts: Vec<usize> = (0..h * n_chan).map(|r| u16::from_be_bytes([d[m + 2 * r], d[m + 2 * r + 1]]) as usize).collect();
        m += 2 * h * n_chan;
        for c in 0..n_chan.min(4) {
            let mut plane = Vec::with_capacity(w * h);
            for r in 0..h {
                let len = counts[c * h + r];
                unpack_bits(d.get(m..m + len)?, &mut plane, w)?;
                m += len;
            }
            planes.push(plane);
        }
    } else {
        return None;
    }
    let composite = (0..w * h)
        .flat_map(|i| [planes[0][i], planes[1][i], planes[2][i], planes.get(3).map_or(255, |p| p[i])])
        .collect();
    Some(ReadPsd { w, h, layers, composite })
}
