//! `wet-color`: RGB colour spaces for document colour management, written from the
//! published colour-space definitions (primaries, white point, transfer curve) and the
//! public ICC profile format.
//!
//! Supported working spaces are the common D65 RGB spaces: sRGB, Display P3, Adobe RGB
//! (1998) and Rec. 2020. Conversions are relative colorimetric with clipping. ICC v2
//! matrix/TRC profiles can be generated for embedding in exported files. CMYK, soft
//! proofing and arbitrary installed profiles are out of scope.

/// A D65 RGB working space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RgbSpace {
    Srgb,
    DisplayP3,
    AdobeRgb,
    Rec2020,
}

type M3 = [[f64; 3]; 3];

const D65: (f64, f64) = (0.3127, 0.3290);
/// ICC profile connection space white (D50) as XYZ.
const D50_XYZ: [f64; 3] = [0.9642, 1.0, 0.8249];

impl RgbSpace {
    pub const ALL: [RgbSpace; 4] = [RgbSpace::Srgb, RgbSpace::DisplayP3, RgbSpace::AdobeRgb, RgbSpace::Rec2020];

    pub fn name(self) -> &'static str {
        match self {
            RgbSpace::Srgb => "sRGB",
            RgbSpace::DisplayP3 => "Display P3",
            RgbSpace::AdobeRgb => "Adobe RGB (1998)",
            RgbSpace::Rec2020 => "Rec. 2020",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.name().eq_ignore_ascii_case(s))
    }

    /// Red, green, blue chromaticities (x, y).
    fn primaries(self) -> [(f64, f64); 3] {
        match self {
            RgbSpace::Srgb => [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)],
            RgbSpace::DisplayP3 => [(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)],
            RgbSpace::AdobeRgb => [(0.64, 0.33), (0.21, 0.71), (0.15, 0.06)],
            RgbSpace::Rec2020 => [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)],
        }
    }

    /// Encoded value (0–1) → linear light.
    pub fn decode(self, v: f64) -> f64 {
        let v = v.clamp(0.0, 1.0);
        match self {
            RgbSpace::Srgb | RgbSpace::DisplayP3 => {
                if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
            }
            RgbSpace::AdobeRgb => v.powf(563.0 / 256.0),
            RgbSpace::Rec2020 => {
                if v < 0.081 { v / 4.5 } else { ((v + 0.099) / 1.099).powf(1.0 / 0.45) }
            }
        }
    }

    /// Linear light → encoded value (0–1).
    pub fn encode(self, l: f64) -> f64 {
        let l = l.clamp(0.0, 1.0);
        match self {
            RgbSpace::Srgb | RgbSpace::DisplayP3 => {
                if l <= 0.0031308 { 12.92 * l } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 }
            }
            RgbSpace::AdobeRgb => l.powf(256.0 / 563.0),
            RgbSpace::Rec2020 => {
                if l < 0.018 { 4.5 * l } else { 1.099 * l.powf(0.45) - 0.099 }
            }
        }
    }

    /// Linear RGB → XYZ (D65, Y = 1 for white).
    fn to_xyz(self) -> M3 {
        let xyz = |(x, y): (f64, f64)| [x / y, 1.0, (1.0 - x - y) / y];
        let p = self.primaries().map(xyz);
        let m = [[p[0][0], p[1][0], p[2][0]], [p[0][1], p[1][1], p[2][1]], [p[0][2], p[1][2], p[2][2]]];
        let s = mul_v(inv(m), xyz(D65));
        [0, 1, 2].map(|r| [0, 1, 2].map(|c| m[r][c] * s[c]))
    }
}

fn mul(a: M3, b: M3) -> M3 {
    [0, 1, 2].map(|r| [0, 1, 2].map(|c| (0..3).map(|k| a[r][k] * b[k][c]).sum()))
}

fn mul_v(a: M3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| (0..3).map(|k| a[r][k] * v[k]).sum())
}

fn inv(m: M3) -> M3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [c(1, 1, 2, 2), -c(0, 1, 2, 2), c(0, 1, 1, 2)],
        [-c(1, 0, 2, 2), c(0, 0, 2, 2), -c(0, 0, 1, 2)],
        [c(1, 0, 2, 1), -c(0, 0, 2, 1), c(0, 0, 1, 1)],
    ]
    .map(|row| row.map(|v| v / det))
}

/// A tone response curve.
#[derive(Clone, Debug, PartialEq)]
pub enum Trc {
    /// One of the built-in spaces' curves.
    Builtin(RgbSpace),
    Gamma(f64),
    /// Sampled curve (encoded 0–1 evenly spaced → linear 0–1).
    Table(Vec<f64>),
    /// ICC parametric curve type 0–4 (`g, a, b, c, d, e, f`).
    Parametric([f64; 7], u16),
}

impl Trc {
    fn decode(&self, v: f64) -> f64 {
        let v = v.clamp(0.0, 1.0);
        match self {
            Trc::Builtin(s) => s.decode(v),
            Trc::Gamma(g) => v.powf(*g),
            Trc::Table(t) => {
                if t.len() < 2 {
                    return v;
                }
                let x = v * (t.len() - 1) as f64;
                let i = (x.floor() as usize).min(t.len() - 2);
                t[i] + (t[i + 1] - t[i]) * (x - i as f64)
            }
            Trc::Parametric([g, a, b, c, d, e, f], kind) => match kind {
                0 => v.powf(*g),
                1 => if v >= -b / a { (a * v + b).powf(*g) } else { 0.0 },
                2 => if v >= -b / a { (a * v + b).powf(*g) + c } else { *c },
                3 => if v >= *d { (a * v + b).powf(*g) } else { c * v },
                _ => if v >= *d { (a * v + b).powf(*g) + e } else { c * v + f },
            }
            .clamp(0.0, 1.0),
        }
    }

    fn encode(&self, l: f64) -> f64 {
        let l = l.clamp(0.0, 1.0);
        match self {
            Trc::Builtin(s) => s.encode(l),
            Trc::Gamma(g) => l.powf(1.0 / g),
            // invert monotone curves by bisection
            _ => {
                let (mut lo, mut hi) = (0.0, 1.0);
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    if self.decode(mid) < l { lo = mid } else { hi = mid }
                }
                0.5 * (lo + hi)
            }
        }
    }
}

/// A general RGB space: linear RGB → XYZ (D65-relative) and a tone curve per channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Space {
    to_xyz: M3,
    trc: [Trc; 3],
}

impl From<RgbSpace> for Space {
    fn from(s: RgbSpace) -> Self {
        Space { to_xyz: s.to_xyz(), trc: [Trc::Builtin(s), Trc::Builtin(s), Trc::Builtin(s)] }
    }
}

impl Space {
    /// A matrix/TRC RGB display profile (ICC v2 or v4, as most installed RGB profiles
    /// are): colourants `rXYZ gXYZ bXYZ` (D50, adapted back to D65) and `rTRC gTRC bTRC`
    /// (`curv` or `para`). `None` for other profile kinds (LUT-based, CMYK, …).
    pub fn from_icc(p: &[u8]) -> Option<Self> {
        let be32 = |o: usize| -> Option<u32> { Some(u32::from_be_bytes(p.get(o..o + 4)?.try_into().ok()?)) };
        if p.get(36..40)? != b"acsp" || p.get(16..20)? != b"RGB " {
            return None;
        }
        let n = be32(128)? as usize;
        let tag = |sig: &[u8; 4]| -> Option<&[u8]> {
            (0..n).find_map(|i| {
                let e = 132 + 12 * i;
                (p.get(e..e + 4)? == sig).then(|| {
                    let (off, len) = (be32(e + 4)? as usize, be32(e + 8)? as usize);
                    p.get(off..off + len)
                })?
            })
        };
        let s15 = |d: &[u8], o: usize| -> Option<f64> { Some(i32::from_be_bytes(d.get(o..o + 4)?.try_into().ok()?) as f64 / 65536.0) };
        let xyz = |sig: &[u8; 4]| -> Option<[f64; 3]> {
            let d = tag(sig)?;
            (d.get(..4)? == b"XYZ ").then_some(())?;
            Some([s15(d, 8)?, s15(d, 12)?, s15(d, 16)?])
        };
        let trc = |sig: &[u8; 4]| -> Option<Trc> {
            let d = tag(sig)?;
            match d.get(..4)? {
                b"curv" => {
                    let count = u32::from_be_bytes(d.get(8..12)?.try_into().ok()?) as usize;
                    Some(match count {
                        0 => Trc::Gamma(1.0),
                        1 => Trc::Gamma(u16::from_be_bytes(d.get(12..14)?.try_into().ok()?) as f64 / 256.0),
                        _ => Trc::Table(
                            (0..count)
                                .map(|i| Some(u16::from_be_bytes(d.get(12 + 2 * i..14 + 2 * i)?.try_into().ok()?) as f64 / 65535.0))
                                .collect::<Option<Vec<_>>>()?,
                        ),
                    })
                }
                b"para" => {
                    let kind = u16::from_be_bytes(d.get(8..10)?.try_into().ok()?);
                    let np = [1, 3, 4, 5, 7][kind.min(4) as usize];
                    let mut v = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    for (i, slot) in v.iter_mut().enumerate().take(np) {
                        *slot = s15(d, 12 + 4 * i)?;
                    }
                    Some(Trc::Parametric(v, kind))
                }
                _ => None,
            }
        };
        let (r, g, b) = (xyz(b"rXYZ")?, xyz(b"gXYZ")?, xyz(b"bXYZ")?);
        let d50 = [[r[0], g[0], b[0]], [r[1], g[1], b[1]], [r[2], g[2], b[2]]];
        let to_xyz = mul(inv(bradford_d65_to_d50()), d50);
        Some(Space { to_xyz, trc: [trc(b"rTRC")?, trc(b"gTRC")?, trc(b"bTRC")?] })
    }
}

/// A converter between two RGB spaces (relative colorimetric, clipped).
pub struct Converter {
    from: Space,
    to: Space,
    m: M3,
    identity: bool,
}

impl Converter {
    pub fn new(from: RgbSpace, to: RgbSpace) -> Self {
        let mut c = Self::between(from.into(), to.into());
        c.identity = from == to;
        c
    }

    /// A converter between any two spaces (built-in or from ICC profiles).
    pub fn between(from: Space, to: Space) -> Self {
        let m = mul(inv(to.to_xyz), from.to_xyz);
        Self { from, to, m, identity: false }
    }

    /// The colour in the destination space before clipping (linear), for gamut checks.
    fn linear_out(&self, rgb: [f32; 3]) -> [f64; 3] {
        let lin = [0, 1, 2].map(|k| self.from.trc[k].decode(rgb[k] as f64));
        mul_v(self.m, lin)
    }

    /// Convert an encoded colour (0–1 per channel).
    pub fn convert(&self, rgb: [f32; 3]) -> [f32; 3] {
        if self.identity {
            return rgb;
        }
        let out = self.linear_out(rgb);
        [0, 1, 2].map(|k| self.to.trc[k].encode(out[k]) as f32)
    }

    /// Whether the colour is representable in the destination space (within `tol`).
    pub fn in_gamut(&self, rgb: [f32; 3], tol: f64) -> bool {
        self.identity || self.linear_out(rgb).iter().all(|&v| v >= -tol && v <= 1.0 + tol)
    }
}

/// Soft proof: how a colour of `working` looks after a round trip through `proof`
/// (clipped to the proof gamut), and whether it was out of that gamut.
pub fn soft_proof(working: &Space, proof: &Space, rgb: [f32; 3]) -> ([f32; 3], bool) {
    let there = Converter::between(working.clone(), proof.clone());
    let back = Converter::between(proof.clone(), working.clone());
    (back.convert(there.convert(rgb)), !there.in_gamut(rgb, 1e-3))
}

/// Bradford chromatic adaptation from D65 to D50 (for ICC colorant tags).
fn bradford_d65_to_d50() -> M3 {
    const B: M3 = [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]];
    let w65 = [D65.0 / D65.1, 1.0, (1.0 - D65.0 - D65.1) / D65.1];
    let (s, d) = (mul_v(B, w65), mul_v(B, D50_XYZ));
    let scale = [[d[0] / s[0], 0.0, 0.0], [0.0, d[1] / s[1], 0.0], [0.0, 0.0, d[2] / s[2]]];
    mul(inv(B), mul(scale, B))
}

/// An ICC v2 display profile (matrix/TRC) describing `space`, for embedding.
pub fn icc_profile(space: RgbSpace) -> Vec<u8> {
    let s15 = |v: f64| ((v * 65536.0).round() as i32).to_be_bytes();
    let xyz_tag = |v: [f64; 3]| {
        let mut t = b"XYZ \0\0\0\0".to_vec();
        v.iter().for_each(|&c| t.extend(s15(c)));
        t
    };
    let m = mul(bradford_d65_to_d50(), space.to_xyz());
    let col = |c: usize| [m[0][c], m[1][c], m[2][c]];
    let trc = {
        let mut t = b"curv\0\0\0\0".to_vec();
        match space {
            RgbSpace::AdobeRgb => {
                t.extend(1u32.to_be_bytes());
                t.extend(((563.0f64 / 256.0 * 256.0).round() as u16).to_be_bytes());
            }
            _ => {
                t.extend(1024u32.to_be_bytes());
                for i in 0..1024 {
                    let v = space.decode(i as f64 / 1023.0);
                    t.extend(((v * 65535.0).round() as u16).to_be_bytes());
                }
            }
        }
        t
    };
    let desc = {
        let name = format!("wet-color {}", space.name());
        let mut t = b"desc\0\0\0\0".to_vec();
        t.extend((name.len() as u32 + 1).to_be_bytes());
        t.extend(name.as_bytes());
        t.push(0);
        t.extend([0u8; 4 + 4 + 2 + 1 + 67]); // no Unicode / ScriptCode descriptions
        t
    };
    let cprt = {
        let mut t = b"text\0\0\0\0".to_vec();
        t.extend(b"No copyright, use freely\0");
        t
    };
    let wtpt = xyz_tag([D65.0 / D65.1, 1.0, (1.0 - D65.0 - D65.1) / D65.1]);
    // (signature, data); the three TRC tags share one curve
    let tags: Vec<(&[u8; 4], Vec<u8>)> = vec![
        (b"desc", desc),
        (b"cprt", cprt),
        (b"wtpt", wtpt),
        (b"rXYZ", xyz_tag(col(0))),
        (b"gXYZ", xyz_tag(col(1))),
        (b"bXYZ", xyz_tag(col(2))),
        (b"rTRC", trc),
    ];
    let n_entries = tags.len() + 2; // gTRC and bTRC point at rTRC's data
    let mut data = Vec::new();
    let mut table = Vec::new();
    let mut offset = 128 + 4 + 12 * n_entries;
    let mut trc_entry = (0, 0);
    for (sig, t) in &tags {
        table.push((**sig, offset, t.len()));
        if *sig == b"rTRC" {
            trc_entry = (offset, t.len());
        }
        data.extend(t);
        offset += t.len();
        while data.len() % 4 != 0 {
            data.push(0);
            offset += 1;
        }
    }
    table.push((*b"gTRC", trc_entry.0, trc_entry.1));
    table.push((*b"bTRC", trc_entry.0, trc_entry.1));
    let size = 128 + 4 + 12 * n_entries + data.len();
    let mut p = Vec::with_capacity(size);
    p.extend((size as u32).to_be_bytes());
    p.extend([0u8; 4]); // preferred CMM
    p.extend(0x0210_0000u32.to_be_bytes()); // version 2.1
    p.extend(b"mntrRGB XYZ ");
    p.extend([0u8; 12]); // date
    p.extend(b"acsp");
    p.extend([0u8; 4 + 4 + 4 + 4 + 8]); // platform, flags, manufacturer, model, attributes
    p.extend(0u32.to_be_bytes()); // perceptual intent
    D50_XYZ.iter().for_each(|&c| p.extend(s15(c)));
    p.extend([0u8; 4 + 16 + 28]); // creator, ID, reserved
    p.extend((n_entries as u32).to_be_bytes());
    for (sig, off, len) in table {
        p.extend(sig);
        p.extend((off as u32).to_be_bytes());
        p.extend((len as u32).to_be_bytes());
    }
    p.extend(data);
    debug_assert_eq!(p.len(), size);
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_matrix_matches_the_standard() {
        let m = RgbSpace::Srgb.to_xyz();
        // published sRGB → XYZ (D65) matrix, first row
        assert!((m[0][0] - 0.4124).abs() < 1e-3 && (m[0][1] - 0.3576).abs() < 1e-3 && (m[0][2] - 0.1805).abs() < 1e-3);
        assert!((m[1][0] + m[1][1] + m[1][2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn round_trips_and_gamut() {
        for s in RgbSpace::ALL {
            for v in [0.0, 0.02, 0.3, 0.75, 1.0] {
                assert!((s.encode(s.decode(v)) - v).abs() < 1e-9, "{s:?} {v}");
            }
        }
        let to_p3 = Converter::new(RgbSpace::Srgb, RgbSpace::DisplayP3);
        let back = Converter::new(RgbSpace::DisplayP3, RgbSpace::Srgb);
        let c = [0.8, 0.3, 0.1];
        let r = back.convert(to_p3.convert(c));
        assert!(c.iter().zip(r).all(|(a, b)| (a - b).abs() < 1e-3), "{r:?}");
        // pure sRGB red is inside P3: less saturated there
        let red = to_p3.convert([1.0, 0.0, 0.0]);
        assert!(red[0] < 1.0 && red[1] > 0.0);
        // white stays white
        let w = to_p3.convert([1.0, 1.0, 1.0]);
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-4), "{w:?}");
    }

    #[test]
    fn matches_an_independent_cmm() {
        // LittleCMS (relative colorimetric) on our generated profiles, sRGB → space
        let cases: [(RgbSpace, [[u8; 3]; 3]); 3] = [
            (RgbSpace::DisplayP3, [[187, 105, 62], [76, 158, 97], [90, 90, 212]]),
            (RgbSpace::AdobeRgb, [[177, 100, 56], [93, 159, 94], [90, 90, 215]]),
            (RgbSpace::Rec2020, [[161, 97, 47], [86, 144, 84], [86, 78, 206]]),
        ];
        let src = [[200u8, 100, 50], [30, 160, 90], [90, 90, 220]];
        for (space, want) in cases {
            let cv = Converter::new(RgbSpace::Srgb, space);
            for (s, w) in src.iter().zip(want) {
                let got = cv.convert(s.map(|v| v as f32 / 255.0)).map(|v| (v * 255.0).round() as i32);
                assert!(got.iter().zip(w).all(|(&g, w)| (g - w as i32).abs() <= 2), "{space:?} {s:?}: {got:?} vs {w:?}");
            }
        }
    }

    #[test]
    fn icc_round_trip_and_soft_proof() {
        for s in RgbSpace::ALL {
            let parsed = Space::from_icc(&icc_profile(s)).unwrap();
            let c = Converter::between(parsed, s.into());
            for rgb in [[0.8, 0.3, 0.1], [0.1, 0.5, 0.9], [0.5, 0.5, 0.5]] {
                let r = c.convert(rgb);
                assert!(rgb.iter().zip(r).all(|(a, b)| (a - b).abs() < 2e-3), "{s:?} {rgb:?} -> {r:?}");
            }
        }
        // saturated P3 green is outside sRGB; a mid grey is inside
        let (p3, srgb): (Space, Space) = (RgbSpace::DisplayP3.into(), RgbSpace::Srgb.into());
        let (_, out) = soft_proof(&p3, &srgb, [0.0, 1.0, 0.0]);
        assert!(out);
        let (g, out) = soft_proof(&p3, &srgb, [0.5, 0.5, 0.5]);
        assert!(!out && g.iter().all(|&v| (v - 0.5).abs() < 2e-3), "{g:?}");
        // parametric curve: the sRGB-style type 3 curve decodes 0.5 like the built-in one
        let t = Trc::Parametric([2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045, 0.0, 0.0], 3);
        assert!((t.decode(0.5) - RgbSpace::Srgb.decode(0.5)).abs() < 1e-6);
    }

    #[test]
    fn icc_profile_layout() {
        for s in RgbSpace::ALL {
            let p = icc_profile(s);
            assert_eq!(u32::from_be_bytes(p[0..4].try_into().unwrap()) as usize, p.len());
            assert_eq!(&p[36..40], b"acsp");
            let n = u32::from_be_bytes(p[128..132].try_into().unwrap()) as usize;
            assert_eq!(n, 9);
            for i in 0..n {
                let e = &p[132 + 12 * i..144 + 12 * i];
                let (off, len) = (u32::from_be_bytes(e[4..8].try_into().unwrap()) as usize, u32::from_be_bytes(e[8..12].try_into().unwrap()) as usize);
                assert!(off % 4 == 0 && off + len <= p.len());
            }
        }
    }
}

#[cfg(test)]
mod installed {
    use super::*;

    /// Installed system profiles (macOS) agree with the built-in spaces of the same name.
    #[test]
    fn system_profiles_match_builtins() {
        let dir = std::path::Path::new("/System/Library/ColorSync/Profiles");
        for (file, space) in [
            ("sRGB Profile.icc", RgbSpace::Srgb),
            ("Display P3.icc", RgbSpace::DisplayP3),
            ("AdobeRGB1998.icc", RgbSpace::AdobeRgb),
        ] {
            let Ok(bytes) = std::fs::read(dir.join(file)) else { continue };
            let parsed = Space::from_icc(&bytes).unwrap_or_else(|| panic!("{file} not parsed"));
            let c = Converter::between(parsed, space.into());
            for rgb in [[0.8f32, 0.3, 0.1], [0.1, 0.5, 0.9], [0.5, 0.5, 0.5], [0.05, 0.9, 0.2]] {
                let r = c.convert(rgb);
                assert!(rgb.iter().zip(r).all(|(a, b)| (a - b).abs() < 0.01), "{file} {rgb:?} -> {r:?}");
            }
        }
    }
}
