//! Brush tip images: the dab shape and the paper-space grain.

/// A single-channel image with values in 0..=1.
#[derive(Clone, Debug)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
    /// Box-filtered half-resolution levels, built on first minified sample (do not
    /// modify `data` after sampling).
    mips: std::sync::OnceLock<Vec<Mask>>,
    /// Maximum of each level (base first), for `sample_lod_normalized`.
    mip_max: std::sync::OnceLock<Vec<f32>>,
}

impl Mask {
    pub fn new(w: usize, h: usize, data: Vec<f32>) -> Self {
        Self { w, h, data, mips: std::sync::OnceLock::new(), mip_max: std::sync::OnceLock::new() }
    }

    /// Sample at normalised coordinates like `sample_clamped`, prefiltered for a texture
    /// minified by `texels_per_px` (trilinear over a box-filtered pyramid). Point sampling
    /// a sparse tip (dotted bristle shapes) at a small dab size misses most of its dots.
    pub fn sample_lod(&self, u: f32, v: f32, texels_per_px: f32) -> f32 {
        if texels_per_px <= 1.0 {
            return self.sample_clamped(u, v);
        }
        let mips = self.mips();
        let lod = texels_per_px.log2().min(mips.len() as f32);
        let l0 = lod.floor() as usize;
        let level = |l: usize| if l == 0 { self.sample_clamped(u, v) } else { mips[l - 1].sample_clamped(u, v) };
        let a = level(l0);
        let f = lod - l0 as f32;
        if f <= 0.0 || l0 >= mips.len() { a } else { a + (level(l0 + 1) - a) * f }
    }

    /// `sample_lod` rescaled so the prefiltered image peaks at 1 (a sparse speckle seen
    /// at a small size becomes a soft mottled mask instead of a faint one).
    pub fn sample_lod_normalized(&self, u: f32, v: f32, texels_per_px: f32) -> f32 {
        let maxes = self.mip_max.get_or_init(|| {
            let max = |m: &Mask| m.data.iter().copied().fold(0.0f32, f32::max);
            std::iter::once(max(self)).chain(self.mips().iter().map(max)).collect()
        });
        let lod = texels_per_px.max(1.0).log2().min((maxes.len() - 1) as f32);
        let (l0, f) = (lod.floor() as usize, lod.fract());
        let peak = maxes[l0] + (maxes[(l0 + 1).min(maxes.len() - 1)] - maxes[l0]) * f;
        if peak <= 0.0 { 0.0 } else { (self.sample_lod(u, v, texels_per_px) / peak).min(1.0) }
    }

    fn mips(&self) -> &Vec<Mask> {
        self.mips.get_or_init(|| {
            let mut out: Vec<Mask> = Vec::new();
            loop {
                let src = out.last().unwrap_or(self);
                if src.w < 2 || src.h < 2 {
                    break;
                }
                let (w, h) = (src.w / 2, src.h / 2);
                let data = (0..w * h)
                    .map(|i| {
                        let (x, y) = (2 * (i % w), 2 * (i / w));
                        let at = |dx: usize, dy: usize| src.data[(y + dy) * src.w + x + dx];
                        0.25 * (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1))
                    })
                    .collect();
                out.push(Mask::new(w, h, data));
            }
            out
        })
    }

    /// Decode an 8-bit (or 16-bit) grayscale/RGB(A) PNG; colour images use the first channel.
    pub fn from_png(bytes: &[u8]) -> Result<Self, png::DecodingError> {
        let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
        dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = dec.read_info()?;
        let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
        let info = reader.next_frame(&mut buf)?;
        let ch = info.color_type.samples();
        let (w, h) = (info.width as usize, info.height as usize);
        let data = (0..w * h)
            .map(|i| buf[i * ch] as f32 / 255.0)
            .collect();
        Ok(Self::new(w, h, data))
    }

    /// A soft round disk (used when a preset has no shape texture).
    pub fn disk(n: usize, hardness: f32) -> Self {
        let c = (n as f32 - 1.0) / 2.0;
        let data = (0..n * n)
            .map(|i| {
                let (x, y) = ((i % n) as f32 - c, (i / n) as f32 - c);
                let r = (x * x + y * y).sqrt() / c;
                if r >= 1.0 {
                    0.0
                } else if r <= hardness {
                    1.0
                } else {
                    let t = (1.0 - r) / (1.0 - hardness).max(1e-6);
                    t * t * (3.0 - 2.0 * t)
                }
            })
            .collect();
        Self::new(n, n, data)
    }

    /// Bilinear sample at normalised coordinates `u, v` in `[0, 1]`; zero outside.
    pub fn sample_clamped(&self, u: f32, v: f32) -> f32 {
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return 0.0;
        }
        self.bilinear(u * (self.w - 1) as f32, v * (self.h - 1) as f32, false)
    }

    /// Bilinear sample at pixel coordinates, wrapping (for tiled grain).
    pub fn sample_tiled(&self, x: f32, y: f32) -> f32 {
        self.bilinear(x, y, true)
    }

    fn bilinear(&self, x: f32, y: f32, wrap: bool) -> f32 {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |xi: i64, yi: i64| -> f32 {
            let (w, h) = (self.w as i64, self.h as i64);
            let (xi, yi) = if wrap {
                (xi.rem_euclid(w), yi.rem_euclid(h))
            } else {
                (xi.clamp(0, w - 1), yi.clamp(0, h - 1))
            };
            self.data[(yi * w + xi) as usize]
        };
        let (xi, yi) = (x0 as i64, y0 as i64);
        let top = at(xi, yi) * (1.0 - fx) + at(xi + 1, yi) * fx;
        let bot = at(xi, yi + 1) * (1.0 - fx) + at(xi + 1, yi + 1) * fx;
        top * (1.0 - fy) + bot * fy
    }

    pub fn mean(&self) -> f32 {
        self.data.iter().sum::<f32>() / self.data.len().max(1) as f32
    }
}
