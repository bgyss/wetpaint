//! Stroke interpolation: pointer samples → evenly spaced dabs.

/// One pointer sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub x: f32,
    pub y: f32,
    pub pressure: f32,
}

/// One dab to stamp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DabPos {
    pub x: f32,
    pub y: f32,
    pub pressure: f32,
    /// Stroke direction in radians (for trajectory-following rotation).
    pub angle: f32,
}

/// Small deterministic generator (xorshift64*), seeded per stroke so strokes replay.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Uniform in `[-1, 1)`.
    pub fn signed(&mut self) -> f32 {
        2.0 * self.next_f32() - 1.0
    }
}

/// Walks a polyline of samples and emits dabs every `spacing(pressure)` pixels.
#[derive(Clone)]
pub struct Spacer {
    last: Option<Sample>,
    /// Distance still to travel before the next dab.
    remaining: f32,
    pub dabs_emitted: usize,
}

impl Default for Spacer {
    fn default() -> Self {
        Self::new()
    }
}

impl Spacer {
    pub fn new() -> Self {
        Self {
            last: None,
            remaining: 0.0,
            dabs_emitted: 0,
        }
    }

    /// The pointer position the next dabs start from.
    pub fn last(&self) -> Option<Sample> {
        self.last
    }

    /// Start a stroke. The press itself places no dab; a release at the same position does
    /// (see [`Spacer::release`]), matching the documented pointer semantics.
    pub fn press(&mut self, s: Sample) {
        self.last = Some(s);
        self.remaining = 0.0;
        self.dabs_emitted = 0;
    }

    /// Advance to `s`, calling `emit` for each dab. `spacing` maps pressure to the next
    /// gap in pixels (it is called once per dab, so it may include jitter).
    pub fn move_to(
        &mut self,
        s: Sample,
        mut spacing: impl FnMut(f32) -> f32,
        mut emit: impl FnMut(DabPos),
    ) {
        let Some(a) = self.last else {
            self.press(s);
            return;
        };
        let (dx, dy) = (s.x - a.x, s.y - a.y);
        let len = (dx * dx + dy * dy).sqrt();
        if len <= 0.0 {
            return;
        }
        let angle = dy.atan2(dx);
        let mut t = self.remaining;
        while t <= len {
            let f = t / len;
            let p = a.pressure + (s.pressure - a.pressure) * f;
            emit(DabPos {
                x: a.x + dx * f,
                y: a.y + dy * f,
                pressure: p,
                angle,
            });
            self.dabs_emitted += 1;
            t += spacing(p).max(0.05);
        }
        self.remaining = t - len;
        self.last = Some(s);
    }

    /// End the stroke; a stroke that never moved stamps a single dab.
    pub fn release(&mut self, s: Sample, mut emit: impl FnMut(DabPos)) {
        if self.dabs_emitted == 0 {
            emit(DabPos {
                x: s.x,
                y: s.y,
                pressure: s.pressure,
                angle: 0.0,
            });
            self.dabs_emitted = 1;
        }
        self.last = None;
    }
}
