//! Opacity LUT `L` mapping pigment amount → coverage weight.
//!
//! `table[0] = 0`, `table[i+1] = 0.998·table[i] + 0.002`
//! (`table[i] = 1 - 0.998^i`).

pub const LUT_LEN: usize = 3000;

#[derive(Clone, Debug)]
pub struct OpacityLut {
    pub table: Vec<f32>,
}

impl OpacityLut {
    /// The standard table, built by the recurrence above.
    pub fn standard() -> Self {
        let mut table = vec![0.0f32; LUT_LEN + 1];
        for i in 0..LUT_LEN {
            table[i + 1] = table[i] * 0.998 + 0.002;
        }
        Self { table }
    }

    /// Inverse mapping coverage → amount: `-499.5063·ln(1 - c)`.
    pub fn inverse(c: f32) -> f32 {
        -499.5063 * (1.0 - c).max(1e-12).ln()
    }

    /// `L(x)`: 0 for `x ≤ 0.01`, 1 for `x ≥ 3000`, otherwise linear interpolation.
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        if !(x > 0.01) {
            return 0.0;
        }
        if x >= LUT_LEN as f32 {
            return 1.0;
        }
        let i = x as usize;
        let t = x - i as f32;
        let a = self.table[i];
        a + t * (self.table[i + 1] - a)
    }
}

impl Default for OpacityLut {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints() {
        let l = OpacityLut::default();
        assert_eq!(l.eval(0.0), 0.0);
        assert_eq!(l.eval(0.01), 0.0);
        assert_eq!(l.eval(5000.0), 1.0);
        assert!(l.eval(100.0) > l.eval(10.0));
        assert!((l.eval(500.0) - (1.0 - 0.998f32.powi(500))).abs() < 1e-4);
        assert!((OpacityLut::inverse(l.eval(800.0)) - 800.0).abs() < 1.0);
    }
}
