//! Volume presets: stored values for the five brush volumes (size, opacity, water or
//! oiliness, pressure, length) that can be applied to a brush in one step.

use crate::BrushParams;

/// Which volume a preset sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Volume {
    Size,
    Opacity,
    Water,
    Pressure,
    Length,
}

/// Lists of preset values per volume (size in pixels, the others 0–1).
#[derive(Clone, Debug, PartialEq)]
pub struct VolumePresets {
    pub size: Vec<f32>,
    pub opacity: Vec<f32>,
    pub water: Vec<f32>,
    pub pressure: Vec<f32>,
    pub length: Vec<f32>,
}

impl Default for VolumePresets {
    fn default() -> Self {
        Self {
            size: vec![5.0, 10.0, 20.0, 40.0, 80.0, 160.0],
            opacity: vec![0.1, 0.25, 0.5, 0.75, 1.0],
            water: vec![0.1, 0.25, 0.5, 0.75, 1.0],
            pressure: vec![0.25, 0.5, 0.75, 1.0],
            length: vec![0.25, 0.5, 0.75, 1.0],
        }
    }
}

impl VolumePresets {
    pub fn list(&self, v: Volume) -> &Vec<f32> {
        match v {
            Volume::Size => &self.size,
            Volume::Opacity => &self.opacity,
            Volume::Water => &self.water,
            Volume::Pressure => &self.pressure,
            Volume::Length => &self.length,
        }
    }

    fn list_mut(&mut self, v: Volume) -> &mut Vec<f32> {
        match v {
            Volume::Size => &mut self.size,
            Volume::Opacity => &mut self.opacity,
            Volume::Water => &mut self.water,
            Volume::Pressure => &mut self.pressure,
            Volume::Length => &mut self.length,
        }
    }

    /// Add the brush's current value of `v` as a preset (kept sorted, no duplicates).
    pub fn add_current(&mut self, v: Volume, p: &BrushParams) {
        let value = Self::current(v, p);
        let l = self.list_mut(v);
        if !l.iter().any(|&x| (x - value).abs() < 1e-6) {
            l.push(value);
            l.sort_by(f32::total_cmp);
        }
    }

    pub fn remove(&mut self, v: Volume, index: usize) {
        let l = self.list_mut(v);
        if index < l.len() {
            l.remove(index);
        }
    }

    /// Apply preset `index` of volume `v` to the brush. Pressure and length set the
    /// pressure multiplier and the stroke length limit where the brush has them.
    pub fn apply(&self, v: Volume, index: usize, p: &mut BrushParams) {
        let Some(&value) = self.list(v).get(index) else { return };
        match v {
            Volume::Size => p.size = value,
            Volume::Opacity => p.opacity = value,
            Volume::Water => p.water = value,
            Volume::Pressure => p.pressure_multiplier = value,
            Volume::Length => {} // paint length is not modelled yet
        }
    }

    /// The next preset after the brush's current value (wrapping), as an index.
    pub fn next_index(&self, v: Volume, p: &BrushParams) -> usize {
        let cur = Self::current(v, p);
        let l = self.list(v);
        l.iter().position(|&x| x > cur + 1e-6).unwrap_or(0)
    }

    fn current(v: Volume, p: &BrushParams) -> f32 {
        match v {
            Volume::Size => p.size,
            Volume::Opacity => p.opacity,
            Volume::Water => p.water,
            Volume::Pressure => p.pressure_multiplier,
            Volume::Length => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_add_and_cycle() {
        let mut vp = VolumePresets::default();
        let mut p = BrushParams::default();
        vp.apply(Volume::Size, 2, &mut p);
        assert_eq!(p.size, 20.0);
        assert_eq!(vp.next_index(Volume::Size, &p), 3);
        p.size = 33.0;
        vp.add_current(Volume::Size, &p);
        assert!(vp.size.contains(&33.0) && vp.size.windows(2).all(|w| w[0] < w[1]));
        p.size = 500.0;
        assert_eq!(vp.next_index(Volume::Size, &p), 0);
        vp.remove(Volume::Opacity, 0);
        assert_eq!(vp.opacity[0], 0.25);
    }
}
