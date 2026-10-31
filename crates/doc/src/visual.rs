//! Watercolour behaviour presets: named sets of the 0–10 behaviour settings (paper
//! influence, edge drying, drying speed, impasto, drips, texture), applied to every layer.

/// One preset's 0–10 settings.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualPreset {
    pub name: String,
    pub paper_influence: f32,
    pub border_dry: f32,
    pub dry_speed: f32,
    pub impasto_depth: f32,
    pub drops_size: f32,
    pub drops_length: f32,
    pub texture_visibility: f32,
    pub texture_on_paint: f32,
    pub rewet: f32,
    /// Diffusion speed (simulation rate); not applied by the document.
    pub fps_simulation: f32,
    pub create_drops: bool,
}

/// The built-in presets.
pub fn presets() -> Vec<VisualPreset> {
    // paper influence, edge drying, drying speed, impasto, drop size, drop length, texture
    // visibility, texture on paint, rewet, simulation speed, drips
    let p = |name: &str, v: [f32; 10], create_drops: bool| VisualPreset {
        name: name.into(),
        paper_influence: v[0],
        border_dry: v[1],
        dry_speed: v[2],
        impasto_depth: v[3],
        drops_size: v[4],
        drops_length: v[5],
        texture_visibility: v[6],
        texture_on_paint: v[7],
        rewet: v[8],
        fps_simulation: v[9],
        create_drops,
    };
    vec![
        p("Default", [5.0; 10], true),
        p("Smooth Paper", [2.0, 4.0, 5.0, 5.0, 5.0, 5.0, 2.0, 3.0, 5.0, 5.0], true),
        p("Textured Paper", [8.0, 6.0, 5.0, 5.0, 5.0, 5.0, 8.0, 7.0, 5.0, 5.0], true),
        p("Slow Drying", [5.0, 3.0, 2.0, 5.0, 6.0, 7.0, 5.0, 5.0, 7.0, 5.0], true),
        p("Quick Drying", [5.0, 7.0, 8.0, 5.0, 4.0, 3.0, 5.0, 5.0, 3.0, 5.0], false),
    ]
}

impl crate::Document {
    /// Apply a Visual Settings preset to every layer and the paper display.
    pub fn apply_visual_preset(&mut self, p: &VisualPreset) {
        for l in &mut self.layers {
            let w = &mut l.wet.params;
            w.absorbency = p.dry_speed;
            w.border_dry = p.border_dry;
            w.paper_influence = p.paper_influence;
            w.rewet = p.rewet;
            w.drops_size = p.drops_size;
            w.drops_length = p.drops_length;
            w.create_drops = p.create_drops && w.create_drops;
        }
        self.impasto = p.impasto_depth / 5.0;
        self.paper.shading = p.texture_visibility / 5.0 * 0.094;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply() {
        let ps = presets();
        assert!(ps.iter().all(|p| (0.0..=10.0).contains(&p.paper_influence)));
        let p = &ps[2];
        let mut d = crate::Document::new(16, 16);
        d.apply_visual_preset(p);
        assert_eq!(d.layers[0].wet.params.absorbency, p.dry_speed);
        assert_eq!(d.layers[0].wet.params.border_dry, p.border_dry);
        assert!((d.paper.shading - 0.094 * p.texture_visibility / 5.0).abs() < 1e-6);
    }
}
