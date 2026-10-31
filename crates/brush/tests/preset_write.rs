//! `BrushParams::write_preset` is the inverse of `BrushParams::from_preset`, through a
//! written `.wetbrush` file.

use wet_preset::BrushPreset;
use wet_brush::{BrushParams, PaintMode, RotationMode};

#[test]
fn edited_parameters_survive_a_written_preset() {
    let p = BrushParams {
        size: 42.0,
        opacity: 0.7,
        water: 0.3,
        spacing: 0.12,
        spacing_jitter: 0.2,
        size_jitter: 0.3,
        opacity_jitter: 0.4,
        rotation: 30.0,
        rotation_jitter: 0.5,
        scatter_jitter: 0.6,
        splat_opacity: 0.8,
        pressure_size_influence: 0.9,
        pressure_opacity_influence: 0.25,
        rotation_mode: RotationMode::FollowTrajectory,
        paint_mode: PaintMode::PaintBlend,
        glaze_blend: wet_compose::MixMode::Multiply,
        grain_scale: 2.0,
        grain_brush_size: 60.0,
        grain_follow_brush_size: true,
        grain_follow_distance: true,
        grain_follow_amount: 0.5,
        grain_follow_shape_rotation: true,
        splat_smudge: true,
        pressure_multiplier: 1.5,
        pressure_max: 0.9,
        paper_structure_influence: 0.35,
        ..BrushParams::default()
    };
    let mut preset = BrushPreset {
        tool: "oil".into(),
        name: "Edited".into(),
        parameters: serde_json::json!({ "unrelated": 7 }),
        ..Default::default()
    };
    p.write_preset(&mut preset);
    let back = BrushPreset::from_bytes(&preset.to_bytes()).unwrap();
    assert_eq!(back.parameters["unrelated"], 7, "other keys are kept");
    let q = BrushParams::from_preset(&back);
    assert_eq!(format!("{q:?}"), format!("{p:?}"));
}
