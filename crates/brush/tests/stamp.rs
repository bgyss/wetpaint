use wet_brush::{Brush, BrushParams, Mask, Sample, Spacer};
use wet_fluid::WetLayer;

fn soft_brush() -> Brush {
    let params = BrushParams {
        size: 20.0,
        spacing: 0.05,
        splat_opacity: 0.5, // as the calibration reference brush
        grain_follow_distance: true, // its streak grain (no anchored-grain gain)
        ..BrushParams::default()
    };
    let mut b = Brush::new(params, Mask::disk(64, 0.6), None);
    b.color = [200.0, 40.0, 40.0];
    b
}

fn paint_line(b: &mut Brush, l: &mut WetLayer, y: f32) {
    b.press(Sample { x: 20.0, y, pressure: 1.0 });
    for i in 1..=20 {
        b.move_to(l, Sample { x: 20.0 + 4.0 * i as f32, y, pressure: 1.0 });
    }
    b.release(l, Sample { x: 100.0, y, pressure: 1.0 });
}

#[test]
fn spacer_emits_evenly() {
    let mut s = Spacer::new();
    s.press(Sample { x: 0.0, y: 0.0, pressure: 1.0 });
    let mut xs = Vec::new();
    s.move_to(Sample { x: 10.0, y: 0.0, pressure: 1.0 }, |_| 2.5, |d| xs.push(d.x));
    s.move_to(Sample { x: 20.0, y: 0.0, pressure: 1.0 }, |_| 2.5, |d| xs.push(d.x));
    assert_eq!(xs, vec![0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]);
}

#[test]
fn single_tap_places_one_dab() {
    let mut b = soft_brush();
    let mut l = WetLayer::new(64, 64);
    b.press(Sample { x: 32.0, y: 32.0, pressure: 1.0 });
    b.release(&mut l, Sample { x: 32.0, y: 32.0, pressure: 1.0 });
    let c = 32 * 64 + 32;
    assert!(l.dens.amount.cur().data[c] > 0.0);
    assert_eq!(l.maps.open.data[c], 255);
    assert_eq!(l.maps.open.data[0], 0);
}

#[test]
fn stroke_saturates_and_scales_with_opacity() {
    let mut l1 = WetLayer::new(128, 64);
    let mut b = soft_brush();
    paint_line(&mut b, &mut l1, 32.0);
    let a1 = l1.dens.amount.cur().data[32 * 128 + 60];
    // flat top: the stroke centre reaches the target, independent of dab overlap
    let target = b.medium.amount;
    assert!((a1 - target).abs() / target < 0.02, "{a1} vs {target}");

    let mut l2 = WetLayer::new(128, 64);
    b.params.opacity = 0.5;
    paint_line(&mut b, &mut l2, 32.0);
    let a2 = l2.dens.amount.cur().data[32 * 128 + 60];
    assert!((a2 / a1 - 0.5).abs() < 0.02, "{a2} / {a1}");
}

#[test]
fn strokes_replay_deterministically() {
    let run = || {
        let mut b = soft_brush();
        b.params.spacing_jitter = 0.5;
        b.params.scatter_jitter = 0.2;
        b.set_seed(7);
        let mut l = WetLayer::new(128, 64);
        paint_line(&mut b, &mut l, 32.0);
        l.dens.amount.cur().data.clone()
    };
    assert_eq!(run(), run());
}

#[test]
fn dry_media_settle_and_do_not_flow() {
    let mut b = soft_brush();
    b.set_tool(wet_brush::Tool::Paint(wet_brush::Medium::Pencil));
    let mut l = WetLayer::new(128, 64);
    paint_line(&mut b, &mut l, 32.0);
    let c = 32 * 128 + 60;
    assert!(l.mid.amount.cur().data[c] > 0.0);
    assert_eq!(l.dens.amount.cur().data[c], 0.0);
    assert_eq!(l.water.cur().data[c].to_f32(), 0.0);
    let before = l.mid.amount.cur().data.clone();
    l.params.tilt_active = true;
    l.params.tilt_y = 1.0;
    for _ in 0..50 {
        l.step();
    }
    assert_eq!(l.mid.amount.cur().data, before);
}

#[test]
fn eraser_and_water_tools() {
    let mut b = soft_brush();
    let mut l = WetLayer::new(128, 64);
    paint_line(&mut b, &mut l, 32.0);
    let c = 32 * 128 + 60;
    let a = l.dens.amount.cur().data[c];
    b.set_tool(wet_brush::Tool::Eraser);
    paint_line(&mut b, &mut l, 32.0);
    // the eraser leaves ≈ 10 % of the amount in one pass
    assert!(l.dens.amount.cur().data[c] < 0.2 * a);

    let mut l2 = WetLayer::new(128, 64);
    b.set_tool(wet_brush::Tool::Water);
    paint_line(&mut b, &mut l2, 32.0);
    assert!(l2.water.cur().data[c].to_f32() > 1.0);
    assert_eq!(l2.maps.open.data[c], 255);
    assert_eq!(l2.dens.amount.cur().data[c], 0.0);
}

#[test]
fn smudge_carries_paint_along_the_stroke() {
    let mut b = soft_brush();
    let mut l = WetLayer::new(160, 64);
    // a short blob at the left, then smudge rightwards through clean paper
    b.press(Sample { x: 20.0, y: 32.0, pressure: 1.0 });
    b.move_to(&mut l, Sample { x: 30.0, y: 32.0, pressure: 1.0 });
    b.release(&mut l, Sample { x: 30.0, y: 32.0, pressure: 1.0 });
    let far = 32 * 160 + 70;
    assert_eq!(l.dens.amount.cur().data[far], 0.0);
    b.set_tool(wet_brush::Tool::Smudge);
    b.press(Sample { x: 22.0, y: 32.0, pressure: 1.0 });
    for i in 1..=15 {
        b.move_to(&mut l, Sample { x: 22.0 + 4.0 * i as f32, y: 32.0, pressure: 1.0 });
    }
    b.release(&mut l, Sample { x: 82.0, y: 32.0, pressure: 1.0 });
    assert!(l.dens.amount.cur().data[far] > 1.0);
}

#[test]
fn bristles_paint_lines_only_when_moving() {
    let mut b = soft_brush();
    b.params.size = 40.0;
    b.bristle = Some(wet_brush::BristleParams {
        thin_density: 0.5,
        thick_density: 0.3,
        thick_size: 5.0,
        thick_size_jitter: 0.5,
        wave_length: 500.0,
        wave_jitter: 0.5,
        random_span: 0.5,
        edge_size: None,
        ribbon_px: None,
    });
    let mut l = WetLayer::new(128, 64);
    // a tap paints nothing
    b.press(Sample { x: 60.0, y: 32.0, pressure: 1.0 });
    b.release(&mut l, Sample { x: 60.0, y: 32.0, pressure: 1.0 });
    assert!(l.dens.amount.cur().data.iter().all(|&a| a == 0.0));
    // a stroke paints overlapping lines covering most of the width, with uneven load
    paint_line(&mut b, &mut l, 32.0);
    let col: Vec<f32> = (12..52).map(|y| l.dens.amount.cur().data[y * 128 + 80]).collect();
    let painted = col.iter().filter(|&&a| a > 100.0).count();
    let (lo, hi) = col.iter().filter(|&&a| a > 100.0).fold((f32::MAX, 0.0f32), |(lo, hi), &a| (lo.min(a), hi.max(a)));
    assert!(painted > 20 && hi > 1.15 * lo, "painted {painted} load {lo}..{hi}");
}

#[test]
fn stencil_protects_cells() {
    let mut b = soft_brush();
    let mut l = WetLayer::new(128, 64);
    let mut st = wet_fluid::grid::PlaneU8::new(128, 64, 255);
    for y in 0..64 {
        for x in 60..128 {
            st.data[y * 128 + x] = 0;
        }
    }
    l.maps.stencil = Some(st);
    paint_line(&mut b, &mut l, 32.0);
    assert!(l.dens.amount.cur().data[32 * 128 + 40] > 0.0);
    let c = 32 * 128 + 80;
    assert_eq!(l.dens.amount.cur().data[c], 0.0);
    assert_eq!(l.maps.open.data[c], 0);
    for _ in 0..50 {
        l.step();
    }
    assert_eq!(l.dens.amount.cur().data[c], 0.0);
}

#[test]
fn liquify_warps_and_reconstructs() {
    use wet_brush::{Liquify, Tool};
    let mut b = soft_brush();
    let mut l = WetLayer::new(128, 64);
    paint_line(&mut b, &mut l, 32.0);
    let before = l.dens.amount.cur().data.clone();
    let total = |d: &[f32]| d.iter().sum::<f32>();
    // bloat pushes paint out of a horizontal band: rows far from the centre line gain
    b.set_tool(Tool::Liquify(Liquify::Bloat));
    b.params.size = 40.0;
    b.press(Sample { x: 60.0, y: 32.0, pressure: 1.0 });
    for i in 0..10 {
        b.move_to(&mut l, Sample { x: 60.0 + i as f32 * 0.5, y: 32.0, pressure: 1.0 });
    }
    b.release(&mut l, Sample { x: 65.0, y: 32.0, pressure: 1.0 });
    let after = l.dens.amount.cur().data.clone();
    assert_ne!(before, after);
    let width = |d: &[f32]| (0..64).filter(|y| d[y * 128 + 62] > 5.0).count();
    assert!(width(&after) > width(&before), "{} vs {}", width(&after), width(&before));
    assert!((total(&after) - total(&before)).abs() / total(&before) < 0.25);
    // reconstruct brings the original back
    b.set_tool(Tool::Liquify(Liquify::Reconstruct));
    b.press(Sample { x: 62.0, y: 32.0, pressure: 1.0 });
    let mut l2 = l;
    for _ in 0..3 {
        // three strokes of reconstruct over the same spot, each restoring part of the way
        for i in 0..40 {
            b.move_to(&mut l2, Sample { x: 62.0 + (i % 4) as f32 * 0.25, y: 32.0, pressure: 1.0 });
        }
    }
    b.release(&mut l2, Sample { x: 62.0, y: 32.0, pressure: 1.0 });
    // the cell bloat changed most is now closer to its original value
    let c = (0..before.len())
        .max_by(|&i, &j| (after[i] - before[i]).abs().total_cmp(&(after[j] - before[j]).abs()))
        .unwrap();
    let (orig, warped, now) = (before[c], after[c], l2.dens.amount.cur().data[c]);
    assert!((now - orig).abs() < 0.5 * (warped - orig).abs(), "orig {orig} warped {warped} now {now}");
}

#[test]
fn effect_marker_blends_with_canvas_colour() {
    use wet_brush::{Medium, Tool};
    let mut b = soft_brush();
    let mut l = WetLayer::new(128, 64);
    paint_line(&mut b, &mut l, 32.0); // red watercolour band at y = 32
    let mut fx = soft_brush();
    fx.set_tool(Tool::Paint(Medium::Marker));
    fx.params.glaze_blend = wet_compose::MixMode::Saturation;
    fx.color = [128.0, 128.0, 128.0]; // grey: saturation 0
    // vertical stroke crossing the band and continuing over white paper
    fx.press(Sample { x: 60.0, y: 5.0, pressure: 1.0 });
    for i in 1..=20 {
        fx.move_to(&mut l, Sample { x: 60.0, y: 5.0 + 2.7 * i as f32, pressure: 1.0 });
    }
    fx.release(&mut l, Sample { x: 60.0, y: 59.0, pressure: 1.0 });
    let rgb = |c: usize| [0, 1, 2].map(|k| l.mid.rgb[k].cur().data[c].to_f32());
    // over the red band the deposited colour is desaturated (r ≈ g ≈ b)
    let on_red = rgb(32 * 128 + 60);
    assert!((on_red[0] - on_red[1]).abs() < 10.0, "{on_red:?}");
    // over blank paper (no paint to blend with) the plain colour is laid: effect
    // markers paint the same tint there whatever their mode
    let on_white = rgb(8 * 128 + 60);
    assert!(on_white.iter().all(|&v| (v - 128.0).abs() < 2.0), "{on_white:?}");
}

#[test]
fn minified_tip_is_prefiltered() {
    // a sparse grid of single-texel dots: point sampling between dots reads 0, while the
    // prefiltered sample at 16x minification reads the local average (1/16)
    let n = 256;
    let data = (0..n * n).map(|i| if i % 4 == 0 && (i / n) % 4 == 0 { 1.0 } else { 0.0 }).collect();
    let m = Mask::new(n, n, data);
    let (u, v) = (130.0 / 255.0, 130.0 / 255.0);
    assert_eq!(m.sample_clamped(u, v), 0.0);
    let s = m.sample_lod(u, v, 16.0);
    assert!((s - 1.0 / 16.0).abs() < 0.01, "{s}");
}

#[test]
fn stroke_path_paints_along_an_svg_path() {
    use wet_brush::path::{Path, Placement};
    let mut b = soft_brush();
    let mut l = WetLayer::new(128, 128);
    let p = Path::from_svg("M0 0 h80 v80");
    b.stroke_path(&mut l, &p.placed(&Placement { offset: (20.0, 20.0), ..Placement::default() }), 1.0);
    let amt = |x: usize, y: usize| l.dens.amount.cur().data[y * 128 + x] + l.mid.amount.cur().data[y * 128 + x];
    assert!(amt(60, 20) > 1.0 && amt(100, 60) > 1.0, "along the path");
    assert_eq!(amt(40, 80), 0.0, "off the path");
}

#[test]
fn multicolour_brush_bands_then_blends() {
    use wet_brush::{Medium, Tool};
    let mut b = soft_brush();
    b.set_tool(Tool::Paint(Medium::Oil));
    b.multi = vec![[220.0, 30.0, 30.0], [30.0, 30.0, 220.0]];
    let mut l = WetLayer::new(400, 64);
    b.press(Sample { x: 10.0, y: 32.0, pressure: 1.0 });
    for i in 1..=120 {
        b.move_to(&mut l, Sample { x: 10.0 + 3.0 * i as f32, y: 32.0, pressure: 1.0 });
    }
    let rgb = |x: usize, y: usize| [0, 1, 2].map(|k| l.mid.rgb[k].cur().data[y * 400 + x].to_f32());
    // near the start the two sides carry different colours (moving +x, "across" is +y)
    let (top, bottom) = (rgb(30, 26), rgb(30, 38));
    assert!(top[0] > top[2] + 50.0 && bottom[2] > bottom[0] + 50.0, "{top:?} {bottom:?}");
    // far along the stroke both sides carry the average
    let (top, bottom) = (rgb(350, 26), rgb(350, 38));
    assert!((top[0] - bottom[0]).abs() < 10.0 && (top[2] - bottom[2]).abs() < 10.0, "{top:?} {bottom:?}");
}

#[test]
fn dirty_brush_keeps_picked_up_paint() {
    use wet_brush::{PaintMode, Tool, Medium};
    let run = |dirty: bool| {
        let mut l = WetLayer::new(160, 64);
        let mut red = soft_brush();
        paint_line(&mut red, &mut l, 32.0); // wet red band at y = 32
        let mut b = soft_brush();
        b.set_tool(Tool::Paint(Medium::Watercolor));
        b.params.paint_mode = PaintMode::Mix;
        b.params.dirty = dirty;
        b.color = [30.0, 60.0, 200.0];
        // first stroke crosses the red band (picks red up), second paints clean paper
        b.press(Sample { x: 30.0, y: 32.0, pressure: 1.0 });
        for i in 1..=30 {
            b.move_to(&mut l, Sample { x: 30.0 + 2.0 * i as f32, y: 32.0, pressure: 1.0 });
        }
        b.release(&mut l, Sample { x: 90.0, y: 32.0, pressure: 1.0 });
        b.press(Sample { x: 130.0, y: 5.0, pressure: 1.0 });
        for i in 1..=10 {
            b.move_to(&mut l, Sample { x: 130.0, y: 5.0 + i as f32, pressure: 1.0 });
        }
        b.release(&mut l, Sample { x: 130.0, y: 15.0, pressure: 1.0 });
        l.dens.rgb[0].cur().data[10 * 160 + 130].to_f32()
    };
    // the dirty brush lays a redder second stroke
    assert!(run(true) > run(false) + 0.05, "{} vs {}", run(true), run(false));
}
