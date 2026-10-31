//! `Brush::reach_bounds` holds every cell a stroke step changes, for every tool: the app's
//! GPU-resident strokes only bring that rectangle back from the GPU.

use wet_brush::{Brush, Liquify, Medium, Sample, Tool};
use wet_fluid::WetLayer;

/// The planes a brush may change: water, velocity, both pigment layers, the wet mask.
fn state(l: &WetLayer) -> Vec<Vec<f32>> {
    let h = |p: &wet_fluid::grid::Plane<half::f16>| p.data.iter().map(|v| v.to_f32()).collect::<Vec<f32>>();
    let mut v = vec![h(l.water.cur()), h(l.u.cur()), h(l.v.cur())];
    for k in 0..3 {
        v.push(h(l.dens.rgb[k].cur()));
        v.push(h(l.mid.rgb[k].cur()));
    }
    for p in [l.dens.amount.cur(), l.dens.thickness.cur(), l.mid.amount.cur(), l.mid.thickness.cur()] {
        v.push(p.data.clone());
    }
    v.push(l.maps.open.data.iter().map(|&b| b as f32).collect());
    v
}

/// Paint a zigzag with changing pressure; fail if a step changes a cell outside its bounds.
fn check(name: &str, mut brush: Brush) {
    let (w, h) = (160, 120);
    let mut layer = WetLayer::new(w, h);
    // paint underneath for pick-up, smudging and blending
    for i in 0..12 {
        layer.dab(20.0 + 10.0 * i as f32, 60.0, 10.0, 30.0, [200.0, 60.0, 40.0], 400.0);
    }
    brush.params.size = brush.params.size.clamp(8.0, 40.0);
    brush.color = [40.0, 90.0, 200.0];
    brush.set_seed(7);
    let at = |k: usize| {
        let t = k as f32 / 24.0;
        Sample { x: 15.0 + 130.0 * t, y: 60.0 + 35.0 * if k.is_multiple_of(2) { 1.0 } else { -1.0 } * t.min(0.6), pressure: 0.2 + 0.8 * (t * 7.0).sin().abs() }
    };
    brush.press(at(0));
    for k in 1..=25 {
        let s = at(k.min(24));
        let Some((x0, y0, x1, y1)) = brush.reach_bounds(s, w, h) else {
            return;
        };
        let before = state(&layer);
        if k == 25 {
            brush.release(&mut layer, s);
        } else {
            brush.move_to(&mut layer, s);
        }
        let after = state(&layer);
        for (p, (a, b)) in before.iter().zip(&after).enumerate() {
            for (c, (x, y)) in a.iter().zip(b).enumerate() {
                if x != y {
                    let (cx, cy) = (c % w, c / w);
                    assert!(
                        cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1,
                        "{name}: step {k} changed plane {p} at ({cx}, {cy}) outside ({x0}, {y0})–({x1}, {y1})"
                    );
                }
            }
        }
    }
}

#[test]
fn built_in_tools_stay_in_their_bounds() {
    let media = [Medium::Watercolor, Medium::Ink, Medium::Oil, Medium::ExpressOil, Medium::Pastel, Medium::Pencil, Medium::Marker, Medium::Airbrush];
    let mut tools: Vec<Tool> = media.into_iter().map(Tool::Paint).collect();
    tools.extend([Tool::Water, Tool::Dry, Tool::Eraser, Tool::Blend, Tool::Smudge, Tool::Blow]);
    for tool in tools {
        let mut b = Brush::new(Default::default(), wet_brush::Mask::disk(64, 0.5), None);
        b.set_tool(tool);
        b.params.size_jitter = 0.4;
        b.params.scatter_jitter = 0.5;
        check(&format!("{tool:?}"), b);
    }
    // the tools that read the whole layer say so
    let mut b = Brush::new(Default::default(), wet_brush::Mask::disk(64, 0.5), None);
    b.set_tool(Tool::Liquify(Liquify::Push));
    assert!(b.reach_bounds(Sample { x: 10.0, y: 10.0, pressure: 1.0 }, 100, 100).is_none());
}
