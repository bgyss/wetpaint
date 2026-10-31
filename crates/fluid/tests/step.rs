//! Whole-step behaviour of the reference simulation.

use wet_fluid::WetLayer;

fn painted_layer() -> WetLayer {
    let mut l = WetLayer::new(96, 96);
    for i in 0..20 {
        let t = i as f32 / 19.0;
        l.dab(
            20.0 + 50.0 * t,
            40.0 + 10.0 * t,
            8.0,
            30.0,
            [40.0, 80.0, 200.0],
            400.0,
        );
    }
    l
}

fn finite(l: &WetLayer) -> bool {
    let h = |p: &wet_fluid::grid::PlaneH| p.data.iter().all(|v| v.to_f32().is_finite());
    let f = |p: &wet_fluid::grid::PlaneF| p.data.iter().all(|v| v.is_finite());
    h(l.water.cur())
        && h(l.u.cur())
        && h(l.v.cur())
        && f(l.dens.amount.cur())
        && f(l.mid.amount.cur())
        && l.dens
            .rgb
            .iter()
            .chain(l.mid.rgb.iter())
            .all(|b| h(b.cur()))
}

#[test]
fn stable_and_conserves_pigment() {
    let mut l = painted_layer();
    let (w0, p0) = l.totals();
    for _ in 0..300 {
        l.step();
        assert!(finite(&l));
    }
    let (w1, p1) = l.totals();
    assert!(w1 < w0, "water evaporates: {w0} -> {w1}");
    assert!(
        (p1 - p0).abs() / p0 < 0.02,
        "pigment conserved: {p0} -> {p1}"
    );
}

#[test]
fn pigment_deposits_as_it_dries() {
    let mut l = painted_layer();
    for _ in 0..300 {
        l.step();
    }
    let mid: f64 = l.mid.amount.cur().data.iter().map(|&v| v as f64).sum();
    assert!(mid > 0.0, "some pigment moved to the deposited layer");
}

#[test]
fn tilt_moves_paint_downhill() {
    let mut l = WetLayer::new(96, 96);
    l.dab(48.0, 30.0, 10.0, 60.0, [200.0, 30.0, 30.0], 300.0);
    l.params.tilt_active = true;
    l.params.tilt_y = 0.005;
    l.params.placeholders.tilt_scale = 1.0; // exaggerate for a quick test
    let centroid = |l: &WetLayer| {
        let a = l.dens.amount.cur();
        let (mut s, mut sy) = (0.0f64, 0.0f64);
        for y in 0..l.h {
            for x in 0..l.w {
                let v = a.get(x, y) as f64;
                s += v;
                sy += v * y as f64;
            }
        }
        sy / s
    };
    let y0 = centroid(&l);
    for _ in 0..60 {
        l.step();
    }
    assert!(centroid(&l) > y0 + 0.5, "suspended pigment drifts in +y");
}
