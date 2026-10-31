//! Find the KM concentration that reproduces a target glaze colour over paper.
use wet_pigment::KmMixer;
fn main() {
    let m = KmMixer::new();
    let paper = [243.0 / 255.0; 3];
    for (name, pig, target) in [
        ("red", [200.0, 40.0, 40.0], [239.5, 149.4, 173.6]),
        ("blue", [40.0, 60.0, 200.0], [151.9, 173.4, 234.8]),
    ] {
        let pig = pig.map(|v: f32| v / 255.0);
        let mut best = (f32::MAX, 0.0, [0.0; 3]);
        for i in 0..=1000 {
            let t = i as f32 / 1000.0;
            let c = m.mix2(paper, pig, t).map(|v| v * 255.0);
            let e: f32 = (0..3).map(|k| (c[k] - target[k]).powi(2)).sum();
            if e < best.0 { best = (e, t, c); }
        }
        println!("{name}: t {:.3} colour {:?} rms {:.1}", best.1, best.2.map(|v| v.round()), (best.0 / 3.0).sqrt());
    }
}
