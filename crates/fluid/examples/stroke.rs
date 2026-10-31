//! Paint a few strokes, simulate, and write PNG snapshots.
//! Usage: cargo run -p wet-fluid --release --example stroke -- <out_dir> [steps]

use std::{fs::File, io::BufWriter, path::Path};

use wet_fluid::{WetLayer, render::render_rgba};

fn save(l: &WetLayer, path: &Path) {
    let rgba = render_rgba(l, [250.0, 248.0, 240.0]);
    let mut enc = png::Encoder::new(
        BufWriter::new(File::create(path).unwrap()),
        l.w as u32,
        l.h as u32,
    );
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&rgba).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = Path::new(args.get(1).map(String::as_str).unwrap_or("."));
    let steps: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);
    std::fs::create_dir_all(out).unwrap();

    let mut l = WetLayer::new(320, 240);
    // A paper height field gives the paper-driven force something to act on.
    for y in 0..l.h {
        for x in 0..l.w {
            let v = ((x as f32 * 0.9).sin() * (y as f32 * 1.3).cos() * 40.0) as i8;
            l.maps.height.set(x, y, v);
        }
    }
    let strokes: [([f32; 3], f32, f32, f32); 3] = [
        ([30.0, 70.0, 190.0], 60.0, 80.0, 0.0),
        ([200.0, 60.0, 40.0], 90.0, 150.0, 12.0),
        ([230.0, 190.0, 40.0], 150.0, 60.0, -8.0),
    ];
    for (color, x0, y0, dy) in strokes {
        for i in 0..40 {
            let t = i as f32 / 39.0;
            l.dab(
                x0 + 150.0 * t,
                y0 + dy * t + 6.0 * (t * 9.0).sin(),
                10.0,
                40.0,
                color,
                500.0,
            );
        }
    }
    l.params.tilt_active = true;
    l.params.tilt_y = 0.003;
    save(&l, &out.join("frame_000.png"));
    for s in 1..=steps {
        l.step();
        if s % 100 == 0 {
            save(&l, &out.join(format!("frame_{s:03}.png")));
            let (w, p) = l.totals();
            println!("step {s}: water {w:.1} pigment {p:.1}");
        }
    }
}
