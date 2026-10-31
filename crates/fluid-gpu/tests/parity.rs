//! GPU backend vs CPU reference on identical inputs.

use wet_fluid::WetLayer;
use wet_fluid::grid::Plane;
use wet_fluid_gpu::GpuLayer;

fn scene() -> WetLayer {
    let mut l = WetLayer::new(128, 96);
    for y in 0..l.h {
        for x in 0..l.w {
            l.maps.height.set(
                x,
                y,
                ((x as f32 * 0.9).sin() * (y as f32 * 1.3).cos() * 40.0) as i8,
            );
            l.maps.absorb.set(x, y, ((x * 7 + y * 3) % 50) as u8);
        }
    }
    for i in 0..24 {
        let t = i as f32 / 23.0;
        l.dab(
            20.0 + 80.0 * t,
            40.0 + 12.0 * (t * 6.0).sin(),
            9.0,
            40.0,
            [40.0, 80.0, 200.0],
            500.0,
        );
        l.dab(
            30.0 + 70.0 * t,
            60.0 - 10.0 * t,
            7.0,
            25.0,
            [210.0, 70.0, 40.0],
            350.0,
        );
    }
    l.params.tilt_active = true;
    l.params.tilt_y = 0.004;
    l
}

fn f16_plane(p: &Plane<half::f16>) -> Vec<f32> {
    p.data.iter().map(|v| v.to_f32()).collect()
}

/// Max abs difference relative to the larger of 1 and the plane's max magnitude.
fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let scale = a.iter().chain(b).fold(1.0f32, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

fn compare(cpu: &WetLayer, gpu: &WetLayer, tol: f32, label: &str) {
    let checks: Vec<(&str, Vec<f32>, Vec<f32>)> = vec![
        ("u", f16_plane(cpu.u.cur()), f16_plane(gpu.u.cur())),
        ("v", f16_plane(cpu.v.cur()), f16_plane(gpu.v.cur())),
        (
            "water",
            f16_plane(cpu.water.cur()),
            f16_plane(gpu.water.cur()),
        ),
        (
            "densR",
            f16_plane(cpu.dens.rgb[0].cur()),
            f16_plane(gpu.dens.rgb[0].cur()),
        ),
        (
            "midB",
            f16_plane(cpu.mid.rgb[2].cur()),
            f16_plane(gpu.mid.rgb[2].cur()),
        ),
        (
            "densA",
            cpu.dens.amount.cur().data.clone(),
            gpu.dens.amount.cur().data.clone(),
        ),
        (
            "midA",
            cpu.mid.amount.cur().data.clone(),
            gpu.mid.amount.cur().data.clone(),
        ),
    ];
    let mut worst = 0.0f32;
    for (name, a, b) in &checks {
        let e = rel_err(a, b);
        eprintln!("{label} {name:6} rel err {e:.2e}");
        worst = worst.max(e);
    }
    let open_diff = cpu
        .maps
        .open
        .data
        .iter()
        .zip(&gpu.maps.open.data)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!("{label} wet-mask cells differing: {open_diff}");
    assert!(
        worst < tol,
        "{label}: worst relative error {worst:.2e} >= {tol:.0e}"
    );
}

/// Over many steps the backends diverge cell-by-cell (thresholds amplify f16 rounding
/// differences), so long runs are compared in aggregate: totals and mean error.
#[test]
fn long_run_agrees_in_aggregate() {
    let mut cpu = scene();
    let mut gpu = match GpuLayer::upload(&cpu) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU available, skipping: {e}");
            return;
        }
    };
    eprintln!(
        "adapter: {} ({:?})",
        gpu.adapter_info.name, gpu.adapter_info.backend
    );
    let mut readback = scene();
    for _ in 0..200 {
        cpu.step();
    }
    gpu.step(200);
    gpu.download(&mut readback);
    let (cw, cp) = cpu.totals();
    let (gw, gp) = readback.totals();
    eprintln!("water cpu {cw:.1} gpu {gw:.1}; pigment cpu {cp:.1} gpu {gp:.1}");
    assert!((cw - gw).abs() / cw < 0.01, "water totals");
    assert!((cp - gp).abs() / cp < 0.01, "pigment totals");
    let mean_rel = |a: &[f32], b: &[f32]| {
        let scale = a.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32 / scale
    };
    let water = mean_rel(
        &f16_plane(cpu.water.cur()),
        &f16_plane(readback.water.cur()),
    );
    let mid = mean_rel(&cpu.mid.amount.cur().data, &readback.mid.amount.cur().data);
    eprintln!("mean rel err: water {water:.2e} midA {mid:.2e}");
    assert!(water < 1e-2 && mid < 1e-2, "mean errors");
}

/// Single-step error with the GPU re-synchronised from the CPU before every step, so
/// kernel mismatches show up without accumulated divergence.
#[test]
fn single_steps_match() {
    let mut cpu = scene();
    let Ok(mut gpu) = GpuLayer::upload(&cpu) else {
        return;
    };
    let mut readback = scene();
    for _ in 0..12 {
        gpu.write_state(&cpu);
        cpu.step();
        gpu.step(1);
        gpu.download(&mut readback);
        compare(
            &cpu,
            &readback,
            2e-3,
            &format!("single step {}", cpu.step_index),
        );
    }
}

#[test]
fn display_download_matches_the_full_download() {
    let layer = scene();
    let Ok(mut gpu) = GpuLayer::upload(&layer) else {
        eprintln!("no GPU adapter; skipped");
        return;
    };
    gpu.step(7);
    let mut full = scene();
    gpu.download(&mut full);
    let mut shown = scene();
    gpu.download_display(&mut shown);
    let render = |l: &WetLayer| wet_fluid::render::render_layer(l);
    assert!(render(&full) == render(&shown), "the display planes are the ones rendering reads");
    assert_eq!(full.dens.thickness.cur().data, shown.dens.thickness.cur().data);
    assert_eq!(full.mid.thickness.cur().data, shown.mid.thickness.cur().data);
}

/// Drips (drip.rs) step by step: the GPU spawns the same drop, picks up the same pigment
/// and lays the same trail as the CPU, with the GPU re-synchronised before every step.
#[test]
fn drips_match_step_by_step() {
    let mut cpu = scene();
    cpu.params.create_drops = true;
    cpu.params.tilt_y = 0.6;
    let Ok(mut gpu) = GpuLayer::upload(&cpu) else {
        return;
    };
    let mut readback = scene();
    let (mut spawned, mut moved) = (false, 0);
    for _ in 0..60 {
        gpu.params = cpu.params.clone();
        gpu.write_state(&cpu);
        let had = !cpu.drips.is_empty();
        cpu.step();
        gpu.step(1);
        gpu.download(&mut readback);
        let label = format!("drip step {}", cpu.step_index);
        assert_eq!(cpu.drips.len(), readback.drips.len(), "{label}: drip count");
        if let (Some(a), Some(b)) = (cpu.drips.first(), readback.drips.first()) {
            spawned |= !had;
            moved += had as usize;
            assert!((a.x - b.x).abs() < 1e-3 && (a.y - b.y).abs() < 1e-3, "{label}: position {a:?} vs {b:?}");
            assert!((a.width - b.width).abs() < 1e-3, "{label}: width");
            assert!((a.load - b.load).abs() <= 1e-3 * a.load.max(1.0), "{label}: load {} vs {}", a.load, b.load);
            for k in 0..3 {
                assert!((a.rgb[k] - b.rgb[k]).abs() < 0.05, "{label}: colour {:?} vs {:?}", a.rgb, b.rgb);
            }
        }
        compare(&cpu, &readback, 2e-3, &label);
    }
    eprintln!("spawned {spawned}, moved {moved} steps");
    assert!(spawned && moved > 5, "a drip formed and ran (spawned {spawned}, moved {moved} steps)");
}

/// A region read from the GPU, changed on the CPU and written back changes exactly that
/// region; the timing shows what a brush step's round trip costs.
#[test]
fn regions_round_trip() {
    let layer = scene();
    let Ok(mut gpu) = GpuLayer::upload(&layer) else { return };
    gpu.step(3);
    let mut full = scene();
    gpu.download(&mut full);
    let rect = (30, 20, 79, 59);
    let mut cpu = scene();
    let t = std::time::Instant::now();
    gpu.read_region(&mut cpu, rect);
    let read = t.elapsed();
    // the region matches the full download
    for y in rect.1..=rect.3 {
        for x in rect.0..=rect.2 {
            let c = y * layer.w + x;
            assert_eq!(cpu.water.cur().data[c], full.water.cur().data[c]);
            assert_eq!(cpu.mid.amount.cur().data[c], full.mid.amount.cur().data[c]);
            assert_eq!(cpu.maps.open.data[c], full.maps.open.data[c]);
        }
    }
    for y in rect.1..=rect.3 {
        for x in rect.0..=rect.2 {
            let c = y * layer.w + x;
            cpu.dens.amount.cur_mut().data[c] += 1.0;
            cpu.maps.open.data[c] = 255;
        }
    }
    let t = std::time::Instant::now();
    gpu.write_region(&cpu, rect);
    gpu.wait();
    let write = t.elapsed();
    let mut after = scene();
    gpu.download(&mut after);
    for c in 0..layer.w * layer.h {
        let (x, y) = (c % layer.w, c / layer.w);
        let inside = x >= rect.0 && x <= rect.2 && y >= rect.1 && y <= rect.3;
        let want = full.dens.amount.cur().data[c] + if inside { 1.0 } else { 0.0 };
        assert_eq!(after.dens.amount.cur().data[c], want, "({x}, {y})");
        if inside {
            assert_eq!(after.maps.open.data[c], 255);
        }
    }
    eprintln!("region 50×40: read {read:?}, write {write:?}");
}
