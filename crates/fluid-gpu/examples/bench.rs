//! Steps/second of the CPU reference vs the GPU backend with the whole canvas wet.
//! Usage: cargo run -p wet-fluid-gpu --release --example bench -- [size ...]

use std::time::Instant;

use wet_fluid::WetLayer;
use wet_fluid_gpu::GpuLayer;

fn wet_canvas(n: usize) -> WetLayer {
    let mut l = WetLayer::new(n, n);
    let r = n as f32 / 12.0;
    for i in 0..60 {
        let t = i as f32 / 59.0;
        for row in 0..6 {
            let y = n as f32 * (0.1 + 0.15 * row as f32);
            l.dab(
                n as f32 * (0.05 + 0.9 * t),
                y,
                r,
                40.0,
                [60.0, 90.0, 180.0],
                400.0,
            );
        }
    }
    l
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let sizes = if sizes.is_empty() {
        vec![256, 512, 1024]
    } else {
        sizes
    };
    for n in sizes {
        let mut cpu = wet_canvas(n);
        let cpu_steps = (2_000_000 / (n * n)).clamp(2, 50);
        let t = Instant::now();
        for _ in 0..cpu_steps {
            cpu.step();
        }
        let cpu_rate = cpu_steps as f64 / t.elapsed().as_secs_f64();

        let layer = wet_canvas(n);
        let mut gpu = GpuLayer::upload(&layer).expect("GPU");
        gpu.step(5);
        gpu.wait();
        let gpu_steps = 200;
        let t = Instant::now();
        gpu.step(gpu_steps);
        gpu.wait();
        let gpu_rate = gpu_steps as f64 / t.elapsed().as_secs_f64();
        println!(
            "{n}x{n}: cpu {cpu_rate:8.1} steps/s   gpu {gpu_rate:8.1} steps/s   ({:.0}x)  [{}]",
            gpu_rate / cpu_rate,
            gpu.adapter_info.name
        );
    }
}
