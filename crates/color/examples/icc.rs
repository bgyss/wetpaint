//! Write the ICC profiles of the supported working spaces into a directory.
//!
//! Usage: cargo run -p wet-color --example icc -- <out_dir>

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("out_dir"));
    std::fs::create_dir_all(&dir).unwrap();
    for s in wet_color::RgbSpace::ALL {
        let name = s.name().replace([' ', '(', ')', '.'], "");
        std::fs::write(dir.join(format!("{name}.icc")), wet_color::icc_profile(s)).unwrap();
    }
}
