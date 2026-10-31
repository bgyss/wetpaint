//! Colour palettes: text files with one `#rrggbb` per line, optionally followed by a name.

/// One swatch: colour (0–255) and optional name.
#[derive(Clone, Debug, PartialEq)]
pub struct Swatch {
    pub rgb: [f32; 3],
    pub name: String,
}

/// Parse `#rrggbb[<TAB or space>name]` lines.
pub fn parse_lines(text: &str) -> Vec<Swatch> {
    text.lines()
        .filter_map(|l| {
            let l = l.trim();
            let hex = l.strip_prefix('#')?;
            let (code, name) = hex.split_once(|c: char| c.is_whitespace()).unwrap_or((hex, ""));
            if code.len() != 6 {
                return None;
            }
            let v = u32::from_str_radix(code, 16).ok()?;
            Some(Swatch { rgb: [(v >> 16) as f32, ((v >> 8) & 255) as f32, (v & 255) as f32], name: name.trim().to_string() })
        })
        .collect()
}

/// Load a palette file.
pub fn load(path: &std::path::Path) -> std::io::Result<Vec<Swatch>> {
    Ok(parse_lines(&std::fs::read_to_string(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hex_lines_with_names() {
        let s = parse_lines("#ff0000\n#00ff80\tSpring\nnot a colour\n#123\n");
        assert_eq!(s.len(), 2);
        assert_eq!(s[1].rgb, [0.0, 255.0, 128.0]);
        assert_eq!(s[1].name, "Spring");
    }
}
