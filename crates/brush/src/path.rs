//! Vector paths: SVG path data (`M L H V C S Q T Z`, absolute and relative) flattened to
//! polylines, placed on the canvas (move, rotate, scale, flip) and painted with a brush
//! ("stroke path").

/// A path as polylines (one per subpath) in its own coordinates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Path {
    pub subpaths: Vec<Vec<(f32, f32)>>,
}

/// Placement of a path on the canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub offset: (f32, f32),
    pub rotation: f32,
    pub scale: f32,
    pub flip_h: bool,
    pub flip_v: bool,
}

impl Default for Placement {
    fn default() -> Self {
        Self { offset: (0.0, 0.0), rotation: 0.0, scale: 1.0, flip_h: false, flip_v: false }
    }
}

impl Placement {
    pub fn apply(&self, (x, y): (f32, f32)) -> (f32, f32) {
        let (x, y) = (if self.flip_h { -x } else { x }, if self.flip_v { -y } else { y });
        let (sn, cs) = self.rotation.sin_cos();
        let (x, y) = (x * self.scale, y * self.scale);
        (x * cs - y * sn + self.offset.0, x * sn + y * cs + self.offset.1)
    }
}

/// Segments per curve, from its control polygon length (about one per 2 px).
fn steps(len: f32) -> usize {
    ((len / 2.0).ceil() as usize).clamp(4, 256)
}

impl Path {
    /// Parse SVG path data. Unknown commands end parsing (what was read is kept).
    pub fn from_svg(d: &str) -> Self {
        let mut toks = Tokens { s: d.as_bytes(), i: 0 };
        let mut subpaths: Vec<Vec<(f32, f32)>> = Vec::new();
        let mut cur = (0.0f32, 0.0f32);
        let mut start = cur;
        let mut last_ctrl: Option<(f32, f32)> = None;
        let mut cmd = b'M';
        loop {
            match toks.command() {
                Some(c) => cmd = c,
                None if toks.done() => break,
                None => {} // implicit repeat of the previous command
            }
            let rel = cmd.is_ascii_lowercase();
            let base = if rel { cur } else { (0.0, 0.0) };
            let pt = |t: &mut Tokens| -> Option<(f32, f32)> { Some((t.num()? + base.0, t.num()? + base.1)) };
            let line_to = |subpaths: &mut Vec<Vec<(f32, f32)>>, p: (f32, f32)| {
                if subpaths.is_empty() {
                    subpaths.push(vec![(0.0, 0.0)]);
                }
                subpaths.last_mut().unwrap().push(p);
            };
            match cmd.to_ascii_uppercase() {
                b'M' => {
                    let Some(p) = pt(&mut toks) else { break };
                    subpaths.push(vec![p]);
                    cur = p;
                    start = p;
                    // further pairs after a moveto are linetos
                    cmd = if rel { b'l' } else { b'L' };
                    last_ctrl = None;
                }
                b'L' => {
                    let Some(p) = pt(&mut toks) else { break };
                    line_to(&mut subpaths, p);
                    cur = p;
                    last_ctrl = None;
                }
                b'H' => {
                    let Some(x) = toks.num() else { break };
                    let p = (x + base.0, cur.1);
                    line_to(&mut subpaths, p);
                    cur = p;
                    last_ctrl = None;
                }
                b'V' => {
                    let Some(y) = toks.num() else { break };
                    let p = (cur.0, y + base.1);
                    line_to(&mut subpaths, p);
                    cur = p;
                    last_ctrl = None;
                }
                b'C' | b'S' => {
                    let c1 = if cmd.eq_ignore_ascii_case(&b'S') {
                        last_ctrl.map_or(cur, |c| (2.0 * cur.0 - c.0, 2.0 * cur.1 - c.1))
                    } else {
                        let Some(c) = pt(&mut toks) else { break };
                        c
                    };
                    let (Some(c2), Some(p)) = (pt(&mut toks), pt(&mut toks)) else { break };
                    let n = steps(dist(cur, c1) + dist(c1, c2) + dist(c2, p));
                    for k in 1..=n {
                        let t = k as f32 / n as f32;
                        let u = 1.0 - t;
                        let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
                        let q = (
                            w[0] * cur.0 + w[1] * c1.0 + w[2] * c2.0 + w[3] * p.0,
                            w[0] * cur.1 + w[1] * c1.1 + w[2] * c2.1 + w[3] * p.1,
                        );
                        line_to(&mut subpaths, q);
                    }
                    cur = p;
                    last_ctrl = Some(c2);
                }
                b'Q' | b'T' => {
                    let c = if cmd.eq_ignore_ascii_case(&b'T') {
                        last_ctrl.map_or(cur, |c| (2.0 * cur.0 - c.0, 2.0 * cur.1 - c.1))
                    } else {
                        let Some(c) = pt(&mut toks) else { break };
                        c
                    };
                    let Some(p) = pt(&mut toks) else { break };
                    let n = steps(dist(cur, c) + dist(c, p));
                    for k in 1..=n {
                        let t = k as f32 / n as f32;
                        let u = 1.0 - t;
                        let q = (u * u * cur.0 + 2.0 * u * t * c.0 + t * t * p.0, u * u * cur.1 + 2.0 * u * t * c.1 + t * t * p.1);
                        line_to(&mut subpaths, q);
                    }
                    cur = p;
                    last_ctrl = Some(c);
                }
                b'Z' => {
                    line_to(&mut subpaths, start);
                    cur = start;
                    last_ctrl = None;
                    if toks.peek_number() {
                        break;
                    }
                }
                _ => break,
            }
        }
        subpaths.retain(|s| s.len() >= 2);
        Self { subpaths }
    }

    /// The subpaths placed on the canvas.
    pub fn placed(&self, at: &Placement) -> Vec<Vec<(f32, f32)>> {
        self.subpaths.iter().map(|s| s.iter().map(|&p| at.apply(p)).collect()).collect()
    }
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

struct Tokens<'a> {
    s: &'a [u8],
    i: usize,
}

impl Tokens<'_> {
    fn skip(&mut self) {
        while self.i < self.s.len() && (self.s[self.i].is_ascii_whitespace() || self.s[self.i] == b',') {
            self.i += 1;
        }
    }

    fn done(&mut self) -> bool {
        self.skip();
        self.i >= self.s.len()
    }

    fn command(&mut self) -> Option<u8> {
        self.skip();
        let c = *self.s.get(self.i)?;
        (c.is_ascii_alphabetic() && c != b'e' && c != b'E').then(|| {
            self.i += 1;
            c
        })
    }

    fn peek_number(&mut self) -> bool {
        self.skip();
        self.s.get(self.i).is_some_and(|c| c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.'))
    }

    fn num(&mut self) -> Option<f32> {
        self.skip();
        let st = self.i;
        let mut seen_dot = false;
        let mut seen_exp = false;
        while self.i < self.s.len() {
            let c = self.s[self.i];
            let ok = c.is_ascii_digit()
                || ((c == b'-' || c == b'+') && (self.i == st || matches!(self.s[self.i - 1], b'e' | b'E')))
                || (c == b'.' && !seen_dot && !seen_exp)
                || ((c == b'e' || c == b'E') && !seen_exp && self.i > st);
            if !ok {
                break;
            }
            seen_dot |= c == b'.';
            seen_exp |= c == b'e' || c == b'E';
            self.i += 1;
        }
        std::str::from_utf8(&self.s[st..self.i]).ok()?.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lines_curves_and_relative_commands() {
        let p = Path::from_svg("M10 10 H 50 v20 l-40,0 Z");
        assert_eq!(p.subpaths.len(), 1);
        assert_eq!(p.subpaths[0], vec![(10.0, 10.0), (50.0, 10.0), (50.0, 30.0), (10.0, 30.0), (10.0, 10.0)]);
        let c = Path::from_svg("M0,0 C0,10 10,10 10,0 m5-5 q5-5 10,0");
        assert_eq!(c.subpaths.len(), 2);
        let end = *c.subpaths[0].last().unwrap();
        assert!((end.0 - 10.0).abs() < 1e-4 && end.1.abs() < 1e-4);
        // the cubic passes through its midpoint (5, 7.5)
        let mid = c.subpaths[0].iter().copied().min_by(|a, b| (a.0 - 5.0).abs().total_cmp(&(b.0 - 5.0).abs())).unwrap();
        assert!((mid.0 - 5.0).abs() < 0.6 && (mid.1 - 7.5).abs() < 0.1, "{mid:?}");
        assert_eq!(c.subpaths[1][0], (15.0, -5.0));
        assert!(c.subpaths[1].last().unwrap().0 - 25.0 < 1e-4);
        // numbers packed SVG-style: "1-2.5.5.5"
        let q = Path::from_svg("M1-2.5.5.5L1e1,0");
        assert_eq!(q.subpaths[0], vec![(1.0, -2.5), (0.5, 0.5), (10.0, 0.0)]);
    }

    #[test]
    fn placement() {
        let at = Placement { offset: (100.0, 50.0), rotation: std::f32::consts::FRAC_PI_2, scale: 2.0, flip_h: false, flip_v: false };
        let (x, y) = at.apply((10.0, 0.0));
        assert!((x - 100.0).abs() < 1e-4 && (y - 70.0).abs() < 1e-4);
    }
}
