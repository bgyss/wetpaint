//! Rulers: guides that pull stroke samples onto a line, an ellipse or perspective lines.
//!
//! A stroke starts where the pen goes down; every later sample is projected onto the
//! guide. With `parallel`, a line ruler draws the parallel through the stroke start, and an
//! ellipse ruler draws the concentric ellipse through it. Perspective rulers draw the line
//! from the stroke start towards the vanishing point that best matches the first movement.

/// A ruler guide in canvas coordinates.
#[derive(Clone, Debug, PartialEq)]
pub enum Ruler {
    Line { a: (f32, f32), b: (f32, f32), parallel: bool },
    Ellipse { centre: (f32, f32), radii: (f32, f32), rotation: f32, parallel: bool },
    /// One, two or three vanishing points.
    Perspective { vanishing: Vec<(f32, f32)> },
}

/// Per-stroke ruler state.
#[derive(Clone, Debug, Default)]
pub struct RulerStroke {
    start: Option<(f32, f32)>,
    /// Chosen perspective direction (unit vector from the start).
    dir: Option<(f32, f32)>,
}

/// Movement (px) before a perspective stroke picks its vanishing point.
const DECIDE: f32 = 4.0;

impl RulerStroke {
    pub fn press(&mut self, p: (f32, f32)) {
        self.start = Some(p);
        self.dir = None;
    }

    /// The guided position for pointer sample `p`.
    pub fn constrain(&mut self, ruler: &Ruler, p: (f32, f32)) -> (f32, f32) {
        let s = *self.start.get_or_insert(p);
        match ruler {
            Ruler::Line { a, b, parallel } => {
                let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                let len2 = (dx * dx + dy * dy).max(1e-9);
                let origin = if *parallel { s } else { *a };
                let t = ((p.0 - origin.0) * dx + (p.1 - origin.1) * dy) / len2;
                (origin.0 + t * dx, origin.1 + t * dy)
            }
            Ruler::Ellipse { centre, radii, rotation, parallel } => {
                let (sn, cs) = rotation.sin_cos();
                let local = |q: (f32, f32)| {
                    let (x, y) = (q.0 - centre.0, q.1 - centre.1);
                    ((x * cs + y * sn) / radii.0.max(1e-6), (-x * sn + y * cs) / radii.1.max(1e-6))
                };
                // ellipse-space radius: 1 on the ruler, the start's radius when parallel
                let k = if *parallel {
                    let (u, v) = local(s);
                    (u * u + v * v).sqrt()
                } else {
                    1.0
                };
                let (u, v) = local(p);
                let r = (u * u + v * v).sqrt();
                let (u, v) = if r > 1e-9 { (u / r * k, v / r * k) } else { (k, 0.0) };
                let (x, y) = (u * radii.0, v * radii.1);
                (centre.0 + x * cs - y * sn, centre.1 + x * sn + y * cs)
            }
            Ruler::Perspective { vanishing } => {
                let (mx, my) = (p.0 - s.0, p.1 - s.1);
                if self.dir.is_none() && (mx * mx + my * my).sqrt() >= DECIDE {
                    let unit = |v: (f32, f32)| {
                        let l = (v.0 * v.0 + v.1 * v.1).sqrt().max(1e-9);
                        (v.0 / l, v.1 / l)
                    };
                    let m = unit((mx, my));
                    self.dir = vanishing
                        .iter()
                        .map(|vp| unit((vp.0 - s.0, vp.1 - s.1)))
                        .max_by(|a, b| {
                            let da = (a.0 * m.0 + a.1 * m.1).abs();
                            let db = (b.0 * m.0 + b.1 * m.1).abs();
                            da.total_cmp(&db)
                        });
                }
                match self.dir {
                    Some(d) => {
                        let t = mx * d.0 + my * d.1;
                        (s.0 + t * d.0, s.1 + t * d.1)
                    }
                    None => s,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: (f32, f32), b: (f32, f32)) -> bool {
        (a.0 - b.0).abs() < 1e-3 && (a.1 - b.1).abs() < 1e-3
    }

    #[test]
    fn line_and_parallel() {
        let r = Ruler::Line { a: (0.0, 0.0), b: (10.0, 0.0), parallel: false };
        let mut st = RulerStroke::default();
        st.press((3.0, 5.0));
        assert!(close(st.constrain(&r, (7.0, 2.0)), (7.0, 0.0)));
        let r = Ruler::Line { a: (0.0, 0.0), b: (10.0, 10.0), parallel: true };
        st.press((0.0, 4.0));
        let q = st.constrain(&r, (5.0, 6.0));
        assert!((q.1 - q.0 - 4.0).abs() < 1e-4, "{q:?}");
    }

    #[test]
    fn ellipse_and_concentric() {
        let r = Ruler::Ellipse { centre: (50.0, 50.0), radii: (20.0, 10.0), rotation: 0.0, parallel: false };
        let mut st = RulerStroke::default();
        st.press((70.0, 50.0));
        assert!(close(st.constrain(&r, (90.0, 50.0)), (70.0, 50.0)));
        assert!(close(st.constrain(&r, (50.0, 30.0)), (50.0, 40.0)));
        let r = Ruler::Ellipse { centre: (50.0, 50.0), radii: (20.0, 10.0), rotation: 0.0, parallel: true };
        st.press((60.0, 50.0)); // half-size concentric ellipse
        assert!(close(st.constrain(&r, (50.0, 0.0)), (50.0, 45.0)));
    }

    #[test]
    fn perspective_picks_the_matching_vanishing_point() {
        let r = Ruler::Perspective { vanishing: vec![(-1000.0, 0.0), (1000.0, 0.0), (0.0, -5000.0)] };
        let mut st = RulerStroke::default();
        st.press((0.0, 100.0));
        // first movement mostly upwards: the vertical vanishing point
        st.constrain(&r, (1.0, 94.0));
        let q = st.constrain(&r, (5.0, 50.0));
        assert!(q.0.abs() < 0.5 && (q.1 - 50.0).abs() < 1.0, "{q:?}");
    }
}
