//! Interactive layer transform: the quad the canvas corners are dragged to, edited with
//! handles on the canvas and applied to the layer as one resample when committed.
//!
//! Drag inside the quad to move it, a corner to scale about the centre (with ⌘/Ctrl: move
//! that corner alone, a perspective distortion), outside the quad to rotate about the
//! centre.

pub type Pt = (f32, f32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Handle {
    Move,
    Corner(usize),
    Rotate,
}

#[derive(Clone, Debug)]
pub struct Xform {
    /// Where the canvas corners go: top-left, top-right, bottom-right, bottom-left.
    pub quad: [Pt; 4],
    /// The canvas corners (the untransformed quad).
    pub rest: [Pt; 4],
    /// Drag in progress: the handle, the quad and the pointer when it started.
    drag: Option<(Handle, [Pt; 4], Pt)>,
}

fn sub(a: Pt, b: Pt) -> Pt {
    (a.0 - b.0, a.1 - b.1)
}

fn len(a: Pt) -> f32 {
    a.0.hypot(a.1)
}

/// Is `p` inside the (convex or not) quad? Even–odd crossing test.
fn inside(quad: &[Pt; 4], p: Pt) -> bool {
    let mut odd = false;
    for k in 0..4 {
        let (a, b) = (quad[k], quad[(k + 1) % 4]);
        if (a.1 > p.1) != (b.1 > p.1) && p.0 < a.0 + (p.1 - a.1) / (b.1 - a.1) * (b.0 - a.0) {
            odd = !odd;
        }
    }
    odd
}

impl Xform {
    /// Start transforming a `w` × `h` layer (corners at pixel centres, as the layer remap
    /// uses).
    pub fn new(w: usize, h: usize) -> Self {
        let (w, h) = ((w - 1) as f32, (h - 1) as f32);
        let rest = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
        Self { quad: rest, rest, drag: None }
    }

    pub fn centre(&self) -> Pt {
        let q = &self.quad;
        ((q[0].0 + q[1].0 + q[2].0 + q[3].0) / 4.0, (q[0].1 + q[1].1 + q[2].1 + q[3].1) / 4.0)
    }

    /// Has the quad moved from the canvas corners?
    pub fn changed(&self) -> bool {
        self.quad != self.rest
    }

    pub fn dragging(&self) -> Option<Handle> {
        self.drag.map(|d| d.0)
    }

    /// The handle under canvas point `p`; `tol` is the grab radius in canvas pixels.
    pub fn handle_at(&self, p: Pt, tol: f32) -> Handle {
        if let Some(k) = (0..4).find(|&k| len(sub(self.quad[k], p)) <= tol) {
            Handle::Corner(k)
        } else if inside(&self.quad, p) {
            Handle::Move
        } else {
            Handle::Rotate
        }
    }

    pub fn press(&mut self, p: Pt, tol: f32) {
        self.drag = Some((self.handle_at(p, tol), self.quad, p));
    }

    /// Pointer at `p` during a drag; `free` moves a dragged corner on its own.
    pub fn drag_to(&mut self, p: Pt, free: bool) {
        let Some((handle, start, from)) = self.drag else { return };
        let c = {
            let s = Self { quad: start, rest: self.rest, drag: None };
            s.centre()
        };
        self.quad = match handle {
            Handle::Move => {
                let d = sub(p, from);
                start.map(|q| (q.0 + d.0, q.1 + d.1))
            }
            Handle::Corner(k) if free => {
                let mut q = start;
                q[k] = (start[k].0 + p.0 - from.0, start[k].1 + p.1 - from.1);
                q
            }
            Handle::Corner(_) => {
                let s = len(sub(p, c)) / len(sub(from, c)).max(1e-3);
                let s = s.max(0.01);
                start.map(|q| (c.0 + (q.0 - c.0) * s, c.1 + (q.1 - c.1) * s))
            }
            Handle::Rotate => {
                let (a, b) = (sub(from, c), sub(p, c));
                let t = b.1.atan2(b.0) - a.1.atan2(a.0);
                let (sn, cs) = t.sin_cos();
                start.map(|q| {
                    let d = sub(q, c);
                    (c.0 + d.0 * cs - d.1 * sn, c.1 + d.0 * sn + d.1 * cs)
                })
            }
        };
    }

    pub fn release(&mut self) {
        self.drag = None;
    }

    /// Move, rotate (radians, clockwise on screen) and scale the quad about its centre, as
    /// the arrow and `<` `>` `{` `}` keys do while transforming.
    pub fn nudge(&mut self, offset: Pt, scale: f32, rotation: f32) {
        let c = self.centre();
        let (sn, cs) = rotation.sin_cos();
        self.quad = self.quad.map(|q| {
            let d = sub(q, c);
            (c.0 + offset.0 + (d.0 * cs - d.1 * sn) * scale, c.1 + offset.1 + (d.0 * sn + d.1 * cs) * scale)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [Pt; 4], b: [Pt; 4]) -> bool {
        a.iter().zip(&b).all(|(p, q)| (p.0 - q.0).abs() < 1e-3 && (p.1 - q.1).abs() < 1e-3)
    }

    #[test]
    fn handles_are_found_by_position() {
        let x = Xform::new(101, 51);
        assert_eq!(x.handle_at((1.0, 1.0), 5.0), Handle::Corner(0));
        assert_eq!(x.handle_at((99.0, 49.0), 5.0), Handle::Corner(2));
        assert_eq!(x.handle_at((50.0, 25.0), 5.0), Handle::Move);
        assert_eq!(x.handle_at((150.0, 25.0), 5.0), Handle::Rotate);
        assert!(!x.changed());
    }

    #[test]
    fn dragging_moves_scales_rotates_and_distorts() {
        let mut x = Xform::new(101, 51);
        x.press((50.0, 25.0), 5.0);
        x.drag_to((60.0, 20.0), false);
        x.release();
        assert!(close(x.quad, [(10.0, -5.0), (110.0, -5.0), (110.0, 45.0), (10.0, 45.0)]));

        // a corner twice as far from the centre doubles the quad about the centre
        let mut x = Xform::new(101, 51);
        x.press((100.0, 50.0), 5.0);
        x.drag_to((150.0, 75.0), false);
        assert!(close(x.quad, [(-50.0, -25.0), (150.0, -25.0), (150.0, 75.0), (-50.0, 75.0)]));

        // a quarter turn about the centre (50, 25), dragging from outside the quad
        let mut x = Xform::new(101, 51);
        x.press((150.0, 25.0), 5.0);
        x.drag_to((50.0, 125.0), false);
        assert!(close(x.quad, [(75.0, -25.0), (75.0, 75.0), (25.0, 75.0), (25.0, -25.0)]));

        // a free corner moves alone
        let mut x = Xform::new(101, 51);
        x.press((100.0, 0.0), 5.0);
        x.drag_to((90.0, 10.0), true);
        assert!(close(x.quad, [(0.0, 0.0), (90.0, 10.0), (100.0, 50.0), (0.0, 50.0)]));
        assert!(x.changed());
    }

    #[test]
    fn nudges_match_the_drag_geometry() {
        let mut x = Xform::new(101, 51);
        x.nudge((0.0, 0.0), 1.0, std::f32::consts::FRAC_PI_2);
        assert!(close(x.quad, [(75.0, -25.0), (75.0, 75.0), (25.0, 75.0), (25.0, -25.0)]));
    }
}
