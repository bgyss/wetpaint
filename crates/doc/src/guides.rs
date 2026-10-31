//! Grid, guides and canvas-bound snap targets (non-printing aids): lines to draw over
//! the canvas and a snapping function for selections, transforms and placed objects.

/// How the grid divides the canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Grid {
    /// A line every `spacing` pixels, each cell split into `subdivisions`.
    Spacing { spacing: f32, subdivisions: u32 },
    /// Exactly `columns × rows` cells.
    Evenly { columns: u32, rows: u32 },
}

/// Which canvas-bound lines snap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoundSnaps {
    pub bounds: bool,
    pub centre: bool,
    pub thirds: bool,
    pub quarters: bool,
}

/// Grid and guides of a document.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Guides {
    pub grid: Option<Grid>,
    pub snap_to_grid: bool,
    /// Vertical guides (x positions) and horizontal guides (y positions).
    pub vertical: Vec<f32>,
    pub horizontal: Vec<f32>,
    /// Free guide segments.
    pub segments: Vec<((f32, f32), (f32, f32))>,
    pub snap_to_guides: bool,
    pub bounds: BoundSnaps,
}

impl Guides {
    /// Main and subdivision grid lines: (vertical xs, horizontal ys), each flagged main.
    pub fn grid_lines(&self, w: usize, h: usize) -> (Vec<(f32, bool)>, Vec<(f32, bool)>) {
        let (w, h) = (w as f32, h as f32);
        let lines = |len: f32, step: f32, sub: u32| -> Vec<(f32, bool)> {
            if step <= 0.0 {
                return Vec::new();
            }
            let sub = sub.max(1);
            let fine = step / sub as f32;
            let n = (len / fine).floor() as u32;
            (0..=n).map(|i| (i as f32 * fine, i % sub == 0)).collect()
        };
        match self.grid {
            None => (Vec::new(), Vec::new()),
            Some(Grid::Spacing { spacing, subdivisions }) => (lines(w, spacing, subdivisions), lines(h, spacing, subdivisions)),
            Some(Grid::Evenly { columns, rows }) => (
                lines(w, w / columns.max(1) as f32, 1),
                lines(h, h / rows.max(1) as f32, 1),
            ),
        }
    }

    /// Snap a canvas point to the nearest active line within `radius` pixels, per axis
    /// (free segments snap to their closest point).
    pub fn snap(&self, (x, y): (f32, f32), w: usize, h: usize, radius: f32) -> (f32, f32) {
        let mut xs: Vec<f32> = Vec::new();
        let mut ys: Vec<f32> = Vec::new();
        if self.snap_to_grid {
            let (gx, gy) = self.grid_lines(w, h);
            xs.extend(gx.into_iter().map(|(v, _)| v));
            ys.extend(gy.into_iter().map(|(v, _)| v));
        }
        if self.snap_to_guides {
            xs.extend(&self.vertical);
            ys.extend(&self.horizontal);
        }
        let (wf, hf) = (w as f32, h as f32);
        let b = self.bounds;
        let frac = |f: &[f32]| f.iter().map(|&k| (k * wf, k * hf)).collect::<Vec<_>>();
        let mut fr: Vec<(f32, f32)> = Vec::new();
        if b.bounds {
            fr.extend(frac(&[0.0, 1.0]));
        }
        if b.centre {
            fr.extend(frac(&[0.5]));
        }
        if b.thirds {
            fr.extend(frac(&[1.0 / 3.0, 2.0 / 3.0]));
        }
        if b.quarters {
            fr.extend(frac(&[0.25, 0.75]));
        }
        xs.extend(fr.iter().map(|p| p.0));
        ys.extend(fr.iter().map(|p| p.1));
        let nearest = |v: f32, cands: &[f32]| {
            cands
                .iter()
                .copied()
                .filter(|c| (c - v).abs() <= radius)
                .min_by(|a, b| (a - v).abs().total_cmp(&(b - v).abs()))
                .unwrap_or(v)
        };
        let mut p = (nearest(x, &xs), nearest(y, &ys));
        if self.snap_to_guides && p == (x, y) {
            // free segments: snap to the closest point if near
            for &(a, c) in &self.segments {
                let (dx, dy) = (c.0 - a.0, c.1 - a.1);
                let t = (((x - a.0) * dx + (y - a.1) * dy) / (dx * dx + dy * dy).max(1e-9)).clamp(0.0, 1.0);
                let q = (a.0 + t * dx, a.1 + t * dy);
                if ((q.0 - x).powi(2) + (q.1 - y).powi(2)).sqrt() <= radius {
                    p = q;
                }
            }
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_lines_and_snapping() {
        let g = Guides { grid: Some(Grid::Spacing { spacing: 100.0, subdivisions: 4 }), snap_to_grid: true, ..Guides::default() };
        let (xs, _) = g.grid_lines(400, 300);
        assert_eq!(xs.len(), 17);
        assert!(xs[4].1 && !xs[1].1 && xs[1].0 == 25.0);
        assert_eq!(g.snap((27.0, 101.0), 400, 300, 4.0), (25.0, 100.0));
        assert_eq!(g.snap((37.0, 140.0), 400, 300, 4.0), (37.0, 140.0));
        let e = Guides { grid: Some(Grid::Evenly { columns: 3, rows: 2 }), ..Guides::default() };
        assert_eq!(e.grid_lines(300, 200).0.len(), 4);
    }

    #[test]
    fn guides_and_bounds() {
        let g = Guides {
            vertical: vec![50.0],
            segments: vec![((0.0, 0.0), (100.0, 100.0))],
            snap_to_guides: true,
            bounds: BoundSnaps { centre: true, thirds: true, ..BoundSnaps::default() },
            ..Guides::default()
        };
        assert_eq!(g.snap((52.0, 10.0), 300, 300, 3.0).0, 50.0);
        assert_eq!(g.snap((148.0, 99.0), 300, 300, 3.0), (150.0, 100.0)); // centre x, third y
        let q = g.snap((22.0, 20.0), 300, 300, 3.0); // the diagonal segment
        assert!((q.0 - 21.0).abs() < 1e-4 && (q.1 - 21.0).abs() < 1e-4, "{q:?}");
    }
}
