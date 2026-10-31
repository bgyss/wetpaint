//! Planes (2-D row-major grids), double buffers and rectangles.

use half::f16;

/// A row-major 2-D grid of `T` with row stride equal to its width.
#[derive(Clone, Debug)]
pub struct Plane<T> {
    pub w: usize,
    pub h: usize,
    pub data: Vec<T>,
}

impl<T: Copy> Plane<T> {
    pub fn new(w: usize, h: usize, fill: T) -> Self {
        Self {
            w,
            h,
            data: vec![fill; w * h],
        }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> T {
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: T) {
        let i = self.idx(x, y);
        self.data[i] = v;
    }

    pub fn fill(&mut self, v: T) {
        self.data.fill(v);
    }

    pub fn copy_from(&mut self, other: &Plane<T>) {
        self.data.copy_from_slice(&other.data);
    }
}

/// Storage type for most per-cell quantities.
pub type PlaneH = Plane<f16>;
pub type PlaneF = Plane<f32>;
pub type PlaneU8 = Plane<u8>;
pub type PlaneI8 = Plane<i8>;

/// Read an f16 cell as f32.
#[inline]
pub fn ld(p: &PlaneH, i: usize) -> f32 {
    p.data[i].to_f32()
}

/// Write an f32 value into an f16 cell (round-to-nearest).
#[inline]
pub fn st(p: &mut PlaneH, i: usize, v: f32) {
    p.data[i] = f16::from_f32(v);
}

/// A channel with two buffers; kernels read `cur()` and write `next_mut()`, then the
/// step driver calls `swap()`.
#[derive(Clone, Debug)]
pub struct Buffered<T> {
    bufs: [Plane<T>; 2],
    cur: usize,
}

impl<T: Copy> Buffered<T> {
    pub fn new(w: usize, h: usize, fill: T) -> Self {
        Self {
            bufs: [Plane::new(w, h, fill), Plane::new(w, h, fill)],
            cur: 0,
        }
    }

    pub fn cur(&self) -> &Plane<T> {
        &self.bufs[self.cur]
    }

    pub fn cur_mut(&mut self) -> &mut Plane<T> {
        &mut self.bufs[self.cur]
    }

    pub fn next_mut(&mut self) -> &mut Plane<T> {
        &mut self.bufs[1 - self.cur]
    }

    /// Current (read) and next (write) buffers at once.
    pub fn split(&mut self) -> (&Plane<T>, &mut Plane<T>) {
        let (a, b) = self.bufs.split_at_mut(1);
        if self.cur == 0 {
            (&a[0], &mut b[0])
        } else {
            (&b[0], &mut a[0])
        }
    }

    pub fn swap(&mut self) {
        self.cur = 1 - self.cur;
    }
}

/// Inclusive cell rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl Rect {
    pub fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }

    pub fn cells(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (self.y0..=self.y1).flat_map(move |y| (self.x0..=self.x1).map(move |x| (x, y)))
    }

    /// Run `f(y)` for every row of the rectangle, in parallel.
    pub fn par_rows(&self, f: impl Fn(usize) + Sync + Send) {
        use rayon::prelude::*;
        (self.y0..=self.y1).into_par_iter().for_each(f);
    }
}

/// Shared write access to a plane for row-parallel kernels.
pub struct Shared<'a, T> {
    ptr: *mut T,
    len: usize,
    _plane: std::marker::PhantomData<&'a mut [T]>,
}

// SAFETY: access goes through `set`/`get` on disjoint cells (contract of `Shared::new`).
unsafe impl<T: Send> Send for Shared<'_, T> {}
unsafe impl<T: Send> Sync for Shared<'_, T> {}

impl<'a, T: Copy> Shared<'a, T> {
    /// # Safety
    /// While the `Shared` is in use, concurrent users must write disjoint cells and must
    /// not read cells other tasks write. Row-parallel kernels satisfy this by writing only
    /// the cells of their own row.
    pub unsafe fn new(p: &'a mut Plane<T>) -> Self {
        Self { ptr: p.data.as_mut_ptr(), len: p.data.len(), _plane: std::marker::PhantomData }
    }

    #[inline]
    pub fn set(&self, i: usize, v: T) {
        assert!(i < self.len);
        // SAFETY: in bounds; disjointness per the constructor's contract.
        unsafe { *self.ptr.add(i) = v }
    }
}

impl<T: Copy + std::ops::Add<Output = T>> Shared<'_, T> {
    /// `cell += v` (same disjointness contract as `set`).
    #[inline]
    pub fn add(&self, i: usize, v: T) {
        assert!(i < self.len);
        // SAFETY: in bounds; disjointness per the constructor's contract.
        unsafe { *self.ptr.add(i) = *self.ptr.add(i) + v }
    }
}

impl Shared<'_, f16> {
    /// Store an f32 into an f16 plane (round to nearest).
    #[inline]
    pub fn st(&self, i: usize, v: f32) {
        self.set(i, f16::from_f32(v));
    }
}

/// Bilinear weights and top-left cell for a sample position, clamped so the 2×2
/// footprint stays inside a `w`×`h` grid.
///
/// The diffuse kernels' guards keep back-traces in range; this clamps defensively, which only matters for out-of-contract inputs.
#[inline]
pub fn bilinear_footprint(px: f32, py: f32, w: usize, h: usize) -> (usize, [f32; 4]) {
    let px = px.clamp(0.0, (w - 2) as f32 + 0.999);
    let py = py.clamp(0.0, (h - 2) as f32 + 0.999);
    let ix = px as usize;
    let iy = py as usize;
    let fx = px - ix as f32;
    let fy = py - iy as f32;
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;
    (iy * w + ix, [w00, w10, w01, w11])
}

/// Cell indices of a bilinear footprint, matching the weight order of
/// [`bilinear_footprint`].
#[inline]
pub fn footprint_cells(base: usize, w: usize) -> [usize; 4] {
    [base, base + 1, base + w, base + w + 1]
}

#[inline]
pub fn bilinear_h(p: &PlaneH, px: f32, py: f32) -> f32 {
    let (b, wt) = bilinear_footprint(px, py, p.w, p.h);
    let c = footprint_cells(b, p.w);
    (0..4).map(|k| wt[k] * ld(p, c[k])).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_weights_sum_to_one() {
        for &(x, y) in &[(3.25, 4.75), (0.0, 0.0), (7.999, 1.5)] {
            let (_, wt) = bilinear_footprint(x, y, 10, 10);
            assert!((wt.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn buffered_swap() {
        let mut b = Buffered::new(2, 2, 0u8);
        b.next_mut().fill(7);
        assert_eq!(b.cur().data[0], 0);
        b.swap();
        assert_eq!(b.cur().data[0], 7);
    }
}
