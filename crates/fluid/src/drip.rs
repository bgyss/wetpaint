//! Drips: drops spawned by full velocity diffusion that run downhill across
//! the canvas and leave a wet trail the fluid can follow.
//!
//! On a tilted canvas with drips enabled, the wet stroke pours down as one wide funnel from the
//! middle of its lower edge, about 80–120 px wide at first for a 136 px stroke, narrowing
//! by about half a pixel per pixel travelled, and reaching the canvas bottom within
//! 50 steps (≈ 5 px/step). throughout.

use crate::grid::ld;
use crate::layer::WetLayer;
use crate::mix::Rgb;

/// One running drop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drip {
    pub x: f32,
    pub y: f32,
    /// Trail width in pixels; shrinks as the drop runs out of water.
    pub width: f32,
    /// Width lost per step.
    pub taper: f32,
    /// Narrowest the trail gets.
    pub min_width: f32,
    /// Suspended pigment the drop carries, and its colour.
    pub load: f32,
    pub rgb: Rgb,
    /// Mean suspended pigment per cell where the drop formed.
    pub density: f32,
}

/// Water a drip leaves in its trail (comparable to a fresh stroke).
pub const TRAIL_WATER: f32 = 3.0 * crate::params::WATER_SCALE;
/// Trail width per unit drop size (≈ 10 / 40 / 55 / 95 px at size 1 / 3 / 5 / 10).
pub const WIDTH_PER_SIZE: f32 = 9.5;
/// Speed in pixels per step per unit drop size at full tilt (≈ 5 px/step at size 10).
pub const SPEED_PER_SIZE: f32 = 0.5;
/// Width lost per pixel travelled at drop length 1 (slower for longer drops).
pub const TAPER_PER_PX: f32 = 0.5;
/// The trail keeps at least this fraction of its starting width.
pub const MIN_WIDTH: f32 = 0.4;
/// Share of the suspended pigment under the drop picked up when it forms.
pub const PICK: f32 = 1.0;
/// Pigment laid per newly reached cell, relative to the source's mean density.
pub const TRAIL_SHARE: f32 = 0.6;

/// Per-layer drip settings derived from the drop parameters and the tilt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DripConsts {
    /// Starting trail width.
    pub width0: f32,
    /// Pixels moved per step, along the unit direction `dir`.
    pub speed: f32,
    pub dir: (f32, f32),
    pub taper: f32,
    pub min_width: f32,
}

/// The drip settings, or `None` when drips are off (no drops or no tilt).
pub fn consts(p: &crate::FluidParams) -> Option<DripConsts> {
    let (tx, ty) = p.effective_tilt();
    let tilt = (tx * tx + ty * ty).sqrt();
    if !p.create_drops || tilt <= 0.0 {
        return None;
    }
    let size = p.drops_size.clamp(0.0, 10.0).max(0.5);
    let length = p.drops_length.clamp(0.0, 10.0).max(0.5);
    let width0 = (WIDTH_PER_SIZE * size).max(4.0);
    let speed = SPEED_PER_SIZE * size * tilt.min(1.0);
    Some(DripConsts {
        width0,
        speed,
        dir: (tx / tilt, ty / tilt),
        taper: TAPER_PER_PX * speed / length,
        min_width: (MIN_WIDTH * width0).max(2.0),
    })
}

/// Spawn drips at this step's spawn points, then move every drip one step.
pub fn advance(l: &mut WetLayer) {
    let Some(k) = consts(&l.params) else {
        l.drips.clear();
        return;
    };
    let DripConsts { width0, speed, dir: (dx, dy), taper, min_width } = k;
    let (w, h) = (l.w as i64, l.h as i64);
    // the kernel offers at most one spawn point per full velocity diffusion step (a hashed
    // wet cell). One drop runs at a time (several at once merge into a blob much wider than one
    // funnel); it takes up the suspended pigment under it and carries it down
    let spawn = if l.drips.is_empty() { l.drops.first().copied() } else { None };
    if let Some((x, y)) = spawn {
        let (mut load, mut acc, mut cells) = (0.0f32, [0.0f32; 3], 0usize);
        disc(x as f32, y as f32, 0.5 * width0, w, h, |c| {
            cells += 1;
            let a = l.dens.amount.cur().data[c] * PICK;
            if a > 0.0 {
                for (k, v) in acc.iter_mut().enumerate() {
                    *v += a * ld(l.dens.rgb[k].cur(), c);
                }
                load += a;
                l.dens.amount.cur_mut().data[c] -= a;
            }
        });
        let rgb = if load > 0.0 { acc.map(|v| v / load) } else { [0.0; 3] };
        l.drips.push(Drip {
            x: x as f32,
            y: y as f32,
            width: width0,
            taper,
            min_width,
            load,
            rgb,
            density: load / cells.max(1) as f32,
        });
    }
    let mut keep = Vec::with_capacity(l.drips.len());
    for mut d in std::mem::take(&mut l.drips) {
        // the drop runs ahead of the flow, so the fluid follows into an already wet trail
        // (a drop at the fluid's own speed leaves tide lines, Q9)
        d.x += dx * speed;
        d.y += dy * speed;
        d.width = (d.width - d.taper).max(d.min_width);
        if d.x < 0.0 || d.y < 0.0 || d.x >= w as f32 || d.y >= h as f32 {
            continue;
        }
        // only newly reached paper is wetted; re-filling the trail behind the drop
        // would send pulses down it (seen as arcs)
        let mut fresh = Vec::new();
        disc(d.x, d.y, 0.5 * d.width, w, h, |c| {
            if l.maps.open.data[c] == 0 && l.maps.stencil_at(c) > 0.0 {
                fresh.push(c);
            }
        });
        // the trail takes a fixed share of the source's pigment density until the load
        // runs out, which sets the trail length
        let per = if fresh.is_empty() { 0.0 } else { (TRAIL_SHARE * d.density).min(d.load / fresh.len() as f32) };
        for &c in &fresh {
            let water = (TRAIL_WATER - ld(l.water.cur(), c)).max(0.0);
            l.deposit(c, water, d.rgb, per);
        }
        d.load -= per * fresh.len() as f32;
        keep.push(d);
    }
    l.drips = keep;
}

/// Calls `f` for every cell inside the disc of radius `r` around `(x, y)`.
fn disc(x: f32, y: f32, r: f32, w: i64, h: i64, mut f: impl FnMut(usize)) {
    let (x0, x1) = (((x - r) as i64).max(0), ((x + r) as i64).min(w - 1));
    let (y0, y1) = (((y - r) as i64).max(0), ((y + r) as i64).min(h - 1));
    for cy in y0..=y1 {
        for cx in x0..=x1 {
            let (ox, oy) = (cx as f32 - x, cy as f32 - y);
            if ox * ox + oy * oy <= r * r {
                f((cy * w + cx) as usize);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drip_runs_downhill_and_wets_its_trail() {
        let mut l = WetLayer::new(64, 128);
        l.params.create_drops = true;
        l.params.tilt_active = true;
        l.params.tilt_y = 1.0;
        l.params.drops_size = 10.0;
        l.drops.push((32, 10));
        advance(&mut l);
        assert_eq!(l.drips.len(), 1);
        for _ in 0..15 {
            l.drops.clear();
            advance(&mut l);
        }
        let d = l.drips[0];
        assert!(d.y > 50.0 && (d.x - 32.0).abs() < 1e-3, "{d:?}");
        let c = 40 * 64 + 32;
        assert_eq!(l.maps.open.data[c], 255);
        assert!(ld(l.water.cur(), c) >= TRAIL_WATER - 0.01);
    }
}
