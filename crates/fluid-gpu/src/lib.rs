//! wgpu compute backend for the wet-fluid simulation.
//!
//! [`GpuLayer`] holds the full state of a [`WetLayer`] in GPU buffers and runs the same
//! step schedule with WGSL ports of the reference kernels. Use
//! [`GpuLayer::upload`] / [`GpuLayer::download`] to move state between the backends.
//!
//! Drips (`wet_fluid::drip`) run on the GPU too: drop spawning in full velocity diffusion
//! and the drip's pickup and trail as two single-workgroup passes after each step, with
//! the drip state carried across [`GpuLayer::write_state`] / [`GpuLayer::download`].
//!
//! [`display::Compositor`] composites rendered layers ([`GpuLayer::render`]) for display
//! without bringing them back to the CPU. Layers share a device through [`GpuContext`].
//!
//! Not yet ported: the pluggable pigment mixer (normal mixing only).

pub mod display;

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use rayon::prelude::*;

use bytemuck::{Pod, Zeroable};
use half::f16;
use wet_fluid::grid::{Buffered, Plane};
use wet_fluid::{FluidParams, WetLayer};

/// Plane slots in the channel buffer (must match `fluid.wgsl`).
const N_PLANES: usize = 32;
const N_MAPS: usize = 6;
const UNIFORM_STRIDE: u64 = 256;
const MAX_PASSES: usize = 16;
const BIG: i32 = 0x3fff_ffff;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Cfg {
    w: u32,
    h: u32,
    n: u32,
    vel: u32,
    wd: u32,
    mid: u32,
    gran_on: u32,
    _p0: u32,
    vmax: f32,
    wthr: f32,
    gx: f32,
    gy: f32,
    pthr: f32,
    kp: f32,
    ks: f32,
    kw: f32,
    wclamp: f32,
    rate: f32,
    offset: f32,
    k_abs: f32,
    k_rewet: f32,
    gran_strength: f32,
    gran_contrast: f32,
    s0: f32,
    kh: f32,
    bx0: f32,
    bx1: f32,
    by0: f32,
    by1: f32,
    _p1: f32,
    _p2: f32,
    _p3: f32,
    seed: [u32; 4],
    drops_on: u32,
    drip_on: u32,
    _p4: u32,
    _p5: u32,
    d_width0: f32,
    d_speed: f32,
    d_taper: f32,
    d_minw: f32,
    d_dx: f32,
    d_dy: f32,
    trail_water: f32,
    d_pick: f32,
    trail_share: f32,
    _p6: f32,
    _p7: f32,
    _p8: f32,
    rx0: u32,
    ry0: u32,
    rw: u32,
    rh: u32,
}

/// Display readback states (`GpuLayer::request_display`).
const DISPLAY_IDLE: u8 = 0;
const DISPLAY_PENDING: u8 = 1;
const DISPLAY_READY: u8 = 2;
/// Planes the display reads.
const DISPLAY_PLANES: usize = 11;

/// Drip state slots in the drip buffer (`D` in `fluid.wgsl`).
const DRIP_FLOATS: usize = 16;

#[derive(Debug)]
pub enum GpuError {
    Adapter(String),
    Device(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Adapter(e) => write!(f, "no suitable GPU adapter: {e}"),
            GpuError::Device(e) => write!(f, "device request failed: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// A GPU device and queue to run layers on. Clones share the device, so every layer of a
/// document (and a renderer drawing them) can use one.
#[derive(Clone)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Open the high-performance adapter with [`Self::limits`].
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .map_err(|e| GpuError::Adapter(e.to_string()))?;
            let (device, queue) = adapter
                .request_device(&Self::device_descriptor(&adapter))
                .await
                .map_err(|e| GpuError::Device(e.to_string()))?;
            Ok(Self { device, queue, adapter_info: adapter.get_info() })
        })
    }

    /// What a device made elsewhere (by a UI renderer, say) must be requested with to run
    /// layers: the adapter's own limits, since a large canvas needs large storage buffers.
    pub fn device_descriptor(adapter: &wgpu::Adapter) -> wgpu::DeviceDescriptor<'static> {
        wgpu::DeviceDescriptor {
            label: Some("wet-fluid"),
            required_limits: adapter.limits(),
            ..Default::default()
        }
    }
}

pub struct GpuLayer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<(&'static str, wgpu::ComputePipeline)>,
    bind_group: wgpu::BindGroup,
    planes: wgpu::Buffer,
    maps: wgpu::Buffer,
    rect: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    readback: wgpu::Buffer,
    drip: wgpu::Buffer,
    /// The layer rendered for display by [`Self::render`]: straight RGBA (4 floats per
    /// cell), then the relief height, then the wetness (1 per cell each).
    out: wgpu::Buffer,
    /// Staging buffer of the asynchronous display readback, its state and the plane slots
    /// it was requested with.
    display: wgpu::Buffer,
    display_state: Arc<AtomicU8>,
    display_slots: Vec<usize>,
    pub w: usize,
    pub h: usize,
    pub params: FluidParams,
    gran_on: bool,
    vel: u32,
    wd: u32,
    mid: u32,
    pub step_index: u64,
    pub adapter_info: wgpu::AdapterInfo,
    /// Region transfer (`k_pack` / `k_unpack`): the packed-region bind group layout, the
    /// two pipelines, and the region buffers sized for the largest rectangle so far.
    region_bgl: wgpu::BindGroupLayout,
    region_pipelines: (wgpu::ComputePipeline, wgpu::ComputePipeline),
    region: Option<Region>,
    /// The undo snapshot copy in `readback` ([`Self::request_snapshot`]), and one that was
    /// finished early because a download needed the buffer.
    snap_state: Arc<AtomicU8>,
    snap_ready: Option<LayerSnapshot>,
    /// The arrived snapshot being converted on a worker thread.
    snap_thread: Option<std::thread::JoinHandle<LayerSnapshot>>,
}

struct Region {
    packed: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind: wgpu::BindGroup,
    floats: usize,
}

/// Floats per cell of a packed region: 13 planes and the wet mask.
const REGION_FLOATS: usize = 14;

/// The planes an undo step restores (as `wet_doc` keeps them): water, u, v, settled and
/// suspended colours (f16), the amounts and thicknesses (f32), and the wet mask.
pub struct LayerSnapshot {
    pub h16: Vec<Vec<f16>>,
    pub f32s: Vec<Vec<f32>>,
    pub open: Vec<u8>,
}

/// An inclusive canvas rectangle: x0, y0, x1, y1.
pub type Rect = (usize, usize, usize, usize);

const ENTRY_POINTS: &[&str] = &[
    "k_bbox",
    "k_bbox_final",
    "k_dry",
    "k_diffuse_fast",
    "k_diffuse_complete",
    "k_advect",
    "k_divergence",
    "k_relax",
    "k_project",
    "k_transport_weights",
    "k_transport_gather",
    "k_wet_mask",
    "k_drip_spawn",
    "k_drip_move",
    "k_render",
];

fn h2f(p: &Plane<f16>) -> impl Iterator<Item = f32> + '_ {
    p.data.iter().map(|v| v.to_f32())
}

impl GpuLayer {
    /// Create GPU state mirroring `layer` (its current buffers become slot 0).
    pub fn upload(layer: &WetLayer) -> Result<Self, GpuError> {
        assert!(
            layer.pigment.is_none(),
            "GPU backend supports normal mixing only"
        );
        Ok(Self::upload_on(&GpuContext::new()?, layer))
    }

    /// [`Self::upload`] onto an existing device.
    pub fn upload_on(ctx: &GpuContext, layer: &WetLayer) -> Self {
        assert!(
            layer.pigment.is_none(),
            "GPU backend supports normal mixing only"
        );
        let (device, queue, adapter_info) = (ctx.device.clone(), ctx.queue.clone(), ctx.adapter_info.clone());

        let (w, h) = (layer.w, layer.h);
        let n = w * h;
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mk = |label: &str, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let planes = mk("planes", (N_PLANES * n * 4) as u64, storage);
        let maps = mk("maps", (N_MAPS * n * 4) as u64, storage);
        let gran_len = layer.maps.gran.as_ref().map_or(1, |g| g.data.len());
        let gran = mk("gran", (gran_len * 4) as u64, storage);
        let lut = mk("lut", (layer.lut.table.len() * 4) as u64, storage);
        let rect = mk("rect", 16 * 4, storage);
        let uniforms = mk(
            "uniforms",
            UNIFORM_STRIDE * MAX_PASSES as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let readback = mk(
            "readback",
            ((N_PLANES + N_MAPS) * n * 4 + DRIP_FLOATS * 4) as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let drip = mk("drip", (DRIP_FLOATS * 4) as u64, storage);
        let out = mk("rendered", (display::LAYER_FLOATS * n * 4) as u64, storage);
        let display = mk(
            "display",
            (DISPLAY_PLANES * n * 4) as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        if let Some(g) = &layer.maps.gran {
            let v: Vec<f32> = g.data.iter().map(|&b| b as f32).collect();
            queue.write_buffer(&gran, 0, bytemuck::cast_slice(&v));
        }
        queue.write_buffer(&lut, 0, bytemuck::cast_slice(&layer.lut.table));

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fluid.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("fluid.wgsl"))),
        });
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Cfg>() as u64),
                    },
                    count: None,
                },
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
                storage_entry(7, false),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniforms,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<Cfg>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: planes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: maps.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gran.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: lut.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: rect.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: drip.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluid"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipelines = ENTRY_POINTS
            .iter()
            .map(|&e| {
                let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(e),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (e, p)
            })
            .collect();
        // the region transfer kernels also bind the packed region (group 1)
        let region_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("region"),
            entries: &[storage_entry(0, false)],
        });
        let region_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("region"),
            bind_group_layouts: &[Some(&bgl), Some(&region_bgl)],
            immediate_size: 0,
        });
        let region_pipeline = |e: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&region_layout),
                module: &module,
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let region_pipelines = (region_pipeline("k_pack"), region_pipeline("k_unpack"));

        let mut g = Self {
            region_bgl,
            region_pipelines,
            region: None,
            snap_state: Arc::new(AtomicU8::new(DISPLAY_IDLE)),
            snap_ready: None,
            snap_thread: None,
            device,
            queue,
            pipelines,
            bind_group,
            planes,
            maps,
            rect,
            uniforms,
            readback,
            drip,
            out,
            display,
            display_state: Arc::new(AtomicU8::new(DISPLAY_IDLE)),
            display_slots: Vec::new(),
            w,
            h,
            params: layer.params.clone(),
            gran_on: layer.maps.gran.is_some(),
            vel: 0,
            wd: 0,
            mid: 0,
            step_index: layer.step_index,
            adapter_info,
        };
        g.write_state(layer);
        g
    }

    /// Overwrite GPU channel and map buffers (and the step counter) from `layer`.
    pub fn write_state(&mut self, layer: &WetLayer) {
        self.discard_display();
        self.step_index = layer.step_index;
        let n = self.w * self.h;
        let mut s = vec![0.0f32; N_PLANES * n];
        let mut put = |slot: usize, it: &mut dyn Iterator<Item = f32>| {
            for (d, v) in s[slot * n..(slot + 1) * n].iter_mut().zip(it) {
                *d = v;
            }
        };
        let bh = |b: &Buffered<f16>| b.cur().clone();
        put(0, &mut h2f(&bh(&layer.u)));
        put(2, &mut h2f(&bh(&layer.v)));
        put(4, &mut h2f(&bh(&layer.water)));
        for k in 0..3 {
            put(6 + 2 * k, &mut h2f(&bh(&layer.dens.rgb[k])));
            put(16 + 2 * k, &mut h2f(&bh(&layer.mid.rgb[k])));
        }
        put(12, &mut layer.dens.amount.cur().data.iter().copied());
        put(14, &mut layer.dens.thickness.cur().data.iter().copied());
        put(22, &mut layer.mid.amount.cur().data.iter().copied());
        put(24, &mut layer.mid.thickness.cur().data.iter().copied());
        put(26, &mut h2f(&layer.us));
        put(27, &mut h2f(&layer.vs));
        put(28, &mut layer.acc_a.data.iter().copied());
        put(29, &mut layer.acc_b.data.iter().copied());
        put(30, &mut h2f(&layer.div));
        put(31, &mut h2f(&layer.pressure));
        self.queue
            .write_buffer(&self.planes, 0, bytemuck::cast_slice(&s));
        let m = &layer.maps;
        let mut mv = Vec::with_capacity(N_MAPS * n);
        mv.extend(m.active.data.iter().map(|&b| b as f32));
        mv.extend(m.open.data.iter().map(|&b| b as f32));
        mv.extend(m.factor.data.iter().map(|&b| b as f32));
        mv.extend(m.absorb.data.iter().map(|&b| b as f32));
        mv.extend(m.height.data.iter().map(|&b| b as f32));
        mv.extend((0..n).map(|c| layer.maps.stencil_at(c)));
        self.queue
            .write_buffer(&self.maps, 0, bytemuck::cast_slice(&mv));
        let mut d = [0.0f32; DRIP_FLOATS];
        if let Some(r) = layer.drips.first() {
            d[..11].copy_from_slice(&[1.0, r.x, r.y, r.width, r.taper, r.min_width, r.load, r.rgb[0], r.rgb[1], r.rgb[2], r.density]);
        }
        self.queue.write_buffer(&self.drip, 0, bytemuck::cast_slice(&d));
        (self.vel, self.wd, self.mid) = (0, 0, 0);
    }

    fn cfg(&self) -> Cfg {
        let p = &self.params;
        let d = p.diffuse_consts();
        let k = p.dry_consts();
        let ph = &p.placeholders;
        Cfg {
            w: self.w as u32,
            h: self.h as u32,
            n: self.step_index as u32,
            vel: self.vel,
            wd: self.wd,
            mid: self.mid,
            gran_on: (self.gran_on && k.gran_enabled) as u32,
            vmax: d.vmax,
            wthr: d.wthr,
            gx: d.gx,
            gy: d.gy,
            pthr: d.pthr,
            kp: d.kp,
            ks: d.ks,
            kw: d.kw,
            wclamp: d.wclamp,
            rate: k.rate,
            offset: k.offset,
            k_abs: k.k_abs,
            k_rewet: k.k_rewet,
            gran_strength: k.gran_strength,
            gran_contrast: k.gran_contrast,
            s0: ph.transport_s0,
            kh: ph.transport_kh,
            bx0: 0.0,
            bx1: (self.w - 1) as f32,
            by0: 0.0,
            by1: (self.h - 1) as f32,
            ..self.drip_cfg()
        }
    }

    /// The drip settings part of the pass configuration (all zero when drips are off).
    fn drip_cfg(&self) -> Cfg {
        let Some(d) = wet_fluid::drip::consts(&self.params) else { return Cfg::default() };
        Cfg {
            drip_on: 1,
            d_width0: d.width0,
            d_speed: d.speed,
            d_taper: d.taper,
            d_minw: d.min_width,
            d_dx: d.dir.0,
            d_dy: d.dir.1,
            trail_water: wet_fluid::drip::TRAIL_WATER,
            d_pick: wet_fluid::drip::PICK,
            trail_share: wet_fluid::drip::TRAIL_SHARE,
            ..Cfg::default()
        }
    }

    fn pipeline(&self, name: &str) -> &wgpu::ComputePipeline {
        &self.pipelines.iter().find(|(n, _)| *n == name).unwrap().1
    }

    /// Run `count` simulation steps. Each step is one queue submission.
    pub fn step(&mut self, count: usize) {
        for _ in 0..count {
            self.step_once();
        }
    }

    fn step_once(&mut self) {
        let n = self.step_index;
        self.step_index += 1;
        let t = self.params.timing();
        // Build the pass list with the buffer parities each pass sees.
        let mut passes: Vec<(&'static str, Cfg)> = Vec::new();
        let full = |name: &str| !matches!(name, "k_bbox_final" | "k_drip_spawn" | "k_drip_move");
        passes.push(("k_bbox", self.cfg()));
        passes.push(("k_bbox_final", self.cfg()));
        if n.is_multiple_of(t.n_dry as u64) {
            passes.push(("k_dry", self.cfg()));
            self.wd ^= 1;
            self.mid ^= 1;
        }
        let diffuse = if n.is_multiple_of(t.n_full as u64) {
            "k_diffuse_complete"
        } else {
            "k_diffuse_fast"
        };
        let mut dc = self.cfg();
        if diffuse == "k_diffuse_complete" && self.params.create_drops && n.is_multiple_of(3) && self.params.tilted() {
            dc.drops_on = 1;
            dc.seed = wet_fluid::layer::step_seed(n);
        }
        passes.push((diffuse, dc));
        passes.push(("k_advect", self.cfg()));
        self.vel ^= 1;
        if n.is_multiple_of(3) && self.params.project_cycles > 0 {
            passes.push(("k_divergence", self.cfg()));
            passes.push(("k_relax", self.cfg()));
            passes.push(("k_project", self.cfg()));
            self.vel ^= 1;
        }
        passes.push(("k_transport_weights", self.cfg()));
        passes.push(("k_transport_gather", self.cfg()));
        self.wd ^= 1;
        passes.push(("k_wet_mask", self.cfg()));
        // drip.rs advance, after the fluid step as on the CPU
        passes.push(("k_drip_spawn", self.cfg()));
        passes.push(("k_drip_move", self.cfg()));

        let mut ubytes = vec![0u8; UNIFORM_STRIDE as usize * passes.len()];
        for (i, (_, c)) in passes.iter().enumerate() {
            let off = i * UNIFORM_STRIDE as usize;
            ubytes[off..off + std::mem::size_of::<Cfg>()].copy_from_slice(bytemuck::bytes_of(c));
        }
        self.queue.write_buffer(&self.uniforms, 0, &ubytes);
        let rect_init: [i32; 16] = [BIG, BIG, -1, -1, 0, 0, 0, 0, 0, 0, BIG, 0, 0, 0, 0, 0];
        self.queue
            .write_buffer(&self.rect, 0, bytemuck::cast_slice(&rect_init));

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            let (gx, gy) = (self.w.div_ceil(8) as u32, self.h.div_ceil(8) as u32);
            for (i, (name, _)) in passes.iter().enumerate() {
                cp.set_pipeline(self.pipeline(name));
                cp.set_bind_group(0, &self.bind_group, &[(i as u64 * UNIFORM_STRIDE) as u32]);
                if full(name) {
                    cp.dispatch_workgroups(gx, gy, 1)
                } else {
                    cp.dispatch_workgroups(1, 1, 1)
                }
            }
        }
        self.queue.submit([enc.finish()]);
    }

    /// Render the current state for display into [`Self::rendered`]: what
    /// `wet_fluid::render::render_layer` computes, plus the relief height impasto shading
    /// reads. Queued after the steps submitted so far; nothing comes back to the CPU.
    pub fn render(&self) {
        // the last uniform slot: a step uses fewer passes
        let slot = (MAX_PASSES - 1) as u64 * UNIFORM_STRIDE;
        self.queue.write_buffer(&self.uniforms, slot, bytemuck::bytes_of(&self.cfg()));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(self.pipeline("k_render"));
            cp.set_bind_group(0, &self.bind_group, &[slot as u32]);
            cp.dispatch_workgroups(self.w.div_ceil(8) as u32, self.h.div_ceil(8) as u32, 1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// The buffer [`Self::render`] writes.
    pub fn rendered(&self) -> &wgpu::Buffer {
        &self.out
    }

    /// The packed-region buffers, grown to hold `floats`.
    fn region_buffers(&mut self, floats: usize) -> &Region {
        if self.region.as_ref().is_none_or(|r| r.floats < floats) {
            let floats = floats.next_power_of_two();
            let mk = |label, usage| {
                self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (floats * 4) as u64, usage, mapped_at_creation: false })
            };
            let packed = mk("region", wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
            let staging = mk("region staging", wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("region"),
                layout: &self.region_bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: packed.as_entire_binding() }],
            });
            self.region = Some(Region { packed, staging, bind, floats });
        }
        self.region.as_ref().expect("made")
    }

    /// Dispatch `k_pack` or `k_unpack` over `rect`.
    fn region_pass(&mut self, rect: Rect, pack: bool, enc: &mut wgpu::CommandEncoder) {
        let (rw, rh) = (rect.2 - rect.0 + 1, rect.3 - rect.1 + 1);
        let slot = (MAX_PASSES - 2) as u64 * UNIFORM_STRIDE;
        let cfg = Cfg { rx0: rect.0 as u32, ry0: rect.1 as u32, rw: rw as u32, rh: rh as u32, ..self.cfg() };
        self.queue.write_buffer(&self.uniforms, slot, bytemuck::bytes_of(&cfg));
        let region = self.region.as_ref().expect("region buffers");
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(if pack { &self.region_pipelines.0 } else { &self.region_pipelines.1 });
        cp.set_bind_group(0, &self.bind_group, &[slot as u32]);
        cp.set_bind_group(1, &region.bind, &[]);
        cp.dispatch_workgroups(rw.div_ceil(8) as u32, rh.div_ceil(8) as u32, 1);
    }

    /// Bring `rect` of the planes a brush stroke works on (water, velocity, both pigment
    /// layers, the wet mask) into `layer`, waiting for the steps submitted so far. The rest
    /// of `layer` is left as it was.
    pub fn read_region(&mut self, layer: &mut WetLayer, rect: Rect) {
        let (rw, rh) = (rect.2 - rect.0 + 1, rect.3 - rect.1 + 1);
        let rn = rw * rh;
        self.region_buffers(REGION_FLOATS * rn);
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.region_pass(rect, true, &mut enc);
        let region = self.region.as_ref().expect("region buffers");
        let bytes = (REGION_FLOATS * rn * 4) as u64;
        enc.copy_buffer_to_buffer(&region.packed, 0, &region.staging, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = region.staging.slice(..bytes);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.wait();
        {
            let data = slice.get_mapped_range().expect("map region");
            let all: &[f32] = bytemuck::cast_slice(&data);
            let plane = |k: usize| &all[k * rn..(k + 1) * rn];
            let w = self.w;
            let rows = |dst: &mut [f16], src: &[f32]| {
                for y in 0..rh {
                    let d = &mut dst[(rect.1 + y) * w + rect.0..][..rw];
                    for (d, &s) in d.iter_mut().zip(&src[y * rw..(y + 1) * rw]) {
                        *d = f16::from_f32(s);
                    }
                }
            };
            let rows32 = |dst: &mut [f32], src: &[f32]| {
                for y in 0..rh {
                    dst[(rect.1 + y) * w + rect.0..][..rw].copy_from_slice(&src[y * rw..(y + 1) * rw]);
                }
            };
            let [dr, dg, db] = &mut layer.dens.rgb;
            let [mr, mg, mb] = &mut layer.mid.rgb;
            let h16 = [layer.water.cur_mut(), layer.u.cur_mut(), layer.v.cur_mut(), dr.cur_mut(), dg.cur_mut(), db.cur_mut(), mr.cur_mut(), mg.cur_mut(), mb.cur_mut()];
            for (k, p) in h16.into_iter().enumerate() {
                rows(&mut p.data, plane(k));
            }
            let f32s = [layer.dens.amount.cur_mut(), layer.dens.thickness.cur_mut(), layer.mid.amount.cur_mut(), layer.mid.thickness.cur_mut()];
            for (k, p) in f32s.into_iter().enumerate() {
                rows32(&mut p.data, plane(9 + k));
            }
            let open = plane(13);
            for y in 0..rh {
                let d = &mut layer.maps.open.data[(rect.1 + y) * w + rect.0..][..rw];
                for (d, &s) in d.iter_mut().zip(&open[y * rw..(y + 1) * rw]) {
                    *d = s as u8;
                }
            }
        }
        region.staging.unmap();
    }

    /// Write `rect` of `layer`'s stroke planes (as [`Self::read_region`] reads them) back to
    /// the GPU, after the work submitted so far.
    pub fn write_region(&mut self, layer: &WetLayer, rect: Rect) {
        let (rw, rh) = (rect.2 - rect.0 + 1, rect.3 - rect.1 + 1);
        let rn = rw * rh;
        let w = self.w;
        let mut v = vec![0.0f32; REGION_FLOATS * rn];
        {
            let (h16, rest) = v.split_at_mut(9 * rn);
            let (f32s, open) = rest.split_at_mut(4 * rn);
            let planes16 = [layer.water.cur(), layer.u.cur(), layer.v.cur(), layer.dens.rgb[0].cur(), layer.dens.rgb[1].cur(), layer.dens.rgb[2].cur(), layer.mid.rgb[0].cur(), layer.mid.rgb[1].cur(), layer.mid.rgb[2].cur()];
            for (k, p) in planes16.into_iter().enumerate() {
                for y in 0..rh {
                    let src = &p.data[(rect.1 + y) * w + rect.0..][..rw];
                    for (d, s) in h16[k * rn + y * rw..][..rw].iter_mut().zip(src) {
                        *d = s.to_f32();
                    }
                }
            }
            let planes32 = [layer.dens.amount.cur(), layer.dens.thickness.cur(), layer.mid.amount.cur(), layer.mid.thickness.cur()];
            for (k, p) in planes32.into_iter().enumerate() {
                for y in 0..rh {
                    f32s[k * rn + y * rw..][..rw].copy_from_slice(&p.data[(rect.1 + y) * w + rect.0..][..rw]);
                }
            }
            for y in 0..rh {
                for (d, &s) in open[y * rw..][..rw].iter_mut().zip(&layer.maps.open.data[(rect.1 + y) * w + rect.0..][..rw]) {
                    *d = s as f32;
                }
            }
        }
        self.region_buffers(REGION_FLOATS * rn);
        let region = self.region.as_ref().expect("region buffers");
        self.queue.write_buffer(&region.packed, 0, bytemuck::cast_slice(&v));
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.region_pass(rect, false, &mut enc);
        self.queue.submit([enc.finish()]);
    }

    /// Start copying the planes an undo step restores, as of the work submitted so far,
    /// without waiting ([`Self::take_snapshot`] collects them).
    pub fn request_snapshot(&mut self) {
        if self.snapshot_pending() {
            return;
        }
        let n = self.w * self.h;
        let bytes = (n * 4) as u64;
        let (vel, wd, mid) = (self.vel as usize, self.wd as usize, self.mid as usize);
        let slots = [4 + wd, vel, 2 + vel, 6 + wd, 8 + wd, 10 + wd, 16 + mid, 18 + mid, 20 + mid, 12 + wd, 14 + wd, 22 + mid, 24 + mid];
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (i, &slot) in slots.iter().enumerate() {
            enc.copy_buffer_to_buffer(&self.planes, slot as u64 * bytes, &self.readback, i as u64 * bytes, bytes);
        }
        enc.copy_buffer_to_buffer(&self.maps, bytes, &self.readback, 13 * bytes, bytes);
        self.queue.submit([enc.finish()]);
        self.snap_state.store(DISPLAY_PENDING, Ordering::Release);
        let state = self.snap_state.clone();
        self.readback.slice(..14 * bytes).map_async(wgpu::MapMode::Read, move |r| {
            state.store(if r.is_ok() { DISPLAY_READY } else { DISPLAY_IDLE }, Ordering::Release);
        });
    }

    /// Is an undo snapshot requested and not yet taken?
    pub fn snapshot_pending(&self) -> bool {
        self.snap_ready.is_some() || self.snap_thread.is_some() || self.snap_state.load(Ordering::Acquire) != DISPLAY_IDLE
    }

    /// The snapshot [`Self::request_snapshot`] started, once it has arrived and been
    /// converted (on a worker thread, so the caller's frame does not pay for it); `wait`:
    /// block until then.
    pub fn take_snapshot(&mut self, wait: bool) -> Option<LayerSnapshot> {
        if let Some(s) = self.snap_ready.take() {
            return Some(s);
        }
        if self.snap_thread.is_none() {
            if self.snap_state.load(Ordering::Acquire) == DISPLAY_PENDING {
                let poll = if wait { wgpu::PollType::wait_indefinitely() } else { wgpu::PollType::Poll };
                let _ = self.device.poll(poll);
            }
            if self.snap_state.load(Ordering::Acquire) != DISPLAY_READY {
                return None;
            }
            let (buf, n) = (self.readback.clone(), self.w * self.h);
            let convert = move || {
                let snap = {
                    let data = buf.slice(..(14 * n * 4) as u64).get_mapped_range().expect("map snapshot");
                    let all: &[f32] = bytemuck::cast_slice(&data);
                    let plane = |k: usize| &all[k * n..(k + 1) * n];
                    LayerSnapshot {
                        h16: (0..9).into_par_iter().map(|k| plane(k).iter().map(|&v| f16::from_f32(v)).collect()).collect(),
                        f32s: (9..13).into_par_iter().map(|k| plane(k).to_vec()).collect(),
                        open: plane(13).par_iter().map(|&v| v as u8).collect(),
                    }
                };
                buf.unmap();
                snap
            };
            // the web has no threads (and its GPU objects stay on the main thread)
            #[cfg(target_arch = "wasm32")]
            {
                self.snap_state.store(DISPLAY_IDLE, Ordering::Release);
                return Some(convert());
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.snap_thread = Some(std::thread::spawn(convert));
            }
        }
        if !wait && !self.snap_thread.as_ref().is_some_and(|t| t.is_finished()) {
            return None;
        }
        let snap = self.snap_thread.take().expect("converting").join().expect("snapshot conversion");
        self.snap_state.store(DISPLAY_IDLE, Ordering::Release);
        Some(snap)
    }

    /// Block until all submitted work has finished.
    pub fn wait(&self) {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
    }

    /// Copy only what display compositing reads (water, colours, amounts and thicknesses
    /// of both pigment layers: 11 of the 37 planes) into `layer`'s current buffers, for
    /// showing a simulation that keeps running on the GPU. The other planes of `layer`
    /// are left stale; use [`Self::download`] before the CPU edits or steps it. Blocks
    /// until the GPU has finished; [`Self::request_display`] does not.
    pub fn download_display(&mut self, layer: &mut WetLayer) {
        self.discard_display();
        self.request_display();
        self.take_display(layer, true);
    }

    /// Start copying the display planes of the state as of the steps submitted so far,
    /// without waiting. Returns false while an earlier request is still outstanding.
    pub fn request_display(&mut self) -> bool {
        if self.display_state.load(Ordering::Acquire) != DISPLAY_IDLE {
            return false;
        }
        let n = self.w * self.h;
        let (wd, mid) = (self.wd as usize, self.mid as usize);
        self.display_slots = vec![4 + wd, 12 + wd, 14 + wd, 22 + mid, 24 + mid];
        for k in 0..3 {
            self.display_slots.push(6 + 2 * k + wd);
            self.display_slots.push(16 + 2 * k + mid);
        }
        let bytes = (n * 4) as u64;
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (i, &slot) in self.display_slots.iter().enumerate() {
            enc.copy_buffer_to_buffer(&self.planes, slot as u64 * bytes, &self.display, i as u64 * bytes, bytes);
        }
        self.queue.submit([enc.finish()]);
        self.display_state.store(DISPLAY_PENDING, Ordering::Release);
        let state = self.display_state.clone();
        self.display.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            state.store(if r.is_ok() { DISPLAY_READY } else { DISPLAY_IDLE }, Ordering::Release);
        });
        true
    }

    /// Copy a finished display request into `layer` (see [`Self::download_display`]);
    /// `wait` blocks until it is ready. Returns whether `layer` was updated.
    pub fn take_display(&mut self, layer: &mut WetLayer, wait: bool) -> bool {
        if self.display_state.load(Ordering::Acquire) == DISPLAY_PENDING {
            let poll = if wait { wgpu::PollType::wait_indefinitely() } else { wgpu::PollType::Poll };
            let _ = self.device.poll(poll);
        }
        if self.display_state.load(Ordering::Acquire) != DISPLAY_READY {
            return false;
        }
        let n = self.w * self.h;
        {
            let data = self.display.slice(..).get_mapped_range().expect("map display");
            let all: &[f32] = bytemuck::cast_slice(&data);
            let plane = |i: usize| &all[i * n..(i + 1) * n];
            let seth = |dst: &mut Plane<f16>, src: &[f32]| {
                dst.data.par_iter_mut().zip(src.par_iter()).for_each(|(d, &s)| *d = f16::from_f32(s));
            };
            seth(layer.water.cur_mut(), plane(0));
            layer.dens.amount.cur_mut().data.copy_from_slice(plane(1));
            layer.dens.thickness.cur_mut().data.copy_from_slice(plane(2));
            layer.mid.amount.cur_mut().data.copy_from_slice(plane(3));
            layer.mid.thickness.cur_mut().data.copy_from_slice(plane(4));
            for k in 0..3 {
                seth(layer.dens.rgb[k].cur_mut(), plane(5 + 2 * k));
                seth(layer.mid.rgb[k].cur_mut(), plane(6 + 2 * k));
            }
        }
        self.display.unmap();
        self.display_state.store(DISPLAY_IDLE, Ordering::Release);
        true
    }

    /// Drop an outstanding display request (its planes may predate a
    /// [`Self::write_state`]).
    pub fn discard_display(&mut self) {
        match self.display_state.load(Ordering::Acquire) {
            DISPLAY_PENDING => {
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
                if self.display_state.load(Ordering::Acquire) == DISPLAY_READY {
                    self.display.unmap();
                }
            }
            DISPLAY_READY => self.display.unmap(),
            _ => {}
        }
        self.display_state.store(DISPLAY_IDLE, Ordering::Release);
    }

    /// Copy the GPU state into `layer`'s current buffers.
    pub fn download(&mut self, layer: &mut WetLayer) {
        // the readback buffer may hold an undo snapshot on its way: keep it
        if self.snap_thread.is_some() || self.snap_state.load(Ordering::Acquire) != DISPLAY_IDLE {
            self.snap_ready = self.take_snapshot(true);
        }
        let n = self.w * self.h;
        let bytes = (n * 4) as u64;
        let (vel, wd, mid) = (self.vel as usize, self.wd as usize, self.mid as usize);
        // the current half of each double-buffered plane, and the single planes the CPU
        // step reads back (acc_b is scratch)
        let slots = [
            vel, 2 + vel, 4 + wd, 6 + wd, 8 + wd, 10 + wd, 16 + mid, 18 + mid, 20 + mid,
            12 + wd, 14 + wd, 22 + mid, 24 + mid, 26, 27, 28, 30, 31,
        ];
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (i, &slot) in slots.iter().enumerate() {
            enc.copy_buffer_to_buffer(&self.planes, slot as u64 * bytes, &self.readback, i as u64 * bytes, bytes);
        }
        let (open_at, drip_at) = (slots.len() as u64 * bytes, (slots.len() + 1) as u64 * bytes);
        enc.copy_buffer_to_buffer(&self.maps, bytes, &self.readback, open_at, bytes);
        enc.copy_buffer_to_buffer(&self.drip, 0, &self.readback, drip_at, (DRIP_FLOATS * 4) as u64);
        self.queue.submit([enc.finish()]);
        let slice = self.readback.slice(..drip_at + (DRIP_FLOATS * 4) as u64);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.wait();
        {
            let data = slice.get_mapped_range().expect("map readback");
            let all: &[f32] = bytemuck::cast_slice(&data);
            let plane = |i: usize| &all[i * n..(i + 1) * n];
            let seth = |dst: &mut Plane<f16>, src: &[f32]| {
                dst.data.par_iter_mut().zip(src.par_iter()).for_each(|(d, &s)| *d = f16::from_f32(s));
            };
            let setf = |dst: &mut Plane<f32>, src: &[f32]| dst.data.copy_from_slice(src);
            seth(layer.u.cur_mut(), plane(0));
            seth(layer.v.cur_mut(), plane(1));
            seth(layer.water.cur_mut(), plane(2));
            for k in 0..3 {
                seth(layer.dens.rgb[k].cur_mut(), plane(3 + k));
                seth(layer.mid.rgb[k].cur_mut(), plane(6 + k));
            }
            setf(layer.dens.amount.cur_mut(), plane(9));
            setf(layer.dens.thickness.cur_mut(), plane(10));
            setf(layer.mid.amount.cur_mut(), plane(11));
            setf(layer.mid.thickness.cur_mut(), plane(12));
            seth(&mut layer.us, plane(13));
            seth(&mut layer.vs, plane(14));
            setf(&mut layer.acc_a, plane(15));
            seth(&mut layer.div, plane(16));
            seth(&mut layer.pressure, plane(17));
            layer.maps.open.data.par_iter_mut().zip(plane(slots.len()).par_iter()).for_each(|(d, &s)| *d = s as u8);
            let d = &all[slots.len() * n + n..slots.len() * n + n + DRIP_FLOATS];
            layer.drips.clear();
            if d[0] != 0.0 {
                layer.drips.push(wet_fluid::drip::Drip {
                    x: d[1],
                    y: d[2],
                    width: d[3],
                    taper: d[4],
                    min_width: d[5],
                    load: d[6],
                    rgb: [d[7], d[8], d[9]],
                    density: d[10],
                });
            }
        }
        self.readback.unmap();
        layer.step_index = self.step_index;
    }
}
