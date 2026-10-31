//! Display compositing on the GPU (`display.wgsl`): a stack of rendered layers over the
//! paper, written to an RGBA8 texture a window can draw directly. Layers simulated on the
//! GPU are read where they are ([`GpuLayer::render`]); layers on the CPU are rendered
//! there and uploaded. Covers what `wet_doc::Document::render` does (blend modes with
//! Kubelka–Munk pigment mixing, masks, clipping, isolated groups, paper colour and relief,
//! impasto, gloss, soft shadows, metallic layers, deckle) plus the app's view overlays.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wet_compose::MixMode;
use wet_fluid::WetLayer;

use crate::{GpuContext, GpuLayer};

/// Floats per cell of a rendered layer: RGBA, relief height, wetness.
pub const LAYER_FLOATS: usize = 6;
const STRIDE: u64 = 256;
/// Slots for the Kubelka–Munk tables in the document buffer.
const KM_SLOTS: usize = 128;

// k_layer flags (display.wgsl)
const F_COMPOSITE: u32 = 1;
const F_SHADE: u32 = 2;
const F_CLIPPED: u32 = 4;
const F_BASE: u32 = 8;
const F_MASK: u32 = 16;
const F_GROUP: u32 = 32;
const F_HEIGHT: u32 = 64;
const F_METAL: u32 = 128;
const F_GLOSS: u32 = 256;
const F_PIGMENT: u32 = 512;
// k_clear / k_out
const F_PAPER: u32 = 1;
const F_GREY: u32 = 1;
const F_STENCIL: u32 = 2;
const F_SELECT: u32 = 4;
const F_DECKLE: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DCfg {
    w: u32,
    h: u32,
    mode: u32,
    flags: u32,
    opacity: f32,
    impasto: f32,
    gloss: f32,
    gloss_k: f32,
    bg: [f32; 4],
    shadow_k: f32,
    shadow_tan: f32,
    metal_ref: f32,
    metal_str: f32,
    metal_zoom: f32,
    env_w: u32,
    env_h: u32,
    _p0: u32,
}

/// Where a layer's rendering comes from.
pub enum Source<'a> {
    /// Simulated on the GPU: rendered there.
    Gpu(&'a GpuLayer),
    /// On the CPU: rendered there and uploaded (`fresh`: it changed since the last frame
    /// it was uploaded in this slot).
    Cpu(&'a WetLayer, bool),
}

pub struct DisplayLayer<'a> {
    pub source: Source<'a>,
    pub mode: MixMode,
    pub opacity: f32,
    /// Its colour is composited (visible, not masking fluid).
    pub composite: bool,
    /// It is shown (visible, in a visible group): its relief shades the painting.
    pub shown: bool,
    /// Its alpha is cut to the last unclipped layer's below it.
    pub clip: bool,
    /// Layer mask, 0–255 per cell.
    pub mask: Option<&'a [u8]>,
    pub metallic: bool,
    /// Index into [`Backdrop::groups`]; adjacent layers of one group composite in isolation.
    pub group: Option<usize>,
}

/// A layer group as it composites onto the layers below.
#[derive(Clone, Copy)]
pub struct GroupBlend {
    pub mode: MixMode,
    pub opacity: f32,
    pub visible: bool,
}

/// `wet_doc` Metallic Materials settings, 0–10, and the reflection map (`None`:
/// procedural).
pub struct Metallic<'a> {
    pub reflectivity: f32,
    pub scale: f32,
    pub strength: f32,
    pub map: Option<(usize, usize, &'a [f32])>,
}

/// Everything about the document the layers are composited with.
pub struct Backdrop<'a> {
    /// Paper colour, 0–1.
    pub color: [f32; 3],
    /// Paper relief shading per cell (`None`: flat).
    pub shade: Option<&'a [f32]>,
    pub impasto: f32,
    /// Glossy highlight strength (0: off).
    pub gloss: f32,
    /// Soft shadows (strength, light altitude), 0–10.
    pub shadows: Option<(f32, f32)>,
    /// Shading of metallic layers, when any is shown.
    pub metallic: Option<Metallic<'a>>,
    /// Kubelka–Munk tables (`KmMixer::tables`) when the Pigment mode mixes pigments.
    pub pigment: Option<&'a [f32]>,
    pub groups: &'a [GroupBlend],
    /// Sheet coverage (the output alpha) when the paper is deckled.
    pub deckle: Option<&'a [f32]>,
    /// View overlays: greyscale, stencil film coverage (0–1) and the selection (0–255).
    pub greyscale: bool,
    pub stencil: Option<&'a [f32]>,
    pub selection: Option<&'a [u8]>,
    /// Masks, deckle, stencil film, selection or reflection map may have changed since the
    /// last frame (they are uploaded again).
    pub changed: bool,
}

pub struct Compositor {
    ctx: GpuContext,
    pub w: usize,
    pub h: usize,
    acc: wgpu::Buffer,
    aux: wgpu::Buffer,
    group: wgpu::Buffer,
    hmax: wgpu::Buffer,
    /// Paper shading, stencil, selection, deckle planes; the KM tables; the reflection map.
    doc: wgpu::Buffer,
    /// The paper shading last uploaded (address, length), to upload it only on change.
    paper_key: (usize, usize),
    uniforms: wgpu::Buffer,
    uniform_slots: usize,
    /// Upload buffers for layers rendered on the CPU and for masks, by stack position.
    uploads: Vec<Option<wgpu::Buffer>>,
    masks: Vec<Option<wgpu::Buffer>>,
    texture: wgpu::Texture,
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<(&'static str, wgpu::ComputePipeline)>,
}

const ENTRY_POINTS: &[&str] = &["k_clear", "k_group_clear", "k_layer", "k_group_out", "k_hmax", "k_shadow", "k_metal", "k_out"];

impl Compositor {
    pub fn new(ctx: &GpuContext, w: usize, h: usize) -> Self {
        let device = &ctx.device;
        let n = (w * h) as u64;
        let buffer = |label: &str, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("display"),
            size: wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            // the window samples it as sRGB, like the 8-bit image the CPU path uploads
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("display.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("display.wgsl"))),
        });
        let buf = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: binding == 0, min_binding_size: None },
            count: None,
        };
        let (read, write) = (wgpu::BufferBindingType::Storage { read_only: true }, wgpu::BufferBindingType::Storage { read_only: false });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display"),
            entries: &[
                buf(0, wgpu::BufferBindingType::Uniform),
                buf(1, write),
                buf(2, read),
                buf(3, read),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                buf(5, read),
                buf(6, write),
                buf(7, write),
                buf(8, write),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ENTRY_POINTS
            .iter()
            .map(|&e| {
                let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(e),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (e, p)
            })
            .collect();
        Self {
            ctx: ctx.clone(),
            w,
            h,
            acc: buffer("display acc", n * 16, wgpu::BufferUsages::STORAGE),
            aux: buffer("display aux", n * 16, wgpu::BufferUsages::STORAGE),
            group: buffer("display group", n * 16, wgpu::BufferUsages::STORAGE),
            hmax: buffer("display hmax", 4, wgpu::BufferUsages::STORAGE),
            doc: buffer("display doc", (4 * n + KM_SLOTS as u64) * 4, storage),
            paper_key: (0, 0),
            uniforms: buffer("display cfg", STRIDE * 8, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            uniform_slots: 8,
            uploads: Vec::new(),
            masks: Vec::new(),
            texture,
            layout,
            pipelines,
        }
    }

    fn pipeline(&self, name: &str) -> &wgpu::ComputePipeline {
        &self.pipelines.iter().find(|(n, _)| *n == name).expect("entry point").1
    }

    /// The composited painting (RGBA8, premultiplied; view it as `Rgba8UnormSrgb` to sample
    /// sRGB).
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// A view of [`Self::texture`] that samples it as sRGB, for drawing it in a window.
    pub fn srgb_view(&self) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            // sampled only: an sRGB format cannot be a storage binding
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        })
    }

    /// Upload the document planes that changed (offsets in floats).
    fn upload_doc(&mut self, b: &Backdrop) {
        let queue = &self.ctx.queue;
        let n = self.w * self.h;
        let at = |plane: usize| (plane * n * 4) as u64;
        if let Some(s) = b.shade {
            let key = (s.as_ptr() as usize, s.len());
            if key != self.paper_key {
                queue.write_buffer(&self.doc, 0, bytemuck::cast_slice(s));
                self.paper_key = key;
            }
        }
        if !b.changed {
            return;
        }
        if let Some(s) = b.stencil {
            queue.write_buffer(&self.doc, at(1), bytemuck::cast_slice(s));
        }
        if let Some(sel) = b.selection {
            let v: Vec<f32> = sel.iter().map(|&m| m as f32).collect();
            queue.write_buffer(&self.doc, at(2), bytemuck::cast_slice(&v));
        }
        if let Some(d) = b.deckle {
            queue.write_buffer(&self.doc, at(3), bytemuck::cast_slice(d));
        }
        if let Some(t) = b.pigment {
            queue.write_buffer(&self.doc, at(4), bytemuck::cast_slice(&t[..t.len().min(KM_SLOTS)]));
        }
        let env = b.metallic.as_ref().and_then(|m| m.map).map(|(_, _, d)| d);
        if let Some(d) = env {
            let need = ((4 * n + KM_SLOTS + d.len()) * 4) as u64;
            if self.doc.size() < need {
                // grow, then upload every plane again
                let bigger = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("display doc"),
                    size: need,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.paper_key = (0, 0);
                self.doc = bigger;
                return self.upload_doc(b);
            }
            queue.write_buffer(&self.doc, ((4 * n + KM_SLOTS) * 4) as u64, bytemuck::cast_slice(d));
        }
    }

    /// Composite `layers` (bottom first) over `backdrop` into [`Self::texture`]. Queued on
    /// the device after work already submitted; returns without waiting.
    pub fn compose(&mut self, layers: &[DisplayLayer], backdrop: &Backdrop) {
        let n = self.w * self.h;
        self.upload_doc(backdrop);
        // layers on the CPU: rendered there, uploaded when they changed; masks with the
        // document's changes
        self.uploads.resize_with(layers.len(), || None);
        self.masks.resize_with(layers.len(), || None);
        let device = self.ctx.device.clone();
        let queue = self.ctx.queue.clone();
        let storage = |label, floats: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (floats * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        for (k, l) in layers.iter().enumerate() {
            if let Source::Cpu(wet, fresh) = l.source {
                let new = self.uploads[k].is_none();
                let buf = self.uploads[k].get_or_insert_with(|| storage("display layer", LAYER_FLOATS * n));
                if fresh || new {
                    queue.write_buffer(buf, 0, &cpu_rendering(wet));
                }
            }
            if let Some(m) = l.mask {
                let new = self.masks[k].is_none();
                let buf = self.masks[k].get_or_insert_with(|| storage("display mask", n));
                if backdrop.changed || new {
                    let v: Vec<f32> = m.iter().map(|&x| x as f32).collect();
                    queue.write_buffer(buf, 0, bytemuck::cast_slice(&v));
                }
            }
        }
        for l in layers {
            if let Source::Gpu(g) = l.source {
                g.render();
            }
        }

        // the pass list: (entry point, uniforms, layer index for its bindings)
        let pigment = backdrop.pigment.is_some();
        let mode_index = |m: MixMode| MixMode::ALL.iter().position(|&x| x == m).unwrap_or(0) as u32;
        let metal = backdrop.metallic.as_ref();
        let env = metal.and_then(|m| m.map);
        let base = DCfg {
            w: self.w as u32,
            h: self.h as u32,
            mode: 0,
            flags: 0,
            opacity: 1.0,
            impasto: backdrop.impasto,
            gloss: backdrop.gloss,
            gloss_k: backdrop.impasto.max(0.2) * 0.5,
            bg: [backdrop.color[0], backdrop.color[1], backdrop.color[2], 0.0],
            shadow_k: backdrop.shadows.map_or(0.0, |(s, _)| 0.06 * s.clamp(0.0, 10.0)),
            shadow_tan: backdrop.shadows.map_or(1.0, |(_, a)| 0.6 - 0.05 * a.clamp(0.0, 10.0)),
            metal_ref: metal.map_or(0.0, |m| m.reflectivity.clamp(0.0, 10.0) / 10.0),
            metal_str: metal.map_or(0.0, |m| m.strength.clamp(0.0, 10.0) / 10.0),
            metal_zoom: metal.map_or(1.0, |m| 0.5 + m.scale.clamp(1.0, 10.0) / 5.0),
            env_w: env.map_or(0, |e| e.0 as u32),
            env_h: env.map_or(0, |e| e.1 as u32),
            _p0: 0,
        };
        let mut passes: Vec<(&'static str, DCfg, Option<usize>)> =
            vec![("k_clear", DCfg { flags: if backdrop.shade.is_some() { F_PAPER } else { 0 }, ..base }, None)];
        let group_of = |k: usize| layers.get(k).and_then(|l| l.group).filter(|&g| g < backdrop.groups.len());
        for (k, l) in layers.iter().enumerate() {
            let g = group_of(k);
            if g.is_some() && (k == 0 || group_of(k - 1) != g) {
                passes.push(("k_group_clear", base, None));
            }
            let mut flags = 0;
            if l.composite && l.opacity > 0.0 {
                flags |= F_COMPOSITE;
            }
            if l.shown {
                flags |= F_SHADE | F_HEIGHT;
                if l.metallic && metal.is_some() {
                    flags |= F_METAL;
                }
                if backdrop.gloss > 0.0 {
                    flags |= F_GLOSS;
                }
            }
            flags |= if l.clip { F_CLIPPED } else { F_BASE };
            if l.mask.is_some() {
                flags |= F_MASK;
            }
            if g.is_some() {
                flags |= F_GROUP;
            }
            if pigment {
                flags |= F_PIGMENT;
            }
            passes.push(("k_layer", DCfg { mode: mode_index(l.mode), flags, opacity: l.opacity, ..base }, Some(k)));
            if let Some(g) = g
                && group_of(k + 1) != Some(g)
            {
                let grp = backdrop.groups[g];
                if grp.visible && grp.opacity > 0.0 {
                    let flags = if pigment { F_PIGMENT } else { 0 };
                    passes.push(("k_group_out", DCfg { mode: mode_index(grp.mode), flags, opacity: grp.opacity, ..base }, None));
                }
            }
        }
        if backdrop.shadows.is_some_and(|(s, _)| s > 0.0) {
            passes.push(("k_hmax", base, None));
            passes.push(("k_shadow", base, None));
        }
        if metal.is_some() {
            passes.push(("k_metal", base, None));
        }
        let mut out = 0;
        for (on, f) in [
            (backdrop.greyscale, F_GREY),
            (backdrop.stencil.is_some(), F_STENCIL),
            (backdrop.selection.is_some(), F_SELECT),
            (backdrop.deckle.is_some(), F_DECKLE),
        ] {
            if on {
                out |= f;
            }
        }
        passes.push(("k_out", DCfg { flags: out, ..base }, None));

        if passes.len() > self.uniform_slots {
            self.uniform_slots = passes.len().next_power_of_two();
            self.uniforms = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("display cfg"),
                size: STRIDE * self.uniform_slots as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        let mut bytes = vec![0u8; STRIDE as usize * passes.len()];
        for (i, (_, c, _)) in passes.iter().enumerate() {
            bytes[i * STRIDE as usize..][..std::mem::size_of::<DCfg>()].copy_from_slice(bytemuck::bytes_of(c));
        }
        queue.write_buffer(&self.uniforms, 0, &bytes);

        let view = self.texture.create_view(&Default::default());
        let bind = |layer: &wgpu::Buffer, mask: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("display"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uniforms,
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<DCfg>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: self.acc.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: layer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.doc.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 5, resource: mask.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.aux.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: self.group.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: self.hmax.as_entire_binding() },
                ],
            })
        };
        // passes without a layer (or mask) bind the document buffer in its place
        let plain = bind(&self.doc, &self.doc);
        let groups: Vec<wgpu::BindGroup> = layers
            .iter()
            .enumerate()
            .map(|(k, l)| {
                let src = match l.source {
                    Source::Gpu(g) => g.rendered(),
                    Source::Cpu(..) => self.uploads[k].as_ref().expect("uploaded"),
                };
                bind(src, self.masks[k].as_ref().filter(|_| l.mask.is_some()).unwrap_or(&self.doc))
            })
            .collect();
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            let (gx, gy) = (self.w.div_ceil(8) as u32, self.h.div_ceil(8) as u32);
            for (i, (name, _, layer)) in passes.iter().enumerate() {
                cp.set_pipeline(self.pipeline(name));
                cp.set_bind_group(0, layer.map_or(&plain, |k| &groups[k]), &[(i as u64 * STRIDE) as u32]);
                cp.dispatch_workgroups(gx, gy, 1);
            }
        }
        queue.submit([enc.finish()]);
    }

    /// Read the composited painting back as premultiplied RGBA8, waiting for the GPU
    /// (tests).
    pub fn read_rgba(&self) -> Vec<u8> {
        let (device, queue) = (&self.ctx.device, &self.ctx.queue);
        let row = (self.w * 4).next_multiple_of(256);
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display readback"),
            size: (row * self.h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row as u32), rows_per_image: None },
            },
            wgpu::Extent3d { width: self.w as u32, height: self.h as u32, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = buf.slice(..).get_mapped_range().expect("map");
        data.chunks(row).flat_map(|r| &r[..self.w * 4]).copied().collect()
    }
}

/// A CPU layer in the layout `k_render` writes: straight RGBA per cell, then heights,
/// then wetness.
fn cpu_rendering(l: &WetLayer) -> Vec<u8> {
    use rayon::prelude::*;
    let n = l.w * l.h;
    let mut v = vec![0.0f32; LAYER_FLOATS * n];
    let (rgba, rest) = v.split_at_mut(4 * n);
    let (heights, wet) = rest.split_at_mut(n);
    rgba.par_chunks_exact_mut(4).zip(wet_fluid::render::render_layer(l)).for_each(|(d, s)| d.copy_from_slice(&s));
    let (mt, dt) = (l.mid.thickness.cur(), l.dens.thickness.cur());
    heights.par_iter_mut().enumerate().for_each(|(c, h)| *h = ((mt.data[c] + dt.data[c]) / 400.0).min(4.0));
    let water = l.water.cur();
    wet.par_iter_mut().enumerate().for_each(|(c, w)| *w = (water.data[c].to_f32() / 2.0).clamp(0.0, 1.0));
    bytemuck::cast_slice(&v).to_vec()
}
