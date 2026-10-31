//! GPU simulation for the app: wet layers step on the wgpu backend (`wet-fluid-gpu`)
//! while the CPU copy of each is kept current enough to display.
//!
//! The CPU [`WetLayer`](wet_fluid::WetLayer) stays the document's state. A layer moves to
//! the GPU when it is wet and nothing is editing it; each frame only the planes display
//! compositing reads come back. Before the CPU edits anything (a stroke, a command, a
//! save) [`GpuSim::release`] brings every layer back in full. Layers the GPU backend does
//! not cover step on the CPU as before: the layer being painted and pigment-mixing layers
//! (the GPU mixes normally only).

use rayon::prelude::*;
use wet_doc::Document;
use wet_fluid_gpu::display::{Backdrop, Compositor, DisplayLayer, GroupBlend, Metallic, Source};
use wet_fluid_gpu::{GpuContext, GpuError, GpuLayer};

type Renderer = std::sync::Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>;

/// Display compositing on the GPU, drawn by the window from the compositor's texture.
struct Display {
    comp: Compositor,
    /// The window's renderer, the texture's id there and the filter it was registered with.
    tex: Option<(Renderer, egui::TextureId, wgpu::FilterMode)>,
    /// [`crate::app::App::epoch`] when the CPU layers were last uploaded.
    epoch: u64,
    /// The paper shading last uploaded, held so its address (the compositor's key for it)
    /// is not reused by a new one.
    paper: Option<std::sync::Arc<Vec<f32>>>,
    /// Stencil film coverage, recomputed when the document changes.
    stencil: Option<Vec<f32>>,
    /// The pigment mixer's tables.
    km: Vec<f32>,
}

/// How the display is viewed and what changed, for [`GpuSim::composite`].
pub struct View {
    pub greyscale: bool,
    /// Pigment mixing is on (Edit > Pigment Mixing).
    pub pigment: bool,
    /// [`crate::app::App::epoch`]: changes whenever a layer may have been edited.
    pub epoch: u64,
    /// The layer a stroke is changing.
    pub painting: Option<usize>,
}

impl Drop for Display {
    fn drop(&mut self) {
        if let Some((r, id, _)) = &self.tex {
            r.write().free_texture(id);
        }
    }
}

pub struct GpuSim {
    /// GPU state per layer index, created on first use and reused while the size and the
    /// granulation texture match.
    layers: Vec<Option<GpuLayer>>,
    /// The GPU holds layer `i`'s latest state (the CPU copy is current only in the
    /// display planes).
    owns: Vec<bool>,
    /// Steps were run on the GPU since the display planes were last read back.
    fresh: Vec<bool>,
    /// The granulation texture each GPU layer was made with (address and length; the GPU
    /// copy is only written at upload).
    gran: Vec<(usize, usize)>,
    /// The device every layer runs on.
    ctx: GpuContext,
    /// Layers stepped on the CPU since the display last uploaded them.
    stepped: Vec<bool>,
    display: Option<Display>,
    /// The last frame was composited on the GPU: the CPU copies of GPU-held layers are only
    /// read for thumbnails and the drying check, so they come back less often.
    gpu_shown: bool,
    frame: u32,
    /// A stroke's undo step on its way from the GPU: the layer and the label.
    pending: Option<(usize, String)>,
    pub adapter: String,
}

impl GpuSim {
    /// Set up the GPU backend on `ctx` (the window's device), or on a device of its own,
    /// failing when there is no usable adapter.
    pub fn new(doc: &Document, ctx: Option<GpuContext>) -> Result<Self, GpuError> {
        let ctx = match ctx {
            Some(c) => c,
            None => GpuContext::new()?,
        };
        let n = doc.layers.len();
        let adapter = ctx.adapter_info.name.clone();
        Ok(Self {
            layers: (0..n).map(|_| None).collect(),
            owns: vec![false; n],
            fresh: vec![false; n],
            gran: vec![(0, 0); n],
            ctx,
            stepped: vec![false; n],
            display: None,
            gpu_shown: false,
            frame: 0,
            pending: None,
            adapter,
        })
    }

    fn fit(&mut self, n: usize) {
        self.layers.resize_with(n, || None);
        self.owns.resize(n, false);
        self.fresh.resize(n, false);
        self.gran.resize(n, (0, 0));
        self.stepped.resize(n, false);
    }

    /// Is any layer's state on the GPU only?
    #[cfg(test)]
    pub fn holds_any(&self) -> bool {
        self.owns.iter().any(|&o| o)
    }

    /// Is layer `i`'s state on the GPU only?
    pub fn holds(&self, i: usize) -> bool {
        self.owns.get(i).copied().unwrap_or(false)
    }

    /// Bring every GPU-held layer back to the CPU in full; call before the CPU reads
    /// anything but the display or changes any layer.
    pub fn release(&mut self, doc: &mut Document) {
        self.flush_snapshot(doc, true);
        for i in 0..self.owns.len() {
            self.release_layer(doc, i);
        }
    }

    /// Bring layer `i` back to the CPU in full (a stroke on it is starting).
    pub fn release_layer(&mut self, doc: &mut Document, i: usize) {
        self.flush_snapshot(doc, true);
        if self.owns.get(i).copied().unwrap_or(false) {
            if let (Some(Some(g)), Some(l)) = (self.layers.get_mut(i), doc.layers.get_mut(i)) {
                g.download(&mut l.wet);
            }
            self.owns[i] = false;
            self.fresh[i] = false;
        }
    }

    /// Run `n` steps of every layer, on the GPU where it can (`skip`: a layer the CPU is
    /// painting on).
    pub fn step(&mut self, doc: &mut Document, n: usize, skip: Option<usize>) -> Result<(), GpuError> {
        self.fit(doc.layers.len());
        for i in 0..doc.layers.len() {
            let wet = &doc.layers[i].wet;
            let gpu_ok = Some(i) != skip && wet.pigment.is_none();
            if !gpu_ok {
                if self.owns[i] {
                    if let Some(g) = &mut self.layers[i] {
                        g.download(&mut doc.layers[i].wet);
                    }
                    self.owns[i] = false;
                }
                for _ in 0..n {
                    doc.layers[i].wet.step();
                }
                self.stepped[i] |= n > 0;
                continue;
            }
            if !self.owns[i] {
                // dry layers have nothing to do; the CPU skips them cheaply
                if wet.active_rect().is_none() {
                    for _ in 0..n {
                        doc.layers[i].wet.step();
                    }
                    continue;
                }
                let reuse = self.layers[i].as_ref().is_some_and(|g| g.w == wet.w && g.h == wet.h) && self.gran[i] == gran_key(wet);
                if reuse {
                    let g = self.layers[i].as_mut().expect("checked");
                    g.params = wet.params.clone();
                    g.write_state(wet);
                } else {
                    self.layers[i] = Some(GpuLayer::upload_on(&self.ctx, wet));
                    self.gran[i] = gran_key(wet);
                }
                self.owns[i] = true;
            }
            if let Some(g) = &mut self.layers[i] {
                // tilt and drop settings can change while the layer runs on the GPU
                g.params = doc.layers[i].wet.params.clone();
                g.step(n);
                self.fresh[i] = true;
            }
        }
        Ok(())
    }

    /// Pick up the display planes of GPU-held layers that have arrived, and ask for the
    /// next ones where steps ran since. Never waits for the GPU: the display trails the
    /// simulation by a frame. A layer whose water and suspended pigment have all gone
    /// returns to the CPU.
    pub fn show(&mut self, doc: &mut Document) {
        self.fit(doc.layers.len());
        self.flush_snapshot(doc, false);
        self.frame = self.frame.wrapping_add(1);
        let read_back = !self.gpu_shown || self.frame.is_multiple_of(15);
        for i in 0..doc.layers.len() {
            if !self.owns[i] {
                continue;
            }
            let Some(g) = &mut self.layers[i] else { continue };
            let l = &mut doc.layers[i].wet;
            if g.take_display(l, false) {
                let wet = l.water.cur().data.par_iter().any(|v| v.to_f32() > 0.0) || l.dens.amount.cur().data.par_iter().any(|&v| v > 0.0);
                if !wet && !self.fresh[i] {
                    g.download(l);
                    self.owns[i] = false;
                    continue;
                }
            }
            if self.fresh[i] && read_back && g.request_display() {
                self.fresh[i] = false;
            }
        }
    }

    /// Start a stroke on GPU-held layer `i` without bringing it back: the undo step
    /// (`label`) is copied on the GPU and added to the document once it arrives.
    pub fn begin_stroke(&mut self, doc: &mut Document, i: usize, label: &str) {
        self.flush_snapshot(doc, true);
        if let Some(Some(g)) = self.layers.get_mut(i) {
            g.request_snapshot();
            self.pending = Some((i, label.to_owned()));
        }
    }

    /// Add a stroke's undo step to the document once its copy has arrived (`wait`: block
    /// until it has). Called every frame, and before anything else edits or reads layers.
    pub fn flush_snapshot(&mut self, doc: &mut Document, wait: bool) {
        let Some((i, label)) = &self.pending else { return };
        let Some(Some(g)) = self.layers.get_mut(*i) else {
            self.pending = None;
            return;
        };
        if let Some(s) = g.take_snapshot(wait) {
            doc.checkpoint_planes(*i, label, s.h16, s.f32s, s.open);
            self.pending = None;
        }
    }

    /// Bring the stroke planes of `rect` of GPU-held layer `i` into the document's copy,
    /// for the brush to work on.
    pub fn read_region(&mut self, doc: &mut Document, i: usize, rect: wet_fluid_gpu::Rect) {
        if let (Some(Some(g)), Some(l)) = (self.layers.get_mut(i), doc.layers.get_mut(i)) {
            g.read_region(&mut l.wet, rect);
        }
    }

    /// Write what the brush did to `rect` of layer `i` back to the GPU.
    pub fn write_region(&mut self, doc: &Document, i: usize, rect: wet_fluid_gpu::Rect) {
        if let (Some(Some(g)), Some(l)) = (self.layers.get_mut(i), doc.layers.get(i)) {
            g.write_region(&l.wet, rect);
            self.fresh[i] = true;
        }
    }

    /// Composite the display on the GPU into the compositor's texture.
    pub fn composite(&mut self, doc: &Document, view: &View) {
        self.fit(doc.layers.len());
        if self.display.as_ref().is_none_or(|d| (d.comp.w, d.comp.h) != (doc.w, doc.h)) {
            self.display = Some(Display {
                comp: Compositor::new(&self.ctx, doc.w, doc.h),
                tex: None,
                epoch: view.epoch.wrapping_sub(1),
                paper: None,
                stencil: None,
                km: wet_pigment::KmMixer::new().tables(),
            });
        }
        let d = self.display.as_mut().expect("made");
        let changed = d.epoch != view.epoch;
        d.epoch = view.epoch;
        if changed {
            // the stencil film the view shows (ui.rs decorate), when stencils changed
            let stencils: Vec<&wet_doc::Stencil> = doc.stencils.iter().filter(|s| s.visible).collect();
            d.stencil = (!stencils.is_empty()).then(|| {
                let w = doc.w;
                (0..doc.w * doc.h)
                    .into_par_iter()
                    .map(|i| stencils.iter().map(|s| s.coverage((i % w) as f32, (i / w) as f32)).fold(0.0f32, f32::max))
                    .collect()
            });
        }
        let group_ok = |g: Option<usize>| g.filter(|&g| g < doc.groups.len());
        let layers: Vec<DisplayLayer> = doc
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let source = match (&self.layers[i], self.owns[i]) {
                    (Some(g), true) => Source::Gpu(g),
                    _ => Source::Cpu(&l.wet, changed || self.stepped[i] || view.painting == Some(i)),
                };
                DisplayLayer {
                    source,
                    mode: if l.pigments { wet_compose::MixMode::Pigment } else { l.mode },
                    opacity: l.opacity,
                    composite: l.visible && !l.masking_fluid,
                    shown: doc.layer_shown(l),
                    clip: l.clip,
                    mask: l.mask.as_deref(),
                    metallic: l.metallic,
                    group: group_ok(l.group),
                }
            })
            .collect();
        self.stepped.fill(false);
        let groups: Vec<GroupBlend> = doc.groups.iter().map(|g| GroupBlend { mode: g.mode, opacity: g.opacity, visible: g.visible }).collect();
        let shade = doc.paper_shading();
        let m = &doc.metallic;
        let metallic = doc.layers.iter().any(|l| l.metallic && doc.layer_shown(l)).then(|| Metallic {
            reflectivity: m.reflectivity,
            scale: m.scale,
            strength: m.strength,
            map: m.map.as_ref().map(|(w, h, v)| (*w, *h, v.as_slice())),
        });
        // a mixer is used when asked for, and always for pigment layers (as Document::render)
        let pigment = view.pigment || doc.layers.iter().any(|l| l.pigments);
        let backdrop = Backdrop {
            color: doc.display_background(),
            shade: shade.as_deref().map(|v| v.as_slice()),
            impasto: doc.impasto,
            gloss: doc.gloss,
            shadows: (doc.soft_shadows > 0.0).then_some((doc.soft_shadows, doc.shadow_altitude)),
            metallic,
            pigment: pigment.then_some(d.km.as_slice()),
            groups: &groups,
            deckle: doc.deckle_map(),
            greyscale: view.greyscale,
            stencil: d.stencil.as_deref(),
            selection: doc.selection.as_deref(),
            changed,
        };
        d.comp.compose(&layers, &backdrop);
        d.paper = shade;
        self.gpu_shown = true;
    }

    /// [`Self::composite`], returning the texture registered with the window's renderer.
    pub fn compose(
        &mut self,
        doc: &Document,
        render: &egui_wgpu::RenderState,
        filter: wgpu::FilterMode,
        view: &View,
    ) -> egui::TextureId {
        self.composite(doc, view);
        let d = self.display.as_mut().expect("composited");
        let view = d.comp.srgb_view();
        match &mut d.tex {
            Some((_, id, f)) if *f == filter => *id,
            Some((r, id, f)) => {
                r.write().update_egui_texture_from_wgpu_texture(&render.device, &view, filter, *id);
                *f = filter;
                *id
            }
            None => {
                let id = render.renderer.write().register_native_texture(&render.device, &view, filter);
                d.tex = Some((render.renderer.clone(), id, filter));
                id
            }
        }
    }

    /// The display was composited on the CPU this frame.
    pub fn shown_on_cpu(&mut self) {
        self.gpu_shown = false;
    }

    /// A full copy of GPU-held layer `i`'s state, leaving it on the GPU (tests).
    #[cfg(test)]
    pub fn download_copy(&mut self, i: usize, into: &mut wet_fluid::WetLayer) {
        if let Some(Some(g)) = self.layers.get_mut(i) {
            g.download(into);
        }
    }

    /// Wait for the GPU to finish what was submitted (benchmarks).
    #[cfg(test)]
    pub fn wait(&self) {
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// The compositor, for tests that read its result back.
    #[cfg(test)]
    pub fn compositor(&self) -> Option<&Compositor> {
        self.display.as_ref().map(|d| &d.comp)
    }

    /// [`Self::show`], waiting for the display planes of every GPU-held layer to be
    /// current (for tests and one-off reads).
    #[cfg(test)]
    pub fn show_now(&mut self, doc: &mut Document) {
        // read back whatever the last frame showed
        self.gpu_shown = false;
        self.show(doc);
        for i in 0..doc.layers.len() {
            if let (true, Some(g)) = (self.owns[i], &mut self.layers[i]) {
                g.take_display(&mut doc.layers[i].wet, true);
            }
        }
    }
}

fn gran_key(l: &wet_fluid::WetLayer) -> (usize, usize) {
    l.maps.gran.as_ref().map_or((0, 0), |g| (g.data.as_ptr() as usize, g.data.len()))
}
