//! The window: egui drawn with wgpu on a device the simulation shares, so layers simulated
//! on the GPU and the display live on one device.

use std::{num::NonZeroU32, sync::Arc};

use egui::{ViewportId, ViewportOutput};
use winit::{
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes},
};
use wet_fluid_gpu::GpuContext;

pub struct EguiWindow {
    pub window: Arc<Window>,
    pub ctx: egui::Context,
    state: egui_winit::State,
    painter: egui_wgpu::winit::Painter,
    viewport: egui::ViewportInfo,
}

impl EguiWindow {
    pub fn new(el: &ActiveEventLoop, attrs: WindowAttributes) -> Self {
        let window = Arc::new(el.create_window(attrs).expect("window"));
        let ctx = egui::Context::default();
        let mut setup = egui_wgpu::WgpuSetupCreateNew::from_display_handle(el.owned_display_handle());
        // compute-capable backends only (not GL), with the limits the simulation asks for
        setup.instance_descriptor.backends = wgpu::Backends::PRIMARY;
        setup.device_descriptor = Arc::new(GpuContext::device_descriptor);
        let config = egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(setup),
            surface: egui_wgpu::SurfaceConfig::LOW_LATENCY,
            ..Default::default()
        };
        let mut painter = pollster::block_on(egui_wgpu::winit::Painter::new(ctx.clone(), config, false, Default::default()));
        pollster::block_on(painter.set_window(ViewportId::ROOT, Some(window.clone()))).expect("GPU surface for the window");
        let state = egui_winit::State::new(
            ctx.clone(),
            ViewportId::ROOT,
            el,
            Some(window.scale_factor() as f32),
            el.system_theme(),
            painter.max_texture_side(),
        );
        Self { window, ctx, state, painter, viewport: Default::default() }
    }

    /// The device and renderer egui draws with, for the simulation and display to share.
    pub fn render_state(&self) -> Option<egui_wgpu::RenderState> {
        self.painter.render_state()
    }

    pub fn on_window_event(&mut self, event: &WindowEvent) -> egui_winit::EventResponse {
        if let WindowEvent::Resized(size) = event
            && let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
        {
            self.painter.on_window_resized(ViewportId::ROOT, w, h);
        }
        self.state.on_window_event(&self.window, event)
    }

    /// Run the UI for one frame and draw it.
    pub fn frame(&mut self, run_ui: impl FnMut(&mut egui::Ui)) {
        let input = self.state.take_egui_input(&self.window);
        let out = self.ctx.run_ui(input, run_ui);
        for (_, ViewportOutput { commands, .. }) in out.viewport_output {
            egui_winit::process_viewport_commands(&self.ctx, &mut self.viewport, commands, &self.window, &mut Default::default());
        }
        self.state.handle_platform_output(&self.window, out.platform_output);
        let primitives = self.ctx.tessellate(out.shapes, out.pixels_per_point);
        let mut textures = out.textures_delta;
        self.painter.paint_and_update_textures(
            ViewportId::ROOT,
            out.pixels_per_point,
            [0.15, 0.15, 0.15, 1.0],
            &primitives,
            &mut textures,
            Vec::new(),
            &self.window,
        );
    }

    pub fn destroy(&mut self) {
        self.painter.destroy();
    }
}
