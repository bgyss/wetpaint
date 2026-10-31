//! CPU reference implementation of a grid-based wet-media (watercolour) simulation.
//!
//! The kernels (one module each under [`kernels`]) favour clarity over speed: they are
//! the correctness reference for the GPU backend (`wet-fluid-gpu`). Tuned constants live
//! in [`params::Placeholders`], the opacity LUT in [`lut`], and display compositing in
//! [`render`].

pub mod drip;
pub mod grid;
pub mod kernels;
pub mod layer;
pub mod lut;
pub mod mix;
pub mod params;
pub mod render;

pub use layer::WetLayer;
pub use params::FluidParams;
