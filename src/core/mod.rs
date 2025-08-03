//! Core physics and mathematical foundations for photonic simulation

pub mod error;
pub mod types;
pub mod waveguide;
pub mod propagation;
pub mod coupling;

pub use error::{PhotonicError, Result};
pub use types::{OpticalField, WaveguideGeometry, DeviceGeometry, Complex64};
pub use waveguide::{WaveguideMode, EffectiveIndexCalculator};
pub use propagation::{PropagationSolver, BeamPropagationMethod};
pub use coupling::{CouplingCalculator, OverlapIntegral};