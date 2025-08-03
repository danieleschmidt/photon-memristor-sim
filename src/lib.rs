//! # Photon-Memristor-Sim
//!
//! High-performance Rust/WASM simulator for neuromorphic photonic-memristor arrays
//! with JAX integration for differentiable device-algorithm co-design.

pub mod core;
pub mod devices;
pub mod simulation;
pub mod optimization;
pub mod utils;

// Python bindings
#[cfg(feature = "python")]
pub mod python_bindings;

// WASM bindings
#[cfg(feature = "wasm")]
pub mod wasm_bindings;

// Re-export main types for convenience
pub use core::{OpticalField, WaveguideGeometry, DeviceGeometry};
pub use devices::{PCMDevice, OxideMemristor, MicroringResonator};
pub use simulation::{PhotonicArray, SimulationResult};

/// Main result type used throughout the library
pub type Result<T> = std::result::Result<T, crate::core::PhotonicError>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_library_loads() {
        assert!(!VERSION.is_empty());
    }
}