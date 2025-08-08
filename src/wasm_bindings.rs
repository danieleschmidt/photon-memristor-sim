//! WebAssembly bindings for photonic neural networks
//!
//! This module provides WASM-compatible bindings for running photonic
//! neural network simulations in web browsers and JavaScript environments.

use crate::core::{Result, PhotonicError, OpticalField, WaveguideGeometry};
use crate::devices::{MachZehnderInterferometer, MicroringResonator};
use crate::simulation::PhotonicArray;
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

/// WASM-compatible configuration for photonic simulations
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmPhotonicConfig {
    /// Simulation wavelength in nanometers
    pub wavelength_nm: f64,
    /// Number of simulation steps
    pub num_steps: usize,
    /// Simulation precision level
    pub precision: String,
}

/// WASM-compatible photonic simulation result
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmSimulationResult {
    /// Final optical power values
    pub power_values: Vec<f64>,
    /// Simulation success status
    pub success: bool,
    /// Error message if simulation failed
    pub error_message: Option<String>,
}

/// WASM-compatible photonic array wrapper
#[wasm_bindgen]
pub struct WasmPhotonicArray {
    /// Internal photonic array
    array: PhotonicArray,
}

#[wasm_bindgen]
impl WasmPhotonicConfig {
    /// Create new WASM photonic configuration
    #[wasm_bindgen(constructor)]
    pub fn new(wavelength_nm: f64, num_steps: usize, precision: String) -> WasmPhotonicConfig {
        WasmPhotonicConfig {
            wavelength_nm,
            num_steps,
            precision,
        }
    }

    /// Get wavelength in nanometers
    #[wasm_bindgen(getter)]
    pub fn wavelength_nm(&self) -> f64 {
        self.wavelength_nm
    }

    /// Set wavelength in nanometers
    #[wasm_bindgen(setter)]
    pub fn set_wavelength_nm(&mut self, value: f64) {
        self.wavelength_nm = value;
    }

    /// Get number of simulation steps
    #[wasm_bindgen(getter)]
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }

    /// Set number of simulation steps
    #[wasm_bindgen(setter)]
    pub fn set_num_steps(&mut self, value: usize) {
        self.num_steps = value;
    }
}

#[wasm_bindgen]
impl WasmSimulationResult {
    /// Get power values as JavaScript array
    #[wasm_bindgen(getter)]
    pub fn power_values(&self) -> Vec<f64> {
        self.power_values.clone()
    }

    /// Get simulation success status
    #[wasm_bindgen(getter)]
    pub fn success(&self) -> bool {
        self.success
    }

    /// Get error message
    #[wasm_bindgen(getter)]
    pub fn error_message(&self) -> Option<String> {
        self.error_message.clone()
    }
}

#[wasm_bindgen]
impl WasmPhotonicArray {
    /// Create new WASM photonic array
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize) -> Result<WasmPhotonicArray, JsValue> {
        let array = PhotonicArray::new(rows, cols)
            .map_err(|e| JsValue::from_str(&format!("Failed to create photonic array: {}", e)))?;
        
        Ok(WasmPhotonicArray { array })
    }

    /// Run simulation with given configuration
    pub fn simulate(&mut self, config: &WasmPhotonicConfig) -> WasmSimulationResult {
        // Stub implementation for WASM simulation
        match self.run_simulation_internal(config) {
            Ok(power_values) => WasmSimulationResult {
                power_values,
                success: true,
                error_message: None,
            },
            Err(e) => WasmSimulationResult {
                power_values: Vec::new(),
                success: false,
                error_message: Some(e.to_string()),
            },
        }
    }

    /// Set input optical field
    pub fn set_input(&mut self, power_values: Vec<f64>) -> Result<(), JsValue> {
        // Stub implementation for WASM input setting
        // Would create optical fields from power values in actual implementation
        Ok(())
    }

    /// Get array dimensions
    pub fn get_dimensions(&self) -> Vec<usize> {
        vec![self.array.rows, self.array.cols]
    }
}

impl WasmPhotonicArray {
    /// Internal simulation runner
    fn run_simulation_internal(&mut self, config: &WasmPhotonicConfig) -> Result<Vec<f64>> {
        // Stub implementation - would run actual simulation
        let num_outputs = self.array.rows * self.array.cols;
        let power_values = vec![1.0; num_outputs]; // Placeholder values
        Ok(power_values)
    }
}

/// WASM-compatible utility functions
#[wasm_bindgen]
pub struct WasmUtils;

#[wasm_bindgen]
impl WasmUtils {
    /// Calculate optical power from complex field
    #[wasm_bindgen]
    pub fn calculate_power(real: f64, imag: f64) -> f64 {
        real * real + imag * imag
    }

    /// Calculate phase from complex field
    #[wasm_bindgen]
    pub fn calculate_phase(real: f64, imag: f64) -> f64 {
        imag.atan2(real)
    }

    /// Convert wavelength to frequency
    #[wasm_bindgen]
    pub fn wavelength_to_frequency(wavelength_nm: f64) -> f64 {
        // Speed of light (m/s) / (wavelength in meters)
        299_792_458.0 / (wavelength_nm * 1e-9)
    }

    /// Convert frequency to wavelength
    #[wasm_bindgen]
    pub fn frequency_to_wavelength(frequency_hz: f64) -> f64 {
        // Speed of light (m/s) / frequency * 1e9 (convert to nm)
        299_792_458.0 / frequency_hz * 1e9
    }
}

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    // Set up panic hook for better error messages in WASM
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    // Initialize logging for WASM
    // Note: wee_alloc setup would be done at compile time with #[global_allocator]
}

/// WASM-compatible error type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmError {
    pub message: String,
    pub code: String,
}

impl From<PhotonicError> for WasmError {
    fn from(error: PhotonicError) -> Self {
        WasmError {
            message: error.to_string(),
            code: "PHOTONIC_ERROR".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_config_creation() {
        let config = WasmPhotonicConfig::new(1550.0, 1000, "high".to_string());
        assert_eq!(config.wavelength_nm, 1550.0);
        assert_eq!(config.num_steps, 1000);
        assert_eq!(config.precision, "high");
    }

    #[test]
    fn test_wasm_utils() {
        let power = WasmUtils::calculate_power(3.0, 4.0);
        assert_eq!(power, 25.0);

        let phase = WasmUtils::calculate_phase(1.0, 1.0);
        assert_eq!(phase, std::f64::consts::PI / 4.0);
    }

    #[test]
    fn test_wavelength_frequency_conversion() {
        let wavelength_nm = 1550.0;
        let frequency = WasmUtils::wavelength_to_frequency(wavelength_nm);
        let back_to_wavelength = WasmUtils::frequency_to_wavelength(frequency);
        
        // Should be approximately equal (allowing for floating point precision)
        assert!((back_to_wavelength - wavelength_nm).abs() < 1e-6);
    }
}