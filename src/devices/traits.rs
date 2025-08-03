//! Common traits for photonic and memristive devices

use crate::core::{Result, OpticalField, Complex64};
use nalgebra::DVector;


/// Trait for all photonic devices
pub trait PhotonicDevice: Send + Sync {
    /// Simulate device response to optical input
    fn simulate(&self, input: &OpticalField) -> Result<OpticalField>;
    
    /// Get device parameters as vector
    fn parameters(&self) -> DVector<f64>;
    
    /// Update device parameters
    fn update_parameters(&mut self, params: &DVector<f64>) -> Result<()>;
    
    /// Get parameter names for optimization
    fn parameter_names(&self) -> Vec<String>;
    
    /// Get parameter bounds for optimization
    fn parameter_bounds(&self) -> Vec<(f64, f64)>;
    
    /// Calculate gradients with respect to parameters
    fn parameter_gradients(&self, input: &OpticalField, grad_output: &OpticalField) -> Result<DVector<f64>>;
    
    /// Get device name/type
    fn device_type(&self) -> &'static str;
    
    /// Validate current device state
    fn validate(&self) -> Result<()>;
}

/// Trait for memristive devices
pub trait MemristiveDevice: PhotonicDevice {
    /// Get current memristive state
    fn get_state(&self) -> f64;
    
    /// Set memristive state
    fn set_state(&mut self, state: f64) -> Result<()>;
    
    /// Get state bounds
    fn state_bounds(&self) -> (f64, f64);
    
    /// Update state based on optical/electrical stimulus
    fn update_state(&mut self, optical_power: f64, electrical_voltage: f64, dt: f64) -> Result<()>;
    
    /// Get switching characteristics
    fn switching_characteristics(&self) -> SwitchingCharacteristics;
}

/// Trait for tunable devices
pub trait TunableDevice: PhotonicDevice {
    /// Get tuning parameter
    fn get_tuning(&self) -> f64;
    
    /// Set tuning parameter
    fn set_tuning(&mut self, tuning: f64) -> Result<()>;
    
    /// Get tuning range
    fn tuning_range(&self) -> (f64, f64);
    
    /// Calculate tuning sensitivity (dλ/dtuning)
    fn tuning_sensitivity(&self, wavelength: f64) -> f64;
}

/// Switching characteristics for memristive devices
#[derive(Debug, Clone)]
pub struct SwitchingCharacteristics {
    /// Set voltage threshold
    pub set_voltage: f64,
    /// Reset voltage threshold
    pub reset_voltage: f64,
    /// Set time constant
    pub set_time: f64,
    /// Reset time constant
    pub reset_time: f64,
    /// State retention time
    pub retention_time: f64,
    /// Endurance (number of cycles)
    pub endurance: u64,
}

impl Default for SwitchingCharacteristics {
    fn default() -> Self {
        Self {
            set_voltage: 1.0,      // 1V
            reset_voltage: -1.0,   // -1V
            set_time: 1e-9,        // 1ns
            reset_time: 1e-9,      // 1ns
            retention_time: 3600.0, // 1 hour
            endurance: 1_000_000,   // 1M cycles
        }
    }
}

/// Device response metrics
#[derive(Debug, Clone)]
pub struct DeviceResponse {
    /// Transmission coefficient
    pub transmission: Complex64,
    /// Reflection coefficient
    pub reflection: Complex64,
    /// Phase shift
    pub phase_shift: f64,
    /// Insertion loss (dB)
    pub insertion_loss: f64,
    /// Extinction ratio (dB)
    pub extinction_ratio: f64,
}

impl DeviceResponse {
    /// Calculate power transmission
    pub fn power_transmission(&self) -> f64 {
        self.transmission.norm_sqr()
    }
    
    /// Calculate power reflection
    pub fn power_reflection(&self) -> f64 {
        self.reflection.norm_sqr()
    }
    
    /// Calculate total loss
    pub fn total_loss(&self) -> f64 {
        1.0 - self.power_transmission() - self.power_reflection()
    }
}

/// Thermal effects trait
pub trait ThermalEffects {
    /// Calculate temperature distribution
    fn calculate_temperature(&self, optical_power: f64, electrical_power: f64) -> Result<f64>;
    
    /// Get thermo-optic coefficient
    fn thermo_optic_coefficient(&self) -> f64;
    
    /// Update refractive index based on temperature
    fn temperature_dependent_index(&self, base_index: Complex64, temperature: f64) -> Complex64 {
        let temp_change = temperature - 300.0; // Reference: 300K
        let delta_n = self.thermo_optic_coefficient() * temp_change;
        Complex64::new(base_index.re + delta_n, base_index.im)
    }
}

/// Nonlinear effects trait
pub trait NonlinearEffects {
    /// Calculate Kerr nonlinearity
    fn kerr_nonlinearity(&self, intensity: f64) -> f64;
    
    /// Calculate two-photon absorption
    fn two_photon_absorption(&self, intensity: f64) -> f64;
    
    /// Calculate free-carrier effects
    fn free_carrier_effects(&self, carrier_density: f64) -> (f64, f64); // (Δn, Δα)
}

/// Noise effects trait
pub trait NoiseEffects {
    /// Add thermal noise
    fn add_thermal_noise(&self, field: &mut OpticalField, temperature: f64) -> Result<()>;
    
    /// Add shot noise
    fn add_shot_noise(&self, field: &mut OpticalField) -> Result<()>;
    
    /// Add phase noise
    fn add_phase_noise(&self, field: &mut OpticalField, noise_power: f64) -> Result<()>;
}

/// Manufacturing variations trait
pub trait ManufacturingVariations {
    /// Apply dimension variations
    fn apply_dimension_variations(&mut self, variations: &DVector<f64>) -> Result<()>;
    
    /// Apply material property variations
    fn apply_material_variations(&mut self, variations: &DVector<f64>) -> Result<()>;
    
    /// Get sensitivity to variations
    fn variation_sensitivity(&self) -> DVector<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_switching_characteristics_default() {
        let characteristics = SwitchingCharacteristics::default();
        assert_eq!(characteristics.set_voltage, 1.0);
        assert_eq!(characteristics.reset_voltage, -1.0);
        assert!(characteristics.endurance > 0);
    }
    
    #[test]
    fn test_device_response() {
        let response = DeviceResponse {
            transmission: Complex64::new(0.8, 0.0),
            reflection: Complex64::new(0.1, 0.0),
            phase_shift: 0.5,
            insertion_loss: 1.0,
            extinction_ratio: 20.0,
        };
        
        assert!((response.power_transmission() - 0.64).abs() < 1e-10);
        assert!((response.power_reflection() - 0.01).abs() < 1e-10);
        assert!((response.total_loss() - 0.35).abs() < 1e-10);
    }
}