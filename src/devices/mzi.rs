//! Mach-Zehnder Interferometer device model

use crate::core::{Result, Complex64, OpticalField};
use crate::devices::traits::{PhotonicDevice, TunableDevice};
use nalgebra::DVector;


/// MZI configuration
#[derive(Debug, Clone)]
pub struct MZIConfiguration {
    pub arm_length_difference: f64,
    pub split_ratio: f64,
    pub phase_shift: f64,
}

/// Mach-Zehnder Interferometer
#[derive(Debug, Clone)]
pub struct MachZehnderInterferometer {
    config: MZIConfiguration,
    effective_index: f64,
}

impl MachZehnderInterferometer {
    pub fn new() -> Self {
        Self {
            config: MZIConfiguration {
                arm_length_difference: 0.0,
                split_ratio: 0.5,
                phase_shift: 0.0,
            },
            effective_index: 2.4,
        }
    }
}

impl PhotonicDevice for MachZehnderInterferometer {
    fn simulate(&self, input: &OpticalField) -> Result<OpticalField> {
        // Simple MZI transfer function
        let k = 2.0 * std::f64::consts::PI / input.wavelength;
        let phase_diff = k * self.effective_index * self.config.arm_length_difference + self.config.phase_shift;
        
        let transmission = (phase_diff / 2.0).cos().powi(2);
        
        let mut output = input.clone();
        output.amplitude *= Complex64::new(transmission.sqrt(), 0.0);
        output.power = output.calculate_power();
        Ok(output)
    }
    
    fn parameters(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.config.arm_length_difference,
            self.config.phase_shift,
            self.effective_index,
        ])
    }
    
    fn update_parameters(&mut self, params: &DVector<f64>) -> Result<()> {
        if params.len() >= 1 {
            self.config.arm_length_difference = params[0];
        }
        if params.len() >= 2 {
            self.config.phase_shift = params[1];
        }
        if params.len() >= 3 {
            self.effective_index = params[2];
        }
        Ok(())
    }
    
    fn parameter_names(&self) -> Vec<String> {
        vec![
            "arm_length_difference".to_string(),
            "phase_shift".to_string(),
            "effective_index".to_string(),
        ]
    }
    
    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![
            (-100e-6, 100e-6),  // arm length difference
            (-2.0 * std::f64::consts::PI, 2.0 * std::f64::consts::PI), // phase shift
            (1.4, 3.5),         // effective index
        ]
    }
    
    fn parameter_gradients(&self, _input: &OpticalField, _grad_output: &OpticalField) -> Result<DVector<f64>> {
        Ok(DVector::zeros(3))
    }
    
    fn device_type(&self) -> &'static str {
        "mach_zehnder_interferometer"
    }
    
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl TunableDevice for MachZehnderInterferometer {
    fn get_tuning(&self) -> f64 {
        self.config.phase_shift
    }
    
    fn set_tuning(&mut self, tuning: f64) -> Result<()> {
        self.config.phase_shift = tuning;
        Ok(())
    }
    
    fn tuning_range(&self) -> (f64, f64) {
        (-2.0 * std::f64::consts::PI, 2.0 * std::f64::consts::PI)
    }
    
    fn tuning_sensitivity(&self, _wavelength: f64) -> f64 {
        1.0 // Phase shift directly controls transmission
    }
}