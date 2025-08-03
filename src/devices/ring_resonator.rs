//! Microring resonator device model

use crate::core::{Result, Complex64, OpticalField};
use crate::devices::traits::{PhotonicDevice, TunableDevice};
use nalgebra::DVector;

use std::f64::consts::PI;

/// Ring geometry parameters
#[derive(Debug, Clone)]
pub struct RingGeometry {
    pub radius: f64,
    pub coupling_gap: f64,
    pub waveguide_width: f64,
}

/// Microring resonator implementation
#[derive(Debug, Clone)]
pub struct MicroringResonator {
    geometry: RingGeometry,
    effective_index: f64,
    quality_factor: f64,
    tuning_parameter: f64, // Thermal or electro-optic tuning
}

impl MicroringResonator {
    pub fn new(radius: f64) -> Self {
        Self {
            geometry: RingGeometry {
                radius,
                coupling_gap: 200e-9,
                waveguide_width: 450e-9,
            },
            effective_index: 2.4,
            quality_factor: 10000.0,
            tuning_parameter: 0.0,
        }
    }
    
    fn transmission_at_wavelength(&self, wavelength: f64) -> Complex64 {
        // Simple Lorentzian response
        let circumference = 2.0 * PI * self.geometry.radius;
        let round_trip_phase = 2.0 * PI * self.effective_index * circumference / wavelength;
        
        // Add tuning
        let tuned_phase = round_trip_phase + self.tuning_parameter;
        
        // Resonance condition
        let detuning = tuned_phase - 2.0 * PI * (tuned_phase / (2.0 * PI)).round();
        
        // Transmission with finite Q
        let finesse = PI * self.quality_factor.sqrt();
        let transmission = 1.0 / (1.0 + finesse * finesse * (detuning / PI).sin().powi(2));
        
        Complex64::new(transmission.sqrt(), 0.0)
    }
}

impl PhotonicDevice for MicroringResonator {
    fn simulate(&self, input: &OpticalField) -> Result<OpticalField> {
        let transmission = self.transmission_at_wavelength(input.wavelength);
        let mut output = input.clone();
        output.amplitude *= transmission;
        output.power = output.calculate_power();
        Ok(output)
    }
    
    fn parameters(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.geometry.radius,
            self.effective_index,
            self.quality_factor,
            self.tuning_parameter,
        ])
    }
    
    fn update_parameters(&mut self, params: &DVector<f64>) -> Result<()> {
        if params.len() >= 1 {
            self.geometry.radius = params[0].max(1e-6);
        }
        if params.len() >= 2 {
            self.effective_index = params[1];
        }
        if params.len() >= 3 {
            self.quality_factor = params[2].max(100.0);
        }
        if params.len() >= 4 {
            self.tuning_parameter = params[3];
        }
        Ok(())
    }
    
    fn parameter_names(&self) -> Vec<String> {
        vec![
            "radius".to_string(),
            "effective_index".to_string(),
            "quality_factor".to_string(),
            "tuning_parameter".to_string(),
        ]
    }
    
    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![
            (1e-6, 100e-6),    // radius
            (1.4, 3.5),        // effective index
            (100.0, 1e6),      // quality factor
            (-PI, PI),         // tuning parameter
        ]
    }
    
    fn parameter_gradients(&self, _input: &OpticalField, _grad_output: &OpticalField) -> Result<DVector<f64>> {
        Ok(DVector::zeros(4))
    }
    
    fn device_type(&self) -> &'static str {
        "microring_resonator"
    }
    
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl TunableDevice for MicroringResonator {
    fn get_tuning(&self) -> f64 {
        self.tuning_parameter
    }
    
    fn set_tuning(&mut self, tuning: f64) -> Result<()> {
        self.tuning_parameter = tuning;
        Ok(())
    }
    
    fn tuning_range(&self) -> (f64, f64) {
        (-PI, PI)
    }
    
    fn tuning_sensitivity(&self, wavelength: f64) -> f64 {
        // dλ/dtuning sensitivity
        wavelength / (2.0 * PI)
    }
}