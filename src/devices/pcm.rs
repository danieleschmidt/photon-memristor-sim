//! Phase Change Material device model

use crate::core::{Result, Complex64, OpticalField};
use crate::devices::traits::{PhotonicDevice, MemristiveDevice, SwitchingCharacteristics};
use nalgebra::DVector;


/// PCM material types
#[derive(Debug, Clone)]
pub enum PCMMaterial {
    GST, // Ge2Sb2Te5
    GSST, // Ge2Sb2Te5-SnTe
}

/// PCM device implementation
#[derive(Debug, Clone)]
pub struct PCMDevice {
    material: PCMMaterial,
    crystallinity: f64, // 0 = amorphous, 1 = crystalline
    temperature: f64,
    dimensions: (f64, f64, f64), // L x W x H
}

impl PCMDevice {
    pub fn new(material: PCMMaterial) -> Self {
        Self {
            material,
            crystallinity: 0.0,
            temperature: 300.0,
            dimensions: (200e-9, 50e-9, 10e-9),
        }
    }
}

impl PhotonicDevice for PCMDevice {
    fn simulate(&self, input: &OpticalField) -> Result<OpticalField> {
        // Simple transmission based on crystallinity
        let transmission = 0.8 + 0.2 * self.crystallinity;
        let mut output = input.clone();
        output.amplitude *= Complex64::new(transmission, 0.0);
        output.power = output.calculate_power();
        Ok(output)
    }
    
    fn parameters(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.crystallinity, self.temperature])
    }
    
    fn update_parameters(&mut self, params: &DVector<f64>) -> Result<()> {
        if params.len() >= 1 {
            self.crystallinity = params[0].max(0.0).min(1.0);
        }
        if params.len() >= 2 {
            self.temperature = params[1];
        }
        Ok(())
    }
    
    fn parameter_names(&self) -> Vec<String> {
        vec!["crystallinity".to_string(), "temperature".to_string()]
    }
    
    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0), (200.0, 1000.0)]
    }
    
    fn parameter_gradients(&self, _input: &OpticalField, _grad_output: &OpticalField) -> Result<DVector<f64>> {
        Ok(DVector::zeros(2))
    }
    
    fn device_type(&self) -> &'static str {
        "pcm"
    }
    
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl MemristiveDevice for PCMDevice {
    fn get_state(&self) -> f64 {
        self.crystallinity
    }
    
    fn set_state(&mut self, state: f64) -> Result<()> {
        self.crystallinity = state.max(0.0).min(1.0);
        Ok(())
    }
    
    fn state_bounds(&self) -> (f64, f64) {
        (0.0, 1.0)
    }
    
    fn update_state(&mut self, _optical_power: f64, _electrical_voltage: f64, _dt: f64) -> Result<()> {
        // Simplified state update
        Ok(())
    }
    
    fn switching_characteristics(&self) -> SwitchingCharacteristics {
        SwitchingCharacteristics::default()
    }
}

/// Crystallization model for PCM
pub struct CrystallizationModel;

impl CrystallizationModel {
    pub fn crystallization_rate(_temperature: f64) -> f64 {
        1e6 // Simplified rate
    }
}