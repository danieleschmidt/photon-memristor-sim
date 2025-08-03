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

/// Crystallization model for PCM based on Johnson-Mehl-Avrami equation
pub struct CrystallizationModel {
    pub activation_energy: f64,    // eV
    pub pre_exponential: f64,      // Hz
    pub avrami_exponent: f64,      // dimensionless
}

impl CrystallizationModel {
    /// GST material parameters
    pub fn gst() -> Self {
        Self {
            activation_energy: 2.2,     // eV
            pre_exponential: 1e13,      // Hz
            avrami_exponent: 2.5,       // typical for nucleation-growth
        }
    }
    
    /// GSST material parameters  
    pub fn gsst() -> Self {
        Self {
            activation_energy: 1.8,     // eV (lower than GST)
            pre_exponential: 1e12,      // Hz
            avrami_exponent: 2.2,
        }
    }
    
    /// Calculate crystallization rate at given temperature
    pub fn crystallization_rate(&self, temperature: f64) -> f64 {
        let k_b = 8.617e-5; // eV/K
        let exp_factor = (-self.activation_energy / (k_b * temperature)).exp();
        self.pre_exponential * exp_factor
    }
    
    /// Calculate crystalline fraction evolution (JMA equation)
    pub fn crystalline_fraction(&self, temperature: f64, time: f64, current_fraction: f64) -> f64 {
        let rate = self.crystallization_rate(temperature);
        let avrami_time = (rate * time).powf(self.avrami_exponent);
        let target_fraction = 1.0 - (-avrami_time).exp();
        
        // Interpolate towards target based on kinetics
        let alpha = 1.0 - (-rate * time).exp();
        current_fraction + alpha * (target_fraction - current_fraction)
    }
}

/// Thermal model for PCM heating
pub struct ThermalModel {
    pub thermal_conductivity: f64,    // W/m/K
    pub specific_heat: f64,           // J/kg/K  
    pub density: f64,                 // kg/m³
    pub melting_point: f64,           // K
    pub crystallization_temp: f64,    // K
}

impl ThermalModel {
    pub fn gst() -> Self {
        Self {
            thermal_conductivity: 0.5,    // W/m/K
            specific_heat: 200.0,         // J/kg/K
            density: 6150.0,              // kg/m³
            melting_point: 888.0,         // K
            crystallization_temp: 423.0,  // K (150°C)
        }
    }
    
    /// Calculate temperature rise from optical power
    pub fn temperature_rise(&self, power: f64, volume: f64, pulse_duration: f64) -> f64 {
        let energy = power * pulse_duration;
        let mass = self.density * volume;
        let thermal_capacity = mass * self.specific_heat;
        
        // Simplified: neglect heat diffusion for short pulses
        energy / thermal_capacity
    }
    
    /// Check if temperature enables phase transition
    pub fn phase_transition_enabled(&self, temperature: f64) -> (bool, bool) {
        let can_melt = temperature > self.melting_point;
        let can_crystallize = temperature > self.crystallization_temp && temperature < self.melting_point;
        (can_melt, can_crystallize)
    }
}

/// Optical constants model for PCM materials
pub struct OpticalModel {
    pub amorphous_n: Complex64,
    pub crystalline_n: Complex64,
    pub wavelength: f64,
}

impl OpticalModel {
    /// GST optical constants at 1550nm
    pub fn gst_1550nm() -> Self {
        Self {
            amorphous_n: Complex64::new(4.0, 0.1),     // n + ik (amorphous)
            crystalline_n: Complex64::new(6.5, 0.5),   // n + ik (crystalline)
            wavelength: 1550e-9,
        }
    }
    
    /// Calculate effective refractive index based on crystallinity
    pub fn effective_index(&self, crystallinity: f64) -> Complex64 {
        // Linear interpolation between amorphous and crystalline
        self.amorphous_n * (1.0 - crystallinity) + self.crystalline_n * crystallinity
    }
    
    /// Calculate transmission coefficient
    pub fn transmission(&self, crystallinity: f64, thickness: f64) -> Complex64 {
        let n_eff = self.effective_index(crystallinity);
        let k = 2.0 * std::f64::consts::PI * n_eff.im / self.wavelength;
        let absorption = (-k * thickness).exp();
        
        // Fresnel transmission (simplified, normal incidence)
        let r = ((n_eff - Complex64::new(1.0, 0.0)) / (n_eff + Complex64::new(1.0, 0.0))).norm_sqr();
        let fresnel_factor = (1.0 - r).sqrt();
        
        Complex64::new(fresnel_factor * absorption, 0.0)
    }
}