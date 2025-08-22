//! Waveguide modeling and mode analysis

use crate::core::{PhotonicError, Result, Complex64, WaveguideGeometry};
use nalgebra::DMatrix;
use std::f64::consts::PI;

/// Waveguide mode structure
#[derive(Debug, Clone)]
pub struct WaveguideMode {
    /// Effective index
    pub effective_index: Complex64,
    /// Mode profile (normalized)
    pub field_profile: DMatrix<Complex64>,
    /// Mode number (0 = fundamental)
    pub mode_number: usize,
    /// Propagation constant
    pub beta: Complex64,
    /// Confinement factor
    pub confinement_factor: f64,
}

impl WaveguideMode {
    /// Create a new waveguide mode
    pub fn new(
        effective_index: Complex64,
        field_profile: DMatrix<Complex64>,
        mode_number: usize,
        wavelength: f64,
    ) -> Self {
        let k0 = 2.0 * PI / wavelength;
        let beta = effective_index * Complex64::new(k0, 0.0);
        
        // Calculate confinement factor (simplified)
        let total_power: f64 = field_profile.iter().map(|c| c.norm_sqr()).sum();
        let core_power: f64 = field_profile.iter().map(|c| c.norm_sqr()).sum(); // Simplified
        let confinement_factor = if total_power > 0.0 {
            core_power / total_power
        } else {
            0.0
        };
        
        Self {
            effective_index,
            field_profile,
            mode_number,
            beta,
            confinement_factor,
        }
    }
    
    /// Calculate group velocity
    pub fn group_velocity(&self, _wavelength: f64, _d_wavelength: f64) -> f64 {
        // Simplified calculation - would need dispersion data for accuracy
        let c = 299792458.0; // Speed of light
        let n_g = self.effective_index.re; // Group index approximation
        c / n_g
    }
    
    /// Calculate mode effective area
    pub fn effective_area(&self, dx: f64, dy: f64) -> f64 {
        let intensity_max = self.field_profile.iter()
            .map(|c| c.norm_sqr())
            .fold(0.0, f64::max);
        
        if intensity_max > 0.0 {
            let total_power: f64 = self.field_profile.iter()
                .map(|c| c.norm_sqr())
                .sum::<f64>() * dx * dy;
            
            (total_power * total_power) / (intensity_max * dx * dy)
        } else {
            0.0
        }
    }
}

/// Effective index calculator for waveguides
pub struct EffectiveIndexCalculator {
    geometry: WaveguideGeometry,
    wavelength: f64,
}

impl EffectiveIndexCalculator {
    /// Create new calculator
    pub fn new(geometry: WaveguideGeometry, wavelength: f64) -> Self {
        Self { geometry, wavelength }
    }
    
    /// Calculate effective index using Marcatili's method for rectangular waveguides
    pub fn calculate_effective_index(&self, mode_number: usize) -> Result<Complex64> {
        if mode_number > 10 {
            return Err(PhotonicError::invalid_parameter(
                "mode_number",
                mode_number,
                "≤ 10 for this implementation"
            ));
        }
        
        let k0 = 2.0 * PI / self.wavelength;
        let n_core = self.geometry.core_index.re;
        let n_clad = self.geometry.cladding_index.re;
        
        // Check if waveguide supports guided modes
        if n_core <= n_clad {
            return Err(PhotonicError::physics_violation(
                "Core index must be higher than cladding for guided modes"
            ));
        }
        
        // Normalized frequency (V-parameter)
        let v_param = k0 * self.geometry.width * (n_core * n_core - n_clad * n_clad).sqrt();
        
        // Single mode cutoff
        if v_param < 2.405 && mode_number == 0 {
            // Fundamental mode approximation
            let gamma = (k0 * k0 * (n_core * n_core - n_clad * n_clad)).sqrt();
            let effective_width = self.geometry.width + 2.0 / gamma;
            
            // Simple effective index approximation
            let n_eff = n_clad + (n_core - n_clad) * 
                (1.0 - (mode_number as f64 * PI / (effective_width * gamma)).cos());
            
            Ok(Complex64::new(n_eff, self.geometry.core_index.im))
        } else if mode_number == 0 {
            // Multimode case - fundamental mode
            let fraction = 1.0 - (2.405 / v_param).powi(2);
            let n_eff = n_clad + (n_core - n_clad) * fraction;
            Ok(Complex64::new(n_eff, self.geometry.core_index.im))
        } else {
            // Higher order modes
            let cutoff_v = mode_number as f64 * PI;
            if v_param > cutoff_v {
                let fraction = 1.0 - (cutoff_v / v_param).powi(2);
                let n_eff = n_clad + (n_core - n_clad) * fraction;
                Ok(Complex64::new(n_eff, self.geometry.core_index.im))
            } else {
                Err(PhotonicError::physics_violation(
                    format!("Mode {} is beyond cutoff", mode_number)
                ))
            }
        }
    }
    
    /// Generate mode profile using Gaussian approximation
    pub fn generate_mode_profile(
        &self,
        _effective_index: Complex64,
        nx: usize,
        ny: usize,
    ) -> Result<DMatrix<Complex64>> {
        let width_x = 3.0 * self.geometry.width; // Simulation window
        let width_y = 3.0 * self.geometry.height;
        
        let dx = width_x / nx as f64;
        let dy = width_y / ny as f64;
        
        let mut profile = DMatrix::zeros(ny, nx);
        
        // Mode width approximation
        let _k0 = 2.0 * PI / self.wavelength;
        let n_core = self.geometry.core_index.re;
        let n_clad = self.geometry.cladding_index.re;
        let delta = (n_core * n_core - n_clad * n_clad) / (2.0 * n_core * n_core);
        
        let mode_width_x = self.geometry.width / (2.0 * (2.0 * delta).sqrt());
        let mode_width_y = self.geometry.height / (2.0 * (2.0 * delta).sqrt());
        
        for i in 0..ny {
            for j in 0..nx {
                let x = (j as f64 - nx as f64 / 2.0) * dx;
                let y = (i as f64 - ny as f64 / 2.0) * dy;
                
                // Gaussian mode approximation
                let amplitude = (-((x / mode_width_x).powi(2) + (y / mode_width_y).powi(2))).exp();
                profile[(i, j)] = Complex64::new(amplitude, 0.0);
            }
        }
        
        // Normalize
        let total_power: f64 = profile.iter().map(|c| c.norm_sqr()).sum();
        if total_power > 0.0 {
            let norm_factor = 1.0 / total_power.sqrt();
            profile *= Complex64::new(norm_factor, 0.0);
        }
        
        Ok(profile)
    }
    
    /// Find all supported modes
    pub fn find_all_modes(&self, max_modes: usize) -> Result<Vec<WaveguideMode>> {
        let mut modes = Vec::new();
        
        for mode_num in 0..max_modes {
            match self.calculate_effective_index(mode_num) {
                Ok(n_eff) => {
                    let profile = self.generate_mode_profile(n_eff, 64, 64)?;
                    let mode = WaveguideMode::new(n_eff, profile, mode_num, self.wavelength);
                    modes.push(mode);
                }
                Err(_) => break, // Mode beyond cutoff
            }
        }
        
        if modes.is_empty() {
            return Err(PhotonicError::physics_violation(
                "No guided modes found in this waveguide"
            ));
        }
        
        Ok(modes)
    }
}

impl PhotonicError {
    pub fn physics_violation(description: impl Into<String>) -> Self {
        Self::PhysicsViolation {
            description: description.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_silicon_waveguide_mode() {
        let geometry = WaveguideGeometry::silicon_photonic_standard();
        let calculator = EffectiveIndexCalculator::new(geometry, 1550e-9);
        
        let n_eff = calculator.calculate_effective_index(0).unwrap();
        
        // Effective index should be between core and cladding
        assert!(n_eff.re > 1.44); // > cladding
        assert!(n_eff.re < 3.47); // < core
    }
    
    #[test]
    fn test_mode_profile_generation() {
        let geometry = WaveguideGeometry::silicon_photonic_standard();
        let calculator = EffectiveIndexCalculator::new(geometry, 1550e-9);
        
        let n_eff = calculator.calculate_effective_index(0).unwrap();
        let profile = calculator.generate_mode_profile(n_eff, 32, 32).unwrap();
        
        assert_eq!(profile.shape(), (32, 32));
        
        // Check normalization
        let total_power: f64 = profile.iter().map(|c| c.norm_sqr()).sum();
        assert!((total_power - 1.0).abs() < 1e-10);
    }
}