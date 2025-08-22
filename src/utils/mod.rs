//! Utility functions and helpers

use crate::core::{Complex64, OpticalField};
use nalgebra::{DMatrix, DVector};

/// Constants and unit conversions
pub mod constants {
    /// Speed of light in vacuum (m/s)
    pub const SPEED_OF_LIGHT: f64 = 299792458.0;
    
    /// Planck's constant (J⋅s)
    pub const PLANCK_CONSTANT: f64 = 6.62607015e-34;
    
    /// Electron charge (C)
    pub const ELECTRON_CHARGE: f64 = 1.602176634e-19;
    
    /// Boltzmann constant (J/K)
    pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;
}

/// Unit conversion utilities
pub fn wavelength_to_frequency(wavelength: f64) -> f64 {
    constants::SPEED_OF_LIGHT / wavelength
}

pub fn frequency_to_wavelength(frequency: f64) -> f64 {
    constants::SPEED_OF_LIGHT / frequency
}

pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 10.0)
}

pub fn linear_to_db(linear: f64) -> f64 {
    10.0 * linear.log10()
}

pub fn wavelength_to_energy(wavelength: f64) -> f64 {
    constants::PLANCK_CONSTANT * constants::SPEED_OF_LIGHT / wavelength
}

/// Create Gaussian beam profile
pub fn create_gaussian_beam(
    nx: usize,
    ny: usize,
    beam_waist: f64,
    wavelength: f64,
    power: f64,
) -> OpticalField {
    let width_x = 5.0 * beam_waist;
    let width_y = 5.0 * beam_waist;
    
    let dx = width_x / nx as f64;
    let dy = width_y / ny as f64;
    
    let mut amplitude = DMatrix::zeros(ny, nx);
    
    for i in 0..ny {
        for j in 0..nx {
            let x = ((j as f64) - (nx as f64 / 2.0)) * dx;
            let y = ((i as f64) - (ny as f64 / 2.0)) * dy;
            
            let r_squared = x * x + y * y;
            let field_amplitude = (-2.0 * r_squared / (beam_waist * beam_waist)).exp();
            
            amplitude[(i, j)] = Complex64::new(field_amplitude, 0.0);
        }
    }
    
    // Normalize to specified power
    let total_intensity: f64 = amplitude.iter().map(|c| c.norm_sqr()).sum();
    let norm_factor = (power / (total_intensity * dx * dy)).sqrt();
    amplitude *= Complex64::new(norm_factor, 0.0);
    
    let x_coords = DVector::from_fn(nx, |i, _| ((i as f64) - (nx as f64 / 2.0)) * dx);
    let y_coords = DVector::from_fn(ny, |i, _| ((i as f64) - (ny as f64 / 2.0)) * dy);
    
    OpticalField::new(amplitude, wavelength, power, x_coords, y_coords)
}

/// Calculate effective area from field profile
pub fn effective_area(field: &OpticalField) -> f64 {
    let intensity_max = field.amplitude.iter()
        .map(|c| c.norm_sqr())
        .fold(0.0, f64::max);
    
    if intensity_max > 0.0 {
        let dx = if field.coordinates.0.len() > 1 {
            field.coordinates.0[1] - field.coordinates.0[0]
        } else { 1.0 };
        let dy = if field.coordinates.1.len() > 1 {
            field.coordinates.1[1] - field.coordinates.1[0]
        } else { 1.0 };
        
        let total_power = field.calculate_power();
        (total_power * total_power) / (intensity_max * dx * dy)
    } else {
        0.0
    }
}

/// Generate random phase screen for atmospheric turbulence
pub fn generate_phase_screen(
    nx: usize,
    ny: usize,
    _coherence_length: f64,
    phase_variance: f64,
) -> DMatrix<f64> {
    let mut phase_screen = DMatrix::zeros(ny, nx);
    
    // Simple random phase generation
    // Real implementation would use Kolmogorov turbulence spectrum
    for i in 0..ny {
        for j in 0..nx {
            let random_value: f64 = rand::random::<f64>() - 0.5;
            phase_screen[(i, j)] = random_value * phase_variance;
        }
    }
    
    phase_screen
}

/// Apply phase modulation to optical field
pub fn apply_phase_modulation(field: &mut OpticalField, phase: &DMatrix<f64>) {
    if field.amplitude.shape() != phase.shape() {
        return; // Dimension mismatch
    }
    
    for i in 0..field.amplitude.nrows() {
        for j in 0..field.amplitude.ncols() {
            let phase_factor = Complex64::new(0.0, phase[(i, j)]).exp();
            field.amplitude[(i, j)] *= phase_factor;
        }
    }
}

/// Calculate overlap integral between two fields
pub fn overlap_integral(field1: &OpticalField, field2: &OpticalField) -> Complex64 {
    if field1.amplitude.shape() != field2.amplitude.shape() {
        return Complex64::new(0.0, 0.0);
    }
    
    let dx = if field1.coordinates.0.len() > 1 {
        field1.coordinates.0[1] - field1.coordinates.0[0]
    } else { 1.0 };
    let dy = if field1.coordinates.1.len() > 1 {
        field1.coordinates.1[1] - field1.coordinates.1[0]
    } else { 1.0 };
    
    let mut overlap = Complex64::new(0.0, 0.0);
    
    for i in 0..field1.amplitude.nrows() {
        for j in 0..field1.amplitude.ncols() {
            let e1 = field1.amplitude[(i, j)];
            let e2_conj = field2.amplitude[(i, j)].conj();
            overlap += e1 * e2_conj * dx * dy;
        }
    }
    
    overlap
}

/// Performance profiler for benchmarking
pub struct Profiler {
    start_time: std::time::Instant,
    measurements: std::collections::HashMap<String, f64>,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            measurements: std::collections::HashMap::new(),
        }
    }
    
    pub fn time<F, R>(&mut self, name: &str, f: F) -> R 
    where 
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed().as_secs_f64();
        self.measurements.insert(name.to_string(), duration);
        result
    }
    
    pub fn get_measurement(&self, name: &str) -> Option<f64> {
        self.measurements.get(name).copied()
    }
    
    pub fn total_time(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
    
    pub fn print_summary(&self) {
        println!("Performance Summary:");
        println!("Total time: {:.3} ms", self.total_time() * 1000.0);
        for (name, duration) in &self.measurements {
            println!("  {}: {:.3} ms", name, duration * 1000.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_unit_conversions() {
        let wavelength = 1550e-9; // 1550 nm
        let frequency = wavelength_to_frequency(wavelength);
        let back_to_wavelength = frequency_to_wavelength(frequency);
        
        assert!((wavelength - back_to_wavelength).abs() < 1e-15);
    }
    
    #[test]
    fn test_db_conversions() {
        let linear = 100.0;
        let db = linear_to_db(linear);
        let back_to_linear = db_to_linear(db);
        
        assert!((linear - back_to_linear).abs() < 1e-10);
    }
    
    #[test]
    fn test_gaussian_beam() {
        let beam = create_gaussian_beam(32, 32, 10e-6, 1550e-9, 1e-3);
        
        assert_eq!(beam.dimensions(), (32, 32));
        assert!((beam.power - 1e-3).abs() < 1e-6);
        assert_eq!(beam.wavelength, 1550e-9);
    }
    
    #[test]
    fn test_effective_area() {
        let beam = create_gaussian_beam(32, 32, 10e-6, 1550e-9, 1e-3);
        let area = effective_area(&beam);
        
        assert!(area > 0.0);
        // For Gaussian beam, effective area should be related to beam waist
        assert!(area > 1e-10 && area < 1e-8); // Reasonable range
    }
    
    #[test]
    fn test_profiler() {
        let mut profiler = Profiler::new();
        
        let result = profiler.time("test_operation", || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(profiler.get_measurement("test_operation").unwrap() >= 0.009); // ~10ms
    }
}